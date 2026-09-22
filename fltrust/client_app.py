import flwr as fl
import torch
import torch.nn as nn
from collections import OrderedDict
from fltrust.task import MNISTNet

class MaliciousFLClient(fl.client.NumPyClient):
    def __init__(self, dataloader, attack_simulator=None, lr=0.006):
        self.dataloader = dataloader
        # Convert dataloader to an iterator to sample a single batch per round
        self.data_iterator = iter(self.dataloader)
        self.attack_simulator = attack_simulator
        self.model = MNISTNet()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.lr = lr

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def _get_next_batch(self):
        try:
            return next(self.data_iterator)
        except StopIteration:
            self.data_iterator = iter(self.dataloader)
            return next(self.data_iterator)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        self.model.train()
        
        # FLTrust constraint: R_l = 1 (Exactly one batch per round)
        images, labels = self._get_next_batch()
        
        if self.attack_simulator:
            images, labels = self.attack_simulator.apply_data_poisoning(images, labels)

        images, labels = images.to(self.device), labels.to(self.device)
        optimizer.zero_grad()
        loss = criterion(self.model(images), labels)
        loss.backward()
        optimizer.step()

        trained_weights = self.get_parameters()

        if self.attack_simulator:
            final_weights = self.attack_simulator.apply_model_poisoning(
                local_weights=trained_weights,
                global_weights=parameters
            )
        else:
            final_weights = trained_weights

        return final_weights, len(labels), {}