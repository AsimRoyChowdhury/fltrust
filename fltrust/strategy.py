import flwr as fl
import numpy as np
import copy
import torch
from collections import OrderedDict
from fltrust.defences import aggregate_krum, aggregate_trimmed_mean, aggregate_median

class FLTrustStrategy(fl.server.strategy.FedAvg):
    def __init__(self, defense_type, global_model, root_loader, num_malicious=20, lr=0.006, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.defense_type = defense_type
        self.global_model = global_model
        self.root_loader = root_loader
        self.root_iterator = iter(self.root_loader)
        self.num_malicious = num_malicious 
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.global_model.to(self.device)
        self.lr = lr
        self.global_weights = [val.cpu().numpy() for _, val in self.global_model.state_dict().items()]

    def _get_next_root_batch(self):
        try:
            return next(self.root_iterator)
        except StopIteration:
            self.root_iterator = iter(self.root_loader)
            return next(self.root_iterator)

    def _get_server_update(self):
        server_model = copy.deepcopy(self.global_model)
        server_model.to(self.device)
        optimizer = torch.optim.SGD(server_model.parameters(), lr=self.lr)
        server_model.train()
        
        # FLTrust constraint: R_l = 1 (Exactly one batch per round)[cite: 5]
        images, labels = self._get_next_root_batch()
        images, labels = images.to(self.device), labels.to(self.device)
        
        optimizer.zero_grad()
        loss = torch.nn.CrossEntropyLoss()(server_model(images), labels)
        loss.backward()
        optimizer.step()
            
        g_0_list = [(p_old - p_new.cpu().numpy()).flatten() for p_new, p_old in zip(server_model.state_dict().values(), self.global_weights)]
        return np.concatenate(g_0_list)

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        # Extract client gradients (g_i = w - w_i)[cite: 5]
        client_gradients = []
        for client, fit_res in results:
            w_i = fl.common.parameters_to_ndarrays(fit_res.parameters)
            g_i_list = [(layer_global - layer_i).flatten() for layer_i, layer_global in zip(w_i, self.global_weights)]
            client_gradients.append(np.concatenate(g_i_list))
            
        if self.defense_type == "FedAvg":
            aggregated_gradient = np.mean(client_gradients, axis=0)
            
        elif self.defense_type == "Krum":
            aggregated_gradient = aggregate_krum(client_gradients, self.num_malicious)
            
        elif self.defense_type == "Trim-mean":
            aggregated_gradient = aggregate_trimmed_mean(client_gradients, self.num_malicious)
            
        elif self.defense_type == "Median":
            aggregated_gradient = aggregate_median(client_gradients)
            
        elif self.defense_type == "FLTrust":
            g_0 = self._get_server_update()
            norm_g_0 = np.linalg.norm(g_0)
            
            trust_scores, normalized_gradients = [], []
            for g_i in client_gradients:
                norm_g_i = np.linalg.norm(g_i)
                # ReLU-clipped cosine similarity based trust score[cite: 5]
                c_i = np.dot(g_i, g_0) / (norm_g_i * norm_g_0) if norm_g_i > 0 and norm_g_0 > 0 else 0.0
                ts_i = max(0.0, float(c_i))
                trust_scores.append(ts_i)
                
                # Normalizing the magnitudes to match the server model update[cite: 5]
                g_bar_i = (norm_g_0 / norm_g_i) * g_i if norm_g_i > 0 else g_i
                normalized_gradients.append(g_bar_i)
                
            total_trust = sum(trust_scores)
            if total_trust == 0:
                aggregated_gradient = np.zeros_like(g_0)
            else:
                # Weighted aggregation[cite: 5]
                aggregated_gradient = sum((ts / total_trust) * g_bar for ts, g_bar in zip(trust_scores, normalized_gradients))

        # Reconstruct layers and apply the global update: w = w - g (since g was extracted as w - w_i)
        pointer, new_global_weights = 0, []
        for layer_global in self.global_weights:
            num_elements = layer_global.size
            layer_grad = aggregated_gradient[pointer:pointer+num_elements].reshape(layer_global.shape)
            new_global_weights.append(layer_global - layer_grad)
            pointer += num_elements
            
        self.global_weights = new_global_weights
        
        # Load back into PyTorch model memory
        params_dict = zip(self.global_model.state_dict().keys(), self.global_weights)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.global_model.load_state_dict(state_dict, strict=True)
        
        return fl.common.ndarrays_to_parameters(self.global_weights), {}