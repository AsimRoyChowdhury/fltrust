import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset, Dataset
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 30, kernel_size=3) 
        self.pool = nn.MaxPool2d(2, 2)               
        self.conv2 = nn.Conv2d(30, 50, kernel_size=3)
        self.fc1 = nn.Linear(50 * 5 * 5, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 50 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class BackdoorDataset(Dataset):
    def __init__(self, dataset, target_label=2):
        self.dataset = dataset
        self.target_label = target_label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = image.clone()
        image[:, -2:, -2:] = 2.5 
        return image, self.target_label

def prepare_datasets(num_clients=100, batch_size=32):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = MNIST(root="./data", train=True, download=True, transform=transform)
    
    # 1. Root Dataset (First 100 samples)
    root_dataset = Subset(train_dataset, range(0, 100))
    root_loader = DataLoader(root_dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Client Datasets (Remaining partitioned data)
    client_dataset = Subset(train_dataset, range(100, len(train_dataset)))
    partition_size = len(client_dataset) // num_clients
    lengths = [partition_size] * num_clients
    lengths[-1] += len(client_dataset) - sum(lengths)
    
    partitions = random_split(client_dataset, lengths, generator=torch.Generator().manual_seed(42))
    client_loaders = [DataLoader(p, batch_size=batch_size, shuffle=True) for p in partitions]
    
    # 3. Clean Test Dataset (For Testing Error Rate)
    test_dataset = MNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 4. Backdoor Test Dataset (For Attack Success Rate)
    non_target_indices = [i for i in range(len(test_dataset)) if test_dataset[i][1] != 2]
    filtered_test_dataset = Subset(test_dataset, non_target_indices)
    backdoor_dataset = BackdoorDataset(filtered_test_dataset, target_label=2)
    backdoor_test_loader = DataLoader(backdoor_dataset, batch_size=batch_size, shuffle=False)
    
    return root_loader, client_loaders, test_loader, backdoor_test_loader

def evaluate_model(model, dataloader, metric="TER"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = correct / total
    
    # ASR is the accuracy on the triggered dataset; TER = (1 - accuracy) on the clean dataset
    if metric == "ASR":
        return accuracy 
    return 1.0 - accuracy