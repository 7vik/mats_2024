import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots
from IPython.display import clear_output
from collections import defaultdict
from itertools import islice
import random
import time
from pathlib import Path
import math

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device

def randomseed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(3072, 64, bias=False)
        self.fc2 = nn.Linear(64, 64, bias=False)
        self.fc3 = nn.Linear(64, 10, bias=False)

    def forward(self, x):
        x = x.view(-1, 3072)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def accuracy(model, data):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data:
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()
    return correct / total

def classwise_accuracy(model, data):
    model.eval()
    correct = defaultdict(int)
    total = defaultdict(int)
    with torch.no_grad():
        for images, labels in data:
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            for i in range(len(labels)):
                label = labels[i].item()
                total[label] += 1
                correct[label] += int(predicted[i] == label)
    return [round(correct[i] / total[i], 3) if total[i] > 0 else 0 for i in range(10)]

train_dataset = CIFAR10(root='.', train=True, download=True, transform=torchvision.transforms.ToTensor())
test_dataset = CIFAR10(root='.', train=False, download=True, transform=torchvision.transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

def cluster_goodness_fast(model, cluster_U_indices, cluster_V_indices, num_clusters):
    A = model.fc2.weight ** 2
    mask = torch.zeros_like(A, dtype=torch.bool)
    
    for cluster_idx in range(num_clusters):
        u_indices = torch.tensor(cluster_U_indices[cluster_idx], dtype=torch.long)
        v_indices = torch.tensor(cluster_V_indices[cluster_idx], dtype=torch.long)
        mask[u_indices.unsqueeze(1), v_indices] = True
    
    intra_cluster_out_sum = torch.sum(A[mask])
    total_out_sum = torch.sum(A)
    
    return intra_cluster_out_sum / total_out_sum

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 16 * 16, 64, bias=False)
        self.fc2 = nn.Linear(64, 64, bias=False)
        self.fc3 = nn.Linear(64, 10, bias=False)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 16 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# model = MLP()
model = CNN()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
ce_losses = []
cluster_losses = []
lomda = 0.7
epochs = 21

# equally divide fc2 into 4 clusters (64 / 4 = 16) as a dict

num_clusters = 4

cluster_size = int(model.fc2.weight.shape[0] / num_clusters)

cluster_U_indices = {i: list(range(i * cluster_size, (i + 1) * cluster_size)) for i in range(num_clusters)}
cluster_V_indices = {i: list(range(i * cluster_size, (i + 1) * cluster_size)) for i in range(num_clusters)}

randomseed(42)

for epoch in range(epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss_ce = criterion(output, target)
        loss_cluster = cluster_goodness_fast(model, cluster_U_indices, cluster_V_indices, num_clusters)
        cluster_losses.append(loss_cluster.item())
        ce_losses.append(loss_ce.item())
        loss = loss_ce - lomda * loss_cluster
        loss.backward()
        optimizer.step()
        if batch_idx % 900 == 0:
            acc = accuracy(model, test_loader)
            print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, Clusterability: {loss_cluster.item():.4f}, Accuracy: {acc:.4f}')

# model_unclustered = MLP()
model_unclustered = CNN()
model_unclustered.to(device)

# Train the model without clustering
optimizer = optim.Adam(model_unclustered.parameters(), lr=1e-3)
ce_losses_unclustered = []
epochs = 10

randomseed(42)

criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    model_unclustered.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model_unclustered(data)
        loss = criterion(output, target)
        ce_losses_unclustered.append(loss.item())
        loss.backward()
        optimizer.step()
        if batch_idx % 900 == 0:
            acc = accuracy(model_unclustered, test_loader)
            print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, Accuracy: {acc:.4f}')

path: str = 'cifar_checkpoints/'

Path(path).mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), path + 'model.pth')
# store the cluster indices
torch.save(cluster_U_indices, path + 'cluster_U_indices.pth')
torch.save(cluster_V_indices, path + 'cluster_V_indices.pth')

model = CNN()
model.load_state_dict(torch.load(path + 'model.pth'))
model.to(device)
cluster_U_indices = torch.load(path + 'cluster_U_indices.pth')
cluster_V_indices = torch.load(path + 'cluster_V_indices.pth')

# Evaluate the model
def evaluate_model(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data.view(-1, 784))
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return correct / total


def evaluate_model_for_label(model, test_loader, label):
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # filter out the data points with the given label
            mask = target == label
            data = data[mask]
            target = target[mask]
            output = model(data.view(-1, 784))
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return correct / total if total > 0 else 0


def fast_label_perf(model, x, label):
    with torch.no_grad():
        output = model(x)
        criterion = nn.CrossEntropyLoss()
        target = torch.tensor([label] * x.size(0)).to(device)
        loss = criterion(output, target)
        accuracy = (output.argmax(dim=1) == target).sum().item() / x.size(0)
    return loss, accuracy


import torch
import torch.nn as nn
import copy

def prune_model(model, x, label, device, verbose=False):
    pruned_model = copy.deepcopy(model)
    
    layers = list(pruned_model.children())
    for layer in reversed(layers):
        if verbose:
            print(f'Pruning layer: {layer}')
        if isinstance(layer, nn.Linear):
            for neuron_idx in tqdm.trange(layer.weight.shape[0]):
                for weight_idx in range(layer.weight.shape[1]):
                    # Create a mask to zero out the weight
                    mask = torch.ones_like(layer.weight)
                    mask[neuron_idx, weight_idx] = 0
                    
                    # Apply the mask
                    original_weight = layer.weight[neuron_idx, weight_idx].item()
                    layer.weight.data[neuron_idx, weight_idx] = 0
                    
                    # Check performance
                    loss_pruned, acc_pruned = fast_label_perf(pruned_model, x, label)
                    loss_original, acc_original = fast_label_perf(model, x, label)
                    
                    # If performance decreases, restore the weight
                    if loss_pruned > loss_original or acc_pruned < acc_original:
                        layer.weight.data[neuron_idx, weight_idx] = original_weight

        # fraction pruned
        num_zeros = torch.sum(layer.weight == 0).item()
        total_params = layer.weight.numel()
        if verbose:
            print(f'Fraction pruned: {num_zeros / total_params:.4f}')
                    
    return pruned_model


def effective_circuit_size(model):
    # fraction of non-zero weights
    total_params = 0
    num_zeros = 0
    for layer in model.children():
        if isinstance(layer, nn.Linear):
            total_params += layer.weight.numel()
            num_zeros += torch.sum(layer.weight == 0).item()
    return round(1 - (num_zeros / total_params), 3)


# effective circuit sizes for pruned models v pruned unclustered models for each label
# this will take a while to run

ecs_pruned_all_labels = []
ecs_pruned_unclustered_all_labels = []

for label in tqdm.trange(10):
    label_data = torch.stack([test_dataset[i][0] for i in range(len(test_dataset)) if test_dataset[i][1] == label])
    label_data = label_data.to(device)
    
    pruned_model = prune_model(model, label_data, label, device, verbose=False)
    pruned_model_unclustered = prune_model(model_unclustered, label_data, label, device, verbose=False)
    
    ecs_pruned_all_labels.append(effective_circuit_size(pruned_model))
    ecs_pruned_unclustered_all_labels.append(effective_circuit_size(pruned_model_unclustered))

    # print the effective circuit sizes for each label
    print(f'Label: {label}, ECS (pruned): {ecs_pruned_all_labels[-1]}, ECS (pruned unclustered): {ecs_pruned_unclustered_all_labels[-1]}')

path = 'cifar_results/'

Path(path).mkdir(parents=True, exist_ok=True)
torch.save(ecs_pruned_all_labels, path + 'ecs_pruned_all_labels.pth')
torch.save(ecs_pruned_unclustered_all_labels, path + 'ecs_pruned_unclustered_all_labels.pth')

print(f'EVERYTHING DONE!')