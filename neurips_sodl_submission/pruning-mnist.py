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

dataset = 'MNIST' # 'MNIST' or 'CIFAR10'

if dataset == 'MNIST':
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = MNIST(root='.', train=True, download=True, transform=transform)
    test_dataset = MNIST(root='.', train=False, download=True, transform=transform)
elif dataset == 'CIFAR10':
    transform = torchvision.transforms.ToTensor()
    train_dataset = CIFAR10(root='.', train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root='.', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# for MNIST

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 64, bias=False)
        self.fc2 = nn.Linear(64, 64, bias=False)
        self.fc3 = nn.Linear(64, 10, bias=False)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# for CIFAR10

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
    
def new_model(dataset, device):
    if dataset == 'MNIST':
        model = MLP()
    elif dataset == 'CIFAR10':
        model = CNN()
    model = model.to(device)
    return model        

path: str = f'results/{dataset}/'

# load the model and cluster indices
model = new_model(dataset, device)
model.load_state_dict(torch.load(f'{path}model.pth'))
cluster_U_indices = torch.load(f'{path}cluster_U_indices.pth')
cluster_V_indices = torch.load(f'{path}cluster_V_indices.pth')

unclustered_model = new_model(dataset, device)
unclustered_model.load_state_dict(torch.load(f'{path}unclustered_model.pth'))

def fast_label_perf(model, x, label):
    with torch.no_grad():
        output = model(x)
        criterion = nn.CrossEntropyLoss()
        target = torch.tensor([label] * x.size(0)).to(device)
        loss = criterion(output, target)
        accuracy = (output.argmax(dim=1) == target).sum().item() / x.size(0)
    return loss, accuracy

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
                    if (loss_pruned - loss_original) > 0:
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
# this will take a while to run, and it would be better to run this on a script in the background

ecs_pruned_all_labels = []
ecs_pruned_unclustered_all_labels = []

for label in tqdm.trange(10):
    label_data = torch.stack([test_dataset[i][0] for i in range(len(test_dataset)) if test_dataset[i][1] == label])
    label_data = label_data.to(device)
    
    pruned_model = prune_model(model, label_data, label, device, verbose=False)
    pruned_model_unclustered = prune_model(unclustered_model, label_data, label, device, verbose=False)
    
    ecs_pruned_all_labels.append(effective_circuit_size(pruned_model))
    ecs_pruned_unclustered_all_labels.append(effective_circuit_size(pruned_model_unclustered))

    # print the effective circuit sizes for each label
    print(f'Label: {label}, ECS (pruned): {ecs_pruned_all_labels[-1]}, ECS (pruned unclustered): {ecs_pruned_unclustered_all_labels[-1]}')

torch.save(ecs_pruned_all_labels, path + 'ecs_pruned_all_labels.pth')
torch.save(ecs_pruned_unclustered_all_labels, path + 'ecs_pruned_unclustered_all_labels.pth')

print(f'EVERYTHING DONE!')