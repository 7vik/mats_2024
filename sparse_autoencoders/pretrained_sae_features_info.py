from datasets import load_dataset  
from transformer_lens import HookedTransformer
from sae_lens import SAE
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import einops
import tqdm
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f'{device}')
model = HookedTransformer.from_pretrained('gemma-2b', device=device)
model.to(device)
print('DonE! Happy InterpreTING!!')

from transformer_lens.utils import tokenize_and_concatenate

dataset = load_dataset(
    path = "NeelNanda/pile-10k",
    split="train",
    streaming=False,
)

batch_str = []
for i in tqdm.trange(len(dataset['text'][:500])):
    batch_str.append(dataset['text'][i][:128])

name = 'blocks.3.hook_resid_post'

def activation_filter(name: str) -> bool:
    return name.endswith('blocks.3.hook_resid_post')

from sae_lens import SAE

sae, cfg_dict, sparsity = SAE.from_pretrained(
    release = 'gemma-2b-res-jb',
    sae_id = 'blocks.6.hook_resid_post',
    device = device
)

num_features = 1000
feature_dim = model.cfg.d_model
# features = torch.randn(num_features, feature_dim).to(device)
# take features from rows of the decoder
features = sae.W_dec[:num_features].to(device)
features = F.normalize(features, p=2, dim=-1)

def activation_turn_off(activations, hook, idx=1):
    global features
    direction = features[idx]
    activations = activations.clone()
    component = activations @ direction
    activations = activations - component[:, :, None] * direction[None, None, :]
    return activations

def entropy(probs):
    eps = 1e-8
    return -torch.sum(probs * torch.log(probs + eps), dim=-1)

torch.cuda.empty_cache()

batch_size = 8
information_gains = [0 for _ in range(num_features)]

for feature_idx in tqdm.trange(num_features):
    for i in tqdm.trange(0, len(batch_str), batch_size):
        str_batch = batch_str[i:i + batch_size]
        model.reset_hooks()
        logits = model.run_with_hooks(str_batch, return_type='logits')
        probs = F.softmax(logits, dim=-1)[:, :, :]
        entropies_before = entropy(probs)
        torch.cuda.empty_cache()
        logits_after = model.run_with_hooks(str_batch, return_type='logits', fwd_hooks=[(
            activation_filter,
            lambda x, hook: activation_turn_off(x, hook, idx=feature_idx)
            )])
        probs_after = F.softmax(logits_after, dim=-1)[:, :, :]
        entropies_after = entropy(probs_after)
        torch.cuda.empty_cache()
        diff_entropies = entropies_before - entropies_after
        feature_importance = - diff_entropies.sum()
        information_gains[feature_idx] += feature_importance.item()
        torch.cuda.empty_cache()

# normalize information gains
information_gains_ = [ig / len(batch_str) for ig in information_gains]

# store them in a txt file
with open('information_gains.txt', 'w') as f:
    for ig in information_gains_:
        f.write(f'{ig}\n')