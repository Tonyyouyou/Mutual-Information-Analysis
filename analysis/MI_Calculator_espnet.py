import torch
import torch.nn as nn
from mine.mine import Mine
import numpy as np
import os
import pdb
import pickle

x_dim = 768
y_dim = 768

statistics_network = nn.Sequential(
    nn.Linear(x_dim + y_dim, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 1)
)

mine = Mine(
    T=statistics_network,
    loss='mine_biased',
    method='concat'
).to('cuda')

with open('/home/xyz/Desktop/Mutual-Information-Analysis/analysis/layer_outputs/layer_outputs.pkl', 'rb') as f:
    representations = pickle.load(f)

X_torch = representations['TransformerEncoder']
Y_torch = representations['ConBiMambaWav2Vec2EncoderLayer_11'].permute(1, 0, 2)

mi = mine.optimize(X_torch[0], Y_torch[0], iters=100, batch_size=64)

print(mi)