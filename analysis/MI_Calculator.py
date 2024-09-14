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



# exp_tensor = torch.load('analysis/data/ssamba_tiny/ssamba_patch400_tiny_batch_5.pt')


tensor_dir = os.listdir("/home/xyz/Desktop/Mutual-Information-Analysis/analysis/data/ssamba_base")


total_mi_dict = {}

for tensor_name in tensor_dir:
    path = os.path.join('/home/xyz/Desktop/Mutual-Information-Analysis/analysis/data/ssamba_base', tensor_name)

    exp_tensor = torch.load(path)

    mi_layer_list = {}

    for layer_index in range(1, len(exp_tensor)):
        mi_list = []
        for index in range(25):
            X_torch = exp_tensor[0][index].to('cuda')
            Y_torch = exp_tensor[layer_index][index].to('cuda')

            mine = Mine(T=statistics_network, loss='mine_biased',method='concat').to('cuda')

            mi = mine.optimize(X_torch, Y_torch, iters=100, batch_size=128)

            print(f"batch {tensor_name} layer {layer_index}, batch {index} mutual information is {mi}" )

            mi_list.append(mi.detach().cpu())

        average_mi = sum(mi_list) / len(mi_list)
        mi_layer_list[layer_index] = average_mi
    
    total_mi_dict[tensor_name] = mi_layer_list

print("Mutual Information:", total_mi_dict)

with open('ssamba_base.pkl', 'wb') as file:
    pickle.dump(total_mi_dict, file)