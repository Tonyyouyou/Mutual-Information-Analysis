import torch.nn as nn
from mine.mine import Mine
import numpy as np

x_dim=100
y_dim=100

statistics_network = nn.Sequential(
    nn.Linear(x_dim + y_dim, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 1)
)

mine = Mine(
    T = statistics_network,
    loss = 'mine' ,#mine_biased, fdiv
    method = 'concat'
)

joint_samples = np.random.multivariate_normal(mu = np.array([0,0]), cov = np.array([[1, 0.2], [0.2, 1]]))

X, Y = joint_samples[:, 0], joint_samples[:, 1]

mi = mine.optimize(X, Y, iters = 100)