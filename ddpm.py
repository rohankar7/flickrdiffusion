import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import tqdm
from tqdm.notebook import trange, tqdm
from torch.optim.lr_scheduler import MultiplicativeLR, LambdaLR
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import math
# x(t+Δt)=x(t)+σ(t)(√Δt)*r
def forward_diff(x0, t0, dt, nsteps):
    x = np.zeros(nsteps + 1)
    x[0] = x0
    t = t0 + np.arange(nsteps + 1) * dt
    for i in range(nsteps):
        random_normal_var = np.random.randn()
        x[i+1] = x[i] + noise_strength_func(t[i]) * math.sqrt(dt) * random_normal_var
    return x, t
# x(t+Δt)=x(t)+σ(T−t)**2*d/dx[logp(x,T−t)]*Δt+σ(T−t)√Δt*r
# s(x,t):=d/dxlogp(x,t)
# if x0=0, score function s(x,t) = −(x−x0)/σ**2*t = −x/σ**2*t
def score_func(x, x0, T, ti):
  score = - (x-x0)/((noise_strength_func(T - ti)**2)*(T-ti))
  return score
def reverse_diff(x0, T, nsteps, dt):
    x = np.zeros(nsteps + 1); x[0] = 0
    t = np.arange(nsteps + 1) * dt
    # Several Euler-Maruyama time steps
    for i in range(nsteps):
        random_normal_var = np.random.randn()
        x[i+1] = x[i] + (noise_strength_func(T-t[i])**2)*score_func(x[i], x0, T, t[i])*dt + (noise_strength_func(T-t[i]) * math.sqrt(dt) * random_normal_var)
    return x, t
def noise_strength_func(t): return 1

if __name__ == "__main__":
    # Forward Diffusion
    # dt = 0.1
    # x0 = 0
    # nsteps = 100
    # t0 = 0
    # for i in range(5):
    #     x, t = forward_diff(x0, t0, dt, nsteps)
    #     plt.plot(t,x)
    # plt.show()
    # Reverse Diffusion
    dt = 0.1
    x0 = 0
    nsteps = 100
    t0 = 0
    T = 11
    for i in range(5):
        x, t = reverse_diff(x0, T, nsteps, dt)
        plt.plot(t,x)
    plt.show()