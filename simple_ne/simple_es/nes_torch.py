import torch
import torch.nn as nn
import numpy as np

class NESOptimizer:
    def __init__(self, solution_dim, npop=50, sigma=0.1, alpha=0.001):
        self.npop = npop
        self.sigma = sigma
        self.alpha = alpha
        self.solution_dim = solution_dim
        self.w = torch.randn(solution_dim, dtype=torch.float32, requires_grad=True)

    def ask(self):
        return self.w + self.sigma * torch.randn(self.npop, self.solution_dim)

    def tell(self, samples):
        rewards = torch.zeros(self.npop)
        for j in range(self.npop):
            w_try = samples[j]
            rewards[j] = self.f(w_try)
        
        rewards_std = (rewards - torch.mean(rewards)) / torch.std(rewards)
        gradient = torch.mean((self.sigma / (self.npop * self.sigma)) * samples.T @ rewards_std)
        self.w.data += self.alpha * gradient.detach()
        return torch.mean(rewards).item()

    def eval(self, eval_func, x):
        with torch.no_grad():
            samples = self.ask()
            rewards = torch.zeros(self.npop)
            for s in range(self.npop):
                w_try = samples[s]
                r = eval_func(w_try, x)