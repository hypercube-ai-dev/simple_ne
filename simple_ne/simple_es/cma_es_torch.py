import torch
import numpy as np

class CMAES:
    def __init__(self, num_params, population_size=10, sigma_init=0.1, regularization=.1):
        self.num_params = num_params
        self.population_size = population_size
        self.sigma = sigma_init
        self.reg = regularization
        self.mean = torch.zeros(num_params, requires_grad=False)
        self.cov_matrix = torch.eye(num_params, requires_grad=False)
        self.population = torch.randn((population_size, num_params), requires_grad=False)

    def ask(self):
        return self.mean + self.sigma * torch.matmul(self.population, torch.linalg.cholesky(self.cov_matrix).T)

    def tell(self, evaluations):
        sorted_indices = torch.argsort(evaluations)
        sorted_population = self.population[sorted_indices]

        # Update mean
        self.mean = torch.mean(sorted_population[:self.population_size // 2], dim=0)

        # Update covariance matrix using CMA formula
        delta = sorted_population - self.mean
        self.cov_matrix = torch.matmul(delta.T, delta) / (self.population_size // 2 - 1)
        # Update step size (sigma)
        self.sigma *= np.exp(1 / 2 * ((torch.norm(delta, dim=1) / self.sigma) ** 2).mean() - self.num_params)
