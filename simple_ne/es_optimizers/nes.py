import torch
import torch.nn.utils as nn_utils
import numpy as np

class NESOptimizer:
    def __init__(self, model, sigma=0.1, population_size=50, lr=0.01):
        self.model = model
        self.sigma = sigma
        self.population_size = population_size
        self.lr = lr
        self.theta = nn_utils.parameters_to_vector(model.parameters()).detach().numpy()

    def _evaluate(self, perturbations, fit_func):
        rewards = []
        for perturbation in perturbations:
            perturbed_theta = self.theta + self.sigma * perturbation
            nn_utils.vector_to_parameters(torch.tensor(perturbed_theta), self.model.parameters())
            reward = self._compute_fitness()
            rewards.append(reward)
        return np.array(rewards)

    def _compute_fitness(self, x, fit_func):
        self.model.eval()
        with torch.no_grad():
            out = self.model(x)
            loss = self.fit_func(out)
        return loss.item()  # Negative loss as fitness

    # get_fit_func should be a function that returns the fitness function with
    # appropriate state
    def step(self, get_fit_func):
        perturbations = np.random.randn(self.population_size, len(self.theta))
        fit_func = get_fit_func()
        rewards = self._evaluate(perturbations, fit_func)
        A = (rewards - np.mean(rewards)) / np.std(rewards)
        gradient = np.dot(perturbations.T, A) / (self.population_size * self.sigma)
        self.theta += self.lr * gradient
        nn_utils.vector_to_parameters(torch.tensor(self.theta), self.model.parameters())
