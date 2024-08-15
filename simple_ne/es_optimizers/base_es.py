import torch
import torch.nn.utils as nn_utils
import numpy as np

# base class to be reused by various es or population based optimizers
class BaseEsOptiizer():
    def __init__(self, model, pop_size, fit_func):
        self.model = model
        self.pop_size = pop_size
        self.fit_func = fit_func

    def _compute_fitness(self, params, episodes):
        nn_utils.vector_to_parameters(params, self.model.parameters())
        self.model.eval()
        total_reward = 0
        for _ in range(episodes):
            reward = self.fit_func(self.model)
            total_reward += reward
        return total_reward / episodes
    
    def initialize(self):
        """Initialize the optimizer (to be overridden by subclasses)."""
        raise NotImplementedError

    def sample_population(self):
        """Sample a population of solutions (to be overridden by subclasses)."""
        raise NotImplementedError

    def evaluate_population(self, population, episodes = 5):
        """Evaluate the population using the objective function."""
        fitnesses = np.array([self._compute_fitness(ind, episodes) for ind in population])
        return fitnesses