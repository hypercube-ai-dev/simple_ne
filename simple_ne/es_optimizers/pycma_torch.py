import cma
import numpy as np
import torch
import torch.nn.utils as nn_utils

class CMAESOptimizer:
    def __init__(self, model, fit_func, sigma=0.5, population_size=50, max_iter=1000, tolx=1e-6):
        self.model = model
        self.fit_func = fit_func
        self.theta = nn_utils.parameters_to_vector(model.parameters()).detach().numpy()
        self.sigma = sigma
        self.population_size = population_size
        self.max_iter = max_iter
        self.tolx = tolx
        self.es = cma.CMAEvolutionStrategy(self.theta, self.sigma, {'popsize': self.population_size, 'maxiter': self.max_iter, 'tolx': self.tolx})

    def _evaluate(self, solutions, episodes=10):
        rewards = []
        for theta in solutions:
            nn_utils.vector_to_parameters(torch.tensor(theta), self.model.parameters())
            reward = self._compute_fitness(episodes)
            rewards.append(-reward)  # CMA-ES minimizes, so we use negative rewards
        return rewards

    def _compute_fitness(self, episodes):
        self.model.eval()
        total_reward = 0
        for _ in range(episodes):
            reward = self.fit_func(self.model)
            total_reward += reward
        return total_reward / episodes

    def step(self, episodes=10):
        solutions = self.es.ask()
        fitnesses = self._evaluate(solutions, episodes)
        self.es.tell(solutions, fitnesses)
        self.es.logger.add()
        self.es.disp()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
