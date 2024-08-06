import cma
import gym
import numpy as np
import torch
import torch.nn.utils as nn_utils

class CMAESOptimizer:
    def __init__(self, model, sigma=0.5, population_size=50, max_iter=1000, tolx=1e-6):
        self.model = model
        self.theta = nn_utils.parameters_to_vector(model.parameters()).detach().numpy()
        self.sigma = sigma
        self.population_size = population_size
        self.max_iter = max_iter
        self.tolx = tolx
        self.es = cma.CMAEvolutionStrategy(self.theta, self.sigma, {'popsize': self.population_size, 'maxiter': self.max_iter, 'tolx': self.tolx})

    def _evaluate(self, solutions, env, episodes=10):
        rewards = []
        for theta in solutions:
            nn_utils.vector_to_parameters(torch.tensor(theta), self.model.parameters())
            reward = self._compute_fitness(env, episodes)
            rewards.append(-reward)  # CMA-ES minimizes, so we use negative rewards
        return rewards

    def _compute_fitness(self, env, episodes):
        self.model.eval()
        total_reward = 0
        for _ in range(episodes):
            state = env.reset()
            done = False
            while not done:
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action_probs = self.model(state)
                action = torch.argmax(action_probs, dim=1).item()
                state, reward, done, _ = env.step(action)
                total_reward += reward
        return total_reward / episodes

    def step(self, env, episodes=10):
        solutions = self.es.ask()
        fitnesses = self._evaluate(solutions, env, episodes)
        self.es.tell(solutions, fitnesses)
        self.es.logger.add()
        self.es.disp()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
