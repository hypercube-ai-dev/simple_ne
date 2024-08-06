import numpy as np
import torch
import torch.nn.utils as nn_utils

class CustomCMAESOptimizer:
    def __init__(self, model, sigma=0.5, population_size=50, max_iter=1000, tolx=1e-6):
        self.model = model
        self.theta = nn_utils.parameters_to_vector(model.parameters()).detach().numpy()
        self.sigma = sigma
        self.population_size = population_size
        self.max_iter = max_iter
        self.tolx = tolx

        self.n_params = len(self.theta)
        self.mean = self.theta.copy()
        self.cov = np.eye(self.n_params)
        self.cov_inv = np.linalg.inv(self.cov)
        self.weights = np.log((self.population_size + 1) / 2) - np.log(np.arange(1, self.population_size + 1))
        self.weights /= self.weights.sum()
        self.mu_eff = 1 / np.sum(self.weights ** 2)
        self.c1 = 2 / ((self.n_params + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(1 - self.c1, 2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((self.n_params + 2) ** 2 + self.mu_eff))
        self.c_sigma = (self.mu_eff + 2) / (self.n_params + self.mu_eff + 3)
        self.d_sigma = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (self.n_params + 1)) - 1) + self.c_sigma
        self.pc = np.zeros(self.n_params)
        self.ps = np.zeros(self.n_params)
        self.B = np.eye(self.n_params)
        self.D = np.ones(self.n_params)
        self.C = self.B @ np.diag(self.D ** 2) @ self.B.T
        self.eigen_eval = 0

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
        solutions = self.sample_population()
        fitnesses = self._evaluate(solutions, env, episodes)
        self.update_distribution(solutions, fitnesses)

    def sample_population(self):
        z = np.random.randn(self.population_size, self.n_params)
        solutions = self.mean + self.sigma * z @ self.B.T @ np.diag(self.D)
        return solutions

    def update_distribution(self, solutions, fitnesses):
        idx = np.argsort(fitnesses)
        z = (solutions[idx[:self.population_size]] - self.mean) / self.sigma
        z_w = np.dot(z.T, self.weights)
        self.mean += self.c_sigma * self.sigma * z_w

        self.ps = (1 - self.c_sigma) * self.ps + np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff) * z_w
        h_sigma = (np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.c_sigma) ** (2 * (self.eigen_eval + 1))) < (1.4 + 2 / (self.n_params + 1)))
        self.pc = (1 - self.c1) * self.pc + h_sigma * np.sqrt(self.c1 * (2 - self.c1) * self.mu_eff) * z_w

        z_rank = z - np.mean(z, axis=0)
        rank_one_update = np.outer(self.pc, self.pc)
        rank_mu_update = z_rank.T @ (z_rank * self.weights[:, np.newaxis])
        self.C = (1 - self.c1 - self.c_mu) * self.C + self.c1 * rank_one_update + self.c_mu * rank_mu_update

        self.sigma *= np.exp((self.c_sigma / self.d_sigma) * (np.linalg.norm(self.ps) / np.sqrt(self.n_params) - 1))
        self.eigen_eval += 1

        if self.eigen_eval > 1 / (self.c1 + self.c_mu) / self.n_params / 10:
            self.eigen_eval = 0
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            self.D, self.B = np.linalg.eigh(self.C)
            self.D = np.sqrt(self.D)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
