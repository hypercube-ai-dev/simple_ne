import torch
import torch.nn.utils as nn_utils
import numpy as np
class CMAESOptimizer:
    def __init__(self, model, compute_fitness, sigma=0.5, population_size=50, max_iter=1000, tolx=1e-6, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.theta = nn_utils.parameters_to_vector(model.parameters()).detach().to(self.device)
        self.sigma = sigma
        self.population_size = population_size
        self.max_iter = max_iter
        self.tolx = tolx
        self.compute_fitness = compute_fitness

        self.n_params = len(self.theta)
        self.mean = self.theta.clone()
        self.cov = torch.eye(self.n_params, device=self.device)
        self.weights = torch.log((torch.arange(1, self.population_size + 1, dtype=torch.float32, device=self.device) + 1) / 2) - torch.log(torch.arange(1, self.population_size + 1, dtype=torch.float32, device=self.device))
        self.weights /= self.weights.sum()
        self.mu_eff = 1 / torch.sum(self.weights ** 2)
        self.c1 = 2 / ((self.n_params + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(1 - self.c1, 2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((self.n_params + 2) ** 2 + self.mu_eff))
        self.c_sigma = (self.mu_eff + 2) / (self.n_params + self.mu_eff + 3)
        self.d_sigma = 1 + 2 * max(0, torch.sqrt((self.mu_eff - 1) / (self.n_params + 1)) - 1) + self.c_sigma
        self.pc = torch.zeros(self.n_params, device=self.device)
        self.ps = torch.zeros(self.n_params, device=self.device)
        self.B = torch.eye(self.n_params, device=self.device)
        self.D = torch.ones(self.n_params, device=self.device)
        self.C = self.B @ torch.diag(self.D ** 2) @ self.B.T
        self.eigen_eval = 0

    def _evaluate(self, solutions):
        rewards = []
        for theta in solutions:
            nn_utils.vector_to_parameters(theta, self.model.parameters())
            reward = self._compute_fitness(self.model)
            rewards.append(-reward)  # CMA-ES minimizes, so we use negative rewards
        return rewards

    def _compute_fitness(self, model, episodes = 10):
        self.model.eval()
        total_reward = 0
        for _ in range(episodes):
            reward = self.compute_fitness(model)
            total_reward += reward
        return total_reward / episodes

    def is_covariance_matrix_stable(self):
        condition_number = torch.linalg.cond(self.C)
        return condition_number < 1e10  # Adjust the threshold as needed

    def update_distribution(self, solutions, fitnesses):
        idx = torch.argsort(torch.tensor(fitnesses))
        z = (solutions[idx[:self.population_size]] - self.mean) / self.sigma
        z_w = torch.sum(z.T * self.weights, dim=1)
        self.mean += self.c_sigma * self.sigma * z_w

        self.ps = (1 - self.c_sigma) * self.ps + torch.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff) * z_w
        h_sigma = (torch.linalg.norm(self.ps) / torch.sqrt(1 - (1 - self.c_sigma) ** (2 * (self.eigen_eval + 1))) < (1.4 + 2 / (self.n_params + 1)))
        self.pc = (1 - self.c1) * self.pc + h_sigma * torch.sqrt(self.c1 * (2 - self.c1) * self.mu_eff) * z_w

        z_rank = z - torch.mean(z, dim=0)
        rank_one_update = torch.outer(self.pc, self.pc)
        rank_mu_update = z_rank.T @ (z_rank * self.weights[:, None])
        self.C = (1 - self.c1 - self.c_mu) * self.C + self.c1 * rank_one_update + self.c_mu * rank_mu_update

        self.sigma *= torch.exp((self.c_sigma / self.d_sigma) * (torch.linalg.norm(self.ps) / torch.sqrt(torch.tensor(self.n_params, dtype=torch.float32)) - 1))
        self.eigen_eval += 1

        # Check if the covariance matrix is stable or perform periodic update
        if not self.is_covariance_matrix_stable() or self.eigen_eval > 1 / (self.c1 + self.c_mu) / self.n_params / 10:
            self.eigen_eval = 0
            self.C = torch.triu(self.C) + torch.triu(self.C, 1).T
            self.D, self.B = torch.linalg.eigh(self.C)
            self.D = torch.sqrt(self.D)

    def step(self, env, episodes=10):
        solutions = self.sample_population()
        fitnesses = self._evaluate(solutions, env, episodes)
        self.update_distribution(solutions, fitnesses)

    def sample_population(self):
        z = torch.randn(self.population_size, self.n_params, device=self.device)
        solutions = self.mean + self.sigma * z @ self.B.T @ torch.diag(self.D)
        return solutions

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
