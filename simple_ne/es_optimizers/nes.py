import torch
import torch.nn.utils as nn_utils
import numpy as np

class NESOptimizer:
    def __init__(self, model, fit_func, sigma=0.5, population_size=500, lr=0.01):
        self.model = model
        self.sigma = sigma
        self.population_size = population_size
        self.lr = lr
        params = nn_utils.parameters_to_vector(model.parameters()).detach()
        print(params.shape)
        self.theta = np.zeros(params.numpy().shape)
        self.fit_func = fit_func

    def _evaluate(self, perturbations):
        rewards = []
        for perturbation in perturbations:
            perturbed_theta = self.theta + self.sigma * perturbation
            nn_utils.vector_to_parameters(torch.tensor(perturbed_theta, dtype=torch.float32), self.model.parameters())
            reward = self._compute_fitness()
            rewards.append(reward)
        return np.array(rewards)

    def _compute_fitness(self):
        self.model.eval()
        with torch.no_grad():
            reward = self.fit_func(self.model)
        return reward.item()  # Negative loss as fitness

    # get_fit_func should be a function that returns the fitness function with
    # appropriate state
    def step(self):
        perturbations = np.random.randn(self.population_size, len(self.theta))
        rewards = self._evaluate(perturbations)
        A = (rewards - np.mean(rewards)) / np.std(rewards)
        gradient = np.dot(perturbations.T, A) / (self.population_size * self.sigma)
        self.theta += self.lr * gradient
        nn_utils.vector_to_parameters(torch.tensor(self.theta, dtype=torch.float32), self.model.parameters())


class AdaptiveNES:
    def __init__(self, model, sigma=0.1, lr=0.01, population_size=50, sigma_lr=0.1, lr_lr=0.1):
        self.model = model
        self.sigma = sigma
        self.lr = lr
        self.population_size = population_size
        self.sigma_lr = sigma_lr
        self.lr_lr = lr_lr

        self.params = list(self.model.parameters())
        self.shapes = [param.size() for param in self.params]
        self.num_params = sum(p.numel() for p in self.params)
        self.best_reward = None
        self.best_params = None

    def _flat_params(self):
        return torch.cat([p.view(-1) for p in self.params])

    def _update_params(self, flat_params):
        index = 0
        for param, shape in zip(self.params, self.shapes):
            numel = param.numel()
            param.data = flat_params[index:index + numel].view(shape).data
            index += numel

    def _evaluate(self, params):
        self._update_params(params)
        return self.model.evaluate()  # Replace this with your model's evaluation method

    def step(self):
        # Sample perturbations
        perturbations = torch.randn(self.population_size, self.num_params) * self.sigma
        
        rewards = torch.zeros(self.population_size)
        flat_params = self._flat_params()

        for i in range(self.population_size):
            params = flat_params + perturbations[i]
            rewards[i] = self._evaluate(params)

        mean_reward = rewards.mean()
        reward_std = rewards.std()
        
        # Normalize rewards
        normalized_rewards = (rewards - mean_reward) / reward_std
        
        # Update parameters
        weighted_sum = torch.sum(perturbations.T * normalized_rewards, axis=1)
        grad = weighted_sum / (self.population_size * self.sigma)
        new_flat_params = flat_params + self.lr * grad
        self._update_params(new_flat_params)
        
        # Adapt sigma and lr
        self.sigma *= np.exp(self.sigma_lr * grad.norm().item() / (self.num_params ** 0.5))
        self.lr *= np.exp(self.lr_lr * grad.norm().item() / (self.num_params ** 0.5))

        current_reward = self._evaluate(flat_params)
        if self.best_reward is None or current_reward > self.best_reward:
            self.best_reward = current_reward
            self.best_params = flat_params.clone()
            
    def set_best_params(self):
        if self.best_params is not None:
            self._update_params(self.best_params)
