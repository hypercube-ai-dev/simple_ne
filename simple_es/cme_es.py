import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the objective function to be optimized
def objective_function(x):
    return -(x**2).sum()  # Example: Maximizing the sum of squares

# Define the CEM-ES optimization algorithm
def cem_es(mu, sigma, population_size, elite_frac, max_iterations):
    mu = torch.tensor(mu, requires_grad=True, dtype=torch.float32)
    sigma = torch.tensor(sigma, requires_grad=True, dtype=torch.float32)

    optimizer = optim.Adam([mu, sigma], lr=0.01)

    for iteration in range(max_iterations):
        # Sample candidate solutions from the current distribution
        candidates = torch.normal(mu, sigma, size=(population_size,))

        # Evaluate the objective function for each candidate
        rewards = -objective_function(candidates)

        # Select the top elite_frac proportion of candidates
        num_elites = int(elite_frac * population_size)
        elite_indices = torch.topk(rewards, num_elites)[1]
        elite_candidates = candidates[elite_indices]

        # Update the distribution parameters based on the elite candidates
        new_mu = elite_candidates.mean(dim=0)
        new_sigma = elite_candidates.std(dim=0)

        # Update the distribution parameters using the optimizer
        optimizer.zero_grad()
        loss = -(mu - new_mu).pow(2).sum() - (sigma - new_sigma).pow(2).sum()
        loss.backward()
        optimizer.step()

        # Print the current best solution
        best_solution = elite_candidates[0].detach().numpy()
        print(f'Iteration {iteration + 1}, Best Solution: {best_solution}, Best Reward: {-objective_function(best_solution)}')

    return mu, sigma

if __name__ == "__main__":
    # Initial parameters
    mu_init = np.random.rand(1)  # Initial mean
    sigma_init = np.random.rand(1)  # Initial standard deviation

    # CEM-ES hyperparameters
    population_size = 50
    elite_frac = 0.2
    max_iterations = 100

    # Run CEM-ES
    final_mu, final_sigma = cem_es(mu_init, sigma_init, population_size, elite_frac, max_iterations)

    print(f'Optimal solution: {final_mu.item()}, Optimal reward: {-objective_function(final_mu).item()}')
