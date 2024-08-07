import gymnasium as gym
import torch
from simple_ne.es_optimizers.cma_torch import CMAESOptimizer
from simple_ne.es_nets.linear import LinearNet
# Initialize environment
env = gym.make('LunarLander-v2')
env.seed(seed)

# Define policy network
input_dim = env.observation_space.shape[0]
hidden_dim = 128  # Example hidden layer size
output_dim = env.action_space.n

policy_network = LinearNet(input_dim, hidden_dim, output_dim)

# Initialize custom CMA-ES optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cmaes_optimizer = CMAESOptimizer(policy_network, sigma=0.5, population_size=50, max_iter=1000, tolx=1e-6, device=device)

# Training loop
num_generations = 500
for generation in range(num_generations):
    cmaes_optimizer.step(env, episodes=10)
    if (generation + 1) % 10 == 0:
        average_reward = cmaes_optimizer._compute_fitness(env, episodes=10)
        print(f'Generation [{generation + 1}/{num_generations}], Average Reward: {average_reward:.4f}')
        # Save model checkpoint
        cmaes_optimizer.save_model(f'lunar_lander_custom_cmaes_model_gen_{generation + 1}.pth')

# Close environment
env.close()
