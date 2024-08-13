import gymnasium as gym
import torch
from simple_ne.es_optimizers.cma_torch import CMAESOptimizer
from simple_ne.es_nets.linear import LinearNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def play_game(net, render=False):
    if render:
        env = gym.make("LunarLander-v2", render_mode="human")
    else:
        env = gym.make("LunarLander-v2")
    obs,_ = env.reset()
    done = False
    rs = 0
    e = 0
    hidden = None
    while (not done and e < 1000):
        out = net(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(dim=0))
        action = torch.argmax(out.squeeze(), 0).item()
        #actions.append(float(action))
        obs, r, done, _, _ = env.step(action)
        rs += r
        e += 1
    env.close()
    return rs

# set up network, env, and optimizer
env = gym.make('LunarLander-v2')

input_dim = 8
hidden_dim = 8  # Example hidden layer size
output_dim = 4

policy_network = LinearNet(input_dim, hidden_dim, output_dim)

cmaes_optimizer = CMAESOptimizer(policy_network, play_game, sigma=0.5, population_size=50, max_iter=1000, tolx=1e-6, device=device)

# Training loop
num_generations = 500
for generation in range(num_generations):
    cmaes_optimizer.step(episodes=5)
    if (generation + 1) % 10 == 0:
        average_reward = -cmaes_optimizer._compute_fitness(10)
        print(f'Generation [{generation + 1}/{num_generations}], Average Reward: {average_reward:.4f}')
        if average_reward < -200:
            cmaes_optimizer.save_model(f'lunar_lander_custom_cmaes_model_gen_{generation + 1}.pth')
            break

play_game(cmaes_optimizer.model, True)