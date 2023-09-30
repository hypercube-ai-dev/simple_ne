import gymnasium as gym
import torch
from simple_ne.attention_neat import SimpleNEPopulation, SimpleNEAgent
from simple_ne.preset_params import get_named_params

def play_game(net : SimpleNEAgent):
    env = gym.make("LunarLander-v2")
    obs,_ = env.reset()
    done = False
    rs = 0
    actions = []
    while not done:
        out = net(torch.tensor(obs, dtype=torch.float32))
        action = torch.argmax(out).item()
        actions.append(float(action))
        obs, r, done, _, _ = env.step(action)
        rs += r
    env.close()
    #print(r)
    return rs

def eval_pop(population):
    fitness_list = []
    for net_idx in range(len(population)):
        reward = play_game(population[net_idx])
        fitness_list.append(reward)
    return torch.tensor(fitness_list, dtype=torch.float32)

if __name__ == '__main__':
    pop_params = get_named_params("bernolli", 4)
    pop_params[0] = .5
    pop = SimpleNEPopulation(8, 4, 200, 100, prob_params = pop_params)

    best_fitness = 0

    epoch_counter = 0

    while best_fitness < 200:
        fits = eval_pop(pop.population)
        avg_fitness = fits.mean()
        best_fitness = fits.max()
        if epoch_counter % 5 == 0:
            print(f"Epoch {epoch_counter} avg fitness: {avg_fitness} best fitness: {fits.max()}")
        epoch_counter += 1
        pop.evolve(fits)
    print(f"solved in {epoch_counter} generations")
