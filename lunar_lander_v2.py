import gymnasium as gym
import torch
from simple_ne.base_population import SimpleNEPopulation, SimpleNEAgent
from simple_ne.preset_params import get_named_params
import pickle

def play_game(net : SimpleNEAgent, render=False):
    if render:
        env = gym.make("LunarLander-v2", render_mode="human")
    else:
        env = gym.make("LunarLander-v2")
    obs,_ = env.reset()
    done = False
    rs = 0
    actions = []
    while not done:
        out = net(torch.tensor(obs, dtype=torch.float32))
        #print(out)
        action = torch.argmax(out, 0).item()
        actions.append(float(action))
        obs, r, done, _, _ = env.step(action)
        rs += r
    env.close()
    net.reset()
    return rs

def eval_pop(population, trials=5):
    fitness_list = []
    print(len(population))
    for net_idx in range(len(population)):
        nf = 0.0
        for i in range(trials):
            nf += play_game(population[net_idx])
        fitness_list.append(nf / trials)
    return torch.tensor(fitness_list, dtype=torch.float32)

if __name__ == '__main__':
    pop_params = get_named_params("bernolli", 4)
    pop = SimpleNEPopulation(8, 4, 200, 20, prob_params = pop_params)

    best_fitness = 0

    epoch_counter = 0
    pop.init_population()
    while best_fitness < 200:
        fits = eval_pop(pop.population)
        avg_fitness = fits.mean()
        best_fitness = fits.max()
        if epoch_counter % 5 == 0:
            print(f"Epoch {epoch_counter} avg fitness: {avg_fitness} best fitness: {fits.max()}")
        epoch_counter += 1
        pop.evolve(fits)
    print(f"solved in {epoch_counter} generations")
    solved_net = pop.population[torch.argmax(fits).item()]
    for x in range(5):
        print(play_game(solved_net, True))
    pickle.dump(solved_net,  open('./saved_models/lunar-lander-solver.pkl', 'wb'))
    solved_net.print_model_details()

