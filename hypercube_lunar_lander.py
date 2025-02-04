import gymnasium as gym
import torch
from simple_ne.base_population import SimpleNEPopulation, SimpleNEAgent
from simple_ne.layers.hypercube_layers import TrasnformerClassifier
from simple_ne.preset_params import get_named_params
from simple_ne.hypercube_helper import HypercubeHelper
import pickle
from simple_ne.hypercube_attention import HyperAttention

input_size = 8
out_size = 4
substrate = {
    "input": [],
    "output": [],
    "hiddens": []
}
input_substrate = []
output_substrate = []
center_coord = [.5,.5,.5]
coords_dim = len(center_coord)
tree_depth = 3
params = {"head_depth": 1,
            "max_weight": 5.0,
            "activation": "elu",
            "safe_baseline_depth": 3,
            "grad_steps": 4}

attention_net_encoder = HyperAttention(substrate, params, 3, .5, .5)

def play_game(net : TrasnformerClassifier, render=False):
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

def eval_pop(population):
    fitness_list = []
    print(len(population))
    for net_idx in range(len(population)):
        ne_net = population[net_idx]
        phenotype = attention_net_encoder.create_phenotype_network_nd(ne_net)
        fitness_list.append(play_game(phenotype))
    return torch.tensor(fitness_list, dtype=torch.float32)

if __name__ == '__main__':
    pop_params = get_named_params("bernolli", 4)
    pop = SimpleNEPopulation(8, 4, 200, 20, prob_params = pop_params)
    substrate["input"] = [
        (-1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
        (1.0, 1.0, -.5),
        (1.0, 1.0, .5),
        (-1.0, 1.0, -.5)
        (-1.0, 1.0, .5)
        (.5, .5, .5),
        (-.5, -.5, -.5)
    ]
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

