import gymnasium as gym
import torch
from simple_ne.base_population import SimpleNEPopulation, SimpleNEAgent, SimpleNENode
from simple_ne.preset_params import get_named_params
import pickle

es_size = 10

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

def get_es_nets(net: SimpleNEAgent):
    es_pop = []
    for x in range(es_size):
        es_nodes = []
        for n in net.nodes:
            es_w = n.weights * (.5 * torch.randn(n.weights.shape))
            es_nodes.append(SimpleNENode(n.activation, n.in_idxs, n.node_key, es_w, n.is_output))
        es_pop.append(SimpleNEAgent(es_nodes, net.in_size, net.out_size, net.batch_size))
    es_pop.append(net)
    return es_pop

def make_es_adjustments(net, scores, nets):
    
    return 

def eval_pop(population):
    fitness_list = []
    print(len(population))
    for net_idx in range(len(population)):
        es_pop = get_es_nets(population[net_idx])
        es_scores = []
        for n in es_pop:
            es_scores.append(play_game(n))
        es_updated = 
        fitness_list.append()
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

