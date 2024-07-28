import gymnasium as gym
from simple_ne.es_nets.recurrent_nets import GruNet
import numpy as np
import torch
from simple_ne.simple_es.elite_es import EliteEsPop
import pickle

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
        out, hidden = net(torch.tensor(obs, dtype=torch.float32).unsqueeze(dim=0), hidden)
        action = torch.argmax(out.squeeze(), 0).item()
        #actions.append(float(action))
        obs, r, done, _, _ = env.step(action)
        rs += r
        e += 1
    env.close()
    return rs

if __name__ == '__main__':
    net = GruNet(8, 4)
    pop_size = 50
    popObj = EliteEsPop(net, sigma=.01, elite_cutoff=.1, size=pop_size)
    with torch.no_grad():
        for generation in range(1000):
            pop = popObj.pop
            print(pop[0])
            # Evaluate the samples (replace this with your own evaluation function)
            evals_sorted = sorted([(s, play_game(net.set_params(pop[s]))) for s in range(pop.shape[0])], key= lambda x: x[1], reverse=True)
            mean_score = np.mean([x[1] for x in evals_sorted[:popObj.cutoff]]) 
            best_value = pop[evals_sorted[0][0]]
            if evals_sorted[0][1] > 200:
                break
            print(f"Generation {generation}, Best Value: {evals_sorted[0][0]} Mean Value: {mean_score}")
            popObj.evolve_pop(torch.tensor([x[0] for x in evals_sorted[:popObj.cutoff]]))
        for x in range(5):
            print(play_game(net.set_params(best_value), True))

