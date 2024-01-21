import gymnasium as gym
from simple_ne.preset_params import get_named_params
from simple_ne.simple_es.cma_es_torch import CMAES
import numpy as np
import torch
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
    while (not done and e < 1000):
        out = torch.matmul(torch.tensor(obs, dtype=torch.float32),net.reshape(8,4))
        #print(out)
        action = torch.argmax(out, 0).item()
        #actions.append(float(action))
        obs, r, done, _, _ = env.step(action)
        rs += r
        e += 1
    env.close()
    return -rs

if __name__ == '__main__':
    cmaes = CMAES(32,50,.5)
    for generation in range(1000):
        samples = cmaes.ask()

        # Evaluate the samples (replace this with your own evaluation function)
        evaluations = torch.tensor([play_game(s) for s in samples], dtype=torch.float32)

        cmaes.tell(evaluations)

        mean_score = play_game(cmaes.mean)
        best_value = evaluations.min().item()
        if mean_score < -200:
            break
        #if (generation % 10 == 0):
        print(f"Generation {generation}, Best Value: {best_value} Mean Value: {mean_score}")
    solved_net = cmaes.mean
    for x in range(5):
        print(play_game(solved_net, True))

