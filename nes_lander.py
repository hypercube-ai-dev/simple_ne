import gymnasium as gym
from simple_ne.es_nets.linear import LinearNet
import numpy as np
import torch
from simple_ne.es_optimizers.nes import NESOptimizer
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
        out = net(torch.tensor(obs, dtype=torch.float32).unsqueeze(dim=0))
        action = torch.argmax(out.squeeze(), 0).item()
        #actions.append(float(action))
        obs, r, done, _, _ = env.step(action)
        rs += r
        e += 1
    env.close()
    return rs

if __name__ == '__main__':
    net = LinearNet(8, 32, 4)
    pop_size = 50
    popObj = NESOptimizer(net, play_game)
    with torch.no_grad():
        for generation in range(1000):
            popObj.step()
            current = play_game(popObj.model)
            if generation % 10:
                print("current score: ", current)
            if current > 200:
                break
        for x in range(5):
            print(play_game(net.set_params(popObj.model), True))

