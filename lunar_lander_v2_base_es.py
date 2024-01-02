import gymnasium as gym
from simple_ne.preset_params import get_named_params
from simple_es.cma_es_numpy import CMAEsPopulation
import numpy as np
import torch
import pickle

es_size = 10
class Normalizer:
    """
    Normalizer standardizes the inputs to have approximately zero mean and unit variance.
    See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance on Welford's online algorithm.
    """

    def __init__(self, nb_inputs):
        self.n = np.zeros(nb_inputs)
        self.mean = np.zeros(nb_inputs)
        self.mean_diff = np.zeros(nb_inputs)
        self.var = np.zeros(nb_inputs)

    def observe(self, x):
        self.n += 1.0
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min=1e-2)

    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std


normalizer = Normalizer(8)

def play_game(net, render=False):
    if render:
        env = gym.make("LunarLander-v2", render_mode="human")
    else:
        env = gym.make("LunarLander-v2")
    obs,_ = env.reset()
    done = False
    rs = 0
    actions = []
    e = 0
    while (not done and e < 1000):
        #print(net)
        normalizer.observe(obs)
        obs = normalizer.normalize(obs)
        out = np.dot(obs, np.reshape(net, (8,4)))
        #print(out)
        action = np.random.choice(np.flatnonzero(out==out.max()))
        actions.append(float(action))
        obs, r, done, _, _ = env.step(action)
        rs += r
        e += 1
    env.close()
    return -rs

if __name__ == '__main__':
    es = CMAEsPopulation(100, 32)
    es.run_population(play_game, 1000)

