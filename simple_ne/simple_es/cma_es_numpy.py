import numpy as np

class CMAEsPopulation:
    def __init__(self, pop_size, dim, sigma=.5, lr=None):
        self.sigma = sigma
        self.dim = dim
        self.pop_size = pop_size
        self.mean = np.zeros(dim)

    def gen_pop(self):
        self.population = np.random.multivariate_normal(self.mean, self.sigma**2 * np.eye(len(self.mean)), self.pop_size)

    def rank_population(self, obj_func):
        return sorted([(i, obj_func(i)) for i in self.population], key=lambda x: x[1])
    
    def update_sigma(self, elites, lr_cov):
        diff = elites - self.mean
        cov_mat = np.cov(diff, rowvar=False)
        self.sigma = (1 - lr_cov) * self.sigma + lr_cov * cov_mat
    
    def ask():
        raise("not implemented")
    
    def tell():
        raise("not implemented")
    
    
    def run_population(self, obj_func, iterations=100):
        solved = False
        i = 0
        learning_rate_mean = 1.0 / (10 * self.dim**0.5)
        learning_rate_covariance = 1.0 / (4 * (self.dim + 1)**0.5)
        while (solved == False and i < iterations):
            self.gen_pop()
            ranked = self.rank_population(obj_func)
            elite = [x[0] for x in ranked[:int(self.pop_size / 10)]]
            elite_reward = np.mean([x[1] for x in ranked[:int(self.pop_size / 2)]])
            #elite_reward = ranked[0][1]
            #if i % 10:
            print(f"episode: {i} avage top half reward: {elite_reward}")
            if elite_reward < -200:
                solved = True
            if ranked[0][1] < -200:
                self.mean = elite[0]
            else:
                self.mean = self.mean + learning_rate_mean * np.mean(elite, axis=0)
            #self.mean = elite[0]
            self.update_sigma(elite, learning_rate_covariance)
            i += 1
        return self.mean