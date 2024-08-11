

class BaseEsOptiizer():
    def __init__(self, model, pop_size, fit_func):
        self.model = model
        self.pop_size = pop_size
        self.fit_func = fit_func
    
    def evaluate(self, solutions, episodes):


