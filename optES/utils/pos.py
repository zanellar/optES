import numpy as np

def getpos(pos, n, seed=None):
    '''
        Get the a position of the system. 
        @ param pos: The position of the system. It can be a dictionary with the following keys:
            - type: The type of the distribution ("uniform" or "gaussian")
            - min: The minimum value of the uniform distribution
            - max: The maximum value of the uniform distribution
            - mean: The mean value of the gaussian distribution
            - std: The standard deviation of the gaussian distribution
        @ param n: The dimension of the system
        @ param seed: The seed of the random number generator. If None, the seed is not set. Default: None
    '''
    if seed is not None:
        np.random.seed(seed)

    if type(pos) is dict:
        if pos['type'] == 'uniform':
            return np.random.uniform(pos['min'], pos['max'], size=(n,1))
        elif pos['type'] == 'gaussian':
            return np.random.normal(pos['mean'], pos['std'], size=(n,1))
    else:
        return pos