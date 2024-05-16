from collections import namedtuple
import jax.random as jrandom
import numpy as np


Params = namedtuple('Params', 'B, w, r, k, qE')


class JaxRKey:
    """
    Helper class for seeding RNG with Jax
    """
    def __init__(self, seed):
        self.key = jrandom.PRNGKey(seed)

    def next_seed(self):
        # Use subkey to seed your functions
        self.key, subkey = jrandom.split(self.key)
        return subkey


class Output:
    def __init__(self, Bs, Vs, Rts):
        self.Bs = np.stack(Bs).squeeze()
        self.Vs = np.stack(Vs).squeeze()
        self.Rts = np.stack(Rts).squeeze()

    def plot(self):
        pass
