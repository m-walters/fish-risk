from collections import namedtuple
import jax.random as jrandom


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
