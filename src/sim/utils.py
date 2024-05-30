from collections import namedtuple
from typing import List, Union

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

Params = namedtuple('Params', 'B, w, r, k, qE')

# For typing -- It's often hard to say if an object is one or the other
Array = Union[jnp.ndarray, np.ndarray]


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


class JaxGaussian:
    @staticmethod
    def log_prob(point, loc, scale):
        var = scale ** 2
        denom = jnp.sqrt(2 * jnp.pi * var)
        log_probs = -0.5 * ((point - loc) ** 2) / var - jnp.log(denom)
        return log_probs

    @staticmethod
    def sample(key, loc, scale):
        sample = jax.random.normal(key, loc.shape) * scale + loc
        log_probs = JaxGaussian.log_prob(sample, loc=loc, scale=scale)
        return sample, log_probs


class Output:
    def __init__(
        self,
        Es: List[Array],
        Bs: List[Array],
        Vs: List[Array],
        Rts: List[Array],
    ):
        """
        Outputs from the main world model sim.
        Each variable is a list of length real_horizon,
        and each element therein is an array of shape either (1, num_param_batches) or (num_param_batches).
        The stored values become numpy ndarrays of shape [real_horizon, num_param_batches].
        """
        self.Es = np.stack(Es).squeeze()
        self.Bs = np.stack(Bs).squeeze()
        self.Vs = np.stack(Vs).squeeze()
        self.Rts = np.stack(Rts).squeeze()

    def plot(self):
        pass
