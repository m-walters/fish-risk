from collections import namedtuple
from typing import List, Union

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import xarray as xr

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


class OmegaResults:
    """
    Class for saving runs that iterate over omega
    Outputs must have dimensions/coordinates [omega, time, param_batch]
    """

    def __init__(
        self,
        omegas: Union[np.ndarray, List[float]],
        outputs: List[Output],
        real_horizon: int,
        num_param_batches: int,
    ):
        self.omegas = omegas
        self.outputs = outputs
        self.real_horizon = real_horizon
        self.num_param_batches = num_param_batches

    def to_dataset(self) -> xr.Dataset:
        """
        For runs with varying omega, generate the xArray Dataset
        Outputs must have dimensions [omega, time, param_batch]
        """
        Es = [output.Es for output in self.outputs]
        Bs = [output.Bs for output in self.outputs]
        Vs = [output.Vs for output in self.outputs]
        Rts = [output.Rts for output in self.outputs]

        ds = xr.Dataset(
            {
                "E": (("omega", "time", "batch"), Es),
                "B": (("omega", "time", "batch"), Bs),
                "V": (("omega", "time", "batch"), Vs),
                "Rt": (("omega", "time", "batch"), Rts),
            },
            coords={
                "omega": self.omegas,
                "time": np.arange(self.real_horizon),
                "batch": np.arange(self.num_param_batches),
            },
        )
        return ds

    @staticmethod
    def save_ds(ds: xr.Dataset, path):
        """
        Save an omega dataset to disk
        """
        ds.to_netcdf(path)

    @staticmethod
    def load_ds(path):
        """
        Load a saved dataset
        """
        return xr.open_dataset(path)
