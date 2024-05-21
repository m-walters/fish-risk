from copy import copy

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import differential_entropy as entr

from sim.utils import Params, Output

import warnings


class EulerMaruyamaDynamics:
    """
    Runs evolution of the fish population via the ODE
    """
    def __init__(self, sim_horizon: int, sim_steps: int, D: float, max_b: float):
        self.time_points = jnp.linspace(0., sim_horizon, sim_steps)
        self.dt = 1 / sim_steps
        self.D = D  # diffusion coefficient
        self.max_b = max_b

    def rhs_pymcode(self, Bs, rs, ks, qEs) -> list:
        """
        Will be passed into DifferentialEquation
        p is our parameter tuple (r, k, qE)
        """
        return rs*Bs - rs*Bs*Bs/ks - qEs*Bs

    def __call__(self, params: Params) -> list:
        # Generate sample data
        observed = []
        Bs = params.B
        observed.append(Bs)
        for t in self.time_points:
            rhs = self.rhs_pymcode(
                      Bs,
                      params.r,
                      params.k,
                      params.qE
            )
            Bs_step = Bs + rhs * self.dt + np.random.normal(0, self.D * self.dt, Bs.shape)
            Bs = np.maximum(0, Bs_step)
            Bs = np.minimum(self.max_b, Bs)
            observed.append(Bs)
        return observed


class RevenueModel:
    def __init__(self, P0: float, rho: float):
        self.P0 = P0
        self.rho = rho

    def __call__(self, B, qE):
        C = np.where(B > 0, B, 1.)  # set those entries in B that vanish to 1. arbitrarily
        PB = np.where(B > 0, self.P0 * C ** self.rho, 0.)  # those entries where C is 1 are 0. in PB
        return PB * qE * B


class CostModel:
    def __init__(self, C0: float, gamma: float):
        self.C0 = C0
        self.gamma = gamma

    def __call__(self, qE):
        return self.C0 * (1 - qE) ** self.gamma


class Policy:
    def __init__(self, revenue_model: RevenueModel, cost_model: CostModel):
        self.revenue_model = revenue_model
        self.cost_model = cost_model

    def sample(self, params: Params):
        raise NotImplementedError


class ProfitMaximizingPolicy(Policy):
    def sample(self, params: Params):
        # note: this models a profit-maximizing agent, and single agent, in particular!
        coef = -self.revenue_model.P0 / (self.cost_model.gamma * self.cost_model.C0)
        # set entries of B which vanish to 1 arbitrarily
        Bp = np.where(params.B > 0, params.B ** (self.revenue_model.rho + 1), 1.)
        inv_gamma_power = 1 / (self.cost_model.gamma - 1)
        # set Es to 0 if B vanishes
        Es = np.where(params.B > 0, 1 - (coef * Bp) ** inv_gamma_power, 0.)
        Es = np.minimum(1, np.maximum(Es, 0.))
        if (Es == 0.).any():
            warnings.warn("Optimal extraction rate qE = 0.")
        return Es


class RiskMitigationPolicy(Policy):
    def __init__(
        self,
        revenue_model: RevenueModel,
        cost_model: CostModel,
        lmbda: float
    ):
        super().__init__(revenue_model, cost_model)
        self.lmbda = lmbda

    def sample(self, params: Params):
        return 0.0


class LossModel:
    def __call__(self, V_t, t, omega):
        loss = (-1 / (1 + omega) ** t) * np.minimum(V_t, 0)
        return loss


class PreferencePrior:
    def __init__(self, l_bar: float):
        self.l_bar = l_bar


class SoftmaxPreferencePrior(PreferencePrior):
    def __call__(self, Lt):
        return jax.nn.softmax(self.l_bar - Lt, axis=0)


class UniformPreferencePrior(PreferencePrior):
    def __call__(self, Lt):
        return np.ones(Lt.shape) / self.l_bar


class RiskModel:
    def __init__(self, preference_prior):
        self.preference_prior = preference_prior

    def __call__(self, Lt, Vt):
        # this printing is important for evolving the preference model
        sample_mean = jnp.log(self.preference_prior(Lt)).mean(axis=0)
        entropy = entr(Lt, axis=0)
        Gt = entropy - sample_mean
        print("sample_mean {} - entropy {} = RISK: {}".format(sample_mean, entropy, Gt))
        return Gt


class Model:
    """
    Main model that computes risk etc. through simulation of fishery evolution.
    Uses `n_montecarlo` MonteCarlo predictive simulations at a given realworld
    time step to calculate statistics.
    """
    def __init__(
        self,
        params,
        n_montecarlo,
        dynamics,
        real_horizon,
        policy,
        revenue_model,
        cost_model,
        loss_model,
        risk_model,
        debug=False,
        omega_scale=1,
    ):
        self.params = params
        self.n_montecarlo = n_montecarlo
        self.dynamics = dynamics
        self.real_horizon = real_horizon
        self.policy = policy
        self.revenue_model = revenue_model
        self.cost_model = cost_model
        self.loss_model = loss_model
        self.risk_model = risk_model
        self.debug = debug
        self.omega_scale = omega_scale

    def print(self, out: str, force: bool = False):
        if self.debug or force:
            print(out)

    def sample_policy(self, params: Params):
        Et = self.policy.sample(params)
        new_params = Params(
            B=params.B,
            w=params.w,
            r=params.r,
            k=params.k,
            qE=Et
        )
        return new_params

    def timestep(self, t: int, old_params: Params):
        params = self.sample_policy(old_params)
        observed = self.dynamics(params)
        Bt = observed[-1]
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            Revt = self.revenue_model(Bt, params.qE)
        Ct = self.cost_model(params.qE)
        Vt = Revt - Ct
        Vt = jnp.nan_to_num(np.array(Vt), copy=False)
        Lt = self.loss_model(Vt, t, params.w)

        # update params with new fish biomass
        params = Params(
            B=Bt,
            w=params.w,
            r=params.r,
            k=params.k,
            qE=params.qE
        )

        return Lt, Vt, Bt, params

    def plan(self, t_sim, params):
        Rt_sim = 0.
        for t_plan in range(self.real_horizon):
            t = t_sim + t_plan
            Lt, Vt, Bt, params = self.timestep(t, params)
            Gt = self.risk_model(Lt, Vt)
            Rt_sim += Gt
        print('\nend plan\n')
        return Rt_sim

    def stack_params(self, params: list) -> Params:
        Bs = jnp.vstack([np.array(p.B) for p in params])
        ws = jnp.vstack([self.omega_scale * np.ones(self.params.w.shape) for p in params])
        rs = jnp.vstack([np.array(p.r) for p in params])
        ks = jnp.vstack([np.array(p.k) for p in params])
        qEs = jnp.vstack([np.array(p.qE) for p in params])
        return Params(Bs, ws, rs, ks, qEs)

    def get_sim_params(self):
        param_list = [copy(self.params) for _ in range(self.n_montecarlo)]
        return self.stack_params(param_list)

    def __call__(self):
        es = []
        bs = []
        vs = []
        rts = []
        for t_sim in range(self.real_horizon):
            es.append(self.params.qE)
            Lt_sim, Vt_sim, Bt_sim, self.params = self.timestep(t_sim, self.params)
            sim_params = self.get_sim_params()
            Rt_sim = self.plan(t_sim, sim_params)
            bs.append(Bt_sim)
            vs.append(Vt_sim)
            rts.append(Rt_sim)
        return Output(Es=es, Bs=bs, Vs=vs, Rts=rts)
