import logging
import warnings
from abc import ABC
from copy import copy

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import differential_entropy as entr

from sim.utils import JaxGaussian, JaxRKey, Output, Params

logger = logging.getLogger(__name__)


class ModelBase(ABC):
    """
    Base class for our models
    """

    def __init__(self, *args, **kwargs):
        # We leave kwargs open
        self.key = JaxRKey(seed=kwargs.get("seed", 8675309))


class EulerMaruyamaDynamics(ModelBase):
    """
    Runs evolution of the fish population via the ODE
    """

    def __init__(self, t_end: int, num_points: int, D: float, max_b: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_points = jnp.linspace(0., t_end, num_points)
        self.dt = t_end / num_points
        self.D = D  # diffusion coefficient
        self.max_b = max_b

    def rhs_pymcode(self, Bs: np.ndarray, rs: np.ndarray, ks: np.ndarray, qEs: np.ndarray) -> np.ndarray:
        """
        Will be passed into DifferentialEquation
        p is our parameter tuple (r, k, qE)
        """
        return rs * Bs - rs * Bs * Bs / ks - qEs * Bs

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
            Bs_step = Bs + rhs * self.dt + Bs * self.D * np.random.normal(0, self.dt, Bs.shape)
            Bs = np.maximum(1, Bs_step)  # Life finds a way
            Bs = np.minimum(self.max_b, Bs)
            observed.append(Bs)
        return observed


class RevenueModel(ModelBase):
    def __init__(self, P0: float, rho: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.P0 = P0
        self.rho = rho

    def __call__(self, B, qE):
        market_price = self.P0 * B ** self.rho
        return market_price * qE * B


class CostModel(ModelBase):
    def __init__(self, C0: float, gamma: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.C0 = C0
        self.gamma = gamma

    def __call__(self, qE):
        return self.C0 * (1 - qE) ** self.gamma


class Policy(ModelBase):
    def __init__(self, revenue_model: RevenueModel, cost_model: CostModel, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        if jnp.logical_and(Es == 0., Bp >= 0.).any():
            warnings.warn("Optimal extraction rate qE = 0 but Bp > 0.")
        return Es


class RiskMitigationPolicy(Policy):
    def __init__(
        self,
        revenue_model: RevenueModel,
        cost_model: CostModel,
        lmbda: float,
        *args,
        **kwargs
    ):
        super().__init__(revenue_model, cost_model, *args, **kwargs)
        self.lmbda = lmbda

    def sample(self, params: Params):
        return 0.0


class LossModel(ModelBase):
    def __call__(self, V_t, t, omega):
        """
        :param V_t: Net profit (revenue - cost) at time t
        :param t: Future time step
        :param omega: Discount factor
        """
        loss = (-1 / (1 + omega) ** t) * np.minimum(V_t, 0)
        # loss = (-1 / (1 + omega) ** t) * V_t
        return loss, jnp.zeros(loss.shape)


class NoisyLossModel(LossModel):
    def __init__(self, scale: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = scale

    def __call__(self, V_t, t, omega):
        loss, _ = super(NoisyLossModel, self).__call__(V_t, t, omega)
        key = self.key.next_seed()
        jax_loss = jnp.asarray(loss)
        rloss, log_probs = JaxGaussian.sample(key, jax_loss, self.scale)
        return rloss, log_probs


class PreferencePrior(ModelBase):
    def __init__(self, l_bar: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l_bar = l_bar


class SigmoidPreferencePrior(PreferencePrior):
    def __call__(self, Lt):
        return jax.nn.sigmoid(self.l_bar - Lt)


class ExponentialPreferencePrior(PreferencePrior):
    """
    k is an empirical constant related to stakeholder loss aversion
    k = -ln(p*)/L* where p* is the stakeholder's probability that loss will surpass L*
    """

    def __init__(self, l_bar: float, p_star: float, l_star: float, *args, **kwargs):
        super().__init__(l_bar)
        self.k = -jnp.log(p_star) / l_star

    def __call__(self, Lt):
        return self.k * jnp.exp(-self.k * Lt)


class UniformPreferencePrior(PreferencePrior):
    def __call__(self, Lt):
        return np.ones(Lt.shape) / self.l_bar


class RiskModel(ModelBase):
    def __init__(self, preference_prior, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preference_prior = preference_prior

    def compute_entropy(self, Lt, Lt_logprob, Vt):
        raise NotImplementedError

    def __call__(self, Lt, Lt_logprob, Vt):
        # this printing is important for evolving the preference model
        sample_mean = jnp.log(self.preference_prior(Lt)).mean(axis=0)
        entropy = self.compute_entropy(Lt, Lt_logprob, Vt)
        Gt = - entropy - sample_mean
        return Gt, entropy, sample_mean


class DifferentialEntropyRiskModel(RiskModel):
    def compute_entropy(self, Lt, Lt_logprob, Vt):
        ent = entr(Lt)
        if np.any(ent == float('-inf')):
            warnings.warn("-inf encountered in entropy")
            ent = np.where(ent == -float('inf'), -10, ent)
        return ent


class MonteCarloRiskModel(RiskModel):
    def compute_entropy(self, Lt, Lt_logprob, Vt):
        return Lt_logprob.mean(axis=0)


class WorldModel(ModelBase):
    """
    Main model that computes risk etc. through simulation of fishery evolution.
    Uses `n_montecarlo` MonteCarlo predictive simulations at a given realworld
    time step to calculate statistics.
    """

    def __init__(
        self,
        params,
        n_montecarlo,
        real_horizon,
        plan_horizon,
        dynamics,
        policy,
        revenue_model,
        cost_model,
        loss_model,
        risk_model,
        debug=False,
        omega_scale=1,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.params = params
        self.n_montecarlo = n_montecarlo
        self.dynamics = dynamics
        self.real_horizon = real_horizon
        self.plan_horizon = plan_horizon
        self.policy = policy
        self.revenue_model = revenue_model
        self.cost_model = cost_model
        self.loss_model = loss_model
        self.risk_model = risk_model
        self.debug = debug
        self.omega_scale = omega_scale

        # Can override logger
        if self.debug:
            logger.setLevel(logging.DEBUG)

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
        """
        Shapes of variables (like Bt, Ct, ...) will be
        (m, num_param_batches)
        where m is either 1 for the "real" timestep or n_montecarlo for planning
        """
        params = self.sample_policy(old_params)
        observed = self.dynamics(params)
        Bt = observed[-1]
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            Revt = self.revenue_model(Bt, params.qE)
        Ct = self.cost_model(params.qE)
        Vt = Revt - Ct
        Vt = jnp.nan_to_num(np.array(Vt), copy=False, nan=0.)
        Lt, Lt_logprob = self.loss_model(Vt, t, params.w)

        # update params with new fish biomass
        params = Params(
            B=Bt,
            w=params.w,
            r=params.r,
            k=params.k,
            qE=params.qE
        )

        return Lt, Lt_logprob, Vt, Bt, params

    def plan(self, params):
        Rt_sim = 0.
        for t_plan in range(self.plan_horizon):
            Lt, Lt_logprob, Vt, Bt, params = self.timestep(t_plan, params)
            Gt, entropy, sample_mean = self.risk_model(Lt, Lt_logprob, Vt)
            Rt_sim += Gt
        logger.debug("plan last: -entropy {} | -sample_mean {} | risk {}".format(-entropy, -sample_mean, Gt))

        return Rt_sim

    def stack_params(self, params: list) -> Params:
        Bs = jnp.vstack([np.array(p.B) for p in params])
        # ws = jnp.vstack([jax.random.lognormal(self.key.next_seed(), shape=self.params.w.shape) for p in params])
        ws = jnp.vstack([np.array(p.w) for p in params])
        rs = jnp.vstack([np.array(p.r) for p in params])
        ks = jnp.vstack([np.array(p.k) for p in params])
        qEs = jnp.vstack([np.array(p.qE) for p in params])
        return Params(Bs, ws, rs, ks, qEs)

    def get_montecarlo_params(self):
        """
        Return a replicated stack of current params for running MC predictive simulations
        Returned object is a Params object where each param is an n_montecarlo x num_param_batches size
        The params are identical across n_montecarlo dimension, but differ across num_param_batches dimension
        """
        param_list = [copy(self.params) for _ in range(self.n_montecarlo)]
        return self.stack_params(param_list)

    def __call__(self):
        es = []
        bs = []
        vs = []
        rts = []
        for t_sim in range(self.real_horizon):
            es.append(self.params.qE)
            # behaviour of agent is myopic in that it doesn't care about planning risk
            _, _, Vt_sim, Bt_sim, self.params = self.timestep(0, self.params)
            sim_params = self.get_montecarlo_params()
            Rt_sim = self.plan(sim_params)
            bs.append(Bt_sim)
            vs.append(Vt_sim)
            rts.append(Rt_sim)
        return Output(Es=es, Bs=bs, Vs=vs, Rts=rts)
