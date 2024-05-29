#!/usr/bin/env python3

import pymc as pm

from sim import models, utils
from sim.utils import Params

sim_horizon = 100
sim_steps = 20
D = 1000
mc = 10
horizon = 20
P0 = 1
rho = -0.9
C0 = 1
gamma = -0.9
# maximum loss occurs when cost = 0, revenue = 10000 ** rho * P0
B_max = 50000
l_bar = P0 * B_max ** rho

dynamics = models.EulerMaruyamaDynamics(sim_horizon, sim_steps, D, B_max)
revenue_model = models.RevenueModel(P0=P0, rho=rho)
cost_model = models.CostModel(C0=C0, gamma=gamma)
policy = models.ProfitMaximizingPolicy(revenue_model, cost_model)
loss_model = models.LossModel()
preference_prior = models.SoftmaxPreferencePrior(l_bar)
risk_model = models.RiskModel(preference_prior)

NUM_PARAM_BATCHES = 10

with pm.Model() as pm_model:  # this is a pymc model and in particular the "with...as..." syntax means all assignments in this block are associated with this model's context!
    B0 = pm.Normal("B", mu=90000, sigma=2000)
    w = pm.LogNormal('w', mu=1., sigma=0.00001)
    r = pm.Normal("r", mu=0.2, sigma=0.06)
    k = pm.Normal("k", mu=100000, sigma=10000)
    qE = pm.Normal("qE", mu=0.01, sigma=0.01)

    samples = pm.sample_prior_predictive(samples=NUM_PARAM_BATCHES)

p = utils.Params(**samples.prior)

# mu = 1.
# for experimental_st_dev in np.arange(0.2, 2.0, 0.2):
#     # we run a full loop of testing and graphing on that range.
#     experimental_model = EvolvedPreferenceModel(
#         p,
#         mc,
#         dynamics,
#         horizon,
#         policy,
#         revenue_model,
#         cost_model,
#         loss_model,
#         risk_model,
#         debug=True,
#         mu=mu,
#         sigma=experimental_st_dev,
#         mu_updater=lambda x: x, # we're not doing the second bullet point just yet
#         sigma_updater=lambda x: x
#     )
#     experimental_model()
