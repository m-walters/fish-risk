#!/usr/bin/env python3

import pymc as pm
import numpy as np
from sim.utils import Params
from sim.models import EulerMaruyamaDynamics, \
    RevenueModel, CostModel, RiskMitigationPolicy, \
    ProfitMaximizingPolicy, \
    LossModel, SoftmaxPreferencePrior, UniformPreferencePrior, \
    RiskModel, Model


t_end = 100
num_points = 20
D = 100  # no Brownian motion
mc = 100
horizon = 20
P0 = 1
rho = -0.9
C0 = 1
gamma = -0.9
# maximum loss occurs when cost = 0, revenue = P0 * B_max ** (1 + rho)
B_max = 100000
l_bar = P0 * B_max ** (1 + rho)

lmbdas = np.linspace(0., 1000., 100)

dynamics = EulerMaruyamaDynamics(t_end, num_points, D, B_max)
revenue_model = RevenueModel(P0=P0, rho=rho)
cost_model = CostModel(C0=C0, gamma=gamma)
# policy = RiskMitigationPolicy(revenue_model, cost_model, lmbda=0.1)
policy = ProfitMaximizingPolicy(revenue_model, cost_model)
loss_model = LossModel()
preference_prior = SoftmaxPreferencePrior(l_bar)
# preference_prior = UniformPreferencePrior(l_bar)
risk_model = RiskModel(preference_prior)

NUM_PARAM_BATCHES = 1

with pm.Model() as pm_model:  # this is a pymc model and in particular the "with...as..." syntax means all assignments in this block are associated with this model's context!
    B0 = pm.Normal("B", mu=90000, sigma=2000)
    # w = pm.LogNormal('w', mu=1., sigma=0.00001)
    w = pm.Uniform('w', lower=0., upper=1.)
    r = pm.Normal("r", mu=0.2, sigma=0.06)
    k = pm.Normal("k", mu=100000, sigma=10000)
    qE = pm.Uniform("qE", lower=0., upper=1.)

    samples = pm.sample_prior_predictive(samples=NUM_PARAM_BATCHES)

p = Params(**samples.prior)

experimental_model = Model(
    p,
    mc,
    dynamics,
    horizon,
    policy,
    revenue_model,
    cost_model,
    loss_model,
    risk_model,
    debug=True,
)
experimental_model()
