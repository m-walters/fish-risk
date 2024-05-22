#!/usr/bin/env python3

import pymc as pm
import numpy as np
from sim.utils import Params, JaxRKey
from sim.models import EulerMaruyamaDynamics, \
    RevenueModel, CostModel, RiskMitigationPolicy, \
    ProfitMaximizingPolicy, \
    LossModel, NoisyLossModel, SigmoidPreferencePrior, \
    UniformPreferencePrior, DifferentialEntropyRiskModel, \
    MonteCarloRiskModel, Model
from sim.plotting import plot_outputs


t_end = 100
num_points = 10
D = 1
mc = 100
horizon = 20
P0 = 1.2
rho = -0.8
C0 = 1.2
gamma = -0.5
# maximum loss occurs when cost = 0, revenue = P0 * B_max ** (1 + rho)
B_max = 100000
l_bar = 0. #P0 * B_max ** (1 + rho)
loss_scale = 0.1

lmbdas = np.linspace(0., 1000., 100)

seed = 1234
jax_rkey = JaxRKey(seed)

dynamics = EulerMaruyamaDynamics(t_end, num_points, D, B_max)
revenue_model = RevenueModel(P0=P0, rho=rho)
cost_model = CostModel(C0=C0, gamma=gamma)
# policy = RiskMitigationPolicy(revenue_model, cost_model, lmbda=0.1)
policy = ProfitMaximizingPolicy(revenue_model, cost_model)
# loss_model = LossModel()
loss_model = NoisyLossModel(jax_rkey, loss_scale)
preference_prior = SigmoidPreferencePrior(l_bar)
# preference_prior = UniformPreferencePrior(l_bar)
risk_model = DifferentialEntropyRiskModel(preference_prior)
# risk_model = MonteCarloRiskModel(preference_prior)

NUM_PARAM_BATCHES = 1

with pm.Model() as pm_model:  # this is a pymc model and in particular the "with...as..." syntax means all assignments in this block are associated with this model's context!
    B0 = pm.Normal("B", mu=90000, sigma=2000)
    w = pm.LogNormal('w', mu=1., sigma=0.01)
    r = pm.Normal("r", mu=0.2, sigma=0.06)
    k = pm.Normal("k", mu=100000, sigma=10000)
    qE = pm.Uniform("qE", lower=0., upper=1.)

    samples = pm.sample_prior_predictive(samples=NUM_PARAM_BATCHES)

p = Params(**samples.prior)

omegas = np.arange(0, 0.5, 0.05)
outputs = []
for w in omegas:
    print('Simulating with omega = {}\n'.format(w))
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
        jax_rkey,
        debug=True,
        omega_scale=w,
    )
    output = experimental_model()
    outputs.append(output)

plot_outputs(outputs, omegas)
