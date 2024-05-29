#!/usr/bin/env python3

import pymc as pm
import numpy as np
from sim import utils
from sim import models
from sim import plotting
import logging

logging.basicConfig(level=logging.INFO)

# Sim params
# > Horizon and num steps for the model's inner simulation of the future
inner_horizon = 100
inner_steps = 10
D = 100  # Inner sim brownian motion diffusion
n_montecarlo = 100
real_horizon = 20  # Realworld horizon steps

P0 = 1.2  # Profit
rho = -0.8  # Profit
C0 = 1.2  # Cost
gamma = -0.5  # Cost

# maximum loss occurs when cost = 0, revenue = P0 * B_max ** (1 + rho)
B_max = 100000
l_bar = 0. #P0 * B_max ** (1 + rho)
loss_scale = 0.1

lmbdas = np.linspace(0., 1000., 100)

seed = 1234
jax_rkey = utils.JaxRKey(seed)

dynamics = models.EulerMaruyamaDynamics(inner_horizon, inner_steps, D, B_max)
revenue_model = models.RevenueModel(P0=P0, rho=rho)
cost_model = models.CostModel(C0=C0, gamma=gamma)
# policy = models.RiskMitigationPolicy(revenue_model, cost_model, lmbda=0.1)
policy = models.ProfitMaximizingPolicy(revenue_model, cost_model)
# loss_model = models.LossModel()
loss_model = models.NoisyLossModel(jax_rkey, loss_scale)
# preference_prior = models.ExponentialPreferencePrior(l_bar, 0.4, 1.0)
preference_prior = models.SigmoidPreferencePrior(l_bar)
# risk_model = models.RiskModel(preference_prior)
risk_model = models.DifferentialEntropyRiskModel(preference_prior)

NUM_PARAM_BATCHES = 1

with pm.Model() as pm_model:  # this is a pymc model and in particular the "with...as..." syntax means all assignments in this block are associated with this model's context!
    B0 = pm.Normal("B", mu=90000, sigma=2000)
    w = pm.LogNormal('w', mu=1., sigma=0.01)
    r = pm.Normal("r", mu=0.2, sigma=0.06)
    k = pm.Normal("k", mu=100000, sigma=10000)
    qE = pm.Uniform("qE", lower=0., upper=1.)

    samples = pm.sample_prior_predictive(samples=NUM_PARAM_BATCHES)

p = utils.Params(**samples.prior)

omegas = np.arange(0, 0.5, 0.05)
outputs = []
for w in omegas:
    print('Simulating with omega = {}\n'.format(w))
    experimental_model = models.Model(
        p,
        n_montecarlo,
        dynamics,
        real_horizon,
        inner_horizon,
        policy,
        revenue_model,
        cost_model,
        loss_model,
        risk_model,
        jax_rkey,
        debug=False,
        omega_scale=w,
    )
    output = experimental_model()
    outputs.append(output)

plotting.plot_outputs(outputs, omegas)