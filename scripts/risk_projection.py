import logging
import os

import hydra
import numpy as np
import pymc as pm
from matplotlib import pyplot as plt
from omegaconf import DictConfig, ListConfig, OmegaConf

from sim import models, plotting, utils

RESULTS_DIR = "../results"


@hydra.main(version_base=None, config_path="../configs")
def main(cfg: DictConfig):
    """
    Set the parameters and run the sim
    """
    log_level = cfg.get("log_level", "INFO")
    if log_level == "DEBUG":
        logging.basicConfig(level=logging.DEBUG)
    elif log_level == "INFO":
        logging.basicConfig(level=logging.INFO)
    elif log_level == "WARNING":
        logging.basicConfig(level=logging.WARNING)

    # Get logger *after* setting the level
    logger = logging.getLogger("main")
    # Print our config
    logger.info(f"CONFIG\n{OmegaConf.to_yaml(cfg)}")

    dynamics = getattr(models, cfg.DE_dynamics.model)(**cfg.DE_dynamics, seed=cfg.seed)
    revenue_model = getattr(models, cfg.revenue.model)(**cfg.revenue, seed=cfg.seed)
    cost_model = getattr(models, cfg.cost.model)(**cfg.cost, seed=cfg.seed)
    policy = getattr(models, cfg.policy.model)(
        revenue_model=revenue_model, cost_model=cost_model, **cfg.policy, seed=cfg.seed
    )
    loss_model = getattr(models, cfg.loss.model)(**cfg.loss, seed=cfg.seed)
    preference_prior = getattr(models, cfg.preference_prior.model)(**cfg.preference_prior, seed=cfg.seed)
    risk_model = getattr(models, cfg.risk.model)(preference_prior=preference_prior, **cfg.risk, seed=cfg.seed)

    fish_params = DictConfig(cfg.fish_params)

    with pm.Model() as pm_model:  # this is a pymc model and in particular the "with...as..." syntax means all assignments in this block are associated with this model's context!
        # B0 = pm.MutableData("B", fish_params.B0)
        r = pm.Uniform("r", 0.1, 0.5)
        # k = pm.ConstantData("k", fish_params.k)
        # w = pm.MutableData("w", cfg.run_params.omega.min)
        samples = pm.sample_prior_predictive(samples=cfg.run_params.num_param_batches)

    # qEs are determined from a list of values
    if isinstance(fish_params.qE, ListConfig):
        # Bespoke list of values
        qEs = np.asarray(fish_params.qE)
    else:
        # Linear spacing
        qEs = np.arange(
            fish_params.qE.min,
            fish_params.qE.max,
            fish_params.qE.step
        )
        qEs = np.round(qEs, 2)

    # There's just one though ;)
    omegas = np.asarray(cfg.run_params.omega)
    w = omegas[0]

    outputs = []
    for qE in qEs:
        print('Simulating with qE = {}\n'.format(qE))

        p = utils.Params(
            B=cfg.fish_params.B0 * np.ones((1, cfg.run_params.num_param_batches)),
            qE=qE * np.ones((1, cfg.run_params.num_param_batches)),
            w=w * np.ones((1, cfg.run_params.num_param_batches)),
            # cfg.fish_params.r * np.ones((1, cfg.run_params.num_param_batches)),
            k=cfg.fish_params.k * np.ones((1, cfg.run_params.num_param_batches)),
            **samples.prior
        )

        experimental_model = models.WorldModel(
            p,
            cfg.run_params.num_param_batches,
            cfg.world_sim.n_montecarlo,
            cfg.world_sim.real_horizon,
            cfg.world_sim.plan_horizon,
            dynamics,
            policy,
            revenue_model,
            cost_model,
            loss_model,
            risk_model,
        )
        output = experimental_model()
        outputs.append(output)

    # Store the outputs from this run as xArray Dataset
    projection_results = utils.ProjectionResults(
        qEs, outputs, cfg.world_sim.real_horizon, cfg.run_params.num_param_batches
    )
    ds = projection_results.to_dataset()
    # Automatically save latest
    latest_dir = RESULTS_DIR + "/latest"
    os.makedirs(latest_dir, exist_ok=True)
    OmegaConf.save(config=cfg, f=f"{latest_dir}/config.yaml")
    projection_results.save_ds(ds, f"{latest_dir}/projection_results.nc")

    # If a name is provided, save there too
    if "name" in cfg:
        save_dir: str = cfg.get("save_dir", RESULTS_DIR)
        run_dir: str = os.path.join(save_dir, cfg.name)
        os.makedirs(run_dir, exist_ok=True)
        OmegaConf.save(config=cfg, f=f"{run_dir}/config.yaml")
        projection_results.save_ds(ds, f"{run_dir}/projection_results.nc")

    # Display a plot of results
    plotter = plotting.Plotter(ds)
    plotter.projection_plot(save_path=f"{latest_dir}/projection_results.png")
    plt.show()


if __name__ == '__main__':
    main()
