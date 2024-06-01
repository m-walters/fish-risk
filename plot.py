from sim.plotting import Plotter
from matplotlib import pyplot as plt


plotter = Plotter(
    "results/tmp/omega_results.nc",
    sns_context="notebook"
)

save_path = "results/tmp/fig2.png"
plotter.omega_quad_plot(
    save_path=save_path,
    plot_kwargs={
        "figsize": (8, 4),
    },
)
plt.show()
