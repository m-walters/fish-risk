from sim.plotting import Plotter
from matplotlib import pyplot as plt


plotter = Plotter(
    "../results/latest/projection_results.nc",
    sns_context="notebook"
)

save_path = "../results/latest/fig.png"
plotter.projection_plot(
    save_path=save_path,
    # plot_kwargs={
    #     "figsize": (8, 4),
    # },
)
plt.show()
