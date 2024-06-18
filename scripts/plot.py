from sim.plotting import Plotter
from matplotlib import pyplot as plt


plotter = Plotter(
    "../results/latest/pref_evolve_results.nc",
    sns_context="notebook"
)

save_path = "../results/latest/fig.png"

fig, axs = plotter.pref_evolve_plot_2(
    "p_star",
    "p*",
    # plot_kwargs={
    #     "figsize": (8, 4),
    # },
)

# axs[1,1].set_ylim(bottom=0)

plt.savefig(save_path)

plt.show()
