from sim.plotting import Plotter
from matplotlib import pyplot as plt


plotter = Plotter("results/latest/omegas_latest.nc")
plotter.omega_quad_plot(save_path=f"results/latest/omega_latest-log.png")
plt.show()
