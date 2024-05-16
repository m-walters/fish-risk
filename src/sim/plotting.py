import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.pyplot import cm
import numpy as np


def plot_outputs(outputs, omegas):
    color = iter(cm.rainbow(np.linspace(0, 1, len(omegas))))
    fig, axs = plt.subplots(3, sharex=True)

    for output, omega in zip(outputs, omegas):
        c = next(color)
        plot_data(output=output, omega=omega, color=c, axs=axs)

    axs[0].set_ylabel('Biomass')
    axs[1].set_ylabel('Revenue')
    axs[2].set_ylabel('Risk')

    axs[2].set_xlabel('Horizon')
    axs[2].xaxis.get_major_locator().set_params(integer=True)

    axs[2].legend(
        bbox_to_anchor=(0.9425, 3.91),
        bbox_transform=axs[2].transAxes,
        ncol=len(omegas)/2,
    )

    plt.savefig('../results/fish_sim.pdf')

def plot_data(output, omega, color, axs):
    time = np.arange(len(output.Rts))
    axs[0].plot(time, output.Bs, color=color, label='w={:.1f}'.format(omega))
    axs[1].plot(time, output.Vs, color=color, label='w={:.1f}'.format(omega))
    axs[2].plot(time, output.Rts, color=color, label='w={:.1f}'.format(omega))
