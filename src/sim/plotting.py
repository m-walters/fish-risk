import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.pyplot import cm
import numpy as np


def plot_outputs(outputs, omegas):
    color = iter(cm.rainbow(np.linspace(0, 1, len(omegas))))
    fig, axs = plt.subplots(2, 2, sharex=True)

    plt.subplots_adjust(wspace=0.4)

    for output, omega in zip(outputs, omegas):
        c = next(color)
        plot_data(output=output, omega=omega, color=c, axs=axs)

    axs[0, 0].set_ylabel('Biomass')
    axs[0, 1].set_ylabel('Profit')
    axs[1, 0].set_ylabel('Risk')
    axs[1, 1].set_ylabel('E*')

    for i in range(2):
        lower_ax = axs[1, i]
        lower_ax.set_xlabel('Horizon')
        lower_ax.xaxis.get_major_locator().set_params(integer=True)

    axs[1, 1].legend(
        bbox_to_anchor=(0.7925, 2.51),
        bbox_transform=axs[1, 1].transAxes,
        ncol=len(omegas),
    )

    plt.savefig('../results/fish_sim.pdf')

def plot_data(output, omega, color, axs):
    time = np.arange(len(output.Rts))
    axs[0, 0].plot(time, output.Bs, color=color, label='w={:.1f}'.format(omega))
    axs[0, 1].plot(time, output.Vs, color=color, label='w={:.1f}'.format(omega))
    axs[1, 0].plot(time, output.Rts, color=color, label='w={:.1f}'.format(omega))
    axs[1, 1].plot(time, output.Es, color=color, label='w={:.1f}'.format(omega))
