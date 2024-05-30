import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import cm
import pandas as pd
import seaborn as sns


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

    plt.savefig('results/fish_sim.pdf')


def plot_data(output, omega, color, axs):
    df = pd.DataFrame(output.Bs).melt()
    sns.lineplot(x="variable", y="value", data=df, color=color, label='w={:.1f}'.format(omega), ax=axs[0, 0], legend=False)
    df = pd.DataFrame(output.Vs).melt()
    sns.lineplot(x="variable", y="value", data=df, color=color, label='w={:.1f}'.format(omega), ax=axs[0, 1], legend=False)
    df = pd.DataFrame(output.Rts).melt()
    sns.lineplot(x="variable", y="value", data=df, color=color, label='w={:.1f}'.format(omega), ax=axs[1, 0], legend=False)
    df = pd.DataFrame(output.Es).melt()
    sns.lineplot(x="variable", y="value", data=df, color=color, label='w={:.1f}'.format(omega), ax=axs[1, 1], legend=False)
