from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr

"""
Note https://docs.xarray.dev/en/stable/user-guide/plotting.html
for plotting with xarray objects
"""


class Plotter:
    """
    Plotting util for xArray Datasets
    """

    def __init__(
        self,
        ds_or_path: Union[xr.Dataset, str],
        sns_context: str = "notebook",  # or "paper", "talk", "poster"
    ):
        if isinstance(ds_or_path, xr.Dataset):
            self.ds = ds_or_path
        else:
            self.ds = xr.open_dataset(ds_or_path)

        # Initialize seaborn
        sns.set()
        sns.set_context(sns_context)
        sns.set_palette("colorblind")

    @staticmethod
    def get_color_wheel():
        """
        Return a color generator for the current seaborn palette
        """
        return iter(sns.color_palette())

    @staticmethod
    def subplots(nrow, ncol, **kwargs):
        return plt.subplots(nrow, ncol, **kwargs)

    def omega_quad_plot(self, fig=None, axs=None, save_path=None):
        """
        For an OmegaResults dataset
        Generate a 2x2 plot with Biomass, Profit, Risk, and E*
        In each plot, we reduce across the batch axis, and color by omega
        """
        if axs is None:
            fig, axs = self.subplots(2, 2, sharex=True, figsize=(12, 6))
            plt.subplots_adjust(wspace=0.4)

        colors = self.get_color_wheel()
        for omega in self.ds.omega:
            c = next(colors)
            for i, var in enumerate(['B', 'V', 'Rt', 'E']):
                # This produces a pivot table with time as index and batch as columns
                pivot = self.ds[var].sel(omega=omega).to_pandas()
                # Melt it to go from wide to long form, with batch as a variable, and our var as value
                melted = pivot.melt(var_name='batch', value_name=var, ignore_index=False)
                label = f"w={np.round(omega.values, 2)}"
                sns.lineplot(
                    x="time", y=var, data=melted, color=c, label=label, ax=axs[i // 2, i % 2],
                    legend=False
                )

        # Title
        # fig.suptitle('')
        # Set Biomass y-scale to be log and min at 0
        # axs[0, 0].set_yscale('log')
        axs[0, 0].set_ylim(bottom=0)

        axs[0, 0].set_ylabel('Biomass')
        axs[0, 1].set_ylabel('Profit')
        axs[1, 0].set_ylabel('Risk')
        axs[1, 1].set_ylabel('E*')

        for i in range(2):
            lower_ax = axs[1, i]
            lower_ax.set_xlabel('Horizon')
            lower_ax.xaxis.get_major_locator().set_params(integer=True)

        axs[0, 1].legend(
            # bbox_to_anchor=(0.7925, 2.51),
            # bbox_transform=axs[1, 1].transAxes,
            # ncol=len(self.ds.omega),
            # loc='center left',
            # title='Omega'
        )

        if save_path:
            plt.savefig(save_path)

        return fig, axs
