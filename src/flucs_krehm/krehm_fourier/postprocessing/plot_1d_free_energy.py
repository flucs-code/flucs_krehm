import argparse
import pathlib as pl

import matplotlib.pyplot as plt
import numpy as np

from flucs.postprocessing import FlucsPostProcessing
from flucs.utilities.messages import flucsprint


def plot_1d_free_energy(post, args):

    # Alias arguments
    dimension_name = args.dimension
    groups = args.groups
    fraction = args.fraction

    variable = f"free_energy_1d/{dimension_name}_spectra/W"

    # Get valid files for the specified variable
    nc_paths = post.get_valid_netcdf_paths(variable)

    # Initialise plotting
    fig_w, ax_w = plt.subplots(1, 1, layout="constrained")

    ymin = 1e-6
    xscale = "log"
    yscale = "log"

    # Components of free energy
    components = {
        "W_uperp": f"free_energy_1d/{dimension_name}_spectra/W_uperp",
        "W_dens" : f"free_energy_1d/{dimension_name}_spectra/W_dens",
        "W_bperp": f"free_energy_1d/{dimension_name}_spectra/W_bperp",
        "W_upar" : f"free_energy_1d/{dimension_name}_spectra/W_upar",
    }

    # Iterate over output files
    for index, nc_path in enumerate(nc_paths):

        # Assign identifiers
        sim_label = pl.Path(nc_path).parent.name
        sim_color = plt.cm.rainbow(np.linspace(0, 1, len(nc_paths)))[index]

        # Read data from netCDF file
        time = post.load_netcdf_variable(
            nc_path, "time", groups=groups,
        )[0]
        free_energy, _, dims_dicts = post.load_netcdf_variable(
            nc_path, variable, groups=groups,
        )

        # Validate dimension
        dims = next(dims for dims in reversed(dims_dicts) if dims)
        if len(dims) != 1:
            raise ValueError(
                f"Expected a 1D variable, but '{variable}' has "
                f"dimensions {list(dims)}."
            )
        _, dimension = next(iter(dims.items()))

        # Mask for logarithmic axes
        mask = dimension >= 0.0
        dimension = dimension[mask]
        free_energy = free_energy[:, mask]

        # Time average
        mask_time = time >= (
            np.min(time) + (1.0 - fraction) * (np.max(time) - np.min(time))
        )
        free_energy_avg = np.nanmean(free_energy[mask_time], axis=0)
        free_energy_std = np.nanstd(free_energy[mask_time], axis=0)
        free_energy_max = np.nanmax(free_energy_avg)
        free_energy_norm = free_energy_avg / free_energy_max

        flucsprint(
            rf"avg(W) = {np.sum(free_energy_avg):+.3e} "
            rf"± {np.sum(free_energy_std):.3e}, "
            rf"max(W_k) = {free_energy_max:.3e} ({sim_label})", 
            source=plot_1d_free_energy
        )

        # Plot data
        ax_w.plot(
            dimension,
            free_energy_norm,
            label=sim_label,
            linewidth=1.5,
            color=sim_color,
            linestyle="solid",
        )

        # Plot contributions if required
        if args.contributions:

            # Check if variables are present
            nc_variables = post.get_netcdf_variables(nc_path)
            missing_variables = [
                v for v in components.values() 
                if v not in nc_variables
            ]

            if missing_variables:
                flucsprint(
                    f"{sim_label}: free-energy contributions were requested but "
                    "are not available. Please check output files.",
                    source=plot_1d_free_energy,
                    message_type="warning"
                )
                continue

            # Initialise plotting
            fig_c, axs_c = plt.subplots(
                2, 2, layout="constrained", sharex=True, sharey=True
            )
            axs_c_dict = dict(zip(components, axs_c.flat))

            # Iterate over components
            for c, variable_c in components.items():
 
                # Extract and process data
                data_c = post.load_netcdf_variable(
                    nc_path,
                    variable_c,
                    groups=groups,
                )[0]

                data_c = data_c[:, mask]
                c_avg = np.nanmean(data_c[mask_time], axis=0)
                c_norm = c_avg / free_energy_max

                # Plot
                ax_c = axs_c_dict[c]

                ax_c.plot(
                    dimension,
                    c_norm,
                    label=None,
                    linewidth=1.5,
                    color=sim_color,
                    linestyle="solid",
                )
                ax_c.set_ylabel(rf"$W_\mathrm{{{c[2:]}}}/\mathrm{{max}}(W)$")

                ax_c.plot(
                    dimension,
                    free_energy_norm,
                    label=None,
                    linewidth=1.5,
                    color=sim_color,
                    linestyle="dashed",
                )

                # Set plotting options
                ax_c.set_xscale(xscale)
                ax_c.set_yscale(yscale)
                ax_c.set_ylim(ymin=ymin)
                ax_c.set_box_aspect(0.75)

            for ax in axs_c[-1, :]:
                ax.set_xlabel(dimension_name)

            # Save figure
            fig_c_name = (
                f"free_energy_contributions_vs_{dimension_name}_{sim_label}"
            )
            fig_c.canvas.manager.set_window_title(fig_c_name)

            post.save(
                fig_c,
                name=fig_c_name,
                suffix="png",
                save_kwargs={"dpi": 300},
            ) 

    # Setting plot options
    ax_w.set_xlabel(dimension_name)
    ax_w.set_ylabel(r"$W/\mathrm{max}(W)$")

    ax_w.set_xscale(xscale)
    ax_w.set_yscale(yscale)

    ax_w.set_ylim(ymin=ymin)

    ax_w.legend()

    # Save figure if required
    fig_w_name = f"free_energy_vs_{dimension_name}"
    fig_w.canvas.manager.set_window_title(fig_w_name)

    post.save(
        fig_w,
        name=fig_w_name,
        suffix="png",
        save_kwargs={"dpi": 300},
    )

    plt.show()

    return


if __name__ == "__main__":

    # Setup parser
    parser = argparse.ArgumentParser(
        parents=[FlucsPostProcessing.parser()],
        description=(
            "Plot the 1D free-energy spectrum for the isothermal KREHM system.",
        )
    )

    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        default=False,
        help="List all available variables to plot and exit.",
    )

    parser.add_argument(
        "--dimension",
        "-d",
        type=str,
        choices=["kz", "kx", "ky", "kperp"],
        default="kperp",
        help="Spectral dimension to plot. Default is kperp.",
    )

    parser.add_argument(
        "--fraction",
        "-f",
        type=float,
        default=0.2,
        help=(
            "Final fraction of the selected time series over which to average. "
            "Default is 0.2, i.e. the final 20 percent."
        ),
    )

    parser.add_argument(
        "--contributions",
        "-c",
        action="store_true",
        default=False,
        help="Additionally plot the four free-energy contribution spectra.",
    )

    args = parser.parse_args()

    # Initialise post-processing object
    post = FlucsPostProcessing(
        io_paths=args.io_path,
        save_directory=args.save_directory,
        output_files=["output.1d.nc"],
        constraint="none",
    )

    if args.list:
        post.list_netcdf_variables()
        exit()

    # Call function
    plot_1d_free_energy(post, args)