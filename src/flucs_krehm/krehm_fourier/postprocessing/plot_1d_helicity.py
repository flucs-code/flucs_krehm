import argparse
import pathlib as pl

import matplotlib.pyplot as plt
import numpy as np

from flucs.postprocessing import FlucsPostProcessing
from flucs.utilities.messages import flucsprint


def plot_1d_helicity(post, args):

    # Alias arguments
    dimension_name = args.dimension
    groups = args.groups
    fraction = args.fraction

    variable = f"helicity_1d/{dimension_name}_spectra/H"

    # Get valid files for the specified variable
    nc_paths = post.get_valid_netcdf_paths(variable)

    # Initialise plotting
    fig_h, ax_h = plt.subplots(1, 1, layout="constrained")

    ymin = 1e-6
    xscale = "log"
    yscale = "log"

    # Components of helicity
    components = {
        "H_apar": f"helicity_1d/{dimension_name}_spectra/H_apar",
        "H_upar": f"helicity_1d/{dimension_name}_spectra/H_upar",
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
        helicity, _, dims_dicts = post.load_netcdf_variable(
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
        helicity = helicity[:, mask]

        # Time average
        mask_time = time >= (
            np.min(time) + (1.0 - fraction) * (np.max(time) - np.min(time))
        )
        helicity_avg = np.nanmean(helicity[mask_time], axis=0)
        helicity_std = np.nanstd(helicity[mask_time], axis=0)
        helicity_max = np.nanmax(np.abs(helicity_avg))
        helicity_norm = helicity_avg / helicity_max

        flucsprint(
            rf"avg(H) = {np.sum(helicity_avg):+.3e} "
            rf"± {np.sum(helicity_std):.3e}, "
            rf"max(|H_k|) = {helicity_max:.3e} ({sim_label})",
            source=plot_1d_helicity
        )

        # Plot data
        ax_h.plot(
            dimension,
            np.abs(helicity_norm),
            label=sim_label,
            linewidth=1.5,
            color=sim_color,
            linestyle="solid"
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
                    f"{sim_label}: helicity contributions were requested but "
                    "are not available. Please check output files.",
                    source=plot_1d_helicity,
                    message_type="warning"
                )
                continue

            # Initialise plotting
            fig_c, axs_c = plt.subplots(
                1, 2, layout="constrained", sharex=True, sharey=True
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
                c_norm = c_avg / helicity_max

                # Plot
                ax_c = axs_c_dict[c]
                ax_c.plot(
                    dimension,
                    np.abs(c_norm),
                    label=None,
                    linewidth=1.5,
                    color=sim_color,
                    linestyle="solid",
                )
                ax_c.plot(
                    dimension,
                    np.abs(helicity_norm),
                    label=None,
                    linewidth=1.5,
                    color=sim_color,
                    linestyle="dashed",
                )

                ax_c.set_ylabel(
                    rf"$|H_\mathrm{{{c[2:]}}}|/\mathrm{{max}}(|H|)$"
                )

                # Set plotting options
                ax_c.set_xscale(xscale)
                ax_c.set_yscale(yscale)
                ax_c.set_ylim(ymin=ymin)
                ax_c.set_box_aspect(0.75)

            for ax in axs_c:
                ax.set_xlabel(dimension_name)

            # Save figure
            fig_c_name = (
                f"helicity_contributions_vs_{dimension_name}_{sim_label}"
            )
            fig_c.canvas.manager.set_window_title(fig_c_name)

            post.save(
                fig_c,
                name=fig_c_name,
                suffix="png",
                save_kwargs={"dpi": 300},
            )

    # Setting plot options
    ax_h.set_xlabel(dimension_name)
    ax_h.set_ylabel(r"$|H|/\mathrm{max}(|H|)$")

    ax_h.set_xscale(xscale)
    ax_h.set_yscale(yscale)
    ax_h.set_ylim(ymin=ymin)
    ax_h.legend()

    # Save figure if required
    fig_h_name = f"helicity_vs_{dimension_name}"
    fig_h.canvas.manager.set_window_title(fig_h_name)

    post.save(
        fig_h,
        name=fig_h_name,
        suffix="png",
        save_kwargs={"dpi": 300},
    )

    plt.show()

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        parents=[FlucsPostProcessing.parser()],
        description=(
            "Plot 1D helicity spectrum for the isothermal KREHM system."
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
        help="Additionally plot the two helicity contribution spectra.",
    )

    args = parser.parse_args()

    post = FlucsPostProcessing(
        io_paths=args.io_path,
        save_directory=args.save_directory,
        output_files=["output.1d.nc"],
        constraint="none",
    )

    if args.list:
        post.list_netcdf_variables()
        exit()

    plot_1d_helicity(post, args)
