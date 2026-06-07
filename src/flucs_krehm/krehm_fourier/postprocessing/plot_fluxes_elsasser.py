import argparse
import pathlib as pl

import matplotlib.pyplot as plt
import numpy as np

from flucs.postprocessing import FlucsPostProcessing


def plot_fluxes_elsasser(post, args):

    # Alias arguments
    dimension_name = args.dimension
    groups = args.groups
    fraction = args.fraction

    variables = {
        "plus": {
            "nonlinear": f"fluxes/{dimension_name}_fluxes/dWpdt_nonlinear",
            "forcing": f"fluxes/{dimension_name}_fluxes/dWpdt_forcing",
            "hyperdissipation": (
                f"fluxes/{dimension_name}_fluxes/dWpdt_hyperdissipation"
            ),
        },
        "minus": {
            "nonlinear": f"fluxes/{dimension_name}_fluxes/dWmdt_nonlinear",
            "forcing": f"fluxes/{dimension_name}_fluxes/dWmdt_forcing",
            "hyperdissipation": (
                f"fluxes/{dimension_name}_fluxes/dWmdt_hyperdissipation"
            ),
        },
    }

    # Get valid files for the specified variable
    nc_paths = post.get_valid_netcdf_paths(variables["plus"]["nonlinear"])

    # Initialise plotting
    fig, axs = plt.subplots(
        2, 1, layout="constrained", sharex=True, sharey=True
    )
    axs_dict = {"plus": axs[0], "minus": axs[1]}

    # Iterate over output files
    for index, nc_path in enumerate(nc_paths):

        # Assign identifiers
        sim_label = pl.Path(nc_path).parent.name
        sim_color = plt.cm.rainbow(np.linspace(0, 1, len(nc_paths)))[index]

        # Get injection rates (assumes same forcing for all groups)
        input_file = post.load_netcdf_input_files(nc_path, groups=groups)[-1]

        forcing_input = input_file["forcing"]
        if forcing_input["method"] == "elsasser":
            energy_injection_rate = forcing_input["energy_injection_rate"]
        elif forcing_input["method"] == "meyrand":
            energy_injection_rate = (
                + forcing_input["energy_injection_rate_phi"]
                + forcing_input["energy_injection_rate_apar"]
            )
        else:
            raise ValueError(
                f"Unsupported forcing method '{forcing_input['method']}'."
            )

        injection_imbalance = forcing_input["injection_imbalance"]
        injection_rates = {
            "plus": (
                energy_injection_rate * (1.0 + injection_imbalance) / 2.0
            ),
            "minus": (
                energy_injection_rate * (1.0 - injection_imbalance) / 2.0
            ),
        }

        # Read time from NetCDF file
        time = post.load_netcdf_variable(nc_path, "time", groups=groups)[0]

        # Time average
        mask_time = time >= (
            np.min(time) + (1.0 - fraction) * (np.max(time) - np.min(time))
        )

        dimensions = {}
        nonlinear_fluxes = {}
        nonlinear_fluxes_avg = {}

        # Iterate over Elsasser fields
        for sign in ["plus", "minus"]:
            nonlinear, _, dims_dicts = post.load_netcdf_variable(
                nc_path,
                variables[sign]["nonlinear"],
                groups=groups,
            )

            # Validate dimension
            dims = next(dims for dims in reversed(dims_dicts) if dims)
            if len(dims) != 1:
                raise ValueError(
                    f"Expected a 1D variable, but "
                    f"'{variables[sign]['nonlinear']}' has dimensions "
                    f"{list(dims)}."
                )
            _, dimension = next(iter(dims.items()))

            # Mask for logarithmic axis
            mask = dimension > 0.0
            dimensions[sign] = dimension[mask]
            nonlinear_fluxes[sign] = -nonlinear[:, mask] / energy_injection_rate
            nonlinear_fluxes_avg[sign] = np.nanmean(
                nonlinear_fluxes[sign][mask_time],
                axis=0,
            )

            # Plot nonlinear flux
            axs_dict[sign].plot(
                dimensions[sign],
                nonlinear_fluxes_avg[sign],
                label=sim_label,
                linewidth=1.5,
                color=sim_color,
                linestyle="solid",
            )

            # Plot budget terms if required
            if args.budget:
                dWdt_forcing = post.load_netcdf_variable(
                    nc_path,
                    variables[sign]["forcing"],
                    groups=groups,
                )[0]
                dWdt_hyperdissipation = post.load_netcdf_variable(
                    nc_path,
                    variables[sign]["hyperdissipation"],
                    groups=groups,
                )[0]

                forcing_flux = (
                    injection_rates[sign] - dWdt_forcing[:, mask]
                ) / energy_injection_rate
                hyperdissipation_flux = (
                    -dWdt_hyperdissipation[:, mask]
                    / energy_injection_rate
                )

                forcing_flux_avg = np.nanmean(forcing_flux[mask_time], axis=0)
                hyperdissipation_flux_avg = np.nanmean(
                    hyperdissipation_flux[mask_time],
                    axis=0,
                )

                axs_dict[sign].plot(
                    dimensions[sign],
                    forcing_flux_avg,
                    label=None,
                    linewidth=1.5,
                    color=sim_color,
                    linestyle="dashed",
                )
                axs_dict[sign].plot(
                    dimensions[sign],
                    hyperdissipation_flux_avg,
                    label=None,
                    linewidth=1.5,
                    color=sim_color,
                    linestyle="dashed",
                )

            # Plot requested Elsasser injection
            axs_dict[sign].axhline(
                injection_rates[sign] / energy_injection_rate,
                color=sim_color,
                linestyle="dotted",
                linewidth=1.0,
            )

        # Plot nonlinear flux evolution over time
        if args.time:

            # Initialise individual plot
            fig_time, axs_time = plt.subplots(
                2, 1, layout="constrained", sharex=True, sharey=True
            )
            axs_time_dict = {"plus": axs_time[0], "minus": axs_time[1]}

            fig_name_time = (
                f"elsasser_fluxes_vs_{dimension_name}_time_{sim_label}"
            )
            fig_time.canvas.manager.set_window_title(fig_name_time)

            # Get data in specified plotting window
            time_plot = time[mask_time]

            # Downsample data to prevent overcrowding
            count = min(50, len(time_plot))
            time_indices = np.linspace(
                0, len(time_plot), count, endpoint=False, dtype=int
            )

            # Set colormap
            norm = plt.Normalize(vmin=np.min(time_plot), vmax=np.max(time_plot))
            cmap = plt.cm.rainbow

            # Iterate over Elsasser fields
            for sign in ["plus", "minus"]:
                nonlinear_flux_plot = nonlinear_fluxes[sign][mask_time]

                # Iterate and plot
                for it in time_indices:
                    axs_time_dict[sign].plot(
                        dimensions[sign],
                        nonlinear_flux_plot[it],
                        linewidth=1.0,
                        color=cmap(norm(time_plot[it])),
                    )

                # Plot time average
                axs_time_dict[sign].plot(
                    dimensions[sign],
                    nonlinear_fluxes_avg[sign],
                    linewidth=2.0,
                    color="black",
                    linestyle="solid",
                )

                # Setting plot options
                axs_time_dict[sign].axhline(
                    0.0,
                    color="grey",
                    linestyle="solid",
                    linewidth=1.0,
                )
                axs_time_dict[sign].axhline(
                    injection_rates[sign] / energy_injection_rate,
                    color="grey",
                    linestyle="dotted",
                    linewidth=1.0,
                )
                axs_time_dict[sign].set_xscale("log")

            axs_time_dict["plus"].set_ylabel(r"$\Pi^+/\epsilon_W$")
            axs_time_dict["minus"].set_ylabel(r"$\Pi^-/\epsilon_W$")
            axs_time_dict["minus"].set_xlabel(dimension_name)

            colorbar = fig_time.colorbar(
                plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=axs_time,
            )
            colorbar.set_label("Time")

            # Save figure if required
            post.save(
                fig_time,
                name=fig_name_time,
                suffix="png",
                save_kwargs={"dpi": 300},
            )

    # Setting plot options
    for sign in ["plus", "minus"]:
        axs_dict[sign].axhline(
            0.0,
            color="grey",
            linestyle="solid",
            linewidth=1.0,
        )
        axs_dict[sign].set_xscale("log")

    axs_dict["plus"].set_ylabel(r"$\Pi^+/\epsilon_W$")
    axs_dict["minus"].set_ylabel(r"$\Pi^-/\epsilon_W$")
    axs_dict["minus"].set_xlabel(dimension_name)
    axs_dict["plus"].legend()

    # Save figure if required
    fig_name = f"elsasser_fluxes_vs_{dimension_name}"
    fig.canvas.manager.set_window_title(fig_name)

    post.save(
        fig,
        name=fig_name,
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
            "Plot the cumulative Elsasser free-energy transfers for the "
            "isothermal KREHM system."
        ),
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
        "--time",
        "-t",
        action="store_true",
        default=False,
        help=(
            "Additionally plot nonlinear transfers over the time interval "
            "used for the time-averaging."
        ),
    )

    parser.add_argument(
        "--budget",
        "-b",
        action="store_true",
        default=False,
        help=(
            "Additionally plot forcing and hyperdissipation terms."
        ),
    )

    args = parser.parse_args()

    post = FlucsPostProcessing(
        io_paths=args.io_path,
        save_directory=args.save_directory,
        output_files=["output.fluxes.nc"],
        constraint="none",
    )

    if args.list:
        post.list_netcdf_variables()
        exit()

    plot_fluxes_elsasser(post, args)
