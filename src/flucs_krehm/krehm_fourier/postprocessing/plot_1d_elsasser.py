import argparse
import pathlib as pl

import matplotlib.pyplot as plt
import numpy as np

from flucs.postprocessing import FlucsPostProcessing
from flucs.utilities.messages import flucsprint


def plot_1d_elsasser(post, args):

    # Alias arguments
    dimension_name = args.dimension
    groups = args.groups
    fraction = args.fraction

    variables = {
        "W" : f"free_energy_1d/{dimension_name}_spectra/W",
        "Wp": f"free_energy_1d/{dimension_name}_spectra/Wp",
        "Wm": f"free_energy_1d/{dimension_name}_spectra/Wm",
        "H" : f"helicity_1d/{dimension_name}_spectra/H",
        "Hp": f"helicity_1d/{dimension_name}_spectra/Hp",
        "Hm": f"helicity_1d/{dimension_name}_spectra/Hm",
    }

    # Get valid files for the specified variable
    nc_paths = post.get_valid_netcdf_paths(variables["Wp"])

    # Initialise plotting
    fig_w, ax_w = plt.subplots(1, 1, layout="constrained")
    fig_h, ax_h = plt.subplots(1, 1, layout="constrained")

    ymin = 1e-6
    xscale = "log"
    yscale = "log"

    # Iterate over output files
    for index, nc_path in enumerate(nc_paths):

        # Assign identifiers
        sim_label = pl.Path(nc_path).parent.name
        sim_color = plt.cm.rainbow(np.linspace(0, 1, len(nc_paths)))[index]

        # Check if variables are present
        nc_variables = post.get_netcdf_variables(nc_path)
        missing_variables = [
            v for v in variables.values()
            if v not in nc_variables
        ]

        if missing_variables:
            flucsprint(
                f"{sim_label}: Elsasser spectra were requested but are not "
                "available. Please check output files.",
                source=plot_1d_elsasser,
                message_type="warning",
            )
            continue

        # Read data from netCDF file
        time = post.load_netcdf_variable(
            nc_path, "time", groups=groups,
        )[0]
        free_energy, _, dims_dicts = post.load_netcdf_variable(
            nc_path, variables["W"], groups=groups,
        )
        helicity = post.load_netcdf_variable(
            nc_path, variables["H"], groups=groups,
        )[0]

        Wp = post.load_netcdf_variable(
            nc_path, variables["Wp"], groups=groups,
        )[0]
        Wm = post.load_netcdf_variable(
            nc_path, variables["Wm"], groups=groups,
        )[0]
        Hp = post.load_netcdf_variable(
            nc_path, variables["Hp"], groups=groups,
        )[0]
        Hm = post.load_netcdf_variable(
            nc_path, variables["Hm"], groups=groups,
        )[0]

        # Validate dimension
        dims = next(dims for dims in reversed(dims_dicts) if dims)
        if len(dims) != 1:
            raise ValueError(
                f"Expected a 1D variable, but '{variables['Wp']}' has "
                f"dimensions {list(dims)}."
            )
        _, dimension = next(iter(dims.items()))

        # Check summation is correct
        rtol = 1e-5
        atol = 1e-6

        assert np.allclose(
            free_energy, (Wp + Wm), rtol=rtol, atol=atol, equal_nan=True
        ), (
            f"Elsasser free-energy spectra do not sum correctly for "
            f"{sim_label}. "
        )

        assert np.allclose(
            helicity, (Hp - Hm), rtol=rtol, atol=atol, equal_nan=True
        ), (
            f"Elsasser helicity spectra do not sum correctly for {sim_label}."
        )

        # Mask for logarithmic axes
        mask = dimension >= 0.0
        dimension = dimension[mask]

        Wp = Wp[:, mask]
        Wm = Wm[:, mask]
        Hp = Hp[:, mask]
        Hm = Hm[:, mask]

        # Time average
        mask_time = time >= (
            np.min(time) + (1.0 - fraction) * (np.max(time) - np.min(time))
        )

        Wp_avg = np.nanmean(Wp[mask_time], axis=0)
        Wm_avg = np.nanmean(Wm[mask_time], axis=0)
        Hp_avg = np.nanmean(Hp[mask_time], axis=0)
        Hm_avg = np.nanmean(Hm[mask_time], axis=0)

        W_max = max(np.nanmax(Wp_avg), np.nanmax(Wm_avg))
        H_max = max(np.nanmax(np.abs(Hp_avg)), np.nanmax(np.abs(Hm_avg)))

        flucsprint(
            rf"Wm/Wp = {np.sum(Wm_avg)/np.sum(Wp_avg):.3e}, "
            rf"Hm/Hp = {np.sum(Hm_avg)/np.sum(Hp_avg):+.3e} ({sim_label})",
            source=plot_1d_elsasser,
        )

        # Free-energy Elsasser spectra
        ax_w.plot(
            dimension,
            Wp_avg / W_max,
            label=sim_label,
            linewidth=1.5,
            color=sim_color,
            linestyle="solid",
        )
        ax_w.plot(
            dimension,
            Wm_avg / W_max,
            label=None,
            linewidth=1.5,
            color=sim_color,
            linestyle="dashed",
        )

        # Helicity Elsasser spectra
        ax_h.plot(
            dimension,
            np.abs(Hp_avg / H_max),
            label=sim_label,
            linewidth=1.5,
            color=sim_color,
            linestyle="solid",
        )
        ax_h.plot(
            dimension,
            np.abs(Hm_avg / H_max),
            label=None,
            linewidth=1.5,
            color=sim_color,
            linestyle="dashed",
        )

    # Setting plot options
    ax_w.set_xlabel(dimension_name)
    ax_w.set_ylabel(r"$W^\pm/\mathrm{max}(W^+, W^-)$")
    ax_w.set_xscale(xscale)
    ax_w.set_yscale(yscale)
    ax_w.set_ylim(ymin=ymin)
    ax_w.legend()

    ax_h.set_xlabel(dimension_name)
    ax_h.set_ylabel(r"$|H^\pm|/\mathrm{max}(|H^+|, |H^-|)$")
    ax_h.set_xscale(xscale)
    ax_h.set_yscale(yscale)
    ax_h.set_ylim(ymin=ymin)
    ax_h.legend()

    # Save figures
    fig_w_name = f"elsasser_free_energy_vs_{dimension_name}"
    fig_w.canvas.manager.set_window_title(fig_w_name)
    post.save(
        fig_w,
        name=fig_w_name,
        suffix="png",
        save_kwargs={"dpi": 300},
    )

    fig_h_name = f"elsasser_helicity_vs_{dimension_name}"
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
            "Plot 1D Elsasser spectra for the isothermal KREHM system.",
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

    plot_1d_elsasser(post, args)