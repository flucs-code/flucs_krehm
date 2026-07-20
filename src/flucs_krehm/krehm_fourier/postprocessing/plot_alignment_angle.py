import argparse
import pathlib as pl

import matplotlib.pyplot as plt
import numpy as np

from flucs.postprocessing import FlucsPostProcessing
from flucs.utilities.messages import flucsprint


def plot_1d_alignment_angle(post, args):

    # Alias arguments
    fraction = args.fraction
    alignedfields = args.alignedfields
    direction = args.direction

    variable_numerator = f'alignment_angle/{alignedfields}/{direction}/numerator'
    variable_denominator = f'alignment_angle/{alignedfields}/{direction}/denominator'

    # Get valid files for the specified variable
    nc_path_numerator = post.get_valid_netcdf_paths(variable_numerator)[0]
    nc_path_denominator = post.get_valid_netcdf_paths(variable_denominator)[0]


    # Initialise plotting
    fig, ax = plt.subplots(1, 1, layout="constrained")


    # Read data from netCDF file
    time = post.load_netcdf_variable(
        nc_path_numerator, "time",
    )[0]
    time_denominator =  post.load_netcdf_variable(
        nc_path_denominator, "time",
    )[0]

    assert (time==time_denominator).all()

    numerator, _, numerator_dims_dicts = post.load_netcdf_variable(
        nc_path_numerator,
        variable_numerator,
    )
    denominator, _, denominator_dims_dicts = post.load_netcdf_variable(
        nc_path_denominator,
        variable_denominator,
    )

    dims_numerator = next(dims_numerator for dims_numerator in reversed(numerator_dims_dicts) if dims_numerator)
    if len(dims_numerator) != 1:
        raise ValueError(
            f"Expected a 1D variable, but '{variable_numerator}' has "
            f"dimensions {list(dims_numerator)}."
        )
    dimension_name_numerator, dimension_numerator = next(iter(dims_numerator.items()))

    dims_denominator = next(dims_denominator for dims_denominator in reversed(denominator_dims_dicts) if dims_denominator)
    if len(dims_denominator) != 1:
        raise ValueError(
            f"Expected a 1D variable, but '{variable_denominator}' has "
            f"dimensions {list(dims_denominator)}."
        )
    dimension_name_denominator, dimension_denominator = next(iter(dims_denominator.items()))

    if dimension_name_numerator != dimension_name_denominator or (dimension_denominator != dimension_numerator).all():
        raise ValueError("Dimensions for numerator and denominator don't match")
    dimension_name = dimension_name_numerator
    dimension = dimension_numerator

    cos_angle = numerator/denominator




    # Mask for logarithmic axes
    mask = dimension >= 0.0
    dimension = dimension[mask]
    cos_angle = cos_angle[:, mask]

    # Time average
    mask_time = time >= (
        np.min(time) + (1.0 - fraction) * (np.max(time) - np.min(time))
    )
    cos_angle_avg = np.nanmean(cos_angle[mask_time], axis=0)
    ax.plot(
        dimension,
        np.abs(cos_angle_avg),
        linewidth=1.5,
        linestyle="solid",
    )   
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(dimension_name)
    ax.set_ylabel(alignedfields)

    fig_name = f'{alignedfields}_vs_{direction}'
    fig.canvas.manager.set_window_title(fig_name)
    post.save(
        fig,
        name=fig_name,
        suffix="png",
        save_kwargs={"dpi": 300},
    )


    return


if __name__ == "__main__":

    # Setup parser
    parser = argparse.ArgumentParser(
        parents=[FlucsPostProcessing.parser()],
        description=(
            "Plot alignment angles."
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
        "--alignedfields",
        "-a",
        type=str,
        default='zp_zm',
        help="Select alignment angle to plot (options 'zp_zm', 'Bperp_gradLaplacianBperp', 'gradBpar_Bperp').",
    )

    parser.add_argument(
        "--direction",
        "-d",
        type=str,
        default='lperp',
        help="Select direction.",
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

    # Initialise post-processing object
    post = FlucsPostProcessing(
        io_paths=args.io_path,
        save_directory=args.save_directory,
        output_files=["output.realspace1d.nc"],
        constraint="none",
    )

    if args.list:
        post.list_netcdf_variables()
        exit()

    # Call function
    plot_1d_alignment_angle(post, args)
