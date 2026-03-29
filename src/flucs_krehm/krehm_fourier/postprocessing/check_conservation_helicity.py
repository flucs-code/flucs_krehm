import argparse
import numpy as np
import pathlib as pl
import matplotlib.pyplot as plt
from flucs.postprocessing import FlucsPostProcessing

def helicity_check(post):

    # Get valid files for the specified variable
    nc_paths = post.get_valid_netcdf_paths("helicity/dHdt")

    # Iterate over output files
    for index, nc_path in enumerate(nc_paths):

        # Separate figure for each output
        fig, axs = plt.subplots(3, 1, layout='constrained', sharex=True)
        ax_helicity, ax_balance, ax_error = axs

        # Set figure title
        figure_name = f"check_conservation_helicity_{pl.Path(nc_path).parent.name}"
        fig.canvas.manager.set_window_title(figure_name)

        # Read data from netCDF file
        variables = post.get_netcdf_variables(nc_path)

        # Mask out initial data at restart boundaries given the calculation
        # of dH/dt is not valid at the first time step
        time, boundaries, _ = post.load_netcdf_variable(nc_path, "time")
        time[0] = np.nan
        for boundary in boundaries:
            time[boundary] = np.nan

        # Load data
        dt = post.load_netcdf_variable(nc_path, "dt")[0]
        helicity = post.load_netcdf_variable(nc_path, "helicity/H")[0]
        dHdt = post.load_netcdf_variable(nc_path, "helicity/dHdt")[0]
        # injection = post.load_netcdf_variable(nc_path, "helicity/dHdt_inj")[0]
        # dissipation = post.load_netcdf_variable(nc_path, "helicity/dHdt_coll")[0]
        injection = 0
        dissipation = 0

        # Add hyperdissipation
        for variable in variables:
            if variable.startswith("helicity/dHdt_hyperdissipation_"):
                dissipation += post.load_netcdf_variable(nc_path, variable)[0]

        # Add vertical lines to mark restart boundaries
        for ax in axs:
            for index in boundaries:
                ax.axvline(time[index], color='black', linestyle="dotted")

        # Plot helicity
        ax_helicity.plot(time, helicity, label="H (helicity)", linewidth=1.5, color='black')

        # Plot helicity balance
        ax_balance.plot(time, dHdt, label="dH/dt", linewidth=1.5, color='black', linestyle='solid')
        # ax_balance.plot(time, injection, label="Injection", linewidth=1.5, color='red', linestyle='solid')
        ax_balance.plot(time, dissipation, label="Dissipation", linewidth=1.5, color='blue', linestyle='solid')
        # ax_balance.plot(time, injection + dissipation, label="Injection + dissipation", linewidth=1.5, color='black', linestyle='dashed')

        # Plot error normalised to the timestep
        error = (dHdt - injection - dissipation)/dt
        ax_error.plot(time, np.abs(error / helicity), label="Error / (dt * H)", linewidth=1.5, color='black')

        # Setting plot options
        ax_error.set_xlim(np.nanmin(time), np.nanmax(time))
        ax_error.set_xlabel(r"$(v_A/L_z)t$")
        ax_error.set_yscale("log")

        ax_helicity.legend()
        ax_balance.legend(ncols=2)
        ax_error.legend()

        # Save figures if required
        post.save(fig, name=figure_name, suffix="png", save_kwargs={"dpi": 300, "close": True})

        plt.show()


    return

if __name__ == "__main__":

    # Setup parser
    parser = argparse.ArgumentParser(
        parents=[FlucsPostProcessing.parser()],
        description="Check helicity conservation for the isothermal KREHM system.",
    )

    args = parser.parse_args()

    # Initialise post-processing object
    post = FlucsPostProcessing(
        io_paths=args.io_path,
        save_directory=args.save_directory,
        output_files=["output.0d.nc"],
        constraint="both"
    )

    # Call function
    helicity_check(post)
