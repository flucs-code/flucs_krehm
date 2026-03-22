"""Pseudospectral Fourier implementation of the Ivanov et al. (2020) 2D fluid
ITG system. The nonlinear term is handled explicitly using the Adams-Bashforth
3-step method.

"""
from typing import ClassVar

import cupy as cp
import numpy as np
from cupy.cuda import cufft
from flucs.diagnostic import FlucsDiagnostic
from flucs.solvers.fourier.fourier_system import FourierSystem
from flucs.utilities.cupy import cupy_set_device_pointer
from flucs.input import InvalidFlucsInputFileError


class KREHMFourier(FourierSystem):
    """Fourier solver for the KREHM system."""
    number_of_fields = 2

    # DFT plans
    plan_r2c: cufft.PlanNd
    plan_c2r: cufft.PlanNd

    # CUDA grids
    nonlinear_bits_shared_mem: int

    # CUDA kernels
    find_derivatives_kernel: cp.RawKernel
    find_nonlinear_bits_kernel: cp.RawKernel

    # CUDA memory

    # Direct pointers to fields
    phi: list[cp.ndarray]
    A: list[cp.ndarray]

    # Nonlinear terms with multistep history
    multistep_nonlinear_terms: cp.ndarray

    # Derivatives and 'bits' used for finding the nonlinear terms
    dft_derivatives_and_bits: cp.ndarray
    real_derivatives_and_bits: cp.ndarray

    # Single-element array for the current CFL rate
    cfl_rate: cp.ndarray

    # Supported diagnostics
    # diags: ClassVar[set[type[FlucsDiagnostic]]] = {
    #     HeatfluxDiag, FreeEnergyDiag
    # }

    def _setup_system(self):
        """Prepares the system for the solver."""

        self.allocate_memory()
        super()._setup_system()

        # kperp = 0 fields do not evolve in a meaningful way
        self.fields_initial[:, :, 0, 0] = 0

    def ready(self):
        # Anything system-specific goes here

        if not self.input["setup.linear"]:
            cupy_set_device_pointer(self.cupy_module,
                                    "multistep_nonlinear_terms",
                                    self.multistep_nonlinear_terms)

        # Setup kernel parameters (grid, block, shared memory)
        self.nonlinear_bits_shared_mem = (
            self.cuda_block_size * self.float().nbytes
        )

        super().ready()

    def allocate_memory(self):
        # GPU arrays

        # For the field arrays, we need to keep the fields
        # at the current time step and the previous one.

        self.fields = [cp.zeros((2, self.nz, self.nx, self.half_ny),
                                dtype=self.complex),
                       cp.zeros((2, self.nz, self.nx, self.half_ny),
                                dtype=self.complex)]

        self.phi = [cp.ndarray((self.nz, self.nx, self.half_ny),
                               dtype=self.complex,
                               memptr=self.fields[0][0, 0, 0, 0].data),
                    cp.ndarray((self.nz, self.nx, self.half_ny),
                               dtype=self.complex,
                               memptr=self.fields[1][0, 0, 0, 0].data),]

        self.A = [cp.ndarray((self.nz, self.nx, self.half_ny),
                             dtype=self.complex,
                             memptr=self.fields[0][1, 0, 0, 0].data),
                  cp.ndarray((self.nz, self.nx, self.half_ny),
                             dtype=self.complex,
                             memptr=self.fields[1][1, 0, 0, 0].data),]

        # when running linearly, need something to pass to the kernels
        # this is unused
        self.dft_bits = cp.zeros(1, dtype=self.complex)

        if not self.input["setup.linear"]:
            # For the nonlinear terms, we need to keep terms at the current
            # time step + terms from the past 2 time steps (since we will be
            # using AB3)
            # The nonlinear terms are indexed as (step, field, kz, kx, ky)
            self.multistep_nonlinear_terms = cp.zeros((3, 2, self.nz, self.nx,
                                                       self.half_ny),
                                                      dtype=self.complex)

            # All fields and derivatives to be transformed to real space
            # The first index indexes the fields and it's meaning is
            # 0 dxphi
            # 1 dyphi,
            # 2 dxA
            # 3 dyA
            # 4 taubarinv_phi
            # 5 Amkperp2de2A
            self.dft_derivatives_and_bits = cp.zeros([6,
                                                      self.padded_nz,
                                                      self.padded_nx,
                                                      self.half_padded_ny],
                                                     dtype=self.complex)

            self.real_derivatives_and_bits = cp.zeros([6,
                                                       self.padded_nz,
                                                       self.padded_nx,
                                                       self.padded_ny],
                                                      dtype=self.float)

            # The above memory is reused for the NL bits.
            # These 'NL bits' are the terms which are calculated in real space.
            # They are transformed back to Fourier space, where any additional
            # derivatives are taken by multiplying the NL bits by the
            # appropriate powers of k. The NL bits here are
            # 0 dxphi * taubarinv_phi + rhos2 * dxA * Amkperp2de2A
            # 1 dyphi * taubarinv_phi + rhos2 * dyA * Amkperp2de2A
            # 2 dxphi * Amkperp2de2A + dxA * taubarinv_phi
            # 3 dyphi * Amkperp2de2A + dyA * taubarinv_phi

            # Still need dft_bits as FourierSystem expects it
            self.dft_bits = self.dft_derivatives_and_bits

            self.cfl_rate = cp.zeros([1], dtype=self.float)

            self.plan_c2r = cufft.PlanNd(
                shape=tuple([self.padded_nz, self.padded_nx, self.padded_ny]),
                istride=1,
                ostride=1,
                inembed=tuple([1, self.padded_nx, self.half_padded_ny]),
                onembed=tuple([1, self.padded_nx, self.padded_ny]),
                idist=self.padded_nz*self.padded_nx*self.half_padded_ny,
                odist=self.padded_nz*self.padded_nx*self.padded_ny,
                fft_type=self.fft_c2r_plan_type,
                batch=6,
                order='C',
                last_axis=3,
                last_size=self.padded_ny)

            self.plan_r2c = cufft.PlanNd(
                shape=tuple([self.padded_nz, self.padded_nx, self.padded_ny]),
                istride=1,
                ostride=1,
                inembed=tuple([1, self.padded_nx, self.padded_ny]),
                onembed=tuple([1, self.padded_nx, self.half_padded_ny]),
                idist=self.padded_nz*self.padded_nx*self.padded_ny,
                odist=self.padded_nz*self.padded_nx*self.half_padded_ny,
                fft_type=self.fft_r2c_plan_type,
                batch=4,
                order='C',
                last_axis=3,
                last_size=self.half_padded_ny)

    def _interpret_input(self):
        """Checks if the input file makes sense"""

        # Make sure to call the parent method to do some standard setup
        # (resolution checks, etc)
        super()._interpret_input()

        # Helper function for a parameter setting
        def _set_or_check(values: dict[str, float | None],
                         name: str,
                         value: float):
            
            current = values[name]

            if current is None:
                values[name] = value
                return True

            if not np.isclose(current, value, rtol=1e-8, atol=0.0):
                raise InvalidFlucsInputFileError(
                    f"Inconsistent value for parameters.{name}: "
                    f"{current} and {value} do not agree."
                )

            return False

        # Check ion charge
        ion_charge = self.input["parameters.ion_charge"]
        if ion_charge <= 0.0 or not float(ion_charge).is_integer():
            raise InvalidFlucsInputFileError(
                "parameters.ion_charge must be a positive integer."
            )
        ion_charge = int(ion_charge)

        # Load inputs
        values = {}
        for name in ("beta_over_mass_ratio", "Ti_over_Te", "rhos", "rhoi", "de"):
            value = self.input[f"parameters.{name}"]
            values[name] = value if value > 0.0 else None

        # If all lengthscales are unset, assume we are simulating RMHD
        rmhd_limit = False
        if (
            values["rhos"] is None
            and values["rhoi"] is None
            and values["de"] is None
            ):
            rmhd_limit = True

            small = 1e-6
            print(f"Assuming RMHD limit, setting all lengthscales to {small}")
            
            values["beta_over_mass_ratio"] = None
            values["Ti_over_Te"] = 1.0
            values["rhoi"] = small
            values["de"] = small

        # Iterate over values and update until we have determined as many as possible\
        changed = True
        while changed:
            changed = False

            # Infer values irrespective of the value of rmhd_limit
            if values["rhoi"] is not None and values["rhos"] is not None:
                changed |= _set_or_check(
                    values,
                    "Ti_over_Te",
                    0.5 * ion_charge * (values["rhoi"] / values["rhos"]) ** 2,
                )

            if values["rhos"] is not None and values["de"] is not None:
                changed |= _set_or_check(
                    values,
                    "beta_over_mass_ratio",
                    2.0 * ion_charge * (values["rhos"] / values["de"]) ** 2,
                )

            if values["rhoi"] is not None and values["Ti_over_Te"] is not None:
                changed |= _set_or_check(
                    values,
                    "rhos",
                    values["rhoi"]
                    / np.sqrt(2.0 * values["Ti_over_Te"] / ion_charge),
                )

            # Only infer values if we are not in the rmhd_limit
            if not rmhd_limit and values["rhos"] is not None and values["Ti_over_Te"] is not None:
                changed |= _set_or_check(
                    values,
                    "rhoi",
                    values["rhos"]
                    * np.sqrt(2.0 * values["Ti_over_Te"] / ion_charge),
                )

            if not rmhd_limit and values["de"] is not None and values["beta_over_mass_ratio"] is not None:
                changed |= _set_or_check(
                    values,
                    "rhos",
                    values["de"]
                    * np.sqrt(values["beta_over_mass_ratio"]
                            / (2.0 * ion_charge)),
                )

            if not rmhd_limit and values["rhos"] is not None and values["beta_over_mass_ratio"] is not None:
                changed |= _set_or_check(
                    values,
                    "de",
                    values["rhos"]
                    * np.sqrt(2.0 * ion_charge
                            / values["beta_over_mass_ratio"]),
                )

        # Report missing parameters, if any
        missing = [
            name for name in ("Ti_over_Te", "rhos", "de")
            if values[name] is None
        ]
        if missing:
            raise InvalidFlucsInputFileError(
                "Insufficient KREHM parameters to determine "
                f"{', '.join(missing)}."
            )

        # Derived parameters
        self.beta_over_mass_ratio = values["beta_over_mass_ratio"]
        self.Ti_over_Te = values["Ti_over_Te"]
        self.ion_charge = ion_charge
        self.rhoi = values["rhoi"]

        # Core parameters
        self.Ti_over_ZTe = self.Ti_over_Te / self.ion_charge
        self.rhos = values["rhos"]
        self.de = values["de"]

    def compile_cupy_module(self) -> None:
        # System-specific constants for the kernels
        self.module_options.define_float("TI_OVER_ZTE", self.Ti_over_ZTe)
        self.module_options.define_float("RHOS2", self.rhos**2)
        self.module_options.define_float("DE2", self.de**2)

        # Call this to compile the module
        super().compile_cupy_module()

        # System-specific kernels
        self.find_derivatives_kernel =\
            self.cupy_module.get_function("find_derivatives")

        self.find_nonlinear_bits_kernel =\
            self.cupy_module.get_function("find_nonlinear_bits")

    def begin_time_step(self) -> None:
        # Do anything model-specific here, then call the parent's method
        super().begin_time_step()

    def calculate_nonlinear_terms(self) -> None:
        """
        Calculates the nonlinear terms. This is the most computationaly
        intensive part of taking a time step. Here, we also determine the
        nonlinear CFL coefficient.

        """
        self.find_derivatives_kernel((self.half_padded_cuda_grid_size,),
                                     (self.cuda_block_size,),
                                     (self.fields[self.current_step % 2 - 1],
                                      self.dft_derivatives_and_bits,
                                      self.cfl_rate))

        self.plan_c2r.fft(self.dft_derivatives_and_bits,
                          self.real_derivatives_and_bits,
                          cufft.CUFFT_INVERSE)

        self.find_nonlinear_bits_kernel(
            (self.full_padded_cuda_grid_size,),
            (self.cuda_block_size,),
            (self.real_derivatives_and_bits,
             self.cfl_rate),
            shared_mem=self.nonlinear_bits_shared_mem
        )

        self.plan_r2c.fft(self.real_derivatives_and_bits,
                          self.dft_derivatives_and_bits,
                          cufft.CUFFT_FORWARD)

        super().calculate_nonlinear_terms()

    def finish_time_step(self) -> None:
        super().finish_time_step()

    def compute_complex_omega():
        pass
