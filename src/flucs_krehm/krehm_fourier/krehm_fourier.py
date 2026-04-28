"""Pseudospectral Fourier implementation of the Ivanov et al. (2020) 2D fluid
ITG system. The nonlinear term is handled explicitly using the Adams-Bashforth
3-step method.

"""
from typing import ClassVar

import cupy as cp
import numpy as np
from scipy.special import i0e
from cupy.cuda import cufft
from flucs.diagnostic import FlucsDiagnostic
from flucs.solvers.fourier.fourier_system import FourierSystem
from flucs.utilities.cupy import cupy_set_device_pointer
from flucs.input import InvalidFlucsInputFileError

from .krehm_fourier_diagnostics import FreeEnergyDiag, HelicityDiag


class KREHMFourier(FourierSystem):
    """Fourier solver for the isothermal KREHM system."""
    number_of_fields = 2
    number_of_fields_nonlinear = 2
    number_of_dft_derivatives = 6
    number_of_dft_bits = 5

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
    apar: list[cp.ndarray]

    # Nonlinear terms with multistep history
    multistep_nonlinear_terms: cp.ndarray

    # Derivatives and 'bits' used for finding the nonlinear terms
    dft_derivatives_and_bits: cp.ndarray
    real_derivatives_and_bits: cp.ndarray

    # Single-element array for the current CFL rate
    cfl_rate: cp.ndarray

    # Supported diagnostics
    diags: ClassVar[set[type[FlucsDiagnostic]]] = {
        FreeEnergyDiag,
        HelicityDiag
    }

    def _setup_system(self):
        """Prepares the system for the solver."""
        super()._setup_system()

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

    def _allocate_memory(self):
        """Allocates runtime arrays."""

        # First, call FourierSystem's method which allocates
        # self.fields among other things.
        super()._allocate_memory(
            allocate_derivatives_and_bits=True,
            combine_derivatives_and_bits=True
        )

        # Pointers to phi and apar for easier access
        self.phi = [cp.ndarray((self.nz, self.nx, self.half_ny),
                               dtype=self.complex,
                               memptr=self.fields[0][0, 0, 0, 0].data),
                    cp.ndarray((self.nz, self.nx, self.half_ny),
                               dtype=self.complex,
                               memptr=self.fields[1][0, 0, 0, 0].data),]

        self.apar = [cp.ndarray((self.nz, self.nx, self.half_ny),
                             dtype=self.complex,
                             memptr=self.fields[0][1, 0, 0, 0].data),
                  cp.ndarray((self.nz, self.nx, self.half_ny),
                             dtype=self.complex,
                             memptr=self.fields[1][1, 0, 0, 0].data),]

        # All fields and derivatives to be transformed to real space
        # are kept in one huge array (dft_derivatives).
        # The first index indexes the fields and it's meaning is
        # 0: dxphi
        # 1: dyphi,
        # 2: dxapar
        # 3: dyapar
        # 4: one_minus_gamma0_over_alpha kperp2 phi
        # 5: kperp2apar

        # The NL bits here are
        # 0: dxphi * one_minus_gamma0_over_alpha * kperp2 * phi 
        #   - dxapar * kperp2apar
        # 1: dyphi * one_minus_gamma0_over_alpha * kperp2 * phi 
        #   - dyapar * kperp2apar
        # 2: de2 * dxphi * kperp2 * apar
        #   - 0.5 (Z/tau) rhoi2 * dxapar * (1-Gamma0)/alpha * kperp2 * phi
        # 3: de2 * dyphi * kperp2 * apar
        #   - 0.5 (Z/tau) rhoi2 * dyapar * (1-Gamma0)/alpha * kperp2 * phi
        # 4: dxphi * dyapar - dyphi * dxapar

        # The arrays for the above are handled by FourierSystem.
        # There are no system-specific arrays that we need to allocate here 

    def _interpret_input(self):
        """Checks if the input file makes sense"""

        # Make sure to call the parent method to do some standard setup
        # (resolution checks, etc)
        super()._interpret_input()

        # Check and set all physical parameters
        self._interpret_physical_parameters()

    def _interpret_physical_parameters(self):
        """
        Makes sure that the user has specified a consistent set of
        parameters and infers any implicit parameters.

        The default parameter values give the standard RMHD system
        of equations.
        """

        # Check positiveness of parameters
        for name in ["Ti_over_Te", "ion_charge"]:
            if self.input[f"parameters.{name}"] <= 0:
                raise InvalidFlucsInputFileError(
                    f"Parameter {name} must be positive."
                )

        # Alias parameters
        Ti_over_Te = self.input["parameters.Ti_over_Te"]
        ion_charge = self.input["parameters.ion_charge"]

        rhoi = self.input["parameters.rhoi"]
        de = self.input["parameters.de"]

        beta_over_mass_ratio = self.input["parameters.beta_over_mass_ratio"]

        # Handle finite beta parameters
        if de >= 0.0 and beta_over_mass_ratio > 0.0:
            raise InvalidFlucsInputFileError(
                "Only one of the parameters.de and "
                "parameters.beta_over_mass_ratio should be specified."
            )

        if de == 0.0:
            beta_over_mass_ratio = np.inf
        elif de > 0.0:
            beta_over_mass_ratio = (
                (ion_charge**2 / Ti_over_Te) * (rhoi / de)**2
            )
        elif beta_over_mass_ratio > 0.0:
            de = ion_charge * rhoi / np.sqrt(
                Ti_over_Te * beta_over_mass_ratio
            )
        else:
            raise InvalidFlucsInputFileError(
                "Please specify at least one of parameters.de and "
                "parameters.beta_over_mass_ratio."
            )

        # Store final parameters
        self.Ti_over_Te = Ti_over_Te
        self.ion_charge = ion_charge
        self.ZTe_over_Ti = self.ion_charge / self.Ti_over_Te

        self.rhoi = rhoi
        self.rhos = np.sqrt(self.ion_charge / (2 * self.Ti_over_Te)) * self.rhoi

        self.de = de
        self.beta_over_mass_ratio = beta_over_mass_ratio

    def compile_cupy_module(self) -> None:
        # System-specific constants for the kernels
        self.module_options.define_float("ZTE_OVER_TI", self.ZTe_over_Ti)
        self.module_options.define_float("RHOI2", self.rhoi**2)
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
                                      self.dft_derivatives,
                                      self.cfl_rate))

        self.plan_derivatives_c2r.fft(self.dft_derivatives,
                          self.real_derivatives,
                          cufft.CUFFT_INVERSE)

        self.find_nonlinear_bits_kernel(
            (self.full_padded_cuda_grid_size,),
            (self.cuda_block_size,),
            (self.real_derivatives,
             self.cfl_rate),
            shared_mem=self.nonlinear_bits_shared_mem
        )

        self.plan_bits_r2c.fft(self.real_derivatives,
                          self.dft_derivatives,
                          cufft.CUFFT_FORWARD)

        super().calculate_nonlinear_terms()

    def finish_time_step(self) -> None:
        super().finish_time_step()

    def compute_linear_matrix_reference(self) -> np.ndarray:
        # Initialise linear matrix
        linear_matrix = np.zeros(
            (
                self.number_of_fields,
                self.number_of_fields,
                *self.half_unpadded_tuple
            ),
            dtype=self.complex,
        )

        # Get wavenumbers
        kx, ky, kz = self.get_broadcast_wavenumbers()
        kperp2 = kx**2 + ky**2

        # Get parameters
        rhoi = self.rhoi
        de = self.de
        ZTe_over_Ti = self.ZTe_over_Ti

        # Construct useful functions
        alpha = 0.5 * (kperp2) * (rhoi**2)
        gamma0 = i0e(alpha)
        taubarinv = ZTe_over_Ti * (1.0 - gamma0)
        one_minus_gamma0_over_alpha = np.divide(
            1.0 - gamma0,
            alpha,
            out=np.ones_like(alpha),
            where=(alpha != 0.0)
        )

        # phi-phi
        linear_matrix[0, 0, :, :, :] = 0.0

        # phi-apar
        linear_matrix[0, 1, :, :, :] = 1j * kz / one_minus_gamma0_over_alpha

        # apar-phi
        linear_matrix[1, 0, :, :, :] = 1j * kz * (
            (1 + taubarinv) / (1 + kperp2 * de**2)
        )

        # apar-apar
        linear_matrix[1, 1, :, :, :] = 0.0

        return linear_matrix
