"""
Pseudospectral Fourier implementation of the isothermal KREHM system from
Adkins et al. (2024). The nonlinear term is handled explicitly using the 
Adams-Bashforth 3-step method.
"""
from typing import ClassVar

import cupy as cp
import numpy as np
from scipy.special import i0e
from cupy.cuda import cufft
from flucs.diagnostic import FlucsDiagnostic
from flucs.solvers.fourier.fourier_system import FourierSystem, FourierSystemForcing
from flucs.utilities.cupy import cupy_set_device_pointer
from flucs.input import InvalidFlucsInputFileError
from flucs.utilities.messages import flucsprint

from .krehm_fourier_diagnostics import FreeEnergyDiag, HelicityDiag

class KREHMFourierElsasserForcing(FourierSystemForcing):
    explicit = True
    linear = False

    def setup_cuda_definitions(self):
        # Alias parameters
        system = self.system
        energy_injection_rate = system.input["forcing.energy_injection_rate"]

        # Forcing bands
        range_kperp = system.input["forcing.range_kperp"]
        range_kz = system.input["forcing.range_kz"]

        if len(range_kperp) != 2:
            raise InvalidFlucsInputFileError(
                "forcing.range_kperp must be a list [kperp_min, kperp_max]."
            )

        if len(range_kz) != 2:
            raise InvalidFlucsInputFileError(
                "forcing.range_kz must be a list [kz_min, kz_max]."
            )

        kperp_min = range_kperp[0]
        kperp_max = range_kperp[1]
        if kperp_max < kperp_min:
            raise InvalidFlucsInputFileError(
                "forcing.kperp_max must be larger than forcing.kperp_min."
            )

        kz_min = range_kz[0]
        kz_max = range_kz[1]
        if kz_max < kz_min:
            raise InvalidFlucsInputFileError(
                "forcing.kz_max must be larger than forcing.kz_min."
            )

        system.module_options.define_float("FORCING_KPERP2_MIN", kperp_min**2)
        system.module_options.define_float("FORCING_KPERP2_MAX", kperp_max**2)
        system.module_options.define_float("FORCING_KZ_MIN", kz_min)
        system.module_options.define_float("FORCING_KZ_MAX", kz_max)

        # Determine number of forced modes
        system._precompute_wavenumbers()
        kx, ky, kz = system.get_broadcast_wavenumbers()
        kperp2 = kx**2 + ky**2

        forced_modes_halfny = (
            (kperp2 > kperp_min**2)
            & (kperp2 < kperp_max**2)
            & (kz > kz_min)
            & (kz < kz_max)
        )
        ky0_modes = ky < 0.5 * ky[0, 0, 1]

        number_of_forced_modes = (
            2 * np.sum(forced_modes_halfny)
            - np.sum(forced_modes_halfny & ky0_modes)
        )

        if number_of_forced_modes == 0:
            raise InvalidFlucsInputFileError(
                "No modes are being forced. Please check your forcing.range_kz "
                "and/or forcing.range_kperp."
            )

        flucsprint(
            "Using Elsasser forcing on a total of "
            f"{number_of_forced_modes} modes.",
            source=self
        )

        # Validate energy injection rate and injection imbalance
        if energy_injection_rate < 0.0:
            raise InvalidFlucsInputFileError(
                "forcing.energy_injection_rate must be positive "
                "semi-definite."
            )

        injection_imbalance = system.input["forcing.injection_imbalance"]
        if injection_imbalance < 0.0 or injection_imbalance > 1.0:
            raise InvalidFlucsInputFileError(
                "forcing.injection_imbalance must be between 0 and 1."
            )

        # Get plus and minus injection
        forcing_epsilon_plus = (
            energy_injection_rate * (1 + injection_imbalance) / 2
        )
        forcing_epsilon_minus = (
            energy_injection_rate * (1 - injection_imbalance) / 2
        )

        system.module_options.define_float(
            "FORCING_EPSILON_PLUS", 
            forcing_epsilon_plus / number_of_forced_modes
        )
        system.module_options.define_float(
            "FORCING_EPSILON_MINUS", 
            forcing_epsilon_minus / number_of_forced_modes
        )

class KREHMFourier(FourierSystem):
    """
    Fourier solver for the isothermal KREHM system.

    """
    number_of_fields = 2
    number_of_fields_explicit = 2
    number_of_dft_derivatives = 6
    number_of_dft_bits = 5

    # Direct pointers to fields
    phi: list[cp.ndarray]
    apar: list[cp.ndarray]

    # CUDA grids and kernels
    nonlinear_bits_shared_mem: int

    find_derivatives_kernel: cp.RawKernel
    find_nonlinear_bits_kernel: cp.RawKernel

    # Supported diagnostics
    diags: ClassVar[set[type[FlucsDiagnostic]]] = {
        FreeEnergyDiag,
        HelicityDiag
    }

    # Supported forcing
    system_forcing_methods: ClassVar[dict[str, FourierSystemForcing]] = {
        "elsasser": KREHMFourierElsasserForcing,
    }

    def ready(self):
        # Anything system-specific goes here
        super().ready()

    def setup_cuda_grids(self):
        super().setup_cuda_grids()
        self.nonlinear_bits_shared_mem = (
            self.cuda_block_size * self.float().nbytes
        )

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

        # NB: real_derivatives and real_bits are the same array
        self.find_nonlinear_bits_kernel(
            (self.full_padded_cuda_grid_size,),
            (self.cuda_block_size,),
            (self.real_derivatives,
             self.cfl_rate),
            shared_mem=self.nonlinear_bits_shared_mem
        )

        # NB: real_derivatives and real_bits are the same array
        self.plan_bits_r2c.fft(self.real_bits, self.dft_bits, cufft.CUFFT_FORWARD)

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
