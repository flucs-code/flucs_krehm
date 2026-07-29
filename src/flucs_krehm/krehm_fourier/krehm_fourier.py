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
from flucs.input import InvalidFlucsInputFileError
from flucs.utilities.messages import flucsprint
from flucs.utilities.cupy import KernelWrapper

from .krehm_fourier_diagnostics import (
    FreeEnergyDiag,
    FreeEnergyDiag1D,
    FluxesDiag,
    HelicityDiag,
    HelicityDiag1D,
)
from .krehm_fourier_forcing import (
    KREHMFourierElsasserForcing,
    KREHMFourierMeyrandForcing,
)

class KREHMFourier(FourierSystem):
    """
    Fourier solver for the isothermal KREHM system.

    """
    number_of_fields = 2
    number_of_dft_derivatives = 6
    number_of_dft_bits = 5

    # Direct pointers to fields
    phi: list[cp.ndarray]
    apar: list[cp.ndarray]

    # CUDA kernels
    find_derivatives_kernel: KernelWrapper
    find_nonlinear_bits_kernel: KernelWrapper

    # Supported diagnostics
    diags: ClassVar[set[type[FlucsDiagnostic]]] = {
        FreeEnergyDiag,
        FreeEnergyDiag1D,
        FluxesDiag,
        HelicityDiag,
        HelicityDiag1D,
    }

    # Supported forcing
    system_forcing_methods: ClassVar[dict[str, FourierSystemForcing]] = {
        "elsasser": KREHMFourierElsasserForcing,
        "meyrand": KREHMFourierMeyrandForcing,
    }

    def ready(self):
        # Anything system-specific goes here
        super().ready()

    def register_kernels(self) -> None:
        super().register_kernels()

        nonlinear_bits_shared_mem = (
            self.cuda_block_size * self.float().nbytes
        )

        self.find_derivatives_kernel = KernelWrapper(
            system=self,
            cuda_kernel_name="find_derivatives",
            grid=(self.half_padded_cuda_grid_size,),
            block=(self.cuda_block_size,),
        )

        self.find_nonlinear_bits_kernel = KernelWrapper(
            system=self,
            cuda_kernel_name="find_nonlinear_bits",
            grid=(self.full_padded_cuda_grid_size,),
            block=(self.cuda_block_size,),
            shared_mem=nonlinear_bits_shared_mem,
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
        of equations. Note that the only paramaters that appear in the evolved
        equations are

        self.ZTe_over_Ti
        self.rhoi
        self.de

        All other parameters are to allow the user flexibility in how they 
        specify the physical input parameters.
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

        betae_over_mass_ratio = self.input["parameters.betae_over_mass_ratio"]

        # Equilibrium parameters
        self.Ti_over_Te = Ti_over_Te
        self.ion_charge = ion_charge

        # Handle eRMHD limit
        if self.input["parameters.ermhd"]:
            flucsprint(
                "Running in electron-RMHD limit, " \
                "overriding existing lengthscale parameters.", 
                source=self
            )

            # Override parameters with appropriate values
            rhoi = 1
            de = 0.0

            # Compute effective value of self.ZTe_over_Ti
            betai = self.input["parameters.ermhd_betai"]

            ZTe_over_Ti = self.ion_charge / self.Ti_over_Te

            self.ZTe_over_Ti = (
                  (ZTe_over_Ti - (betai * (1.0 + ZTe_over_Ti) / 2.0))
                / (1.0         + (betai * (1.0 + ZTe_over_Ti) / 2.0))
            )
        else:
            # Use the usual definition
            self.ZTe_over_Ti = self.ion_charge / self.Ti_over_Te

        # Ion-scale parameters
        self.rhoi = rhoi
        self.rhos = np.sqrt(self.ion_charge / (2 * self.Ti_over_Te)) * self.rhoi

        # Finite-electron-inertia parameters
        if de >= 0.0 and betae_over_mass_ratio > 0.0:
            raise InvalidFlucsInputFileError(
                "Only one of the parameters.de and "
                "parameters.betae_over_mass_ratio should be specified."
            )

        if de == 0.0:
            betae_over_mass_ratio = np.inf
        elif de > 0.0:
            betae_over_mass_ratio = (
                (ion_charge**2 / Ti_over_Te) * (rhoi / de)**2
            )
        elif betae_over_mass_ratio > 0.0:
            de = ion_charge * rhoi / np.sqrt(
                Ti_over_Te * betae_over_mass_ratio
            )
        else:
            raise InvalidFlucsInputFileError(
                "Please specify at least one of parameters.de and "
                "parameters.betae_over_mass_ratio."
            )
        
        self.de = de
        self.betae_over_mass_ratio = betae_over_mass_ratio

    def _set_initial_conditions(self) -> None:
        """
        System-specific initial conditions go here
        """

        # Use restart data
        if self.restart_manager.data is not None:
            super()._set_initial_conditions()
            return

        # Handle known initialisation types
        match self.input["init.method"]:

            case "elsasser":
                # Validate parameters
                energy = self.input["init.energy"]
                imbalance = self.input["init.imbalance"]

                if energy <= 0.0:
                    raise InvalidFlucsInputFileError(
                        "init.energy must be positive."
                    )
                if (imbalance < -1.0) or (imbalance > +1.0):
                    raise InvalidFlucsInputFileError(
                        "init.imbalance must be between -1.0 and 1.0."
                    )

                # Target Elsasser energies
                Wp_target = 0.5 * (1.0 + imbalance) * energy
                Wm_target = 0.5 * (1.0 - imbalance) * energy

                # Construct wavenumbers
                kx, ky, kz = self.get_broadcast_wavenumbers()
                kperp2 = kx**2 + ky**2

                valid_kz = np.zeros_like(kz, dtype=bool)
                valid_kz[+1, :, :] = True
                valid_kz[-1, :, :] = True

                # Envelope
                envelope = (kperp2 ** self.input["init.power"]) * np.exp(
                    -2.0 * (kperp2 /  self.input["init.width"] ** 2)
                )
                envelope[~((kperp2 > 0.0) & valid_kz)] = 0.0

                # Weight for wavenumbe summation
                weight = np.ones((1, 1, self.half_ny), dtype=kperp2.dtype)
                weight[..., 1:] = 2.0

                # Construct thetas with random phases
                random = np.random.default_rng(self.input["init.rand_seed"])
                thetas = []

                for target in (Wp_target, Wm_target):
                    # Base object
                    theta = (
                        envelope
                        * np.exp(
                            1j * random.uniform(
                                0.0,
                                2.0 * np.pi,
                                size=self.half_unpadded_tuple,
                            )
                        )
                    ).astype(self.complex)

                    # Handle reality condition
                    theta_ky0 = theta[:, :, 0]
                    theta_ky0[0, 0] = 0  # should be zero already
                    theta_ky0[0, self.half_nx:] = np.conj(
                        theta_ky0[0, 1:self.half_nx][::-1]
                    )
                    theta_ky0[self.half_nz:, 0] = np.conj(
                        theta_ky0[1:self.half_nz, 0][::-1]
                    )
                    theta_ky0[self.half_nz:, 1:] = np.conj(
                        theta_ky0[1:self.half_nz, 1:][::-1, ::-1]
                    )

                    theta[:, :, 0] = theta_ky0[:, :]

                    # Scale by free-energy contribution
                    W_theta = 0.5 * np.sum(
                        weight * kperp2 * theta * np.conj(theta)
                    ).real
                    thetas.append(np.sqrt(target / W_theta) * theta)

                # Unpack thetas
                thetap, thetam = thetas

                # Convert to evolved fields
                phi, apar = self.compute_fields_from_thetas(thetap, thetam)

                self.fields_initial = np.stack(
                    (phi, apar), axis=0
                ).astype(self.complex)

            case _:
                # Fallback to solver-side initial conditions
                super()._set_initial_conditions()

    def setup_cuda_definitions(self) -> None:
        # System-specific constants for the kernels
        self.module_options.define_float("ZTE_OVER_TI", self.ZTe_over_Ti)
        self.module_options.define_float("RHOI2", self.rhoi**2)
        self.module_options.define_float("DE2", self.de**2)
        if self.input["parameters.ermhd"]:
            self.module_options.define_flag("ERMHD")

        super().setup_cuda_definitions()

    def begin_time_step(self) -> None:
        # Do anything model-specific here, then call the parent's method
        super().begin_time_step()

    def compute_nonlinear_terms(self, fields: cp.ndarray) -> None:
        """
        Computes the nonlinear terms for the supplied fields. Here, we also
        determine the nonlinear CFL coefficient.

        """
        self.find_derivatives_kernel(
            fields,
            self.dft_derivatives,
            self.cfl_rate
        )

        self.plan_derivatives_c2r.fft(
            self.dft_derivatives,
            self.real_derivatives,
            cufft.CUFFT_INVERSE
        )

        # NB: real_derivatives and real_bits are the same array
        self.find_nonlinear_bits_kernel(
            self.real_derivatives,
            self.cfl_rate,
        )

        # NB: real_derivatives and real_bits are the same array
        self.plan_bits_r2c.fft(self.real_bits, self.dft_bits, cufft.CUFFT_FORWARD)

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
        taubarinv, one_minus_gamma0_over_alpha = (
            self.compute_ion_flr_terms(kperp2)
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

    def compute_ion_flr_terms(
            self, 
            kperp2: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the useful functions taubarinv and one_minus_gamma0_over_alpha
        that appear in various places in the equations using CPU memory. 
        """

        # Construct functions
        alpha = 0.5 * (kperp2) * (self.rhoi**2)
        gamma0 = 0 if self.input["parameters.ermhd"] else i0e(alpha)

        taubarinv = self.ZTe_over_Ti * (1.0 - gamma0)

        one_minus_gamma0_over_alpha = np.divide(
            1.0 - gamma0,
            alpha,
            out=np.ones_like(alpha),
            where=(alpha != 0.0)
        )

        return taubarinv, one_minus_gamma0_over_alpha

    def compute_phase_velocity(
            self,
            kperp2: np.ndarray
        ) -> np.ndarray:
        """
        Computes the phase velocity (normalised to the Alfven speed) using 
        CPU memory.

        """

        # Get ion FLR functions
        one_minus_gamma0_over_alpha = (self.compute_ion_flr_terms(kperp2))[1]
        alpha = 0.5 * (kperp2) * (self.rhoi**2)

        # Construct phase velocity
        vphase = np.sqrt(
            (self.ZTe_over_Ti * alpha + 1.0/one_minus_gamma0_over_alpha) /
            (1.0 + kperp2 * self.de**2)  
        )

        return vphase

    def compute_thetas_from_fields(
            self, 
            phi: np.ndarray, 
            apar: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
        """
        Given the Fourier-space fields, computes the corresponding Elsasser
        potentials using CPU memory. 

        """
        # Construct wavenumbers
        kx, ky, kz = self.get_broadcast_wavenumbers()
        kperp2 = kx**2 + ky**2

        # Construct ion FLR functions
        one_minus_gamma0_over_alpha = (self.compute_ion_flr_terms(kperp2))[1]

        # Construct phase velocity
        vphase = self.compute_phase_velocity(kperp2)

        # Construct thetas
        phi_factor = vphase * one_minus_gamma0_over_alpha

        thetap = np.sqrt(1 + kperp2 * self.de**2) * (phi_factor * phi + apar)
        thetam = np.sqrt(1 + kperp2 * self.de**2) * (phi_factor * phi - apar)

        return thetap, thetam

    def compute_fields_from_thetas(
            self, 
            thetap: np.ndarray, 
            thetam: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
        """
        Given the Fourier-space Elsasser potentials, computes the corresponding 
        fields using CPU memory.

        """
        # Construct wavenumbers
        kx, ky, kz = self.get_broadcast_wavenumbers()
        kperp2 = kx**2 + ky**2

        # Construct ion FLR functions
        taubarinv, one_minus_gamma0_over_alpha = (
            self.compute_ion_flr_terms(kperp2)
        )

        # Construct phase velocity
        vphase = self.compute_phase_velocity(kperp2)

        # Construct fields
        phi_factor = vphase * one_minus_gamma0_over_alpha

        phi  = 0.5 * (thetap + thetam) / (
            np.sqrt(1 + kperp2 * self.de**2) * phi_factor
        )
        apar = 0.5 * (thetap - thetam) / (
            np.sqrt(1 + kperp2 * self.de**2)
        )

        return phi, apar
