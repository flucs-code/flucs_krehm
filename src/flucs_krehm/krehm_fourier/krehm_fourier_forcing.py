import numpy as np

from flucs.input import InvalidFlucsInputFileError
from flucs.solvers.fourier.fourier_system_forcing import FourierSystemForcing
from flucs.utilities.messages import flucsprint


class KREHMFourierElsasserForcing(FourierSystemForcing):
    """
    Negative-damping type forcing for the generalised Elsasser potentials.

    Parameters
    ----------
    energy_injection_rate: float
        The total energy injection rate (summed over both Elsasser fields).
    injection_imbalance: float
        Defined as epsilon_H / epsilon_W, where epsilon_H is the helicity 
        injection rate and epsilon_W is the free energy injection rate. 
        Must be between 0 and 1.
    range_kperp: list[float, float]
        The range of perpendicular wavenumbers to force, defined as
        [kperp_min, kperp_max]. 
    range_kz: list[float, float]
        The range of parallel wavenumbers to force, defined as [kz_min, kz_max].
    """
        
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
        kz_abs = np.abs(kz)

        forced_modes_halfny = (
            (kperp2 > kperp_min**2)
            & (kperp2 < kperp_max**2)
            & (kz_abs > kz_min)
            & (kz_abs < kz_max)
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
            f"Forcing applied on a total of {number_of_forced_modes} modes.",
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