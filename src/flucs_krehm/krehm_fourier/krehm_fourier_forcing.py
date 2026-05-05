import numpy as np

from flucs.input import InvalidFlucsInputFileError
from flucs.solvers.fourier.fourier_system_forcing import FourierSystemForcing


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
        # Alias system
        system = self.system

        # Set ranges and number of forced modes
        self.setup_forcing_range_kz_kperp()

        # Validate energy injection rate and injection imbalance
        energy_injection_rate = system.input["forcing.energy_injection_rate"]
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
            forcing_epsilon_plus / self.forced_mode_count
        )
        system.module_options.define_float(
            "FORCING_EPSILON_MINUS", 
            forcing_epsilon_minus / self.forced_mode_count
        )