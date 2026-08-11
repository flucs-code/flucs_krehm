from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from collections.abc import Callable

import cupy as cp

from flucs.diagnostic import FlucsDiagnostic, FlucsDiagnosticVariable
from flucs.solvers.fourier.fourier_system_reductions import FourierReductions
from flucs.utilities.messages import flucsprint

if TYPE_CHECKING:
    from flucs_krehm.krehm_fourier.krehm_fourier import KREHMFourier


def _half_complex_blocks(system):
    """
    Returns the (unpadded_slice, padded_slice) pairs for the kz and kx axes of
    a half-complex array.

    kx and kz are fftfreq-ordered, so the negative-frequency half of each axis
    lives at the *end* of the axis. Zero-padding therefore means moving those
    blocks out to the end of the longer axis rather than leaving them in place,
    which is why cp.fft's `s=` argument cannot be used to do this.
    """
    blocks = []
    for n, half_n, padded_n in (
        (system.nz, system.half_nz, system.padded_nz),
        (system.nx, system.half_nx, system.padded_nx),
    ):
        n_negative = n - half_n
        blocks.append(
            (
                (slice(0, half_n), slice(0, half_n)),
                (slice(half_n, n), slice(padded_n - n_negative, padded_n)),
            )
        )

    return blocks


def pad_half_complex(array, system):
    """
    Zero-pads a half-complex array with trailing shape
    system.half_unpadded_tuple onto the padded Fourier grid, so that products
    formed from it in real space are free of aliasing.
    """
    padded = cp.zeros(
        array.shape[:-3] + system.half_padded_tuple, dtype=array.dtype
    )

    (z_blocks, x_blocks) = _half_complex_blocks(system)
    iky = slice(0, system.half_ny)

    for ikz, ikz_padded in z_blocks:
        for ikx, ikx_padded in x_blocks:
            padded[..., ikz_padded, ikx_padded, iky] = array[..., ikz, ikx, iky]

    return padded


def unpad_half_complex(array, system):
    """
    Inverse of pad_half_complex: truncates a padded half-complex array back
    onto the unpadded Fourier grid.
    """
    unpadded = cp.zeros(
        array.shape[:-3] + system.half_unpadded_tuple, dtype=array.dtype
    )

    (z_blocks, x_blocks) = _half_complex_blocks(system)
    iky = slice(0, system.half_ny)

    for ikz, ikz_padded in z_blocks:
        for ikx, ikx_padded in x_blocks:
            unpadded[..., ikz, ikx, iky] = array[..., ikz_padded, ikx_padded, iky]

    return unpadded


class FreeEnergyDiag(FlucsDiagnostic):
    """
    Computes quantities related to the free energy, including contributions to
    the free-energy budget from injection and dissipation
    """

    name = "free_energy"
    system: KREHMFourier
    option_defaults: ClassVar[dict[str, object]] = {
        "save_contributions": False,
        "save_elsasser": False,
    }

    get_W: Callable[..., cp.ndarray]
    get_dWdt_forcing: Callable[..., cp.ndarray]
    get_dWdt_hyperdissipation_component: Callable[..., cp.ndarray]

    get_W_uperp: Callable[..., cp.ndarray]
    get_W_dens: Callable[..., cp.ndarray]
    get_W_bperp: Callable[..., cp.ndarray]
    get_W_upar: Callable[..., cp.ndarray]

    get_Wp: Callable[..., cp.ndarray]
    get_dWpdt_forcing: Callable[..., cp.ndarray]
    get_dWpdt_hyperdissipation_component: Callable[..., cp.ndarray]

    get_Wm: Callable[..., cp.ndarray]
    get_dWmdt_forcing: Callable[..., cp.ndarray]
    get_dWmdt_hyperdissipation_component: Callable[..., cp.ndarray]

    def init_vars(self) -> None:
        reductions = FourierReductions(self.system)

        # Add variables for free energy and its time derivative
        for name in ["W", "dWdt_forcing", "dWdt", "dWdt_error"]:
            self.add_var(FlucsDiagnosticVariable(
                name=name,
                shape=(),
                dimensions={},
                is_complex=False
            ))

        # Add variables for hyperdissipation components
        for component in self.system.hyperdissipation_components:
            self.add_var(FlucsDiagnosticVariable(
                name=f"dWdt_hyperdissipation_{component}",
                shape=(),
                dimensions={},
                is_complex=False
            ))

        # Register reductions
        self.get_W = reductions.get_reduction(
            reduction_output="scalar",
            functor="FreeEnergy_Functor",
            input_args="FLUCS_COMPLEX*",
            complex_output=False,
        )
        self.get_dWdt_forcing = reductions.get_reduction(
            reduction_output="scalar",
            functor="FreeEnergyForcing_Functor",
            input_args="const FLUCS_COMPLEX (*)[HALFUNPADDEDSIZE],FLUCS_FLOAT,FLUCS_FLOAT,long long",
            complex_output=False,
        )
        self.get_dWdt_hyperdissipation_component = reductions.get_reduction(
            reduction_output="scalar",
            functor="FreeEnergyHyperdissipationComponent_Functor",
            input_args="FLUCS_COMPLEX*,FLUCS_FLOAT,int",
            complex_output=False,
        )

        # Add contributions diagnostics
        if self.save_contributions:
            # Add variables
            for name in ["W_uperp", "W_dens", "W_bperp", "W_upar"]:
                self.add_var(FlucsDiagnosticVariable(
                    name=name,
                    shape=(),
                    dimensions={},
                    is_complex=False,
                ))

            # Register reductions
            self.get_W_uperp = reductions.get_reduction(
                reduction_output="scalar",
                functor="FreeEnergyUperp_Functor",
                input_args="FLUCS_COMPLEX*",
                complex_output=False
            )

            self.get_W_dens = reductions.get_reduction(
                reduction_output="scalar",
                functor="FreeEnergyDens_Functor",
                input_args="FLUCS_COMPLEX*",
                complex_output=False
            )

            self.get_W_bperp = reductions.get_reduction(
                reduction_output="scalar",
                functor="FreeEnergyBperp_Functor",
                input_args="FLUCS_COMPLEX*",
                complex_output=False
            )

            self.get_W_upar = reductions.get_reduction(
                reduction_output="scalar",
                functor="FreeEnergyUpar_Functor",
                input_args="FLUCS_COMPLEX*",
                complex_output=False
            )

        # Add elsasser diagnostics
        if self.save_elsasser:

            for name in [
                "Wp",
                "dWpdt",
                "dWpdt_forcing",
                "Wm",
                "dWmdt",
                "dWmdt_forcing",
            ]:
                self.add_var(FlucsDiagnosticVariable(
                    name=name,
                    shape=(),
                    dimensions={},
                    is_complex=False
                ))

            for component in self.system.hyperdissipation_components:
                self.add_var(FlucsDiagnosticVariable(
                    name=f"dWpdt_hyperdissipation_{component}",
                    shape=(),
                    dimensions={},
                    is_complex=False
                ))
                self.add_var(FlucsDiagnosticVariable(
                    name=f"dWmdt_hyperdissipation_{component}",
                    shape=(),
                    dimensions={},
                    is_complex=False
                ))

            # Register reductions
            self.get_Wp = reductions.get_reduction(
                reduction_output="scalar",
                functor="FreeEnergyThetap_Functor",
                input_args="FLUCS_COMPLEX*",
                complex_output=False,
            )
            self.get_dWpdt_forcing = reductions.get_reduction(
                reduction_output="scalar",
                functor="FreeEnergyThetapForcing_Functor",
                input_args="const FLUCS_COMPLEX (*)[HALFUNPADDEDSIZE],FLUCS_FLOAT,FLUCS_FLOAT,long long",
                complex_output=False,
            )
            self.get_dWpdt_hyperdissipation_component = reductions.get_reduction(
                reduction_output="scalar",
                functor="FreeEnergyThetapHyperdissipationComponent_Functor",
                input_args="FLUCS_COMPLEX*,FLUCS_FLOAT,int",
                complex_output=False,
            )
            self.get_Wm = reductions.get_reduction(
                reduction_output="scalar",
                functor="FreeEnergyThetam_Functor",
                input_args="FLUCS_COMPLEX*",
                complex_output=False,
            )
            self.get_dWmdt_forcing = reductions.get_reduction(
                reduction_output="scalar",
                functor="FreeEnergyThetamForcing_Functor",
                input_args="const FLUCS_COMPLEX (*)[HALFUNPADDEDSIZE],FLUCS_FLOAT,FLUCS_FLOAT,long long",
                complex_output=False,
            )
            self.get_dWmdt_hyperdissipation_component = reductions.get_reduction(
                reduction_output="scalar",
                functor="FreeEnergyThetamHyperdissipationComponent_Functor",
                input_args="FLUCS_COMPLEX*,FLUCS_FLOAT,int",
                complex_output=False,
            )

    def ready(self) -> None:
        pass

    def execute(self) -> None:
        # Useful aliases
        current_dt = self.system.float(self.system.current_dt)
        current_time = self.system.float(self.system.current_time)
        current_step = self.system.int(self.system.current_step)
        adaptive_rate = self.system.float(self.system.adaptive_rate)

        fields = self.system.get_fields()
        fields_prev = self.system.get_fields(1)

        # W
        W = self.get_W(fields).get().item()
        self.save_data("W", W)

        # dWdt_forcing
        dWdt_forcing = self.get_dWdt_forcing(
            fields, current_dt, current_time, current_step
        ).get().item()
        self.save_data("dWdt_forcing", dWdt_forcing)

        # dWdt
        W_prev = self.get_W(fields_prev)
        dWdt = (W - W_prev.get().item()) / current_dt
        self.save_data("dWdt", dWdt)

        # dWdt_hyperdissipation
        dWdt_hyperdissipation_total = 0.0
        for index, component in enumerate(self.system.hyperdissipation_components):
            result = self.get_dWdt_hyperdissipation_component(
                fields, adaptive_rate, index
            )
            dWdt_hyperdissipation_component = -result.get().item()
            self.save_data(
                f"dWdt_hyperdissipation_{component}",
                dWdt_hyperdissipation_component
            )
            dWdt_hyperdissipation_total += dWdt_hyperdissipation_component

        # dWdt_error
        self.save_data(
            "dWdt_error",
            dWdt - dWdt_forcing - dWdt_hyperdissipation_total,
        )

        # Saving contributions if required
        if self.save_contributions:
            
            # (delta u_perp)**2
            self.save_data("W_uperp", self.get_W_uperp(fields).get().item())

            # (delta n_e)**2
            self.save_data("W_dens", self.get_W_dens(fields).get().item())

            # (delta b_perp)**2
            self.save_data("W_bperp", self.get_W_bperp(fields).get().item())

            # (delta u_parallel)**2
            self.save_data("W_upar", self.get_W_upar(fields).get().item())

        # Saving elsasser fields if required
        if self.save_elsasser:

            # Wp
            Wp = self.get_Wp(fields).get().item()
            self.save_data("Wp", Wp)

            # dWpdt_forcing
            dWpdt_forcing = self.get_dWpdt_forcing(
                fields, current_dt, current_time, current_step
            ).get().item()
            self.save_data("dWpdt_forcing", dWpdt_forcing)

            # dWpdt
            Wp_prev = self.get_Wp(fields_prev)
            dWpdt = (Wp - Wp_prev.get().item()) / current_dt
            self.save_data("dWpdt", dWpdt)

            # dWpdt_hyperdissipation
            for index, component in enumerate(self.system.hyperdissipation_components):
                result = self.get_dWpdt_hyperdissipation_component(
                    fields, adaptive_rate, index
                )
                self.save_data(
                    f"dWpdt_hyperdissipation_{component}",
                    -result.get().item(),
                )

            # Wm
            Wm = self.get_Wm(fields).get().item()
            self.save_data("Wm", Wm)

            # dWmdt_forcing
            dWmdt_forcing = self.get_dWmdt_forcing(
                fields, current_dt, current_time, current_step
            ).get().item()
            self.save_data("dWmdt_forcing", dWmdt_forcing)

            # dWmdt
            Wm_prev = self.get_Wm(fields_prev)
            dWmdt = (Wm - Wm_prev.get().item()) / current_dt
            self.save_data("dWmdt", dWmdt)

            # dWmdt_hyperdissipation
            for index, component in enumerate(self.system.hyperdissipation_components):
                result = self.get_dWmdt_hyperdissipation_component(
                    fields, adaptive_rate, index
                )
                self.save_data(
                    f"dWmdt_hyperdissipation_{component}",
                    -result.get().item(),
                )


class HelicityDiag(FlucsDiagnostic):
    """
    Computes quantities related to the helicity, including contributions to the
    helicity budget from injection and dissipation
    """

    name = "helicity"
    system: KREHMFourier
    option_defaults: ClassVar[dict[str, object]] = {
        "save_contributions": False,
        "save_elsasser": False,
    }

    get_H: Callable[..., cp.ndarray]
    get_dHdt_forcing: Callable[..., cp.ndarray]
    get_dHdt_hyperdissipation_component: Callable[..., cp.ndarray]

    get_H_apar: Callable[..., cp.ndarray]
    get_H_upar: Callable[..., cp.ndarray]

    get_Hp: Callable[..., cp.ndarray]
    get_dHpdt_forcing: Callable[..., cp.ndarray]
    get_dHpdt_hyperdissipation_component: Callable[..., cp.ndarray]

    get_Hm: Callable[..., cp.ndarray]
    get_dHmdt_forcing: Callable[..., cp.ndarray]
    get_dHmdt_hyperdissipation_component: Callable[..., cp.ndarray]

    def init_vars(self) -> None:
        reductions = FourierReductions(self.system)

        # Add variables for helicity and its time derivative
        self.add_var(FlucsDiagnosticVariable(
            name="H",
            shape=(),
            dimensions={},
            is_complex=False
        ))
        self.add_var(FlucsDiagnosticVariable(
            name="dHdt",
            shape=(),
            dimensions={},
            is_complex=False
        ))
        self.add_var(FlucsDiagnosticVariable(
            name="dHdt_forcing",
            shape=(),
            dimensions={},
            is_complex=False
        ))
        self.add_var(FlucsDiagnosticVariable(
            name="dHdt_error",
            shape=(),
            dimensions={},
            is_complex=False
        ))

        # Add variables for hyperdissipation components
        for component in self.system.hyperdissipation_components:
            self.add_var(FlucsDiagnosticVariable(
                name=f"dHdt_hyperdissipation_{component}",
                shape=(),
                dimensions={},
                is_complex=False
            ))

        # Register reductions
        self.get_H = reductions.get_reduction(
            reduction_output="scalar",
            functor="Helicity_Functor",
            input_args="FLUCS_COMPLEX*",
            complex_output=False,
        )
        self.get_dHdt_forcing = reductions.get_reduction(
            reduction_output="scalar",
            functor="HelicityForcing_Functor",
            input_args="const FLUCS_COMPLEX (*)[HALFUNPADDEDSIZE],FLUCS_FLOAT,FLUCS_FLOAT,long long",
            complex_output=False,
        )
        self.get_dHdt_hyperdissipation_component = reductions.get_reduction(
            reduction_output="scalar",
            functor="HelicityHyperdissipationComponent_Functor",
            input_args="FLUCS_COMPLEX*,FLUCS_FLOAT,int",
            complex_output=False,
        )

        # Add contributions diagnostics
        if self.save_contributions:
            # Add variables
            for name in ["H_apar", "H_upar"]:
                self.add_var(FlucsDiagnosticVariable(
                    name=name,
                    shape=(),
                    dimensions={},
                    is_complex=False,
                ))

            # Register reductions
            self.get_H_apar = reductions.get_reduction(
                reduction_output="scalar",
                functor="HelicityApar_Functor",
                input_args="FLUCS_COMPLEX*",
                complex_output=False
            )

            self.get_H_upar = reductions.get_reduction(
                reduction_output="scalar",
                functor="HelicityUpar_Functor",
                input_args="FLUCS_COMPLEX*",
                complex_output=False
            )

        # Add elsasser diagnostics
        if self.save_elsasser:
            self.add_var(FlucsDiagnosticVariable(
                name="Hp",
                shape=(),
                dimensions={},
                is_complex=False
            ))
            self.add_var(FlucsDiagnosticVariable(
                name="Hm",
                shape=(),
                dimensions={},
                is_complex=False
            ))
            self.add_var(FlucsDiagnosticVariable(
                name="dHpdt",
                shape=(),
                dimensions={},
                is_complex=False
            ))
            self.add_var(FlucsDiagnosticVariable(
                name="dHpdt_forcing",
                shape=(),
                dimensions={},
                is_complex=False
            ))
            self.add_var(FlucsDiagnosticVariable(
                name="dHmdt",
                shape=(),
                dimensions={},
                is_complex=False
            ))
            self.add_var(FlucsDiagnosticVariable(
                name="dHmdt_forcing",
                shape=(),
                dimensions={},
                is_complex=False
            ))

            for component in self.system.hyperdissipation_components:
                self.add_var(FlucsDiagnosticVariable(
                    name=f"dHpdt_hyperdissipation_{component}",
                    shape=(),
                    dimensions={},
                    is_complex=False
                ))
                self.add_var(FlucsDiagnosticVariable(
                    name=f"dHmdt_hyperdissipation_{component}",
                    shape=(),
                    dimensions={},
                    is_complex=False
                ))

            # Register reductions
            self.get_Hp = reductions.get_reduction(
                reduction_output="scalar",
                functor="HelicityThetap_Functor",
                input_args="FLUCS_COMPLEX*",
                complex_output=False,
            )
            self.get_dHpdt_forcing = reductions.get_reduction(
                reduction_output="scalar",
                functor="HelicityThetapForcing_Functor",
                input_args="const FLUCS_COMPLEX (*)[HALFUNPADDEDSIZE],FLUCS_FLOAT,FLUCS_FLOAT,long long",
                complex_output=False,
            )
            self.get_dHpdt_hyperdissipation_component = reductions.get_reduction(
                reduction_output="scalar",
                functor="HelicityThetapHyperdissipationComponent_Functor",
                input_args="FLUCS_COMPLEX*,FLUCS_FLOAT,int",
                complex_output=False,
            )
            self.get_Hm = reductions.get_reduction(
                reduction_output="scalar",
                functor="HelicityThetam_Functor",
                input_args="FLUCS_COMPLEX*",
                complex_output=False,
            )
            self.get_dHmdt_forcing = reductions.get_reduction(
                reduction_output="scalar",
                functor="HelicityThetamForcing_Functor",
                input_args="const FLUCS_COMPLEX (*)[HALFUNPADDEDSIZE],FLUCS_FLOAT,FLUCS_FLOAT,long long",
                complex_output=False,
            )
            self.get_dHmdt_hyperdissipation_component = reductions.get_reduction(
                reduction_output="scalar",
                functor="HelicityThetamHyperdissipationComponent_Functor",
                input_args="FLUCS_COMPLEX*,FLUCS_FLOAT,int",
                complex_output=False,
            )

    def ready(self) -> None:
        pass

    def execute(self) -> None:
        # Useful aliases
        current_dt = self.system.float(self.system.current_dt)
        current_time = self.system.float(self.system.current_time)
        current_step = self.system.int(self.system.current_step)
        adaptive_rate = self.system.float(self.system.adaptive_rate)

        fields = self.system.get_fields()
        fields_prev = self.system.get_fields(1)

        # H
        H = self.get_H(fields).get().item()
        self.save_data("H", H)

        # dHdt_forcing
        dHdt_forcing = self.get_dHdt_forcing(
            fields, current_dt, current_time, current_step
        ).get().item()
        self.save_data("dHdt_forcing", dHdt_forcing)

        # dHdt
        H_prev = self.get_H(fields_prev)
        dHdt = (H - H_prev.get().item()) / current_dt
        self.save_data("dHdt", dHdt)

        # dHdt_hyperdissipation
        dHdt_hyperdissipation_total = 0.0
        for index, component in enumerate(self.system.hyperdissipation_components):
            result = self.get_dHdt_hyperdissipation_component(
                fields, adaptive_rate, index
            )
            dHdt_hyperdissipation_component = -result.get().item()
            self.save_data(
                f"dHdt_hyperdissipation_{component}",
                dHdt_hyperdissipation_component
            )
            dHdt_hyperdissipation_total += dHdt_hyperdissipation_component

        # dHdt_error
        self.save_data(
            "dHdt_error",
            dHdt - dHdt_forcing - dHdt_hyperdissipation_total,
        )

        # Saving contributions if required
        if self.save_contributions:
            
            # (delta n_e) * apar
            self.save_data("H_apar", self.get_H_apar(fields).get().item())

            # (delta n_e) * upar
            self.save_data("H_upar", self.get_H_upar(fields).get().item())

        # Saving elsasser fields if required
        if self.save_elsasser:

            # Hp
            Hp = self.get_Hp(fields).get().item()
            self.save_data("Hp", Hp)

            # dHpdt_forcing
            dHpdt_forcing = self.get_dHpdt_forcing(
                fields, current_dt, current_time, current_step
            ).get().item()
            self.save_data("dHpdt_forcing", dHpdt_forcing)

            # dHpdt
            Hp_prev = self.get_Hp(fields_prev)
            dHpdt = (Hp - Hp_prev.get().item()) / current_dt
            self.save_data("dHpdt", dHpdt)

            # dHpdt_hyperdissipation
            for index, component in enumerate(self.system.hyperdissipation_components):
                result = self.get_dHpdt_hyperdissipation_component(
                    fields, adaptive_rate, index
                )
                self.save_data(
                    f"dHpdt_hyperdissipation_{component}",
                    -result.get().item(),
                )

            # Hm
            Hm = self.get_Hm(fields).get().item()
            self.save_data("Hm", Hm)

            # dHmdt_forcing
            dHmdt_forcing = self.get_dHmdt_forcing(
                fields, current_dt, current_time, current_step
            ).get().item()
            self.save_data("dHmdt_forcing", dHmdt_forcing)

            # dHmdt
            Hm_prev = self.get_Hm(fields_prev)
            dHmdt = (Hm - Hm_prev.get().item()) / current_dt
            self.save_data("dHmdt", dHmdt)

            # dHmdt_hyperdissipation
            for index, component in enumerate(self.system.hyperdissipation_components):
                result = self.get_dHmdt_hyperdissipation_component(
                    fields, adaptive_rate, index
                )
                self.save_data(
                    f"dHmdt_hyperdissipation_{component}",
                    -result.get().item(),
                )


class FreeEnergyDiag1D(FlucsDiagnostic):
    """
    Computes 1D spectra of KREHM free-energy quantities and budget terms.
    """

    name = "free_energy_1d"
    system: KREHMFourier
    option_defaults: ClassVar[dict[str, object]] = {
        "spectra": ["kperp"],
        "save_contributions": False,
        "save_elsasser": False,
    }

    get_W: dict[str, Callable[..., cp.ndarray]]
    get_dWdt_forcing: dict[str, Callable[..., cp.ndarray]]
    get_dWdt_hyperdissipation: dict[str, Callable[..., cp.ndarray]]

    get_W_uperp: dict[str, Callable[..., cp.ndarray]]
    get_W_dens: dict[str, Callable[..., cp.ndarray]]
    get_W_bperp: dict[str, Callable[..., cp.ndarray]]
    get_W_upar: dict[str, Callable[..., cp.ndarray]]

    get_Wp: dict[str, Callable[..., cp.ndarray]]
    get_dWpdt_forcing: dict[str, Callable[..., cp.ndarray]]
    get_dWpdt_hyperdissipation: dict[str, Callable[..., cp.ndarray]]
    get_Wm: dict[str, Callable[..., cp.ndarray]]
    get_dWmdt_forcing: dict[str, Callable[..., cp.ndarray]]
    get_dWmdt_hyperdissipation: dict[str, Callable[..., cp.ndarray]]

    def init_vars(self) -> None:
        reductions = FourierReductions(self.system)

        # Parse valid spectra (enforce 1D)
        valid_spectra = ("kz", "kx", "ky", "kperp")
        spectra = self.spectra
        if isinstance(spectra, str):
            spectra = [spectra]
        spectra = tuple(dict.fromkeys(spectra))

        invalid_spectra = set(spectra) - set(valid_spectra)
        if invalid_spectra:
            raise ValueError(
                f"{self.name} only supports 1D spectra {valid_spectra}."
            )

        # Initialise dicts
        self.get_W = {}
        self.get_dWdt_forcing = {}
        self.get_dWdt_hyperdissipation = {}

        self.get_W_uperp = {}
        self.get_W_dens = {}
        self.get_W_bperp = {}
        self.get_W_upar = {}

        self.get_Wp = {}
        self.get_dWpdt_forcing = {}
        self.get_dWpdt_hyperdissipation = {}
        self.get_Wm = {}
        self.get_dWmdt_forcing = {}
        self.get_dWmdt_hyperdissipation = {}

        # Iterate over spectra types and init variables
        for spectrum in spectra:
            dimensions = reductions.get_dimensions(spectrum)
            shape = tuple(dimensions)

            for name in ["W", "dWdt_forcing", "dWdt_hyperdissipation"]:
                self.add_var(FlucsDiagnosticVariable(
                    name=f"{spectrum}_spectra/{name}",
                    shape=shape,
                    dimensions=dimensions,
                    is_complex=False,
                ))

            self.get_W[spectrum] = reductions.get_reduction(
                reduction_output=spectrum,
                functor="FreeEnergy_Functor",
                input_args="FLUCS_COMPLEX*",
                complex_output=False,
            )
            self.get_dWdt_forcing[spectrum] = reductions.get_reduction(
                reduction_output=spectrum,
                functor="FreeEnergyForcing_Functor",
                input_args="const FLUCS_COMPLEX (*)[HALFUNPADDEDSIZE],FLUCS_FLOAT,FLUCS_FLOAT,long long",
                complex_output=False,
            )
            self.get_dWdt_hyperdissipation[spectrum] = reductions.get_reduction(
                reduction_output=spectrum,
                functor="FreeEnergyHyperdissipation_Functor",
                input_args="FLUCS_COMPLEX*,FLUCS_FLOAT",
                complex_output=False,
            )

            # Save contributions if required
            if self.save_contributions:
                for name in ["W_uperp", "W_dens", "W_bperp", "W_upar"]:
                    self.add_var(FlucsDiagnosticVariable(
                        name=f"{spectrum}_spectra/{name}",
                        shape=shape,
                        dimensions=dimensions,
                        is_complex=False,
                    ))

                self.get_W_uperp[spectrum] = reductions.get_reduction(
                    reduction_output=spectrum,
                    functor="FreeEnergyUperp_Functor",
                    input_args="FLUCS_COMPLEX*",
                    complex_output=False
                )
                self.get_W_dens[spectrum] = reductions.get_reduction(
                    reduction_output=spectrum,
                    functor="FreeEnergyDens_Functor",
                    input_args="FLUCS_COMPLEX*",
                    complex_output=False
                )
                self.get_W_bperp[spectrum] = reductions.get_reduction(
                    reduction_output=spectrum,
                    functor="FreeEnergyBperp_Functor",
                    input_args="FLUCS_COMPLEX*",
                    complex_output=False
                )
                self.get_W_upar[spectrum] = reductions.get_reduction(
                    reduction_output=spectrum,
                    functor="FreeEnergyUpar_Functor",
                    input_args="FLUCS_COMPLEX*",
                    complex_output=False
                )

            # Save Elsasser contributions if required
            if self.save_elsasser:
                for name in [
                    "Wp",
                    "dWpdt_forcing",
                    "dWpdt_hyperdissipation",
                    "Wm",
                    "dWmdt_forcing",
                    "dWmdt_hyperdissipation",
                ]:
                    self.add_var(FlucsDiagnosticVariable(
                        name=f"{spectrum}_spectra/{name}",
                        shape=shape,
                        dimensions=dimensions,
                        is_complex=False,
                    ))

                self.get_Wp[spectrum] = reductions.get_reduction(
                    reduction_output=spectrum,
                    functor="FreeEnergyThetap_Functor",
                    input_args="FLUCS_COMPLEX*",
                    complex_output=False
                )
                self.get_dWpdt_forcing[spectrum] = reductions.get_reduction(
                    reduction_output=spectrum,
                    functor="FreeEnergyThetapForcing_Functor",
                    input_args="const FLUCS_COMPLEX (*)[HALFUNPADDEDSIZE],FLUCS_FLOAT,FLUCS_FLOAT,long long",
                    complex_output=False,
                )
                self.get_dWpdt_hyperdissipation[spectrum] = reductions.get_reduction(
                    reduction_output=spectrum,
                    functor="FreeEnergyThetapHyperdissipation_Functor",
                    input_args="FLUCS_COMPLEX*,FLUCS_FLOAT",
                    complex_output=False,
                )
                self.get_Wm[spectrum] = reductions.get_reduction(
                    reduction_output=spectrum,
                    functor="FreeEnergyThetam_Functor",
                    input_args="FLUCS_COMPLEX*",
                    complex_output=False
                )
                self.get_dWmdt_forcing[spectrum] = reductions.get_reduction(
                    reduction_output=spectrum,
                    functor="FreeEnergyThetamForcing_Functor",
                    input_args="const FLUCS_COMPLEX (*)[HALFUNPADDEDSIZE],FLUCS_FLOAT,FLUCS_FLOAT,long long",
                    complex_output=False,
                )
                self.get_dWmdt_hyperdissipation[spectrum] = reductions.get_reduction(
                    reduction_output=spectrum,
                    functor="FreeEnergyThetamHyperdissipation_Functor",
                    input_args="FLUCS_COMPLEX*,FLUCS_FLOAT",
                    complex_output=False,
                )

    def ready(self) -> None:
        pass

    def execute(self) -> None:
        current_dt = self.system.float(self.system.current_dt)
        current_time = self.system.float(self.system.current_time)
        current_step = self.system.int(self.system.current_step)
        adaptive_rate = self.system.float(self.system.adaptive_rate)
        fields = self.system.get_fields()

        # Iterate over spectra to save
        for spectrum in self.get_W:

            # Save raw spectra
            self.save_data(
                f"{spectrum}_spectra/W", self.get_W[spectrum](fields).get()
            )

            # dWdt_forcing
            self.save_data(
                f"{spectrum}_spectra/dWdt_forcing",
                self.get_dWdt_forcing[spectrum](
                    fields, current_dt, current_time, current_step
                ).get(),
            )

            # dWdt_hyperdissipation
            result = self.get_dWdt_hyperdissipation[spectrum](
                fields, adaptive_rate
            )
            self.save_data(
                f"{spectrum}_spectra/dWdt_hyperdissipation",
                -result.get(),
            )

            if self.save_contributions:
                self.save_data(
                    f"{spectrum}_spectra/W_uperp", 
                    self.get_W_uperp[spectrum](fields).get()
                )
                self.save_data(
                    f"{spectrum}_spectra/W_dens", 
                    self.get_W_dens[spectrum](fields).get()
                )
                self.save_data(
                    f"{spectrum}_spectra/W_bperp", 
                    self.get_W_bperp[spectrum](fields).get()
                )
                self.save_data(
                    f"{spectrum}_spectra/W_upar", 
                    self.get_W_upar[spectrum](fields).get()
                )

            if self.save_elsasser:

                # Wp
                self.save_data(
                    f"{spectrum}_spectra/Wp", 
                    self.get_Wp[spectrum](fields).get()
                )

                # dWpdt_forcing
                self.save_data(
                    f"{spectrum}_spectra/dWpdt_forcing",
                    self.get_dWpdt_forcing[spectrum](
                        fields, current_dt, current_time, current_step
                    ).get(),
                )

                # dWpdt_hyperdissipation
                result_p = self.get_dWpdt_hyperdissipation[spectrum](
                    fields, adaptive_rate
                )
                self.save_data(
                    f"{spectrum}_spectra/dWpdt_hyperdissipation",
                    -result_p.get(),
                )

                # Wm
                self.save_data(
                    f"{spectrum}_spectra/Wm", 
                    self.get_Wm[spectrum](fields).get()
                )

                # dWmdt_forcing
                self.save_data(
                    f"{spectrum}_spectra/dWmdt_forcing",
                    self.get_dWmdt_forcing[spectrum](
                        fields, current_dt, current_time, current_step
                    ).get(),
                )

                # dWmdt_hyperdissipation
                result_m = self.get_dWmdt_hyperdissipation[spectrum](
                    fields, adaptive_rate
                )
                self.save_data(
                    f"{spectrum}_spectra/dWmdt_hyperdissipation",
                    -result_m.get(),
                )


class HelicityDiag1D(FlucsDiagnostic):
    """
    Computes 1D spectra of KREHM helicity quantities and budget terms.
    """

    name = "helicity_1d"
    system: KREHMFourier
    option_defaults: ClassVar[dict[str, object]] = {
        "spectra": ["kperp"],
        "save_contributions": False,
        "save_elsasser": False,
    }

    get_H: dict[str, Callable[..., cp.ndarray]]
    get_dHdt_forcing: dict[str, Callable[..., cp.ndarray]]
    get_dHdt_hyperdissipation: dict[str, Callable[..., cp.ndarray]]

    get_H_apar: dict[str, Callable[..., cp.ndarray]]
    get_H_upar: dict[str, Callable[..., cp.ndarray]]

    get_Hp: dict[str, Callable[..., cp.ndarray]]
    get_dHpdt_forcing: dict[str, Callable[..., cp.ndarray]]
    get_dHpdt_hyperdissipation: dict[str, Callable[..., cp.ndarray]]
    get_Hm: dict[str, Callable[..., cp.ndarray]]
    get_dHmdt_forcing: dict[str, Callable[..., cp.ndarray]]
    get_dHmdt_hyperdissipation: dict[str, Callable[..., cp.ndarray]]

    def init_vars(self) -> None:
        reductions = FourierReductions(self.system)

        # Parse valid spectra
        valid_spectra = ("kz", "kx", "ky", "kperp")
        spectra = self.spectra
        if isinstance(spectra, str):
            spectra = [spectra]
        spectra = tuple(dict.fromkeys(spectra))

        invalid_spectra = set(spectra) - set(valid_spectra)
        if invalid_spectra:
            raise ValueError(
                f"{self.name} only supports 1D spectra {valid_spectra}."
            )

        # Initialise dicts to iterate over
        self.get_H = {}
        self.get_dHdt_forcing = {}
        self.get_dHdt_hyperdissipation = {}

        self.get_H_apar = {}
        self.get_H_upar = {}

        self.get_Hp = {}
        self.get_dHpdt_forcing = {}
        self.get_dHpdt_hyperdissipation = {}
        self.get_Hm = {}
        self.get_dHmdt_forcing = {}
        self.get_dHmdt_hyperdissipation = {}

        # Iterate over spectra types and init variables
        for spectrum in spectra:
            dimensions = reductions.get_dimensions(spectrum)
            shape = tuple(dimensions)

            for name in ["H", "dHdt_forcing", "dHdt_hyperdissipation"]:
                self.add_var(FlucsDiagnosticVariable(
                    name=f"{spectrum}_spectra/{name}",
                    shape=shape,
                    dimensions=dimensions,
                    is_complex=False,
                ))

            self.get_H[spectrum] = reductions.get_reduction(
                reduction_output=spectrum,
                functor="Helicity_Functor",
                input_args="FLUCS_COMPLEX*",
                complex_output=False,
            )
            self.get_dHdt_forcing[spectrum] = reductions.get_reduction(
                reduction_output=spectrum,
                functor="HelicityForcing_Functor",
                input_args="const FLUCS_COMPLEX (*)[HALFUNPADDEDSIZE],FLUCS_FLOAT,FLUCS_FLOAT,long long",
                complex_output=False,
            )
            self.get_dHdt_hyperdissipation[spectrum] = reductions.get_reduction(
                reduction_output=spectrum,
                functor="HelicityHyperdissipation_Functor",
                input_args="FLUCS_COMPLEX*,FLUCS_FLOAT",
                complex_output=False,
            )

            # Save contributions if required
            if self.save_contributions:
                for name in ["H_apar", "H_upar"]:
                    self.add_var(FlucsDiagnosticVariable(
                        name=f"{spectrum}_spectra/{name}",
                        shape=shape,
                        dimensions=dimensions,
                        is_complex=False,
                    ))

                self.get_H_apar[spectrum] = reductions.get_reduction(
                    reduction_output=spectrum,
                    functor="HelicityApar_Functor",
                    input_args="FLUCS_COMPLEX*",
                    complex_output=False,
                )
                self.get_H_upar[spectrum] = reductions.get_reduction(
                    reduction_output=spectrum,
                    functor="HelicityUpar_Functor",
                    input_args="FLUCS_COMPLEX*",
                    complex_output=False,
                )

            # Save Elsasser contributions if required
            if self.save_elsasser:
                for name in [
                    "Hp",
                    "dHpdt_forcing",
                    "dHpdt_hyperdissipation",
                    "Hm",
                    "dHmdt_forcing",
                    "dHmdt_hyperdissipation",
                ]:
                    self.add_var(FlucsDiagnosticVariable(
                        name=f"{spectrum}_spectra/{name}",
                        shape=shape,
                        dimensions=dimensions,
                        is_complex=False,
                    ))

                self.get_Hp[spectrum] = reductions.get_reduction(
                    reduction_output=spectrum,
                    functor="HelicityThetap_Functor",
                    input_args="FLUCS_COMPLEX*",
                    complex_output=False,
                )
                self.get_dHpdt_forcing[spectrum] = reductions.get_reduction(
                    reduction_output=spectrum,
                    functor="HelicityThetapForcing_Functor",
                    input_args="const FLUCS_COMPLEX (*)[HALFUNPADDEDSIZE],FLUCS_FLOAT,FLUCS_FLOAT,long long",
                    complex_output=False,
                )
                self.get_dHpdt_hyperdissipation[spectrum] = reductions.get_reduction(
                    reduction_output=spectrum,
                    functor="HelicityThetapHyperdissipation_Functor",
                    input_args="FLUCS_COMPLEX*,FLUCS_FLOAT",
                    complex_output=False,
                )
                self.get_Hm[spectrum] = reductions.get_reduction(
                    reduction_output=spectrum,
                    functor="HelicityThetam_Functor",
                    input_args="FLUCS_COMPLEX*",
                    complex_output=False,
                )
                self.get_dHmdt_forcing[spectrum] = reductions.get_reduction(
                    reduction_output=spectrum,
                    functor="HelicityThetamForcing_Functor",
                    input_args="const FLUCS_COMPLEX (*)[HALFUNPADDEDSIZE],FLUCS_FLOAT,FLUCS_FLOAT,long long",
                    complex_output=False,
                )
                self.get_dHmdt_hyperdissipation[spectrum] = reductions.get_reduction(
                    reduction_output=spectrum,
                    functor="HelicityThetamHyperdissipation_Functor",
                    input_args="FLUCS_COMPLEX*,FLUCS_FLOAT",
                    complex_output=False,
                )

    def ready(self) -> None:
        pass

    def execute(self) -> None:
        current_dt = self.system.float(self.system.current_dt)
        current_time = self.system.float(self.system.current_time)
        current_step = self.system.int(self.system.current_step)
        adaptive_rate = self.system.float(self.system.adaptive_rate)
        fields = self.system.get_fields()

        # Iterate over spectra to save
        for spectrum in self.get_H:

            # H
            self.save_data(
                f"{spectrum}_spectra/H", self.get_H[spectrum](fields).get()
            )

            # dHdt_forcing
            self.save_data(
                f"{spectrum}_spectra/dHdt_forcing",
                self.get_dHdt_forcing[spectrum](
                    fields, current_dt, current_time, current_step
                ).get(),
            )

            # dHdt_hyperdissipation
            result = self.get_dHdt_hyperdissipation[spectrum](
                fields, adaptive_rate
            )
            self.save_data(
                f"{spectrum}_spectra/dHdt_hyperdissipation",
                -result.get(),
            )

            if self.save_contributions:
                self.save_data(
                    f"{spectrum}_spectra/H_apar",
                    self.get_H_apar[spectrum](fields).get(),
                )
                self.save_data(
                    f"{spectrum}_spectra/H_upar",
                    self.get_H_upar[spectrum](fields).get(),
                )

            if self.save_elsasser:
                # Hp
                self.save_data(
                    f"{spectrum}_spectra/Hp", self.get_Hp[spectrum](fields).get()
                )

                # dHpdt_forcing
                self.save_data(
                    f"{spectrum}_spectra/dHpdt_forcing",
                    self.get_dHpdt_forcing[spectrum](
                        fields, current_dt, current_time, current_step
                    ).get(),
                )

                # dHpdt_hyperdissipation
                result_p = self.get_dHpdt_hyperdissipation[spectrum](
                    fields, adaptive_rate
                )
                self.save_data(
                    f"{spectrum}_spectra/dHpdt_hyperdissipation",
                    -result_p.get(),
                )

                # Hm
                self.save_data(
                    f"{spectrum}_spectra/Hm", self.get_Hm[spectrum](fields).get()
                )

                # dHmdt_forcing
                self.save_data(
                    f"{spectrum}_spectra/dHmdt_forcing",
                    self.get_dHmdt_forcing[spectrum](
                        fields, current_dt, current_time, current_step
                    ).get(),
                )

                # dHmdt_hyperdissipation
                result_m = self.get_dHmdt_hyperdissipation[spectrum](
                    fields, adaptive_rate
                )
                self.save_data(
                    f"{spectrum}_spectra/dHmdt_hyperdissipation",
                    -result_m.get(),
                )


class FluxesDiag(FlucsDiagnostic):
    """
    Computes cumulative transfer diagnostics for the KREHM invariants.
    """

    name = "fluxes"
    system: KREHMFourier
    option_defaults: ClassVar[dict[str, object]] = {
        "fluxes": ["kperp"],
        "save_free_energy": True,
        "save_helicity": False,
        "save_elsasser": False,
    }

    get_W: dict[str, Callable[..., cp.ndarray]]
    get_dWdt_nonlinear: dict[str, Callable[..., cp.ndarray]]
    get_dWdt_forcing: dict[str, Callable[..., cp.ndarray]]
    get_dWdt_hyperdissipation: dict[str, Callable[..., cp.ndarray]]

    get_H: dict[str, Callable[..., cp.ndarray]]
    get_dHdt_nonlinear: dict[str, Callable[..., cp.ndarray]]
    get_dHdt_forcing: dict[str, Callable[..., cp.ndarray]]
    get_dHdt_hyperdissipation: dict[str, Callable[..., cp.ndarray]]

    get_Wp: dict[str, Callable[..., cp.ndarray]]
    get_dWpdt_nonlinear: dict[str, Callable[..., cp.ndarray]]
    get_dWpdt_forcing: dict[str, Callable[..., cp.ndarray]]
    get_dWpdt_hyperdissipation: dict[str, Callable[..., cp.ndarray]]

    get_Wm: dict[str, Callable[..., cp.ndarray]]
    get_dWmdt_nonlinear: dict[str, Callable[..., cp.ndarray]]
    get_dWmdt_forcing: dict[str, Callable[..., cp.ndarray]]
    get_dWmdt_hyperdissipation: dict[str, Callable[..., cp.ndarray]]

    get_Hp: dict[str, Callable[..., cp.ndarray]]
    get_dHpdt_nonlinear: dict[str, Callable[..., cp.ndarray]]
    get_dHpdt_forcing: dict[str, Callable[..., cp.ndarray]]
    get_dHpdt_hyperdissipation: dict[str, Callable[..., cp.ndarray]]

    get_Hm: dict[str, Callable[..., cp.ndarray]]
    get_dHmdt_nonlinear: dict[str, Callable[..., cp.ndarray]]
    get_dHmdt_forcing: dict[str, Callable[..., cp.ndarray]]
    get_dHmdt_hyperdissipation: dict[str, Callable[..., cp.ndarray]]

    def init_vars(self) -> None:
        reductions = FourierReductions(self.system)

        # Check whether the run is a nonlinear one
        self.disabled = self.system.input["setup.linear"]
        if self.disabled:
            flucsprint(
                "Disabled for linear simulations.",
                source=self,
                message_type="warning",
            )
            return

        # Parse valid fluxes
        valid_fluxes = ("kz", "kx", "ky", "kperp")
        fluxes = self.fluxes

        invalid_fluxes = set(fluxes) - set(valid_fluxes)
        if invalid_fluxes:
            raise ValueError(
                f"fluxes={fluxes} is invalid, "
                f"{self.name} only supports 1D fluxes {valid_fluxes}."
            )

        # Initialise dicts
        self.get_W = {}
        self.get_dWdt_nonlinear = {}
        self.get_dWdt_forcing = {}
        self.get_dWdt_hyperdissipation = {}

        self.get_H = {}
        self.get_dHdt_nonlinear = {}
        self.get_dHdt_forcing = {}
        self.get_dHdt_hyperdissipation = {}

        self.get_Wp = {}
        self.get_dWpdt_nonlinear = {}
        self.get_dWpdt_forcing = {}
        self.get_dWpdt_hyperdissipation = {}

        self.get_Wm = {}
        self.get_dWmdt_nonlinear = {}
        self.get_dWmdt_forcing = {}
        self.get_dWmdt_hyperdissipation = {}

        self.get_Hp = {}
        self.get_dHpdt_nonlinear = {}
        self.get_dHpdt_forcing = {}
        self.get_dHpdt_hyperdissipation = {}

        self.get_Hm = {}
        self.get_dHmdt_nonlinear = {}
        self.get_dHmdt_forcing = {}
        self.get_dHmdt_hyperdissipation = {}

        # Iterate over flux types and init variables
        for flux in fluxes:
            reduction_output = f"{flux}_cumulative"
            dimensions = reductions.get_dimensions(reduction_output)
            shape = tuple(dimensions)

            # Save free-energy diagnostics if required
            if self.save_free_energy:
                for name in [
                    "dWdt",
                    "dWdt_nonlinear",
                    "dWdt_forcing",
                    "dWdt_hyperdissipation",
                    "dWdt_error",
                ]:
                    self.add_var(FlucsDiagnosticVariable(
                        name=f"{flux}_fluxes/{name}",
                        shape=shape,
                        dimensions=dimensions,
                        is_complex=False,
                    ))

                # Register free-energy reductions
                self.get_W[flux] = reductions.get_reduction(
                    reduction_output=reduction_output,
                    functor="FreeEnergy_Functor",
                    input_args="FLUCS_COMPLEX*",
                    complex_output=False,
                )
                self.get_dWdt_nonlinear[flux] = reductions.get_reduction(
                    reduction_output=reduction_output,
                    functor="FreeEnergyNonlinear_Functor",
                    input_args="FLUCS_COMPLEX*,FLUCS_FLOAT,FLUCS_FLOAT,long long,const FLUCS_COMPLEX (*)[HALFPADDEDSIZE]",
                    complex_output=False,
                )
                self.get_dWdt_forcing[flux] = reductions.get_reduction(
                    reduction_output=reduction_output,
                    functor="FreeEnergyForcing_Functor",
                    input_args="const FLUCS_COMPLEX (*)[HALFUNPADDEDSIZE],FLUCS_FLOAT,FLUCS_FLOAT,long long",
                    complex_output=False,
                )
                self.get_dWdt_hyperdissipation[flux] = (
                    reductions.get_reduction(
                        reduction_output=reduction_output,
                        functor="FreeEnergyHyperdissipation_Functor",
                        input_args="FLUCS_COMPLEX*,FLUCS_FLOAT",
                        complex_output=False,
                    )
                )

                # Save Elsasser diagnostics if required
                if self.save_elsasser:
                    for name in [
                        "dWpdt",
                        "dWpdt_nonlinear",
                        "dWpdt_forcing",
                        "dWpdt_hyperdissipation",
                        "dWpdt_error",
                        "dWmdt",
                        "dWmdt_nonlinear",
                        "dWmdt_forcing",
                        "dWmdt_hyperdissipation",
                        "dWmdt_error",
                    ]:
                        self.add_var(FlucsDiagnosticVariable(
                            name=f"{flux}_fluxes/{name}",
                            shape=shape,
                            dimensions=dimensions,
                            is_complex=False,
                        ))

                    # Register Wp reductions
                    self.get_Wp[flux] = reductions.get_reduction(
                        reduction_output=reduction_output,
                        functor="FreeEnergyThetap_Functor",
                        input_args="FLUCS_COMPLEX*",
                        complex_output=False,
                    )
                    self.get_dWpdt_nonlinear[flux] = (
                        reductions.get_reduction(
                            reduction_output=reduction_output,
                            functor="FreeEnergyThetapNonlinear_Functor",
                            input_args="FLUCS_COMPLEX*,FLUCS_FLOAT,FLUCS_FLOAT,long long,const FLUCS_COMPLEX (*)[HALFPADDEDSIZE]",
                            complex_output=False,
                        )
                    )
                    self.get_dWpdt_forcing[flux] = (
                        reductions.get_reduction(
                            reduction_output=reduction_output,
                            functor="FreeEnergyThetapForcing_Functor",
                            input_args="const FLUCS_COMPLEX (*)[HALFUNPADDEDSIZE],FLUCS_FLOAT,FLUCS_FLOAT,long long",
                            complex_output=False,
                        )
                    )
                    self.get_dWpdt_hyperdissipation[flux] = (
                        reductions.get_reduction(
                            reduction_output=reduction_output,
                            functor=(
                                "FreeEnergyThetapHyperdissipation_Functor"
                            ),
                            input_args="FLUCS_COMPLEX*,FLUCS_FLOAT",
                            complex_output=False,
                        )
                    )

                    # Register Wm reductions
                    self.get_Wm[flux] = reductions.get_reduction(
                        reduction_output=reduction_output,
                        functor="FreeEnergyThetam_Functor",
                        input_args="FLUCS_COMPLEX*",
                        complex_output=False,
                    )
                    self.get_dWmdt_nonlinear[flux] = (
                        reductions.get_reduction(
                            reduction_output=reduction_output,
                            functor="FreeEnergyThetamNonlinear_Functor",
                            input_args="FLUCS_COMPLEX*,FLUCS_FLOAT,FLUCS_FLOAT,long long,const FLUCS_COMPLEX (*)[HALFPADDEDSIZE]",
                            complex_output=False,
                        )
                    )
                    self.get_dWmdt_forcing[flux] = (
                        reductions.get_reduction(
                            reduction_output=reduction_output,
                            functor="FreeEnergyThetamForcing_Functor",
                            input_args="const FLUCS_COMPLEX (*)[HALFUNPADDEDSIZE],FLUCS_FLOAT,FLUCS_FLOAT,long long",
                            complex_output=False,
                        )
                    )
                    self.get_dWmdt_hyperdissipation[flux] = (
                        reductions.get_reduction(
                            reduction_output=reduction_output,
                            functor=(
                                "FreeEnergyThetamHyperdissipation_Functor"
                            ),
                            input_args="FLUCS_COMPLEX*,FLUCS_FLOAT",
                            complex_output=False,
                        )
                    )

            # Save helicity diagnostics if required
            if self.save_helicity:
                for name in [
                    "dHdt",
                    "dHdt_nonlinear",
                    "dHdt_forcing",
                    "dHdt_hyperdissipation",
                    "dHdt_error",
                ]:
                    self.add_var(FlucsDiagnosticVariable(
                        name=f"{flux}_fluxes/{name}",
                        shape=shape,
                        dimensions=dimensions,
                        is_complex=False,
                    ))

                # Register helicity reductions
                self.get_H[flux] = reductions.get_reduction(
                    reduction_output=reduction_output,
                    functor="Helicity_Functor",
                    input_args="FLUCS_COMPLEX*",
                    complex_output=False,
                )
                self.get_dHdt_nonlinear[flux] = reductions.get_reduction(
                    reduction_output=reduction_output,
                    functor="HelicityNonlinear_Functor",
                    input_args="FLUCS_COMPLEX*,FLUCS_FLOAT,FLUCS_FLOAT,long long,const FLUCS_COMPLEX (*)[HALFPADDEDSIZE]",
                    complex_output=False,
                )
                self.get_dHdt_forcing[flux] = reductions.get_reduction(
                    reduction_output=reduction_output,
                    functor="HelicityForcing_Functor",
                    input_args="const FLUCS_COMPLEX (*)[HALFUNPADDEDSIZE],FLUCS_FLOAT,FLUCS_FLOAT,long long",
                    complex_output=False,
                )
                self.get_dHdt_hyperdissipation[flux] = (
                    reductions.get_reduction(
                        reduction_output=reduction_output,
                        functor="HelicityHyperdissipation_Functor",
                        input_args="FLUCS_COMPLEX*,FLUCS_FLOAT",
                        complex_output=False,
                    )
                )

                # Save Elsasser diagnostics if required
                if self.save_elsasser:
                    for name in [
                        "dHpdt",
                        "dHpdt_nonlinear",
                        "dHpdt_forcing",
                        "dHpdt_hyperdissipation",
                        "dHpdt_error",
                        "dHmdt",
                        "dHmdt_nonlinear",
                        "dHmdt_forcing",
                        "dHmdt_hyperdissipation",
                        "dHmdt_error",
                    ]:
                        self.add_var(FlucsDiagnosticVariable(
                            name=f"{flux}_fluxes/{name}",
                            shape=shape,
                            dimensions=dimensions,
                            is_complex=False,
                        ))

                    # Register Hp reductions
                    self.get_Hp[flux] = reductions.get_reduction(
                        reduction_output=reduction_output,
                        functor="HelicityThetap_Functor",
                        input_args="FLUCS_COMPLEX*",
                        complex_output=False,
                    )
                    self.get_dHpdt_nonlinear[flux] = (
                        reductions.get_reduction(
                            reduction_output=reduction_output,
                            functor="HelicityThetapNonlinear_Functor",
                            input_args="FLUCS_COMPLEX*,FLUCS_FLOAT,FLUCS_FLOAT,long long,const FLUCS_COMPLEX (*)[HALFPADDEDSIZE]",
                            complex_output=False,
                        )
                    )
                    self.get_dHpdt_forcing[flux] = (
                        reductions.get_reduction(
                            reduction_output=reduction_output,
                            functor="HelicityThetapForcing_Functor",
                            input_args="const FLUCS_COMPLEX (*)[HALFUNPADDEDSIZE],FLUCS_FLOAT,FLUCS_FLOAT,long long",
                            complex_output=False,
                        )
                    )
                    self.get_dHpdt_hyperdissipation[flux] = (
                        reductions.get_reduction(
                            reduction_output=reduction_output,
                            functor=(
                                "HelicityThetapHyperdissipation_Functor"
                            ),
                            input_args="FLUCS_COMPLEX*,FLUCS_FLOAT",
                            complex_output=False,
                        )
                    )

                    # Register Hm reductions
                    self.get_Hm[flux] = reductions.get_reduction(
                        reduction_output=reduction_output,
                        functor="HelicityThetam_Functor",
                        input_args="FLUCS_COMPLEX*",
                        complex_output=False,
                    )
                    self.get_dHmdt_nonlinear[flux] = (
                        reductions.get_reduction(
                            reduction_output=reduction_output,
                            functor="HelicityThetamNonlinear_Functor",
                            input_args="FLUCS_COMPLEX*,FLUCS_FLOAT,FLUCS_FLOAT,long long,const FLUCS_COMPLEX (*)[HALFPADDEDSIZE]",
                            complex_output=False,
                        )
                    )
                    self.get_dHmdt_forcing[flux] = (
                        reductions.get_reduction(
                            reduction_output=reduction_output,
                            functor="HelicityThetamForcing_Functor",
                            input_args="const FLUCS_COMPLEX (*)[HALFUNPADDEDSIZE],FLUCS_FLOAT,FLUCS_FLOAT,long long",
                            complex_output=False,
                        )
                    )
                    self.get_dHmdt_hyperdissipation[flux] = (
                        reductions.get_reduction(
                            reduction_output=reduction_output,
                            functor=(
                                "HelicityThetamHyperdissipation_Functor"
                            ),
                            input_args="FLUCS_COMPLEX*,FLUCS_FLOAT",
                            complex_output=False,
                        )
                    )

    def ready(self) -> None:
        pass

    def execute(self) -> None:
        if self.disabled:
            return
        if not self.save_free_energy and not self.save_helicity:
            return

        # Useful aliases
        current_dt = self.system.float(self.system.current_dt)
        current_time = self.system.float(self.system.current_time)
        current_step = self.system.int(self.system.current_step)
        adaptive_rate = self.system.float(self.system.adaptive_rate)

        fields = self.system.get_fields()
        fields_prev = self.system.get_fields(1)

        # Calculate nonlinear terms for the current fields
        self.system.compute_nonlinear_terms(fields)
        dft_bits = self.system.dft_bits

        # Iterate over free-energy fluxes to save
        for flux in self.get_W:

            # dWdt
            W = self.get_W[flux](fields).get()
            W_prev = self.get_W[flux](fields_prev).get()
            dWdt = (W - W_prev) / current_dt
            self.save_data(f"{flux}_fluxes/dWdt", dWdt)

            # dWdt_nonlinear
            dWdt_nonlinear = self.get_dWdt_nonlinear[flux](
                fields, current_dt, current_time, current_step, dft_bits
            ).get()
            self.save_data(
                f"{flux}_fluxes/dWdt_nonlinear",
                dWdt_nonlinear,
            )

            # dWdt_forcing
            dWdt_forcing = self.get_dWdt_forcing[flux](
                fields, current_dt, current_time, current_step
            ).get()
            self.save_data(
                f"{flux}_fluxes/dWdt_forcing",
                dWdt_forcing,
            )

            # dWdt_hyperdissipation
            result = self.get_dWdt_hyperdissipation[flux](
                fields, adaptive_rate
            )
            dWdt_hyperdissipation = -result.get()
            self.save_data(
                f"{flux}_fluxes/dWdt_hyperdissipation",
                dWdt_hyperdissipation,
            )

            # dWdt_error
            dWdt_error = (
                dWdt
                - dWdt_nonlinear
                - dWdt_forcing
                - dWdt_hyperdissipation
            )
            self.save_data(
                f"{flux}_fluxes/dWdt_error",
                dWdt_error,
            )

            # Save Elsasser diagnostics if required
            if self.save_elsasser:

                # dWpdt
                Wp = self.get_Wp[flux](fields).get()
                Wp_prev = self.get_Wp[flux](fields_prev).get()
                dWpdt = (Wp - Wp_prev) / current_dt
                self.save_data(f"{flux}_fluxes/dWpdt", dWpdt)

                # dWpdt_nonlinear
                dWpdt_nonlinear = self.get_dWpdt_nonlinear[flux](
                    fields, current_dt, current_time, current_step, dft_bits
                ).get()
                self.save_data(
                    f"{flux}_fluxes/dWpdt_nonlinear",
                    dWpdt_nonlinear,
                )

                # dWpdt_forcing
                dWpdt_forcing = self.get_dWpdt_forcing[flux](
                    fields, current_dt, current_time, current_step
                ).get()
                self.save_data(
                    f"{flux}_fluxes/dWpdt_forcing",
                    dWpdt_forcing,
                )

                # dWpdt_hyperdissipation
                result = self.get_dWpdt_hyperdissipation[flux](
                    fields, adaptive_rate
                )
                dWpdt_hyperdissipation = -result.get()
                self.save_data(
                    f"{flux}_fluxes/dWpdt_hyperdissipation",
                    dWpdt_hyperdissipation,
                )

                # dWpdt_error
                dWpdt_error = (
                    dWpdt
                    - dWpdt_nonlinear
                    - dWpdt_forcing
                    - dWpdt_hyperdissipation
                )
                self.save_data(
                    f"{flux}_fluxes/dWpdt_error",
                    dWpdt_error,
                )

                # dWmdt
                Wm = self.get_Wm[flux](fields).get()
                Wm_prev = self.get_Wm[flux](fields_prev).get()
                dWmdt = (Wm - Wm_prev) / current_dt
                self.save_data(f"{flux}_fluxes/dWmdt", dWmdt)

                # dWmdt_nonlinear
                dWmdt_nonlinear = self.get_dWmdt_nonlinear[flux](
                    fields, current_dt, current_time, current_step, dft_bits
                ).get()
                self.save_data(
                    f"{flux}_fluxes/dWmdt_nonlinear",
                    dWmdt_nonlinear,
                )

                # dWmdt_forcing
                dWmdt_forcing = self.get_dWmdt_forcing[flux](
                    fields, current_dt, current_time, current_step
                ).get()
                self.save_data(
                    f"{flux}_fluxes/dWmdt_forcing",
                    dWmdt_forcing,
                )

                # dWmdt_hyperdissipation
                result = self.get_dWmdt_hyperdissipation[flux](
                    fields, adaptive_rate
                )
                dWmdt_hyperdissipation = -result.get()
                self.save_data(
                    f"{flux}_fluxes/dWmdt_hyperdissipation",
                    dWmdt_hyperdissipation,
                )

                # dWmdt_error
                dWmdt_error = (
                    dWmdt
                    - dWmdt_nonlinear
                    - dWmdt_forcing
                    - dWmdt_hyperdissipation
                )
                self.save_data(
                    f"{flux}_fluxes/dWmdt_error",
                    dWmdt_error,
                )

        # Iterate over helicity fluxes to save
        for flux in self.get_H:

            # dHdt
            H = self.get_H[flux](fields).get()
            H_prev = self.get_H[flux](fields_prev).get()
            dHdt = (H - H_prev) / current_dt
            self.save_data(f"{flux}_fluxes/dHdt", dHdt)

            # dHdt_nonlinear
            dHdt_nonlinear = self.get_dHdt_nonlinear[flux](
                fields, current_dt, current_time, current_step, dft_bits
            ).get()
            self.save_data(
                f"{flux}_fluxes/dHdt_nonlinear",
                dHdt_nonlinear,
            )

            # dHdt_forcing
            dHdt_forcing = self.get_dHdt_forcing[flux](
                fields, current_dt, current_time, current_step
            ).get()
            self.save_data(
                f"{flux}_fluxes/dHdt_forcing",
                dHdt_forcing,
            )

            # dHdt_hyperdissipation
            result = self.get_dHdt_hyperdissipation[flux](
                fields, adaptive_rate
            )
            dHdt_hyperdissipation = -result.get()
            self.save_data(
                f"{flux}_fluxes/dHdt_hyperdissipation",
                dHdt_hyperdissipation,
            )

            # dHdt_error
            dHdt_error = (
                dHdt
                - dHdt_nonlinear
                - dHdt_forcing
                - dHdt_hyperdissipation
            )
            self.save_data(
                f"{flux}_fluxes/dHdt_error",
                dHdt_error,
            )

            # Save Elsasser diagnostics if required
            if self.save_elsasser:

                # dHpdt
                Hp = self.get_Hp[flux](fields).get()
                Hp_prev = self.get_Hp[flux](fields_prev).get()
                dHpdt = (Hp - Hp_prev) / current_dt
                self.save_data(f"{flux}_fluxes/dHpdt", dHpdt)

                # dHpdt_nonlinear
                dHpdt_nonlinear = self.get_dHpdt_nonlinear[flux](
                    fields, current_dt, current_time, current_step, dft_bits
                ).get()
                self.save_data(
                    f"{flux}_fluxes/dHpdt_nonlinear",
                    dHpdt_nonlinear,
                )

                # dHpdt_forcing
                dHpdt_forcing = self.get_dHpdt_forcing[flux](
                    fields, current_dt, current_time, current_step
                ).get()
                self.save_data(
                    f"{flux}_fluxes/dHpdt_forcing",
                    dHpdt_forcing,
                )

                # dHpdt_hyperdissipation
                result = self.get_dHpdt_hyperdissipation[flux](
                    fields, adaptive_rate
                )
                dHpdt_hyperdissipation = -result.get()
                self.save_data(
                    f"{flux}_fluxes/dHpdt_hyperdissipation",
                    dHpdt_hyperdissipation,
                )

                # dHpdt_error
                dHpdt_error = (
                    dHpdt
                    - dHpdt_nonlinear
                    - dHpdt_forcing
                    - dHpdt_hyperdissipation
                )
                self.save_data(
                    f"{flux}_fluxes/dHpdt_error",
                    dHpdt_error,
                )

                # dHmdt
                Hm = self.get_Hm[flux](fields).get()
                Hm_prev = self.get_Hm[flux](fields_prev).get()
                dHmdt = (Hm - Hm_prev) / current_dt
                self.save_data(f"{flux}_fluxes/dHmdt", dHmdt)

                # dHmdt_nonlinear
                dHmdt_nonlinear = self.get_dHmdt_nonlinear[flux](
                    fields, current_dt, current_time, current_step, dft_bits
                ).get()
                self.save_data(
                    f"{flux}_fluxes/dHmdt_nonlinear",
                    dHmdt_nonlinear,
                )

                # dHmdt_forcing
                dHmdt_forcing = self.get_dHmdt_forcing[flux](
                    fields, current_dt, current_time, current_step
                ).get()
                self.save_data(
                    f"{flux}_fluxes/dHmdt_forcing",
                    dHmdt_forcing,
                )

                # dHmdt_hyperdissipation
                result = self.get_dHmdt_hyperdissipation[flux](
                    fields, adaptive_rate
                )
                dHmdt_hyperdissipation = -result.get()
                self.save_data(
                    f"{flux}_fluxes/dHmdt_hyperdissipation",
                    dHmdt_hyperdissipation,
                )

                # dHmdt_error
                dHmdt_error = (
                    dHmdt
                    - dHmdt_nonlinear
                    - dHmdt_forcing
                    - dHmdt_hyperdissipation
                )
                self.save_data(
                    f"{flux}_fluxes/dHmdt_error",
                    dHmdt_error,
                )

class CL04Anisotropy(FlucsDiagnostic):
    """
    Computes kpar as a function of kperp using the prescription of Cho & Lazarian (2004).
    """

    name = "cl04_anisotropy"
    system: KREHMFourier
    option_defaults: ClassVar[dict[str, object]] = {
        "compute_on_gpu": True,
    }

    def init_vars(self) -> None:
        #TODO: generalise to KREHM
        if not self.system.input["parameters.ermhd"]:
            raise Exception("CL04 anisotropy not configured for isothermal KREHM, only ERMHD")
        
        self.system._compute_kperp_shells()
        dimensions = {'kperp':self.system.shell_kperp}
        shape = tuple(dimensions)
        self.add_var(FlucsDiagnosticVariable(
            name="kpar",
            shape=shape,
            dimensions=dimensions,
            is_complex=False,
        ))

    def ready(self) -> None:
        pass

    def execute(self) -> None:

        fields = self.system.fields[
            self.system.current_step % self.system.fields_history_size
        ]

        phi = fields[0]
        apar = fields[1]

        kx, ky, kz = self.system.get_broadcast_wavenumbers()
        kx = cp.asarray(kx)
        ky = cp.asarray(ky)
        kz = cp.asarray(kz)
        kvec = cp.stack([kx,ky,kz],axis=0)

        nkperp = self.system.shell_nkperp
        kperp = cp.asarray(self.system.shell_kperp)

        # Width of the kperp shells. This must match the binning used by the
        # framework's shell reductions, i.e. the half-open bins
        # [kperp[i], kperp[i] + bin_width) of _compute_kperp_shells().
        bin_width = (
            self.system.shell_kperp_max - self.system.shell_kperp_min
        ) / nkperp

        # Modes with ky > 0 stand in for their conjugate partner in the
        # half-complex representation, exactly as the CUDA shell-sum kernels
        # handle it (see cuda/reductions.cuh).
        hermitian_weight = cp.where(ky > 0, 2.0, 1.0)

        kperp_grid = cp.sqrt(kx**2 + ky**2)

        kpar = cp.zeros(nkperp)

        # delta_B_par follows from the density contribution to the free energy,
        # which is 2 * ZTe_over_Ti * |phi|^2 in the eRMHD limit (see
        # FreeEnergyDens_Functor in krehm_fourier.cu).
        deltaBz = cp.sqrt(2.0 * self.system.ZTe_over_Ti) * phi
        deltaBx = 1j * ky * apar
        deltaBy = - 1j * kx * apar

        deltaB = cp.stack([deltaBx,deltaBy,deltaBz],axis=0)
        kperp_dash = kperp_grid[None,:,:,:]


        #done as loop because otherwise too much memory required
        #TODO: use CUDA kernel for speed
        for i,kperp_single in enumerate(kperp):

            shell_mask = (kperp_grid >= kperp_single) & (
                kperp_grid < kperp_single + bin_width
            )

            deltaB_locmean = cp.where(kperp_dash < kperp_single/2,deltaB,cp.zeros_like(deltaB))
            deltaB_locmean[2] = cp.zeros_like(deltaB[0])#the z-component of mean-local field is just the guide field


            deltaB_locfluc = cp.where(kperp_dash >= kperp_single/2,deltaB,cp.zeros_like(deltaB))

            grad_deltaB_locfluc = 1j * cp.einsum('imln,jmln->ijmln',kvec,deltaB_locfluc)

            # The product below is quadratic, so it is formed on the padded
            # grid to avoid aliasing power back onto the resolved modes, in the
            # same way the solver evaluates its nonlinear terms.
            deltaB_locmean_realspace = cp.fft.irfftn(
                pad_half_complex(deltaB_locmean, self.system),
                norm="forward",
                axes=(-3,-2,-1),
                s=self.system.full_padded_tuple,
            )

            grad_deltaB_locfluc_realspace = cp.fft.irfftn(
                pad_half_complex(grad_deltaB_locfluc, self.system),
                norm="forward",
                axes=(-3,-2,-1),
                s=self.system.full_padded_tuple,
            )

            B_locmean_realspace = deltaB_locmean_realspace.copy()
            B_locmean_realspace[2] = cp.ones_like(B_locmean_realspace[2])

            nl_term_realspace = cp.einsum('ilmn,ijlmn->jlmn',B_locmean_realspace,grad_deltaB_locfluc_realspace)

            nl_term = unpad_half_complex(
                cp.fft.rfftn(
                    nl_term_realspace,
                    norm="forward",
                    axes = (-3,-2,-1),
                ),
                self.system,
            )

            nl_term_sqrd = cp.einsum(
                'ijkm,ijkm->jkm', cp.conj(nl_term), nl_term
            ).real

            deltaB_locfluc_sqrd = cp.einsum(
                'ijkm,ijkm->jkm', cp.conj(deltaB_locfluc), deltaB_locfluc
            ).real

            weight = cp.where(shell_mask, hermitian_weight, 0.0)

            numerator = cp.sum(weight * nl_term_sqrd)
            denominator = cp.sum(weight * deltaB_locfluc_sqrd)

            # An empty (or zero-power) shell is reported as NaN rather than
            # silently becoming 0/0.
            kpar[i] = cp.where(
                denominator > 0.0,
                cp.sqrt(cp.abs(numerator / cp.where(denominator > 0.0, denominator, 1.0))),
                cp.nan,
            )

        self.save_data('kpar',kpar.get())

class StructureFunctionDiag(FlucsDiagnostic):
    """
    Computes structure functions of the fields in real space.

    Uses the same formatting as the RealspaceDataDiag locations.
    """

    name = "struct_func"
    system: KREHMFourier
    option_defaults: ClassVar[dict[str, object]] = {
        "difference_locations": list(),
        "orders": [2,3,4],
        "number_points": int(1e5),
        "resample_points": False,
        "compute_on_gpu": True,
    }

    slice_calculators: list[Callable[[], None]]


    def init_vars(self) -> None:
        self.slice_calculators = []

        def parse_slice(s: str) -> slice:
            # Check if slice syntax or just a single index
            if ":" not in s:
                index = int(s.strip())
                return slice(index, index + 1)

            parts = s.split(":")
            if len(parts) > 3:
                raise ValueError(f"Invalid slice string: {s}")

            def get_index(x):
                return None if x == "" else int(x)

            return slice(*(get_index(p) for p in parts))
        
        self.izs,self.ixs,self.iys = generate_random_gridpoints_in_realspace(self.number_points,self.system)

        for location in self.difference_locations:
            for order in self.orders:
                loc = ",".join(part.strip() for part in location.split(","))
                loc_name = f"location_{loc}"
                try:
                    loc_parts = loc.split(",")

                    ifield = parse_slice(loc_parts[0])
                    ilz = parse_slice(loc_parts[1])
                    ilx = parse_slice(loc_parts[2])
                    ily = parse_slice(loc_parts[3])

                except (IndexError, ValueError):
                    raise ValueError(
                        f"'{loc}' is not a valid realspace location. "
                        r"The correct location format is '(ifield, iz, ix, iy)' "
                        r"where the indices are integers or slices 'a:b:c'."
                    )

                dimensions = {
                    f"{loc_name}/field": cp.arange(self.system.number_of_fields)[
                        ifield
                    ].get(),
                    f"{loc_name}/lz": cp.linspace(
                        0,
                        self.system.input["dimensions.Lz"],
                        self.system.nz,
                        endpoint=False,
                    )[ilz].get(),
                    f"{loc_name}/lx": cp.linspace(
                        0,
                        self.system.input["dimensions.Lx"],
                        self.system.nx,
                        endpoint=False,
                    )[ilx].get(),
                    f"{loc_name}/ly": cp.linspace(
                        0,
                        self.system.input["dimensions.Ly"],
                        self.system.ny,
                        endpoint=False,
                    )[ily].get(),
                }

                self.add_var(
                    FlucsDiagnosticVariable(
                        name=f"{loc_name}/{order}/data",
                        shape=("field", "lz", "lx", "ly"),
                        dimensions=dimensions,
                        is_complex=False,
                    )
                )


            # Every free variable used in the body must be captured by value
            # here, otherwise all the calculators end up sharing the last
            # location's slices.
            def slice_calculator(
                loc_name=loc_name,
                orders=self.orders,
                ifield=ifield,
                ilz=ilz,
                ilx=ilx,
                ily=ily,
            ):
                fields = cp.asarray(self.system.realspace_fields)
                ilz_grid = cp.arange(self.system.nz)[ilz]
                ilx_grid = cp.arange(self.system.nx)[ilx]
                ily_grid = cp.arange(self.system.ny)[ily]

                nf, nz, nx, ny = fields[ifield].shape[0],ilz_grid.size, ilx_grid.size, ily_grid.size
                accs = {p: cp.zeros((nf, nz, nx, ny)) for p in orders}
                budget = 512 * 1024**2
                bytes_per_point = nf * nz * nx * ny * fields.dtype.itemsize
                chunk = max(1, budget // (2 * bytes_per_point))

                for s in range(0, self.number_points, chunk):
                    zc, xc, yc = (
                        self.izs[s:s+chunk],
                        self.ixs[s:s+chunk],
                        self.iys[s:s+chunk],
                    )

                    diff = fields[
                        cp.arange(self.system.number_of_fields)[ifield][None, :, None, None, None],
                        ((zc[:, None] + ilz_grid)[:, None, :, None, None]) % self.system.nz,
                        ((xc[:, None] + ilx_grid)[:, None, None, :, None]) % self.system.nx,
                        ((yc[:, None] + ily_grid)[:, None, None, None, :]) % self.system.ny,
                    ]                                                          # (p, nf, nz, nx, ny)

                    centers = fields[cp.arange(self.system.number_of_fields)[ifield][None, :], zc[:, None], xc[:, None], yc[:, None]]   # (p, nf)
                    diff -= centers[:, :, None, None, None]
                    mag = cp.abs(diff)
                    for p in orders:
                        accs[p] += (mag ** p).sum(axis=0)

                for p in orders:
                    structure_fn = accs[p] / self.number_points   

                    self.vars[f"{loc_name}/{p}/data"].data_cache.append(
                        structure_fn.get()
                    )

            self.slice_calculators.append(slice_calculator)

    def ready(self) -> None:
        pass
    
    def execute(self) -> None:
        if self.resample_points:
            self.izs,self.ixs,self.iys = (
                generate_random_gridpoints_in_realspace(
                    self.number_points, self.system
                )
            )

        if self.compute_on_gpu:
            self.system.get_realspace_fields_gpu()
        else:
            self.system.get_realspace_fields_cpu()

        for slice_calculator in self.slice_calculators:
            slice_calculator()


def calculate_lperp(lx,ly,system):
    """
    Returns the grid of perpendicular separations |lperp|.

    Separations are taken in the minimum-image convention, so the largest
    separation resolved by a periodic box is min(Lx, Ly) / 2. Beyond that only
    the corners of the box contribute and the shells are no longer sampled
    isotropically.
    """
    dlperp = min(
        (l[1] for l in (lx,ly)),
    )
    lperp_max = 0.5 * min(
        system.input["dimensions.Lx"],
        system.input["dimensions.Ly"],
    )
    nlperp = int(cp.rint(lperp_max / dlperp).item()) + 1
    return dlperp * cp.arange(nlperp,dtype=system.float)

def calculate_lperp_bins(lx,ly,lperp,system,name=""):
    """
    Bins the perpendicular separation vectors of the grid into |lperp| shells.

    The separations are taken in the minimum-image convention: an offset of lx
    close to Lx is really a separation of lx - Lx. Separations that fall
    outside the lperp grid (the corners of the box) are dropped, and, since the
    structure functions are even in the separation vector, only the ly >= 0
    half of the remainder is kept.

    Returns
    -------
    ilx_v, ily_v : cp.ndarray
        Grid-index offsets of the retained separation vectors.
    bins : cp.ndarray
        Index of the lperp shell each retained separation vector belongs to.
    counts : cp.ndarray
        Number of retained separation vectors in each lperp shell.
    """
    Lx = system.input["dimensions.Lx"]
    Ly = system.input["dimensions.Ly"]

    # Minimum-image separations
    dx = lx - Lx * cp.rint(lx / Lx)
    dy = ly - Ly * cp.rint(ly / Ly)

    dx_grid, dy_grid = cp.meshgrid(dx, dy, indexing='ij')

    R = cp.sqrt(dx_grid**2 + dy_grid**2)
    dlperp = lperp[1] - lperp[0]
    idx = cp.rint(R / dlperp).astype(cp.int64)

    valid = (idx < lperp.size) & (dy_grid >= 0)
    bins = idx[valid]

    # Number of separation offsets in each shell; there should be no empty
    # shells.
    counts = cp.bincount(bins, minlength=lperp.size)
    if bool((counts == 0).any()):
        raise ValueError(
            f"{name}: empty lperp shell(s) encountered."
        )

    ilx,ily = cp.meshgrid(
        cp.arange(system.nx),
        cp.arange(system.ny),
        indexing='ij'
    )

    return ilx[valid], ily[valid], bins, counts

def generate_random_gridpoints_in_realspace(number_points,system):
    rng = cp.random.default_rng()
    izs = rng.integers(0,system.nz,size=number_points)
    ixs = rng.integers(0,system.nx,size=number_points)
    iys = rng.integers(0,system.ny,size=number_points)
    return izs, ixs, iys

class StructureFunctionDiag1D(FlucsDiagnostic):
    """
    Computes structure functions of the fields in real space in 1D.

    The only new functionality from StructureFunctionDiag is the ability to calculate structure functions in the perpendicular direction.
    """

    name = "struct_func_1d"
    system: KREHMFourier
    option_defaults: ClassVar[dict[str, object]] = {
        "directions": ["lperp"],
        "fields": [0,1],
        "orders": [2,3,4],
        "number_points": int(1e5),
        "resample_points": False,
        "compute_on_gpu": True,
    }

    slice_calculators: list[Callable[[], None]]


    def init_vars(self) -> None:

        valid_directions = ("lz", "lperp")#TODO: add lpar,ly,lz functionality
        directions = self.directions
        if isinstance(directions, str):
            directions = [directions]
        directions = tuple(dict.fromkeys(directions))

        invalid_directions = set(directions) - set(valid_directions)
        if invalid_directions:
            raise ValueError(
                f"{self.name} only supports 1D spectra {valid_directions}. ly and lx are yet to be added."
            )

        self.directions = directions

        self.izs,self.ixs,self.iys = generate_random_gridpoints_in_realspace(self.number_points,self.system)


        for direction in self.directions:

            match direction:
                case 'lz':
                    lz = cp.linspace(
                        0,
                        self.system.input["dimensions.Lz"],
                        self.system.nz,
                        endpoint=False,
                    )
                    dimensions = {'lz': lz.get()}
                    shape = tuple(dimensions)

                case 'lperp':
                    self.lx = cp.linspace(
                        0,
                        self.system.input["dimensions.Lx"],
                        self.system.nx,
                        endpoint=False,
                    )
                    self.ly = cp.linspace(
                        0,
                        self.system.input["dimensions.Ly"],
                        self.system.ny,
                        endpoint=False,
                    )
                    self.lperp = calculate_lperp(self.lx,self.ly,self.system)
                    dimensions = {'lperp': self.lperp.get()}
                    shape = tuple(dimensions)

            for order in self.orders:
                for field in self.fields:
                    self.add_var(
                        FlucsDiagnosticVariable(
                            name=f"{field}/{direction}/{order}",
                            shape=shape,
                            dimensions=dimensions,
                            is_complex=False,
                        )
                    )

    def ready(self) -> None:
        pass
    
    def execute(self) -> None:
        if self.resample_points:
            self.izs,self.ixs,self.iys = (
                generate_random_gridpoints_in_realspace(
                    self.number_points, self.system
                )
            )

        if self.compute_on_gpu:
            self.system.get_realspace_fields_gpu()
        else:
            self.system.get_realspace_fields_cpu()

        for direction in self.directions:
            for field in self.fields:
                field_realspace = cp.asarray(self.system.realspace_fields[field])

                match direction:
                    case "lz":
                        ilz = cp.arange(self.system.nz)
                        diff = field_realspace[
                            (self.izs[:,None] +ilz[None,:]) % self.system.nz,
                            self.ixs[:,None],
                            self.iys[:,None]
                        ]

                        centers = field_realspace[
                            self.izs[:,None],
                            self.ixs[:,None],
                            self.iys[:,None]
                        ]

                        diff -= centers
                        mag = cp.abs(diff)
                        for p in self.orders:
                            acc = (mag**p).sum(axis=0) / self.number_points
                            self.save_data(
                                f"{field}/{direction}/{p}",
                                acc.get()
                            )

                    case "lperp":

                        ilx_v, ily_v, bins, counts = calculate_lperp_bins(
                            self.lx,
                            self.ly,
                            self.lperp,
                            self.system,
                            name=self.name,
                        )

                        acc = {p: cp.zeros(ilx_v.size, dtype=cp.float64) for p in self.orders}

                        pow_buf = None

                        for ipt in range(self.number_points):
                            iz = self.izs[ipt]
                            ix = self.ixs[ipt]
                            iy = self.iys[ipt]

                            diff = field_realspace[
                                iz,
                                (ix + ilx_v) % self.system.nx,
                                (iy + ily_v) % self.system.ny,
                            ]
                            center = field_realspace[iz,ix,iy]
                            diff -= center

                            mag = cp.abs(diff)

                            if pow_buf is None or pow_buf.shape != mag.shape:
                                pow_buf = cp.empty_like(mag)

                            for p in self.orders:
                                cp.power(mag,p,out=pow_buf)
                                acc[p] += pow_buf

                        for p in self.orders:
                            sums = cp.bincount(bins,weights=acc[p],minlength = self.lperp.size)
                            self.save_data(f"{field}/{direction}/{p}", (sums / (counts * self.number_points)).get())


class AlignmentDiag(FlucsDiagnostic):
    """
    Computes alignment angle quantities, for sin(theta) = <|delta z_1 x delta z_2|>/ <|delta z1||delta z2|>, where z1 and z2 are fields.
    The numerator and denominator are saved separately (so they can be time averaged in post).
    """
    name = "alignment_angle"
    system: KREHMFourier
    option_defaults: ClassVar[dict[str, object]] = {
        "directions": ["lperp"],
        "angle_between": ["zp_zm","gradBpar_Bperp","Bperp_gradLaplacianBperp"],
        "number_points": int(1e5),
        "resample_points": False,
    }

    def init_vars(self):
        valid_directions = ("lz", "lperp")#TODO: add lpar,ly,lz functionality
        directions = self.directions
        if isinstance(directions, str):
            directions = [directions]
        directions = tuple(dict.fromkeys(directions))

        invalid_directions = set(directions) - set(valid_directions)
        if invalid_directions:
            raise ValueError(
                f"{self.name} only supports 1D spectra {valid_directions}. ly and lx are yet to be added."
            )
        
        valid_angle_between = ("zp_zm","gradBpar_Bperp","Bperp_gradLaplacianBperp")
        angle_between = self.angle_between
        if isinstance(angle_between, str):
            angle_between = [angle_between]
        angle_between = tuple(dict.fromkeys(angle_between))

        invalid_angle_between = set(angle_between) - set(valid_angle_between)
        if invalid_angle_between:
            raise ValueError(
                f"{self.name} only supports the following field options {valid_angle_between}."
            )

        self.directions = directions
        self.angle_between = angle_between

        self.izs,self.ixs,self.iys = generate_random_gridpoints_in_realspace(self.number_points,self.system)


        for direction in self.directions:
            match direction:
                case 'lz':
                    lz = cp.linspace(
                        0,
                        self.system.input["dimensions.Lz"],
                        self.system.nz,
                        endpoint=False,
                    )

                    dimensions = {'lz': lz.get()}
                    shape = tuple(dimensions)
                    for angle_between in self.angle_between:
                        self.add_var(
                            FlucsDiagnosticVariable(
                                name=f"{angle_between}/{direction}/numerator",
                                shape=shape,
                                dimensions=dimensions,
                                is_complex=False,
                            )
                        )
                        self.add_var(
                            FlucsDiagnosticVariable(
                                name=f"{angle_between}/{direction}/denominator",
                                shape=shape,
                                dimensions=dimensions,
                                is_complex=False,
                            )
                        )
                case 'lperp':
                    self.lx = cp.linspace(
                        0,
                        self.system.input["dimensions.Lx"],
                        self.system.nx,
                        endpoint=False,
                    )
                    self.ly = cp.linspace(
                        0,
                        self.system.input["dimensions.Ly"],
                        self.system.ny,
                        endpoint=False,
                    )
                    
                    self.lperp = calculate_lperp(self.lx,self.ly,self.system)

                    dimensions = {'lperp': self.lperp.get()}
                    shape = tuple(dimensions)
                    for angle_between in self.angle_between:
                        self.add_var(
                            FlucsDiagnosticVariable(
                                name=f"{angle_between}/{direction}/numerator",
                                shape=shape,
                                dimensions=dimensions,
                                is_complex=False,
                            )
                        )
                        self.add_var(
                            FlucsDiagnosticVariable(
                                name=f"{angle_between}/{direction}/denominator",
                                shape=shape,
                                dimensions=dimensions,
                                is_complex=False,
                            )
                        )

    
    def ready(self):
        pass

    def get_relevant_fields(self,angle_between):
        fields = self.system.fields[
            self.system.current_step % self.system.fields_history_size
        ]
        kx, ky, kz = self.system.get_broadcast_wavenumbers()
        kx = cp.asarray(kx)
        ky = cp.asarray(ky)
        kz = cp.asarray(kz)
        match angle_between:
            case 'zp_zm':
                thetap,thetam = self.system.compute_thetas_from_fields(
                    fields[0].get(),
                    fields[1].get()
                )
                thetap = cp.asarray(thetap)
                thetam = cp.asarray(thetam)

                field1 = cp.stack([-1j*ky*thetap,1j*kx*thetap,cp.zeros_like(thetap)],axis=0)
                field2 = cp.stack([-1j*ky*thetam,1j*kx*thetam,cp.zeros_like(thetam)],axis=0)

            case 'gradBpar_Bperp':
                if not self.system.input["parameters.ermhd"]:
                    raise Exception("not configured for isothermal KREHM, only ERMHD")
                
                phi = fields[0]
                apar = fields[1]

                # See the comment in CL04Anisotropy.execute() for the
                # delta_B_par normalisation.
                deltaBz = cp.sqrt(2.0 * self.system.ZTe_over_Ti) * phi
                deltaBx = 1j * ky * apar
                deltaBy = - 1j * kx * apar

                field1 = 1j * cp.stack([kx * deltaBz,ky * deltaBz,cp.zeros_like(deltaBz)],axis=0)

                field2 = cp.stack([deltaBx,deltaBy,cp.zeros_like(deltaBx)],axis=0)
            case 'Bperp_gradLaplacianBperp':
                if not self.system.input["parameters.ermhd"]:
                    raise Exception("not configured for isothermal KREHM, only ERMHD")
                apar = fields[1]
                deltaBx = 1j * ky * apar
                deltaBy = - 1j * kx * apar
                laplacian_apar = - (kx**2 + ky**2) * apar
                field1 = cp.stack([deltaBx,deltaBy,cp.zeros_like(deltaBx)],axis=0)

                field2 = 1j * cp.stack([kx * laplacian_apar, ky * laplacian_apar,cp.zeros_like(laplacian_apar)],axis=0)

        field1_realspace = cp.fft.irfftn(
            field1,
            norm="forward",
            axes=(1, 2, 3),
            s=self.system.full_unpadded_tuple,
        )
        field2_realspace = cp.fft.irfftn(
            field2,
            norm="forward",
            axes=(1, 2, 3),
            s=self.system.full_unpadded_tuple,
        )

        return field1_realspace, field2_realspace
        
                


    
    def execute(self):
        if self.resample_points:
            self.izs,self.ixs,self.iys = (
                generate_random_gridpoints_in_realspace(
                    self.number_points, self.system
                )
            )

        for angle_between in self.angle_between:
            vec_field1,vec_field2 = self.get_relevant_fields(angle_between) #fields should have shape (3,nx,ny,nz) 
            for direction in self.directions:
                match direction:
                    case 'lz':
                        ilz = cp.arange(self.system.nz)

                        diff1 = vec_field1[
                            :,
                            (self.izs[:,None] +ilz[None,:]) % self.system.nz,
                            self.ixs[:,None],
                            self.iys[:,None]
                        ]
                        diff1 -= vec_field1[
                            :,
                            self.izs[:,None],
                            self.ixs[:,None],
                            self.iys[:,None]
                        ]

                        diff2 = vec_field2[
                            :,
                            (self.izs[:,None] +ilz[None,:]) % self.system.nz,
                            self.ixs[:,None],
                            self.iys[:,None]
                        ]
                        diff2 -= vec_field2[
                            :,
                            self.izs[:,None],
                            self.ixs[:,None],
                            self.iys[:,None]
                        ]

                        cross_product = cp.cross(diff1, diff2, axis=0)
                        cross_product_mag = cp.sqrt(
                            cp.einsum('ijk,ijk->jk', cross_product, cross_product)
                        )
                        numerator = cross_product_mag.sum(axis=0) / self.number_points

                        diff1_mag = cp.sqrt(cp.einsum('ijk,ijk->jk', diff1, diff1))
                        diff2_mag = cp.sqrt(cp.einsum('ijk,ijk->jk', diff2, diff2))
                        denominator = (
                            (diff1_mag * diff2_mag).sum(axis=0) / self.number_points
                        )

                    case 'lperp':
                        ilx_v, ily_v, bins, counts = calculate_lperp_bins(
                            self.lx,
                            self.ly,
                            self.lperp,
                            self.system,
                            name=self.name,
                        )

                        num_acc = cp.zeros(ilx_v.size, dtype=cp.float64)
                        den_acc = cp.zeros(ilx_v.size, dtype=cp.float64)

                        for ipt in range(self.number_points):
                            iz = self.izs[ipt]
                            ix = self.ixs[ipt]
                            iy = self.iys[ipt]

                            diff1 = vec_field1[
                                :,
                                iz,
                                (ix + ilx_v) % self.system.nx,
                                (iy + ily_v) % self.system.ny,
                            ]
                            diff1 -= vec_field1[:, iz, ix, iy][:, None]

                            diff2 = vec_field2[
                                :,
                                iz,
                                (ix + ilx_v) % self.system.nx,
                                (iy + ily_v) % self.system.ny,
                            ]
                            diff2 -= vec_field2[:, iz, ix, iy][:, None]

                            # diff1, diff2 have shape (3, ilx_v.size)
                            cross_product = cp.cross(diff1, diff2, axis=0)
                            num_acc += cp.sqrt(
                                cp.einsum('ij,ij->j', cross_product, cross_product)
                            )

                            diff1_mag = cp.sqrt(cp.einsum('ij,ij->j', diff1, diff1))
                            diff2_mag = cp.sqrt(cp.einsum('ij,ij->j', diff2, diff2))
                            den_acc += diff1_mag * diff2_mag


                        norm = counts * self.number_points
                        numerator = cp.bincount(
                            bins, weights=num_acc, minlength=self.lperp.size
                        ) / norm
                        denominator = cp.bincount(
                            bins, weights=den_acc, minlength=self.lperp.size
                        ) / norm

                self.save_data(
                    f"{angle_between}/{direction}/numerator", numerator.get()
                )
                self.save_data(
                    f"{angle_between}/{direction}/denominator", denominator.get()
                )



