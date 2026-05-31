from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from collections.abc import Callable

import cupy as cp

from flucs.diagnostic import FlucsDiagnostic, FlucsDiagnosticVariable
from flucs.solvers.fourier.fourier_system_reductions import reduce_unpadded_to_scalar

if TYPE_CHECKING:
    from flucs_krehm.krehm_fourier.krehm_fourier import KREHMFourier

class FreeEnergyDiag(FlucsDiagnostic):
    """
    Computes quantities related to the free energy, including contributions to
    the free-energy budget from injection and dissipation
    """

    name = "free_energy"
    system: KREHMFourier
    option_defaults: ClassVar[dict[str, object]] = {
        "save_elsasser": False
    }

    get_W: Callable[..., cp.ndarray]
    get_dWdt_forcing: Callable[..., cp.ndarray]
    get_dWdt_hyperdissipation: Callable[..., cp.ndarray]

    get_Wp: Callable[..., cp.ndarray]
    get_dWpdt_hyperdissipation: Callable[..., cp.ndarray]

    get_Wm: Callable[..., cp.ndarray]
    get_dWmdt_hyperdissipation: Callable[..., cp.ndarray]

    def init_vars(self) -> None:
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
        self.get_W = reduce_unpadded_to_scalar(
            self.system,
            functor="FreeEnergy_Functor",
            input_args="FLUCS_COMPLEX*",
            complex_output=False,
        )
        self.get_dWdt_forcing = reduce_unpadded_to_scalar(
            self.system,
            functor="FreeEnergyForcing_Functor",
            input_args="FLUCS_COMPLEX*",
            complex_output=False,
        )
        self.get_dWdt_hyperdissipation = reduce_unpadded_to_scalar(
            self.system,
            functor="FreeEnergyHyperdissipation_Functor",
            input_args="FLUCS_COMPLEX*,FLUCS_FLOAT,int",
            complex_output=False,
        )

        # Add elsasser diagnostics
        if self.save_elsasser:

            for name in ["Wp", "Wm", "dWpdt", "dWmdt"]:
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
            self.get_Wp = reduce_unpadded_to_scalar(
                self.system,
                functor="FreeEnergyThetap_Functor",
                input_args="FLUCS_COMPLEX*",
                complex_output=False,
            )
            self.get_dWpdt_hyperdissipation = reduce_unpadded_to_scalar(
                self.system,
                functor="FreeEnergyThetapHyperdissipation_Functor",
                input_args="FLUCS_COMPLEX*,FLUCS_FLOAT,int",
                complex_output=False,
            )
            self.get_Wm = reduce_unpadded_to_scalar(
                self.system,
                functor="FreeEnergyThetam_Functor",
                input_args="FLUCS_COMPLEX*",
                complex_output=False,
            )
            self.get_dWmdt_hyperdissipation = reduce_unpadded_to_scalar(
                self.system,
                functor="FreeEnergyThetamHyperdissipation_Functor",
                input_args="FLUCS_COMPLEX*,FLUCS_FLOAT,int",
                complex_output=False,
            )

    def ready(self) -> None:
        pass

    def execute(self) -> None:
        # Useful aliases
        current_dt = self.system.float(self.system.current_dt)
        adaptive_rate = self.system.float(self.system.adaptive_rate)

        fields = self.system.fields[
            self.system.current_step % self.system.fields_history_size
        ]
        fields_prev = self.system.fields[
            (self.system.current_step - 1) % self.system.fields_history_size
        ]

        # W
        W = self.get_W(fields).get().item()
        self.save_data("W", W)

        # dWdt_forcing
        dWdt_forcing = self.get_dWdt_forcing(fields).get().item()
        self.save_data("dWdt_forcing", dWdt_forcing)

        # dWdt
        W_prev = self.get_W(fields_prev)
        dWdt = (W - W_prev.get().item()) / current_dt
        self.save_data("dWdt", dWdt)

        # dWdt_hyperdissipation
        dWdt_hyperdissipation_total = 0.0
        for index, component in enumerate(self.system.hyperdissipation_components):
            result = self.get_dWdt_hyperdissipation(
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

        # Saving elsasser fields if required
        if self.save_elsasser:

            # Wp
            Wp = self.get_Wp(fields).get().item()
            self.save_data("Wp", Wp)

            # dWpdt
            Wp_prev = self.get_Wp(fields_prev)
            dWpdt = (Wp - Wp_prev.get().item()) / current_dt
            self.save_data("dWpdt", dWpdt)

            # dWpdt_hyperdissipation
            for index, component in enumerate(self.system.hyperdissipation_components):
                result = self.get_dWpdt_hyperdissipation(
                    fields, adaptive_rate, index
                )
                self.save_data(
                    f"dWpdt_hyperdissipation_{component}",
                    -result.get().item(),
                )

            # Wm
            Wm = self.get_Wm(fields).get().item()
            self.save_data("Wm", Wm)

            # dWmdt
            Wm_prev = self.get_Wm(fields_prev)
            dWmdt = (Wm - Wm_prev.get().item()) / current_dt
            self.save_data("dWmdt", dWmdt)

            # dWmdt_hyperdissipation
            for index, component in enumerate(self.system.hyperdissipation_components):
                result = self.get_dWmdt_hyperdissipation(
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
        "save_elsasser": False
    }

    get_H: Callable[..., cp.ndarray]
    get_dHdt_forcing: Callable[..., cp.ndarray]
    get_dHdt_hyperdissipation: Callable[..., cp.ndarray]

    get_Hp: Callable[..., cp.ndarray]
    get_dHpdt_hyperdissipation: Callable[..., cp.ndarray]

    get_Hm: Callable[..., cp.ndarray]
    get_dHmdt_hyperdissipation: Callable[..., cp.ndarray]

    def init_vars(self) -> None:
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
        self.get_H = reduce_unpadded_to_scalar(
            self.system,
            functor="Helicity_Functor",
            input_args="FLUCS_COMPLEX*",
            complex_output=False,
        )
        self.get_dHdt_forcing = reduce_unpadded_to_scalar(
            self.system,
            functor="HelicityForcing_Functor",
            input_args="FLUCS_COMPLEX*",
            complex_output=False,
        )
        self.get_dHdt_hyperdissipation = reduce_unpadded_to_scalar(
            self.system,
            functor="HelicityHyperdissipation_Functor",
            input_args="FLUCS_COMPLEX*,FLUCS_FLOAT,int",
            complex_output=False,
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
                name="dHmdt",
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
            self.get_Hp = reduce_unpadded_to_scalar(
                self.system,
                functor="HelicityThetap_Functor",
                input_args="FLUCS_COMPLEX*",
                complex_output=False,
            )
            self.get_dHpdt_hyperdissipation = reduce_unpadded_to_scalar(
                self.system,
                functor="HelicityThetapHyperdissipation_Functor",
                input_args="FLUCS_COMPLEX*,FLUCS_FLOAT,int",
                complex_output=False,
            )
            self.get_Hm = reduce_unpadded_to_scalar(
                self.system,
                functor="HelicityThetam_Functor",
                input_args="FLUCS_COMPLEX*",
                complex_output=False,
            )
            self.get_dHmdt_hyperdissipation = reduce_unpadded_to_scalar(
                self.system,
                functor="HelicityThetamHyperdissipation_Functor",
                input_args="FLUCS_COMPLEX*,FLUCS_FLOAT,int",
                complex_output=False,
            )

    def ready(self) -> None:
        pass

    def execute(self) -> None:
        # Useful aliases
        current_dt = self.system.float(self.system.current_dt)
        adaptive_rate = self.system.float(self.system.adaptive_rate)

        fields = self.system.fields[
            self.system.current_step % self.system.fields_history_size
        ]
        fields_prev = self.system.fields[
            (self.system.current_step - 1) % self.system.fields_history_size
        ]

        # H
        H = self.get_H(fields).get().item()
        self.save_data("H", H)

        # dHdt_forcing
        dHdt_forcing = self.get_dHdt_forcing(fields).get().item()
        self.save_data("dHdt_forcing", dHdt_forcing)

        # dHdt
        H_prev = self.get_H(fields_prev)
        dHdt = (H - H_prev.get().item()) / current_dt
        self.save_data("dHdt", dHdt)

        # dHdt_hyperdissipation
        dHdt_hyperdissipation_total = 0.0
        for index, component in enumerate(self.system.hyperdissipation_components):
            result = self.get_dHdt_hyperdissipation(
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

        # Saving elsasser fields if required
        if self.save_elsasser:

            # Hp
            Hp = self.get_Hp(fields).get().item()
            self.save_data("Hp", Hp)

            # dHpdt
            Hp_prev = self.get_Hp(fields_prev)
            dHpdt = (Hp - Hp_prev.get().item()) / current_dt
            self.save_data("dHpdt",dHpdt)

            # dHpdt_hyperdissipation
            for index, component in enumerate(self.system.hyperdissipation_components):
                result = self.get_dHpdt_hyperdissipation(
                    fields, adaptive_rate, index
                )
                self.save_data(
                    f"dHpdt_hyperdissipation_{component}",
                    -result.get().item(),
                )

            # Hm
            Hm = self.get_Hm(fields).get().item()
            self.save_data("Hm", Hm)

            # dHmdt
            Hm_prev = self.get_Hm(fields_prev)
            dHmdt = (Hm - Hm_prev.get().item()) / current_dt
            self.save_data("dHmdt",dHmdt)

            # dHmdt_hyperdissipation
            for index, component in enumerate(self.system.hyperdissipation_components):
                result = self.get_dHmdt_hyperdissipation(
                    fields, adaptive_rate, index
                )
                self.save_data(
                    f"dHmdt_hyperdissipation_{component}",
                    -result.get().item(),
                )
