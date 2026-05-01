from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import cupy as cp

from flucs.diagnostic import FlucsDiagnostic, FlucsDiagnosticVariable

if TYPE_CHECKING:
    from flucs_krehm.krehm_fourier.krehm_fourier import KREHMFourier
#TODO remove these when coding optimisation wrapper
BLOCK_SIZE = int(256)
THREADS_PER_WARP = int(32) 

class FreeEnergyDiag(FlucsDiagnostic):
    """
    Computes quantities related to the free energy, including contributions to
    the free-energy budget from injection and dissipation
    """

    name = "free_energy"
    system: KREHMFourier
    option_defaults: ClassVar[dict[str, object]] = {
        "save_thetas": False
    }

    # Temporary arrays and kernels
    temp_zx: cp.ndarray
    temp_z: cp.ndarray
    result: cp.ndarray

    # General reduction kernels
    real_last_axis_sum_nx_kernel: cp.RawKernel
    real_last_axis_sum_nz_kernel: cp.RawKernel

    # Free energy kernels
    free_energy_kzkx_kernel: cp.RawKernel
    dWdt_kzkx_kernel: cp.RawKernel
    dWdt_hyperdissipation_magnitude_kernels: dict[str, cp.RawKernel]

    hyperdissipation_components: ClassVar[tuple[str, ...]] = (
        "perp", "kx", "ky", "kz"
    )

    def init_vars(self) -> None:
        # Add variables for free energy and its time derivative
        for name in ["W", "dWdt", "dWdt_error"]:
             self.add_var(FlucsDiagnosticVariable(
                name=name,
                shape=(),
                dimensions={},
                is_complex=False
            ))

        # Add variables for hyperdissipation components
        for component in self.hyperdissipation_components:
            self.add_var(FlucsDiagnosticVariable(
                name=f"dWdt_hyperdissipation_{component}",
                shape=(),
                dimensions={},
                is_complex=False
            ))

        # Add theta diagnostics
        if self.save_thetas:

            for name in ["Wp", "Wm", "dWpdt", "dWmdt"]:
                 self.add_var(FlucsDiagnosticVariable(
                    name=name,
                    shape=(),
                    dimensions={},
                    is_complex=False
                ))

            for component in self.hyperdissipation_components:
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


    def ready(self) -> None:
        # Allocate temporary memory
        self.temp_zx = cp.zeros(
            self.system.nz * self.system.nx, dtype=self.system.float
        )
        self.temp_z = cp.zeros(self.system.nz, dtype=self.system.float)
        self.result = cp.zeros((1,), dtype=self.system.float)

        # General reduction kernels
        self.real_last_axis_sum_nx_kernel = (
            self.system.cupy_module.get_function("real_last_axis_sum_nx")
        )
        self.real_last_axis_sum_nz_kernel = (
            self.system.cupy_module.get_function("real_last_axis_sum_nz")
        )

        # Free energy kernels
        self.free_energy_kzkx_kernel = (
            self.system.cupy_module.get_function("free_energy_kzkx")
        )
        self.dWdt_kzkx_kernel = (
            self.system.cupy_module.get_function("dWdt_kzkx")
        )

        self.dWdt_hyperdissipation_magnitude_kernels = {
            component: self.system.cupy_module.get_function(
                f"dWdt_hyperdissipation_{component}_kzkx"
            )
            for component in self.hyperdissipation_components
        }

    def execute(self) -> None:
        current_dt = self.system.float(self.system.current_dt)

        # Free energy W
        fields = self.system.fields[
            self.system.current_step % self.system.fields_history_size
        ]
        fields_prev = self.system.fields[
            (self.system.current_step - 1) % self.system.fields_history_size
        ]

        self.free_energy_kzkx_kernel(
                (self.system.nx * self.system.nz,),
                (BLOCK_SIZE,),
                (fields, self.temp_zx),
                shared_mem=THREADS_PER_WARP * self.system.float().nbytes)

        self.real_last_axis_sum_nx_kernel(
                (self.system.nz,),
                (BLOCK_SIZE,),
                (self.temp_zx, self.temp_z),
                shared_mem=THREADS_PER_WARP * self.system.float().nbytes)

        self.real_last_axis_sum_nz_kernel(
                (1,),
                (BLOCK_SIZE,),
                (self.temp_z, self.result),
                shared_mem=THREADS_PER_WARP * self.system.float().nbytes)

        self.save_data("W", self.result.get().item())

        # dW/dt
        self.dWdt_kzkx_kernel(
                (self.system.nx * self.system.nz,),
                (BLOCK_SIZE,),
                (fields, fields_prev, current_dt, self.temp_zx),
                shared_mem=THREADS_PER_WARP * self.system.float().nbytes)

        self.real_last_axis_sum_nx_kernel(
                (self.system.nz,),
                (BLOCK_SIZE,),
                (self.temp_zx, self.temp_z),
                shared_mem=THREADS_PER_WARP * self.system.float().nbytes)

        self.real_last_axis_sum_nz_kernel(
                (1,),
                (BLOCK_SIZE,),
                (self.temp_z, self.result),
                shared_mem=THREADS_PER_WARP * self.system.float().nbytes)

        dWdt = self.result.get().item()
        self.save_data("dWdt", dWdt)

        # Hyperdissipation
        dWdt_hyperdissipation_total = 0.0
        for component, kernel in self.dWdt_hyperdissipation_magnitude_kernels.items():
            kernel(
                (self.system.nx * self.system.nz,),
                (BLOCK_SIZE,),
                (fields, current_dt, self.temp_zx),
                shared_mem=THREADS_PER_WARP * self.system.float().nbytes)

            self.real_last_axis_sum_nx_kernel(
                (self.system.nz,),
                (BLOCK_SIZE,),
                (self.temp_zx, self.temp_z),
                shared_mem=THREADS_PER_WARP * self.system.float().nbytes)

            self.real_last_axis_sum_nz_kernel(
                    (1,),
                    (BLOCK_SIZE,),
                    (self.temp_z, self.result),
                    shared_mem=THREADS_PER_WARP * self.system.float().nbytes)

            dWdt_hyperdissipation_component = -self.result.get().item()

            self.save_data(
                f"dWdt_hyperdissipation_{component}",
                dWdt_hyperdissipation_component
            )
            dWdt_hyperdissipation_total += dWdt_hyperdissipation_component

        # Error in free-energy budget
        self.save_data("dWdt_error", dWdt - dWdt_hyperdissipation_total)


class HelicityDiag(FlucsDiagnostic):
    """
    Computes quantities related to the helicity, including contributions to the
    helicity budget from injection and dissipation
    """

    name = "helicity"
    system: KREHMFourier
    option_defaults: ClassVar[dict[str, object]] = {
        "save_thetas": False
    }

    # Temporary arrays and kernels
    temp_zx: cp.ndarray
    temp_z: cp.ndarray
    result: cp.ndarray

    # General reduction kernels
    real_last_axis_sum_nx_kernel: cp.RawKernel
    real_last_axis_sum_nz_kernel: cp.RawKernel

    # Helicity kernels
    helicity_kzkx_kernel: cp.RawKernel
    dHdt_kzkx_kernel: cp.RawKernel
    dHdt_hyperdissipation_magnitude_kernels: dict[str, cp.RawKernel]

    hyperdissipation_components: ClassVar[tuple[str, ...]] = (
        "perp", "kx", "ky", "kz"
    )

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
            name="dHdt_error",
            shape=(),
            dimensions={},
            is_complex=False
        ))

        # Add variables for hyperdissipation components
        for component in self.hyperdissipation_components:
            self.add_var(FlucsDiagnosticVariable(
                name=f"dHdt_hyperdissipation_{component}",
                shape=(),
                dimensions={},
                is_complex=False
            ))

        # Add theta diagnostics
        if self.save_thetas:
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

            for component in self.hyperdissipation_components:
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

    def ready(self) -> None:
        # Allocate temporary memory
        self.temp_zx = cp.zeros(
            self.system.nz * self.system.nx, dtype=self.system.float
        )
        self.temp_z = cp.zeros(self.system.nz, dtype=self.system.float)
        self.result = cp.zeros((1,), dtype=self.system.float)

        # General reduction kernels
        self.real_last_axis_sum_nx_kernel = (
            self.system.cupy_module.get_function("real_last_axis_sum_nx")
        )
        self.real_last_axis_sum_nz_kernel = (
            self.system.cupy_module.get_function("real_last_axis_sum_nz")
        )

        # Helicity kernels
        self.helicity_kzkx_kernel = (
            self.system.cupy_module.get_function("helicity_kzkx")
        )
        self.dHdt_kzkx_kernel = (
            self.system.cupy_module.get_function("dHdt_kzkx")
        )

        self.dHdt_hyperdissipation_magnitude_kernels = {
            component: self.system.cupy_module.get_function(
                f"dHdt_hyperdissipation_{component}_kzkx"
            )
            for component in self.hyperdissipation_components
        }

    def execute(self) -> None:
        current_dt = self.system.float(self.system.current_dt)

        fields = self.system.fields[
            self.system.current_step % self.system.fields_history_size
        ]
        fields_prev = self.system.fields[
            (self.system.current_step - 1) % self.system.fields_history_size
        ]

        # Helicity H
        self.helicity_kzkx_kernel(
            (self.system.nx * self.system.nz,),
            (BLOCK_SIZE,),
            (fields, self.temp_zx),
            shared_mem=THREADS_PER_WARP * self.system.float().nbytes
        )

        self.real_last_axis_sum_nx_kernel(
            (self.system.nz,),
            (BLOCK_SIZE,),
            (self.temp_zx, self.temp_z),
            shared_mem=THREADS_PER_WARP * self.system.float().nbytes
        )

        self.real_last_axis_sum_nz_kernel(
            (1,),
            (BLOCK_SIZE,),
            (self.temp_z, self.result),
            shared_mem=THREADS_PER_WARP * self.system.float().nbytes
        )

        self.save_data("H", self.result.get().item())

        # dHdt
        self.dHdt_kzkx_kernel(
            (self.system.nx * self.system.nz,),
            (BLOCK_SIZE,),
            (fields, fields_prev, current_dt, self.temp_zx),
            shared_mem=THREADS_PER_WARP * self.system.float().nbytes
        )

        self.real_last_axis_sum_nx_kernel(
            (self.system.nz,),
            (BLOCK_SIZE,),
            (self.temp_zx, self.temp_z),
            shared_mem=THREADS_PER_WARP * self.system.float().nbytes
        )

        self.real_last_axis_sum_nz_kernel(
            (1,),
            (BLOCK_SIZE,),
            (self.temp_z, self.result),
            shared_mem=THREADS_PER_WARP * self.system.float().nbytes
        )

        dHdt = self.result.get().item() 
        self.save_data("dHdt", dHdt)

        # Hyperdissipation
        dHdt_hyperdissipation_total = 0.0
        for component, kernel in self.dHdt_hyperdissipation_magnitude_kernels.items():
            kernel(
                (self.system.nx * self.system.nz,),
                (BLOCK_SIZE,),
                (fields, current_dt, self.temp_zx),
                shared_mem=THREADS_PER_WARP * self.system.float().nbytes
            )

            self.real_last_axis_sum_nx_kernel(
                (self.system.nz,),
                (BLOCK_SIZE,),
                (self.temp_zx, self.temp_z),
                shared_mem=THREADS_PER_WARP * self.system.float().nbytes
            )

            self.real_last_axis_sum_nz_kernel(
                (1,),
                (BLOCK_SIZE,),
                (self.temp_z, self.result),
                shared_mem=THREADS_PER_WARP * self.system.float().nbytes
            )

            dHdt_hyperdissipation_component = -self.result.get().item()

            self.save_data(
                f"dHdt_hyperdissipation_{component}",
                dHdt_hyperdissipation_component
            )
            dHdt_hyperdissipation_total += dHdt_hyperdissipation_component

        # Error in helicity budget
        self.save_data("dHdt_error", dHdt - dHdt_hyperdissipation_total)
