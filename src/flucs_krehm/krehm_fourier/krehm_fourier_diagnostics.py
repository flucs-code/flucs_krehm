import cupy as cp
import numpy as np
from flucs.diagnostic import FlucsDiagnostic, FlucsDiagnosticVariable
#TODO remove these when coding optimisation wrapper
BLOCK_SIZE = int(256)
THREADS_PER_WARP = int(32) 

class FreeEnergyDiag(FlucsDiagnostic):
    name = "free_energy"

    temp: cp.ndarray
    result: cp.ndarray
    free_energy_kx_kernel: cp.RawKernel
    last_axis_sum_nx_kernel: cp.RawKernel

    def init_vars(self):
        self.add_var(FlucsDiagnosticVariable(
            name="W",
            shape=(),
            dimensions={},
            is_complex=False
        ))

        self.add_var(FlucsDiagnosticVariable(
            name="dWdt",
            shape=(),
            dimensions={},
            is_complex=False
        ))

        self.add_var(FlucsDiagnosticVariable(
            name="dWdt_hyperdissipation_perp",
            shape=(),
            dimensions={},
            is_complex=False
            )
        )

        self.add_var(FlucsDiagnosticVariable(
            name="dWdt_hyperdissipation_kx",
            shape=(),
            dimensions={},
            is_complex=False
            )
        )

        self.add_var(FlucsDiagnosticVariable(
            name="dWdt_hyperdissipation_ky",
            shape=(),
            dimensions={},
            is_complex=False
            )
        )

        self.add_var(FlucsDiagnosticVariable(
            name="dWdt_hyperdissipation_kz",
            shape=(),
            dimensions={},
            is_complex=False
            )
        )

        self.add_var(FlucsDiagnosticVariable(
            name="dWdt_error",
            shape=(),
            dimensions={},
            is_complex=False
        ))


    def ready(self):
        # Allocate temporary memory
        self.temp_zx = cp.zeros(self.system.nz * self.system.nx,
                                dtype=self.system.complex)

        self.temp_z = cp.zeros(self.system.nz, dtype=self.system.complex)

        self.result = cp.zeros((1,), dtype=self.system.float)
        self.complex_result = cp.zeros((1,), dtype=self.system.complex)

        # Get kernels
        self.dW_kzkx_kernel = self.system.cupy_module.get_function("dW_kzkx")
        self.free_energy_kzkx_kernel = self.system.cupy_module.get_function("free_energy_kzkx")
        self.last_axis_sum_nx_kernel = self.system.cupy_module.get_function("last_axis_sum_nx")
        self.last_axis_sum_nz_kernel = self.system.cupy_module.get_function("last_axis_sum_nz")
        self.real_last_axis_sum_nx_kernel = self.system.cupy_module.get_function("real_last_axis_sum_nx")
        self.real_last_axis_sum_nz_kernel = self.system.cupy_module.get_function("real_last_axis_sum_nz")

        self.hyperdissipation_magnitude_kernels = {
            "perp": self.system.cupy_module.get_function("W_hyperdissipation_perp_kzkx"),
            "kx": self.system.cupy_module.get_function("W_hyperdissipation_kx_kzkx"),
            "ky": self.system.cupy_module.get_function("W_hyperdissipation_ky_kzkx"),
            "kz": self.system.cupy_module.get_function("W_hyperdissipation_kz_kzkx"),
        }

    def execute(self):
        # W
        fields = self.system.fields[self.system.current_step % 2]
        fields_prev = self.system.fields[(self.system.current_step - 1) % 2]

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
        self.dW_kzkx_kernel(
                (self.system.nx * self.system.nz,),
                (BLOCK_SIZE,),
                (fields, fields_prev, self.temp_zx),
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

        dWdt = self.result.get().item() / self.system.current_dt
        self.save_data("dWdt", dWdt)

        # Hyperdissipation
        dWdt_hyperdissipation_total = 0.0
        for component, kernel in self.hyperdissipation_magnitude_kernels.items():
            kernel(
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

            dWdt_hyperdissipation_component = -self.result.get().item()

            self.save_data(
                f"dWdt_hyperdissipation_{component}",
                dWdt_hyperdissipation_component
            )
            dWdt_hyperdissipation_total += dWdt_hyperdissipation_component


        self.save_data("dWdt_error", dWdt - dWdt_hyperdissipation_total)


class HelicityDiag(FlucsDiagnostic):
    name = "helicity"

    temp: cp.ndarray
    result: cp.ndarray
    helicity_kzkx_kernel: cp.RawKernel
    last_axis_sum_nx_kernel: cp.RawKernel

    def init_vars(self):
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
            name="dHdt_hyperdissipation_perp",
            shape=(),
            dimensions={},
            is_complex=False
        ))

        self.add_var(FlucsDiagnosticVariable(
            name="dHdt_hyperdissipation_kx",
            shape=(),
            dimensions={},
            is_complex=False
        ))

        self.add_var(FlucsDiagnosticVariable(
            name="dHdt_hyperdissipation_ky",
            shape=(),
            dimensions={},
            is_complex=False
        ))

        self.add_var(FlucsDiagnosticVariable(
            name="dHdt_hyperdissipation_kz",
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

    def ready(self):
        self.temp_zx = cp.zeros(
            self.system.nz * self.system.nx,
            dtype=self.system.float
        )
        self.temp_z = cp.zeros(self.system.nz, dtype=self.system.float)
        self.result = cp.zeros((1,), dtype=self.system.float)

        self.helicity_kzkx_kernel = self.system.cupy_module.get_function(
            "helicity_kzkx"
        )
        self.dH_kzkx_kernel = self.system.cupy_module.get_function("dH_kzkx")
        self.real_last_axis_sum_nx_kernel = self.system.cupy_module.get_function(
            "real_last_axis_sum_nx"
        )
        self.real_last_axis_sum_nz_kernel = self.system.cupy_module.get_function(
            "real_last_axis_sum_nz"
        )

        self.hyperdissipation_magnitude_kernels = {
            "perp": self.system.cupy_module.get_function(
                "H_hyperdissipation_perp_kzkx"
            ),
            "kx": self.system.cupy_module.get_function(
                "H_hyperdissipation_kx_kzkx"
            ),
            "ky": self.system.cupy_module.get_function(
                "H_hyperdissipation_ky_kzkx"
            ),
            "kz": self.system.cupy_module.get_function(
                "H_hyperdissipation_kz_kzkx"
            ),
        }

    def execute(self):
        fields = self.system.fields[self.system.current_step % 2]
        fields_prev = self.system.fields[(self.system.current_step - 1) % 2]

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

        self.dH_kzkx_kernel(
            (self.system.nx * self.system.nz,),
            (BLOCK_SIZE,),
            (fields, fields_prev, self.temp_zx),
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

        dHdt = self.result.get().item() / self.system.current_dt
        self.save_data("dHdt", dHdt)

        dHdt_hyperdissipation_total = 0.0
        for component, kernel in self.hyperdissipation_magnitude_kernels.items():
            kernel(
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

            dHdt_hyperdissipation_component = -self.result.get().item()

            self.save_data(
                f"dHdt_hyperdissipation_{component}",
                dHdt_hyperdissipation_component
            )
            dHdt_hyperdissipation_total += dHdt_hyperdissipation_component

        self.save_data("dHdt_error", dHdt - dHdt_hyperdissipation_total)
