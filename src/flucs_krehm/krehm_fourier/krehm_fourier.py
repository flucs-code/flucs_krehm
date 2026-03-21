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

        # Anything custom goes here

    def compile_cupy_module(self) -> None:
        # System-specific constants for the kernels
        tau = self.input["parameters.tau"]
        Z = self.input["parameters.Z"]
        self.module_options.define_float("TAU_OVER_Z", tau / Z)
        self.module_options.define_float("RHOS2", self.input["parameters.rhos"]**2)
        self.module_options.define_float("DE2", self.input["parameters.de"]**2)

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
