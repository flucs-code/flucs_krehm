from pathlib import Path
import time

import cupy as cp
import numpy as np
from scipy.special import i0e


RHOI2 = 1.0
BLOCK_SIZE = 512
INCLUDE_DIR = Path(__file__).resolve().parent

cuda_test_code = """
// This is usually done by fourier_system.cuh
#ifdef DOUBLE_PRECISION
    #define FLUCS_FLOAT double
    #define flucs_fabs(x) fabs(x)
    #define FLUCS_COMPLEX_FLOAT_EQUIV double2
#else
    #define FLUCS_FLOAT float
    #define flucs_fabs(x) fabsf(x)
    #define FLUCS_COMPLEX_FLOAT_EQUIV float2
#endif

#define FLOAT_ONE ((FLUCS_FLOAT)1.0)

#ifndef ZTE_OVER_TI
    #define ZTE_OVER_TI ((FLUCS_FLOAT)1.0)
#endif

#include "i0e_cuda.cuh"

extern "C" {

__global__ void i0e_test_kernel(
    FLUCS_FLOAT* __restrict__ input,
    FLUCS_FLOAT* __restrict__ output,
    const int n
) {
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (!(index < n))
        return;

    output[index] = flucs_i0e(input[index]);
}

__global__ void one_minus_gamma0_over_alpha_test_kernel(
    FLUCS_FLOAT* __restrict__ input,
    FLUCS_FLOAT* __restrict__ output,
    const int n
) {
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (!(index < n))
        return;

    output[index] = one_minus_gamma0_over_alpha(input[index]);
}

}
"""

def get_float_type(precision: str):
    if precision == "single":
        return np.float32
    return np.float64

def compile_module(cutoff: float, precision: str):
    options = [
        "--ptxas-options=-O3",
        "--use_fast_math",
        f"-I{INCLUDE_DIR}",
        f"-DGAMMA0_ALPHA_CUTOFF={cutoff}",
        f"-DRHOI2={RHOI2}",
    ]

    if precision == "double":
        options.append("-DDOUBLE_PRECISION")

    module = cp.RawModule(
        code=cuda_test_code,
        options=tuple(options),
    )
    module.compile()
    return module

def run_i0e_test(
    n: int,
    interval_lim: float,
    iterations: int,
    precision: str,
    cutoff: float,
):
    if precision == "single":
        float_type = np.float32
        print("SINGLE PRECISION")
        print("----------------")
    else:
        float_type = np.float64
        print("DOUBLE PRECISION")
        print("----------------")

    module = compile_module(cutoff, precision)
    i0e_kernel = module.get_function("i0e_test_kernel")

    input_vals = np.linspace(-interval_lim, interval_lim, n, dtype=float_type)
    input_cuda = cp.asarray(input_vals, dtype=float_type)
    output_cuda = cp.zeros(n, dtype=float_type)

    output = i0e(input_vals)
    grid_size = (n + BLOCK_SIZE - 1) // BLOCK_SIZE

    i0e_kernel(
        (grid_size,),
        (BLOCK_SIZE,),
        (input_cuda, output_cuda, np.int32(n))
    )

    max_relative_error = np.max(np.abs((output - output_cuda.get()) / output))
    print(f"Max relative error: {max_relative_error:.16e}")

    start_time = time.time()
    for _ in range(iterations):
        i0e_kernel(
            (grid_size,),
            (BLOCK_SIZE,),
            (input_cuda, output_cuda, np.int32(n))
        )

    cp.cuda.runtime.deviceSynchronize()
    end_time = time.time()

    print(f"Time per iteration: {((end_time - start_time) / iterations):.16e} seconds.")
    print()

def run_gamma0_test(
    n: int,
    interval_lim: float,
    precision: str,
    cutoff: float,
):
    if precision == "single":
        float_type = np.float32
        print("SINGLE PRECISION")
        print("----------------")
    else:
        float_type = np.float64
        print("DOUBLE PRECISION")
        print("----------------")

    module = compile_module(cutoff, precision)

    print(
        f"Taylor cut off is set to alpha = {cutoff:.5e}"
    )

    i0e_kernel = module.get_function("i0e_test_kernel")
    one_minus_gamma0_over_alpha_test_kernel = module.get_function(
        "one_minus_gamma0_over_alpha_test_kernel"
    )

    alpha = cp.linspace(0, interval_lim, n, dtype=float_type)
    kperp2 = alpha / (0.5 * RHOI2)
    output_direct = cp.zeros(n, dtype=float_type)
    output_taylor = cp.zeros(n, dtype=float_type)

    input_scipy_double = np.zeros(n, dtype=np.float64)
    input_scipy_double[:] = alpha.get()[:]
    
    output_scipy_double = np.empty(n, dtype=np.float64)
    output_scipy_double[0] = 1.0
    output_scipy_double[1:] = (
        1 - i0e(input_scipy_double[1:])
    ) / input_scipy_double[1:]

    grid_size = (n + BLOCK_SIZE - 1) // BLOCK_SIZE

    i0e_kernel(
        (grid_size,),
        (BLOCK_SIZE,),
        (alpha, output_direct, np.int32(n))
    )
    one_minus_gamma0_over_alpha_test_kernel(
        (grid_size,),
        (BLOCK_SIZE,),
        (kperp2, output_taylor, np.int32(n))
    )

    output_direct[0] = 1.0
    output_direct[1:] = (1 - output_direct[1:]) / alpha[1:]

    num_alphas_with_taylor = alpha[alpha < cutoff].shape[0]

    print(f"Number of alphas using Taylor approximation: {num_alphas_with_taylor:.5e}")
    print("")
    print("Transition samples:")

    for i in range(num_alphas_with_taylor - 2, num_alphas_with_taylor + 2):
        print(
            f"alpha = {alpha[i]:.16e}\n"
            f"  direct: {output_direct[i]:.16e}\n"
            f"  taylor: {output_taylor[i]:.16e}\n"
            f"  scipy : {output_scipy_double[i]:.16e}\n"
        )

    # Get averaged statistics
    diff = output_taylor - output_direct
    max_relative_error_direct = cp.max(
        cp.abs(diff[1:] / output_taylor[1:])
    ).get().item()

    output_taylor_as_double = np.zeros(n, dtype=np.float64)
    output_taylor_as_double[:] = output_taylor.get()[:]

    diff = output_taylor_as_double - output_scipy_double
    max_relative_error_scipy = np.max(
        np.abs(diff[1:] / output_scipy_double[1:])
    )
    max_relative_error_index = (
        np.argmax(np.abs(diff[1:] / output_scipy_double[1:])) + 1
    )

    print(f"Max relative difference at alpha = {alpha[max_relative_error_index]:.16e}\n"
          f"Associated errors:\n"
          f"  direct: {max_relative_error_direct:.16e}\n"
          f"  scipy : {max_relative_error_scipy:.16e}\n"
          )

# Run tests
if __name__ == "__main__":
    
    print("")

    # Number of iterations to time for
    iterations = 1000

    # Setting options for i0e test
    n_i0e = 10_000_000
    interval_lim_i0e = 10_000

    print("-" * 73)
    print(
        "Testing flucs' CUDA implementation of i0e against SciPy.\n"
        f"Using {n_i0e:,} equally spaced points in the interval "
        f"[-{interval_lim_i0e:,}, {interval_lim_i0e:,}].\n"
        f"Timing the calculation for {iterations} iterations.\n"
    )

    i0e_cases = (
        ("single", 0.15),
        ("double", 5e-3),
    )

    for precision, cutoff in i0e_cases:
        run_i0e_test(
            n=n_i0e,
            interval_lim=interval_lim_i0e,
            iterations=iterations,
            precision=precision,
            cutoff=cutoff,
        )

    # Setting options for gamma0 test
    n_gamma0 = 100_000
    interval_lim_gamma0 = 1

    print("-" * 59)
    print(
        "Testing flucs' CUDA implementation of gamma0 against SciPy.\n"
        f"Using {n_gamma0:,} equally spaced points in the interval "
        f"[0, {interval_lim_gamma0}].\n"
        f"Using parameter value RHOI2 = {RHOI2}.\n"
    )

    gamma0_cases = (
        ("single", 0.15),
        ("double", 5e-3),
    )

    for precision, cutoff in gamma0_cases:
        run_gamma0_test(
            n=n_gamma0,
            interval_lim=interval_lim_gamma0,
            precision=precision,
            cutoff=cutoff,
        )