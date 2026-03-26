import numpy as np
from scipy.special import i0e
import cupy as cp
import time

N = 10000000
interval_lim = 10000
iterations = 1000
RHOI2 = 1

test_code = """
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

#include "i0e_cuda.cuh"


extern "C" {

__global__ void i0e_test_kernel(FLUCS_FLOAT* __restrict__ input, FLUCS_FLOAT* __restrict__ output) {
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (!(index < N))
        return;

    output[index] = flucs_i0e(input[index]);
}

__global__ void one_minus_gamma0_over_alpha_test_kernel(FLUCS_FLOAT* __restrict__ input, FLUCS_FLOAT* __restrict__ output) {
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (!(index < N))
        return;

    output[index] = one_minus_gamma0_over_alpha(input[index]);
}

}
"""


print(f"""
Testing flucs' CUDA implementation of i0e against SciPy.

The accuracy test uses {N:,} equally spaced points
in the interval [-{interval_lim:,}, {interval_lim:,}].

The speed test repeats the above calculation {iterations} times.

""")


def test_i0e(precision="single"):
    options = (
        "--ptxas-options=-O3",
        "--use_fast_math",
        "-I.",
        f"-DN={N}",
        f"-DGAMMA0_ALPHA_CUTOFF={GAMMA0_ALPHA_CUTOFF}",
        f"-DRHOI2={RHOI2}",
    )
    if precision == "single":
        float_type = np.float32
        print("SINGLE PRECISION")
        print("----------------")
        print()
    else:
        float_type = np.float64
        print("DOUBLE PRECISION")
        print("----------------")
        print()
        options += ("-DDOUBLE_PRECISION",)

    global test_code
    test_code += f"// {time.time()}"

    module = cp.RawModule(code=test_code, options = options)
    module.compile()

    i0e_kernel = module.get_function("i0e_test_kernel")

    input = np.linspace(-interval_lim, interval_lim, N, dtype=float_type)
    input_cuda = cp.asarray(input, dtype=float_type)
    output_cuda = cp.zeros(N, dtype=float_type)

    output = i0e(input)
    block_size = 512
    grid_size = (N + block_size - 1) // block_size
    i0e_kernel(
        (grid_size,),
        (block_size,),
        (input_cuda, output_cuda)
    )

    max_relative_error = np.max(np.abs((output - output_cuda.get()) / output))
    print(f"Max relative error is {max_relative_error}")

    start_time = time.time()
    for i in range(iterations):
        i0e_kernel(
            (grid_size,),
            (block_size,),
            (input_cuda, output_cuda)
        )

    cp.cuda.runtime.deviceSynchronize()
    end_time = time.time()

    print(f"{iterations} iterations took {end_time - start_time} seconds.")
    print()
    print()


def test_one_minus_gamma0_over_alpha(precision):
    options = (
        "--ptxas-options=-O3",
        "--use_fast_math",
        "-I.",
        f"-DN={N}",
        f"-DGAMMA0_ALPHA_CUTOFF={GAMMA0_ALPHA_CUTOFF}",
        f"-DRHOI2={RHOI2}",
    )
    if precision == "single":
        float_type = np.float32
        print("SINGLE PRECISION")
        print("----------------")
        print()
    else:
        float_type = np.float64
        print("DOUBLE PRECISION")
        print("----------------")
        print()
        options += ("-DDOUBLE_PRECISION",)

    print(
        f"Testing (1 - Gamma0) / alpha for {N=} and {RHOI2=}.\n"
        f"Taylor cut off is at alpha = {GAMMA0_ALPHA_CUTOFF}"
    )

    global test_code
    test_code += f"// {time.time()}"

    module = cp.RawModule(code=test_code, options=options)
    module.compile()

    i0e_kernel = module.get_function("i0e_test_kernel")
    one_minus_gamma0_over_alpha_test_kernel = module.get_function("one_minus_gamma0_over_alpha_test_kernel")

    alpha = cp.linspace(0, interval_lim, N, dtype=float_type)
    kperp2 = alpha / (0.5 * RHOI2)
    output_direct = cp.zeros(N, dtype=float_type)
    output_taylor = cp.zeros(N, dtype=float_type)

    input_scipy_double = np.zeros(N, dtype=np.float64)
    input_scipy_double[:] = alpha.get()[:]
    output_scipy_double = (1 - i0e(input_scipy_double)) / input_scipy_double

    block_size = 512
    grid_size = (N + block_size - 1) // block_size
    i0e_kernel(
        (grid_size,),
        (block_size,),
        (alpha, output_direct)
    )
    one_minus_gamma0_over_alpha_test_kernel(
        (grid_size,),
        (block_size,),
        (kperp2, output_taylor)
    )

    output_direct[:] = (1 - output_direct[:]) / alpha[:]

    num_alphas_with_taylor = alpha[alpha < GAMMA0_ALPHA_CUTOFF].shape[0]

    print(f"We have {num_alphas_with_taylor} alphas using the Taylor approximation.")
    print()

    for i in range(num_alphas_with_taylor - 2, num_alphas_with_taylor + 2):
        print(f"k2 = {kperp2[i]} -> alpha = {alpha[i]} -> output_direct = {output_direct[i]}")
        print(f"k2 = {kperp2[i]} -> alpha = {alpha[i]} -> output_taylor = {output_taylor[i]}")
        print(f"k2 = {kperp2[i]} -> alpha = {alpha[i]} -> scipy_double  = {output_scipy_double[i]}")
        print()

    diff = output_taylor - output_direct
    max_relative_error = np.max(np.abs(diff[1:] / output_taylor[1:]))
    print(f"Max relative diff is {max_relative_error}")

    output_taylor_as_double = np.zeros(N, dtype=np.float64)
    output_taylor_as_double[:] = output_taylor.get()[:]

    diff = output_taylor_as_double - output_scipy_double
    max_relative_error = np.max(np.abs(diff[1:] / output_scipy_double[1:]))
    max_relative_error_index = np.argmax(np.abs(diff[1:] / output_scipy_double[1:])) + 1

    print(f"Max relative diff with scipy double is {max_relative_error}")
    print(f"at alpha = {alpha[max_relative_error_index]} where Taylor gives {output_taylor_as_double[max_relative_error_index]}")
    print(f"while scipy double gives {output_scipy_double[max_relative_error_index]}")
    print()

# test_i0e("single")
# test_i0e("double")


N = 100000
interval_lim = 1
print(f"""
-------------------------------------------------

Now testing the small-alpha implementation of the (1 - Gamma0) / alpha operator.

The accuracy test uses {N:,} equally spaced points
in the interval [0, {interval_lim}] for alpha.

""")
GAMMA0_ALPHA_CUTOFF = 0.15
test_one_minus_gamma0_over_alpha("single")
GAMMA0_ALPHA_CUTOFF = 5e-3
test_one_minus_gamma0_over_alpha("double")
