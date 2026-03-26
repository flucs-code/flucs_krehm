/*
 * Contains all the CUDA kernels for the 2D ITG model of Ivanov et al. (2020).
 */

// A lot of basic functionality is already implemented here.
#include "flucs/solvers/fourier/fourier_system.cuh"
#include "flucs_krehm/i0e_cuda.cuh"

extern "C" {

// Array for AB3 nonlinear terms
__constant__ FLUCS_COMPLEX* multistep_nonlinear_terms = NULL;

__device__ void get_linear_matrix(const size_t index, const FLUCS_FLOAT dt, FLUCS_COMPLEX matrix[2][2]){
    indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
    const size_t ikx = indices.ikx;
    const size_t iky = indices.iky;
    const size_t ikz = indices.ikz;

    const FLUCS_FLOAT kx = kx_from_ikx(ikx);
    const FLUCS_FLOAT ky = ky_from_iky(iky);
    const FLUCS_FLOAT kz = kz_from_ikz(ikz);

    const FLUCS_FLOAT kperp2 = kx*kx + ky*ky;


    // Generate the linear matrix
    matrix[0][0] = FLUCS_COMPLEX(0, 0);
    matrix[0][1] = FLUCS_COMPLEX(0, kz / one_minus_gamma0_over_alpha(kperp2));
    matrix[1][0] = FLUCS_COMPLEX(0, kz * (FLOAT_ONE + taubarinv(kperp2)) / (FLOAT_ONE + kperp2 * DE2));
    matrix[1][1] = FLUCS_COMPLEX(0, 0);
}


__global__ void find_derivatives(const FLUCS_COMPLEX* fields,
                                 FLUCS_COMPLEX* dft_derivatives,
                                 FLUCS_FLOAT* cfl_rate){
    const size_t padded_index = blockDim.x * blockIdx.x + threadIdx.x;

    // Check if we are within bounds
    if (!(padded_index < HALFPADDEDSIZE))
        return;

    indices3d_t padded_indices = get_indices3d<PADDED_NZ, PADDED_NX, HALF_PADDED_NY>(padded_index);
    const size_t padded_ikx = padded_indices.padded_ikx;
    const size_t padded_iky = padded_indices.padded_iky;
    const size_t padded_ikz = padded_indices.padded_ikz;

    if (padded_index == 0)
        cfl_rate[0] = 0;

    // Check if mode should be zeroed
    if (   (padded_ikx >= HALF_NX && padded_ikx < (HALF_NX + PADDED_NX) - NX)
        || (padded_ikz >= HALF_NZ && padded_ikz < (HALF_NZ + PADDED_NZ) - NZ)
        || padded_iky >= HALF_NY){

        dft_derivatives[padded_index] = 0;
        dft_derivatives[padded_index + HALFPADDEDSIZE] = 0;
        dft_derivatives[padded_index + 2*HALFPADDEDSIZE] = 0;
        dft_derivatives[padded_index + 3*HALFPADDEDSIZE] = 0;
        dft_derivatives[padded_index + 4*HALFPADDEDSIZE] = 0;
        dft_derivatives[padded_index + 5*HALFPADDEDSIZE] = 0;
        return;
    }
    
    const size_t ikx = ikx_from_padded_ikx(padded_ikx);
    const size_t ikz = ikz_from_padded_ikz(padded_ikz);

    const size_t index = index_from_3d<NZ, NX, HALF_NY>(ikz, ikx, padded_iky);

    const FLUCS_FLOAT kx = kx_from_ikx(ikx);

    // padded_iky and iky are the same for nonzero modes
    const FLUCS_FLOAT ky = ky_from_iky(padded_iky);

    const FLUCS_FLOAT kperp2 = kx*kx + ky*ky;

    const FLUCS_COMPLEX phi = fields[index];
    const FLUCS_COMPLEX A = fields[index + HALFUNPADDEDSIZE];

    // dxphi
    dft_derivatives[padded_index]\
        = FLUCS_COMPLEX(-kx * phi.imag(), kx * phi.real());

    // dyphi
    dft_derivatives[padded_index + HALFPADDEDSIZE]\
        = FLUCS_COMPLEX(-ky * phi.imag(), ky * phi.real());

    // dxA
    dft_derivatives[padded_index + 2*HALFPADDEDSIZE]\
        = FLUCS_COMPLEX(-kx * A.imag(), kx * A.real());

    // dyA
    dft_derivatives[padded_index + 3*HALFPADDEDSIZE]\
        = FLUCS_COMPLEX(-ky * A.imag(), ky * A.real());

    // [(1 - Gamma0) / alpha] kperp2 phi
    dft_derivatives[padded_index + 4*HALFPADDEDSIZE]\
        = one_minus_gamma0_over_alpha(kperp2) * kperp2 * phi;

    // kperp2 A
    dft_derivatives[padded_index + 5*HALFPADDEDSIZE]\
        = kperp2 * A;

}


__global__ void find_nonlinear_bits(FLUCS_FLOAT* real_derivatives_and_bits,
                                    FLUCS_FLOAT* cfl_rate){
    // Shared memory for CFL calculations
    extern __shared__ float cfl_shared[];

    const size_t real_index = blockDim.x * blockIdx.x + threadIdx.x;

    // Check if we are within bounds
    if (!(real_index < PADDEDSIZE))
        return;

    const FLUCS_FLOAT dxphi = real_derivatives_and_bits[real_index];
    const FLUCS_FLOAT dyphi = real_derivatives_and_bits[real_index + PADDEDSIZE];
    const FLUCS_FLOAT dxA = real_derivatives_and_bits[real_index + 2*PADDEDSIZE];
    const FLUCS_FLOAT dyA = real_derivatives_and_bits[real_index + 3*PADDEDSIZE];
    const FLUCS_FLOAT one_minus_gamma0_over_alpha_kperp2phi = real_derivatives_and_bits[real_index + 4*PADDEDSIZE];
    const FLUCS_FLOAT kperp2A = real_derivatives_and_bits[real_index + 5*PADDEDSIZE];


    const FLUCS_FLOAT cfl = flucs_fabs(dxphi) * (NY / LY) + flucs_fabs(dyphi) * (NX / LX);

    // Find max CFL using shared memory
    // TODO: Could we speed this up by reducing over warps?
    cfl_shared[threadIdx.x] = cfl;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            cfl_shared[threadIdx.x] = fmaxf(cfl_shared[threadIdx.x], cfl_shared[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    // First thread in block writes to global max via atomic
    if (threadIdx.x == 0) {
        atomicMaxFloat(cfl_rate, cfl_shared[0]); // custom atomic for float
    }

    real_derivatives_and_bits[real_index]               = (
        dxphi * one_minus_gamma0_over_alpha_kperp2phi  - dxA * kperp2A
    );
    real_derivatives_and_bits[real_index + PADDEDSIZE]  = (
        dyphi * one_minus_gamma0_over_alpha_kperp2phi  - dyA * kperp2A
    );
    real_derivatives_and_bits[real_index + 2*PADDEDSIZE] = (
        DE2 * dxphi * kperp2A
        - (0.5 * RHOI2 * ZTE_OVER_TI) * dxA * one_minus_gamma0_over_alpha_kperp2phi
    );
    real_derivatives_and_bits[real_index + 3*PADDEDSIZE] = (
        DE2 * dyphi * kperp2A
        - (0.5 * RHOI2 * ZTE_OVER_TI) * dyA * one_minus_gamma0_over_alpha_kperp2phi
    );
    real_derivatives_and_bits[real_index + 4*PADDEDSIZE] = dxphi * dyA - dyphi * dxA;
}

__device__ void add_nonlinear_terms(const size_t index,
                                    const FLUCS_FLOAT dt,
                                    const long long current_step,
                                    const FLUCS_FLOAT AB0,
                                    const FLUCS_FLOAT AB1,
                                    const FLUCS_FLOAT AB2,
                                    const FLUCS_COMPLEX* dft_bits,
                                    FLUCS_COMPLEX* rhs_fields){

    indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
    const size_t ikx = indices.ikx;
    const size_t iky = indices.iky;
    const size_t ikz = indices.ikz;

    // Ignore kperp2 = 0 modes
    if (ikx == 0 && iky == 0)
        return;

    const FLUCS_FLOAT kx = kx_from_ikx(ikx);
    const FLUCS_FLOAT ky = ky_from_iky(iky);

    const size_t padded_ikx = padded_ikx_from_ikx(ikx);
    const size_t padded_ikz = padded_ikz_from_ikz(ikz);

    const size_t padded_index = index_from_3d<PADDED_NZ, PADDED_NX, HALF_PADDED_NY>(padded_ikz, padded_ikx, iky);

    const FLUCS_FLOAT kperp2 = kx*kx + ky*ky;
    
    const FLUCS_COMPLEX phiNL = DFT_PADDEDSIZE_FACTOR * (
        FLUCS_COMPLEX(-ky * dft_bits[padded_index].imag(),
                      ky * dft_bits[padded_index].real())
        + FLUCS_COMPLEX(kx * dft_bits[padded_index + HALFPADDEDSIZE].imag(),
                        -kx * dft_bits[padded_index + HALFPADDEDSIZE].real())
    ) / (one_minus_gamma0_over_alpha(kperp2) * kperp2);

    const FLUCS_COMPLEX ANL = DFT_PADDEDSIZE_FACTOR * (
        dft_bits[padded_index + 4*HALFPADDEDSIZE]
        + FLUCS_COMPLEX(-ky * dft_bits[padded_index + 2*HALFPADDEDSIZE].imag(),
                      ky * dft_bits[padded_index + 2*HALFPADDEDSIZE].real())
        + FLUCS_COMPLEX(kx * dft_bits[padded_index + 3*HALFPADDEDSIZE].imag(),
                        -kx * dft_bits[padded_index + 3*HALFPADDEDSIZE].real())
    ) / (FLOAT_ONE + kperp2*DE2);


    const size_t multistep_index_0 = ((current_step      % 3 + 3) % 3) * 2 * HALFUNPADDEDSIZE + index;
    const size_t multistep_index_1 = ((current_step + 2) % 3)          * 2 * HALFUNPADDEDSIZE + index;
    const size_t multistep_index_2 = ((current_step + 1) % 3)          * 2 * HALFUNPADDEDSIZE + index;

    // phi
    rhs_fields[0] -= dt * (AB0*phiNL
                           +AB1*multistep_nonlinear_terms[multistep_index_1]
                           +AB2*multistep_nonlinear_terms[multistep_index_2]);

    multistep_nonlinear_terms[multistep_index_0] = phiNL;

    // A
    rhs_fields[1] -= dt * (AB0*ANL
                           +AB1*multistep_nonlinear_terms[multistep_index_1 + HALFUNPADDEDSIZE]
                           +AB2*multistep_nonlinear_terms[multistep_index_2 + HALFUNPADDEDSIZE]);

    multistep_nonlinear_terms[multistep_index_0 + HALFUNPADDEDSIZE] = ANL;
}

struct FreeEnergy_Functor {
    const FLUCS_COMPLEX* fields;
    const FLUCS_FLOAT multiplier;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        const FLUCS_COMPLEX phi = fields[index];
        const FLUCS_COMPLEX A = fields[index + HALFUNPADDEDSIZE];

        indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
        const size_t ikx = indices.ikx;
        const size_t iky = indices.iky;

        const FLUCS_FLOAT kx = kx_from_ikx(ikx);
        const FLUCS_FLOAT ky = ky_from_iky(iky);
        const FLUCS_FLOAT kperp2 = kx*kx + ky*ky;

        const FLUCS_FLOAT phi_bit = (
            (1 + taubarinv(kperp2)) * one_minus_gamma0_over_alpha(kperp2) * kperp2 
        ) * (phi.real() * phi.real() + phi.imag() * phi.imag());

        const FLUCS_FLOAT A_bit = (
            kperp2 * (1 + DE2 * kperp2)
        ) * (A.real() * A.real() + A.imag() * A.imag());

        return multiplier * (phi_bit + A_bit);
    }
};

__global__
void free_energy_kzkx(
    const FLUCS_COMPLEX* fields,
    FLUCS_FLOAT* output){

    add_and_sum_last_axis<HALF_NY, true>(
            FLOAT_ONE,
            output,
            FreeEnergy_Functor{fields, FLOAT_ONE}
        );

}


__global__
void dW_kzkx(
    const FLUCS_COMPLEX* fields_now,
    const FLUCS_COMPLEX* fields_prev,
    FLUCS_FLOAT* output){

    add_and_sum_last_axis<HALF_NY, true>(
            (FLUCS_FLOAT)1.0,
            output,
            FreeEnergy_Functor{fields_now, FLOAT_ONE},
            FreeEnergy_Functor{fields_prev, -FLOAT_ONE}
        );

}



} // extern "C"
