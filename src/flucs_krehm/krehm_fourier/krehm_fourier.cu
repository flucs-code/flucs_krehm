/*
 * Contains all the CUDA kernels for the isothermal KREHM model outlined in Adkins et al. (2024).
 */

// A lot of basic functionality is already implemented here.
#include "flucs/solvers/fourier/fourier_system.cuh"
#include "flucs_krehm/i0e_cuda.cuh"

extern "C" {

////////////////////////////////////////////////////////////////////////////////
// Core solver functions
////////////////////////////////////////////////////////////////////////////////

// Array for AB3 nonlinear terms
__constant__ FLUCS_COMPLEX* multistep_nonlinear_terms = NULL;

// Fetches the linear matrix for a given mode
__device__ void get_linear_matrix(
    const size_t index, 
    const FLUCS_FLOAT dt, 
    FLUCS_COMPLEX matrix[2][2]
){
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

// Finds the derivatives (in Fourier space) required to construct the nonlinear
// terms entering through the poisson brackets
__global__ void find_derivatives(
    const FLUCS_COMPLEX* fields,
    FLUCS_COMPLEX* dft_derivatives,
    FLUCS_FLOAT* cfl_rate
){
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
    const FLUCS_COMPLEX apar = fields[index + HALFUNPADDEDSIZE];

    // dxphi
    dft_derivatives[padded_index]\
        = FLUCS_COMPLEX(-kx * phi.imag(), kx * phi.real());

    // dyphi
    dft_derivatives[padded_index + HALFPADDEDSIZE]\
        = FLUCS_COMPLEX(-ky * phi.imag(), ky * phi.real());

    // dxapar
    dft_derivatives[padded_index + 2*HALFPADDEDSIZE]\
        = FLUCS_COMPLEX(-kx * apar.imag(), kx * apar.real());

    // dyapar
    dft_derivatives[padded_index + 3*HALFPADDEDSIZE]\
        = FLUCS_COMPLEX(-ky * apar.imag(), ky * apar.real());

    // [(1 - Gamma0) / alpha] kperp2phi
    dft_derivatives[padded_index + 4*HALFPADDEDSIZE]\
        = one_minus_gamma0_over_alpha(kperp2) * kperp2 * phi;

    // kperp2apar
    dft_derivatives[padded_index + 5*HALFPADDEDSIZE]\
        = kperp2 * apar;

}

// Finds the nonlinear combinations of (real-space) derivatives required to 
// construct the nonlinear terms
__global__ void find_nonlinear_bits(
    FLUCS_FLOAT* real_derivatives_and_bits,
    FLUCS_FLOAT* cfl_rate
){
    // Shared memory for CFL calculations
    extern __shared__ FLUCS_FLOAT cfl_shared[];

    const size_t real_index = blockDim.x * blockIdx.x + threadIdx.x;
    const bool in_bounds = real_index < PADDEDSIZE;

    // Inactive threads do not contribute to the cfl reduction 
    const FLUCS_FLOAT dxphi = in_bounds
        ? real_derivatives_and_bits[real_index]
        : (FLUCS_FLOAT)0;
    const FLUCS_FLOAT dyphi = in_bounds
        ? real_derivatives_and_bits[real_index + PADDEDSIZE]
        : (FLUCS_FLOAT)0;

    const FLUCS_FLOAT cfl = flucs_fabs(dxphi) * (NY / LY)
        + flucs_fabs(dyphi) * (NX / LX);

    // Find max CFL using shared memory
    // TODO: Could we speed this up by reducing over warps?
    cfl_shared[threadIdx.x] = cfl;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            cfl_shared[threadIdx.x] = flucs_fmax(cfl_shared[threadIdx.x], cfl_shared[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    // First thread in block writes to global max via atomic
    if (threadIdx.x == 0) {
        atomicMaxFloat(cfl_rate, cfl_shared[0]); // custom atomic for float
    }

    // Out-of-bounds threads should not contribute to nonlinear bits
    if (!in_bounds)
        return;

    const FLUCS_FLOAT dxapar = real_derivatives_and_bits[real_index + 2*PADDEDSIZE];
    const FLUCS_FLOAT dyapar = real_derivatives_and_bits[real_index + 3*PADDEDSIZE];
    const FLUCS_FLOAT one_minus_gamma0_over_alpha_kperp2phi = real_derivatives_and_bits[real_index + 4*PADDEDSIZE];
    const FLUCS_FLOAT kperp2apar = real_derivatives_and_bits[real_index + 5*PADDEDSIZE];

    real_derivatives_and_bits[real_index]               = (
        dxphi * one_minus_gamma0_over_alpha_kperp2phi  - dxapar * kperp2apar
    );
    real_derivatives_and_bits[real_index + PADDEDSIZE]  = (
        dyphi * one_minus_gamma0_over_alpha_kperp2phi  - dyapar * kperp2apar
    );
    real_derivatives_and_bits[real_index + 2*PADDEDSIZE] = (
        dxphi * (DE2 * kperp2apar)
        - (((FLUCS_FLOAT)0.5) * RHOI2 * ZTE_OVER_TI) * dxapar * one_minus_gamma0_over_alpha_kperp2phi
    );
    real_derivatives_and_bits[real_index + 3*PADDEDSIZE] = (
        dyphi * (DE2 * kperp2apar)
        - (((FLUCS_FLOAT)0.5) * RHOI2 * ZTE_OVER_TI) * dyapar * one_minus_gamma0_over_alpha_kperp2phi
    );
    real_derivatives_and_bits[real_index + 4*PADDEDSIZE] = dxphi * dyapar - dyphi * dxapar;
}

// Returns the nonlinear terms for a given mode
__device__ void add_nonlinear_terms(
    const size_t index,
    const FLUCS_COMPLEX* dft_bits,
    FLUCS_COMPLEX* explicit_terms
){
    // Indices
    indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
    const size_t ikx = indices.ikx;
    const size_t iky = indices.iky;
    const size_t ikz = indices.ikz;

    // Ignore kperp2 = 0 modes
    if (ikx == 0 && iky == 0)
        return;

    // Wavenumbers and indices 
    const FLUCS_FLOAT kx = kx_from_ikx(ikx);
    const FLUCS_FLOAT ky = ky_from_iky(iky);

    const size_t padded_ikx = padded_ikx_from_ikx(ikx);
    const size_t padded_ikz = padded_ikz_from_ikz(ikz);
    const size_t padded_index = index_from_3d<PADDED_NZ, PADDED_NX, HALF_PADDED_NY>(padded_ikz, padded_ikx, iky);

    const FLUCS_FLOAT kperp2 = kx*kx + ky*ky;
    
    // Calculate nonnlinear terms
    explicit_terms[0] += DFT_PADDEDSIZE_FACTOR * (
        + FLUCS_COMPLEX(-ky * dft_bits[padded_index].imag(),
                         ky * dft_bits[padded_index].real())
        - FLUCS_COMPLEX(-kx * dft_bits[padded_index + HALFPADDEDSIZE].imag(),
                         kx * dft_bits[padded_index + HALFPADDEDSIZE].real())
    ) / (one_minus_gamma0_over_alpha(kperp2) * kperp2);

    explicit_terms[1] += DFT_PADDEDSIZE_FACTOR * (
        + dft_bits[padded_index + 4*HALFPADDEDSIZE]
        + FLUCS_COMPLEX(-ky * dft_bits[padded_index + 2*HALFPADDEDSIZE].imag(),
                         ky * dft_bits[padded_index + 2*HALFPADDEDSIZE].real())
        - FLUCS_COMPLEX(-kx * dft_bits[padded_index + 3*HALFPADDEDSIZE].imag(),
                         kx * dft_bits[padded_index + 3*HALFPADDEDSIZE].real())
    ) / (FLOAT_ONE + kperp2*DE2);

}

// Mapping of nonlinear terms to fields
__device__ __forceinline__
int explicit_term_field_index(const int term_index) {
    return term_index; // Trivial indexing in this case
}

////////////////////////////////////////////////////////////////////////////////
// General helper functions
////////////////////////////////////////////////////////////////////////////////

// Phase velocity (normalised to the Alfven speed)
// Note that we supply one_minus_gamma0_over_alpha as an argument to avoid 
// duplicating calls to this in order locations. 
__device__ __forceinline__
FLUCS_FLOAT get_phase_velocity(
    const FLUCS_FLOAT kperp2,
    const FLUCS_FLOAT one_minus_gamma0_over_alpha
){
    const FLUCS_FLOAT alpha = ((FLUCS_FLOAT)0.5) * RHOI2 * kperp2;

    return sqrt(
        (ZTE_OVER_TI * alpha + FLOAT_ONE/one_minus_gamma0_over_alpha)
        / (FLOAT_ONE + DE2 * kperp2)
    );
}

// Generalised Elsasser potentials
// Note that we adopt the sign convention that the "plus" field propagates in
// the positive z direction, which is the opposite convention to that used in,
// e.g., Adkins et al. (2024).
__device__ __forceinline__
void get_thetas_from_fields(
    const size_t index,
    const FLUCS_COMPLEX* fields,
    FLUCS_COMPLEX& thetap,
    FLUCS_COMPLEX& thetam,
    FLUCS_FLOAT& vphase
){
    // Indices
    indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
    const size_t ikx = indices.ikx;
    const size_t iky = indices.iky;

    // Wavenumbers 
    const FLUCS_FLOAT kx = kx_from_ikx(ikx);
    const FLUCS_FLOAT ky = ky_from_iky(iky);

    const FLUCS_FLOAT kperp2 = kx*kx + ky*ky;

    // Fields
    const FLUCS_COMPLEX phi = fields[index];
    const FLUCS_COMPLEX apar = fields[index + HALFUNPADDEDSIZE];

    // Useful intermediate quantities
    const FLUCS_FLOAT gamma_factor = one_minus_gamma0_over_alpha(kperp2);
    vphase = get_phase_velocity(kperp2, gamma_factor);

    // Construct Elsasser potentials
    const FLUCS_FLOAT phi_factor = vphase * gamma_factor;
    const FLUCS_FLOAT prefactor = sqrt(FLOAT_ONE + kperp2 * DE2);

    thetap = prefactor * (phi_factor * phi + apar);
    thetam = prefactor * (phi_factor * phi - apar);
}

////////////////////////////////////////////////////////////////////////////////
// Forcing
////////////////////////////////////////////////////////////////////////////////

#ifdef FORCING

#if defined(FORCING_METHOD_ELSASSER)
__device__ __forceinline__
void add_forcing_elsasser(
    const size_t index,
    const FLUCS_FLOAT dt,
    const long long current_step,
    const FLUCS_COMPLEX* previous_fields,
    FLUCS_COMPLEX explicit_terms[NUMBER_OF_FIELDS_EXPLICIT]
)
{
    // Unused variables
    (void)dt;
    (void)current_step;

    // Indices
    indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
    const size_t ikx = indices.ikx;
    const size_t iky = indices.iky;
    const size_t ikz = indices.ikz;

    // Wavenumbers 
    const FLUCS_FLOAT kx = kx_from_ikx(ikx);
    const FLUCS_FLOAT ky = ky_from_iky(iky);
    const FLUCS_FLOAT kz = kz_from_ikz(ikz);

    const FLUCS_FLOAT kperp2 = kx*kx + ky*ky;

    if (kperp2 == ((FLUCS_FLOAT)0.0))
        return;
    
    if (!(kperp2 > FORCING_KPERP2_MIN &&
          kperp2 < FORCING_KPERP2_MAX &&
          kz > FORCING_KZ_MIN &&
          kz < FORCING_KZ_MAX))
        return;

    // Get fields and matrices
    FLUCS_COMPLEX thetap, thetam;
    FLUCS_FLOAT vphase;
    get_thetas_from_fields(index, previous_fields, thetap, thetam, vphase);

    const FLUCS_FLOAT gamma_factor = one_minus_gamma0_over_alpha(kperp2);

    // Energies
    const FLUCS_FLOAT Wp = (
          ((FLUCS_FLOAT)0.5) * kperp2 
        * (thetap.real()*thetap.real() + thetap.imag()*thetap.imag())
    );
    const FLUCS_FLOAT Wm = (
          ((FLUCS_FLOAT)0.5) * kperp2 
        * (thetam.real()*thetam.real() + thetam.imag()*thetam.imag())
    );

    // Forcing in thetas
    FLUCS_COMPLEX forcing_p = FLUCS_COMPLEX(0, 0);
    FLUCS_COMPLEX forcing_m = FLUCS_COMPLEX(0, 0);

    if (Wp > ((FLUCS_FLOAT)0.0))
        forcing_p = ((FLUCS_FLOAT)0.5) * FORCING_EPSILON_PLUS  * thetap / Wp;
    if (Wm > ((FLUCS_FLOAT)0.0))
        forcing_m = ((FLUCS_FLOAT)0.5) * FORCING_EPSILON_MINUS * thetam / Wm;

    const FLUCS_FLOAT sqrt_one_plus_kperpde2 = sqrt(FLOAT_ONE + kperp2 * DE2);

    // Construct forcing
    explicit_terms[0] -= (
        (forcing_p + forcing_m) / (
            (FLUCS_FLOAT)2.0
            * vphase * gamma_factor
            * sqrt_one_plus_kperpde2
        )
    );
    explicit_terms[1] -= (
        (forcing_p - forcing_m) / (
            (FLUCS_FLOAT)2.0
            * sqrt_one_plus_kperpde2
        )
    );
}
#endif

__device__ void add_forcing_explicit(
    const size_t index,
    const FLUCS_FLOAT dt, 
    const long long current_step,
    const FLUCS_COMPLEX* previous_fields,
    FLUCS_COMPLEX explicit_terms[NUMBER_OF_FIELDS_EXPLICIT] 
){
    #if defined(FORCING_METHOD_ELSASSER)
        add_forcing_elsasser(
            index, dt, current_step, previous_fields, explicit_terms
        );
    #endif
}

#endif


////////////////////////////////////////////////////////////////////////////////
// Diagnostics: Free Energy (W)
////////////////////////////////////////////////////////////////////////////////

struct FreeEnergy_Functor {
    const FLUCS_COMPLEX* fields;
    const FLUCS_FLOAT multiplier;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        const FLUCS_COMPLEX phi = fields[index];
        const FLUCS_COMPLEX apar = fields[index + HALFUNPADDEDSIZE];

        indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
        const size_t ikx = indices.ikx;
        const size_t iky = indices.iky;

        const FLUCS_FLOAT kx = kx_from_ikx(ikx);
        const FLUCS_FLOAT ky = ky_from_iky(iky);
        const FLUCS_FLOAT kperp2 = kx*kx + ky*ky;

        const FLUCS_FLOAT phi_contribution = (
            (1 + taubarinv(kperp2)) * one_minus_gamma0_over_alpha(kperp2) * kperp2 
        ) * (phi.real() * phi.real() + phi.imag() * phi.imag());

        const FLUCS_FLOAT apar_contribution = (
            kperp2 * (1 + DE2 * kperp2)
        ) * (apar.real() * apar.real() + apar.imag() * apar.imag());

        return multiplier * (phi_contribution + apar_contribution);
    }
};

struct FreeEnergyForcing_Functor {
    const FLUCS_COMPLEX* fields;
    const FLUCS_FLOAT multiplier;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        const FLUCS_COMPLEX phi = fields[index];
        const FLUCS_COMPLEX apar = fields[index + HALFUNPADDEDSIZE];
        FLUCS_COMPLEX forcing_terms[NUMBER_OF_FIELDS] = {0};

        add_forcing_elsasser(index, (FLUCS_FLOAT)0, 0, fields, forcing_terms);

        indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
        const size_t ikx = indices.ikx;
        const size_t iky = indices.iky;

        const FLUCS_FLOAT kx = kx_from_ikx(ikx);
        const FLUCS_FLOAT ky = ky_from_iky(iky);
        const FLUCS_FLOAT kperp2 = kx*kx + ky*ky;

        const FLUCS_FLOAT phi_contribution = -2.0 * (
            (1 + taubarinv(kperp2)) * one_minus_gamma0_over_alpha(kperp2) * kperp2 
        ) * (phi.real() * forcing_terms[0].real() + phi.imag() * forcing_terms[0].imag());

        const FLUCS_FLOAT apar_contribution = -2.0 * (
            kperp2 * (1 + DE2 * kperp2)
        ) * (apar.real() * forcing_terms[1].real() + apar.imag() * forcing_terms[1].imag());

        return multiplier * (phi_contribution + apar_contribution);
    }
};

struct FreeEnergyThetap_Functor {
    const FLUCS_COMPLEX* fields;
    const FLUCS_FLOAT multiplier;

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        
        // Indices
        indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
        const FLUCS_FLOAT kx = kx_from_ikx(indices.ikx);
        const FLUCS_FLOAT ky = ky_from_iky(indices.iky);
        const FLUCS_FLOAT kperp2 = kx*kx + ky*ky;

        // Thetas
        FLUCS_COMPLEX thetap, thetam;
        FLUCS_FLOAT vphase;
        get_thetas_from_fields(index, fields, thetap, thetam, vphase);

        return multiplier * ((FLUCS_FLOAT)0.5) * kperp2
            * (thetap.real()*thetap.real() + thetap.imag()*thetap.imag());
    }
};

struct FreeEnergyThetam_Functor {
    const FLUCS_COMPLEX* fields;
    const FLUCS_FLOAT multiplier;

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        
        // Indices
        indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
        const FLUCS_FLOAT kx = kx_from_ikx(indices.ikx);
        const FLUCS_FLOAT ky = ky_from_iky(indices.iky);
        const FLUCS_FLOAT kperp2 = kx*kx + ky*ky;

        // Thetas
        FLUCS_COMPLEX thetap, thetam;
        FLUCS_FLOAT vphase;
        get_thetas_from_fields(index, fields, thetap, thetam, vphase);

        return multiplier * ((FLUCS_FLOAT)0.5) * kperp2
            * (thetam.real()*thetam.real() + thetam.imag()*thetam.imag());
    }
};

// W
__global__
void W_kzkx(
    const FLUCS_COMPLEX* fields,
    FLUCS_FLOAT* output
){

    add_and_sum_last_axis<HALF_NY, true>(
            FLOAT_ONE,
            output,
            FreeEnergy_Functor{fields, FLOAT_ONE}
        );

}

__global__
void Wp_kzkx(
    const FLUCS_COMPLEX* fields,
    FLUCS_FLOAT* output
){

    add_and_sum_last_axis<HALF_NY, true>(
            FLOAT_ONE,
            output,
            FreeEnergyThetap_Functor{fields, FLOAT_ONE}
        );

}

__global__
void Wm_kzkx(
    const FLUCS_COMPLEX* fields,
    FLUCS_FLOAT* output
){

    add_and_sum_last_axis<HALF_NY, true>(
            FLOAT_ONE,
            output,
            FreeEnergyThetam_Functor{fields, FLOAT_ONE}
        );

}

// dWdt
__global__
void dWdt_kzkx(
    const FLUCS_COMPLEX* fields_now,
    const FLUCS_COMPLEX* fields_prev,
    const FLUCS_FLOAT dt, 
    FLUCS_FLOAT* output
){

    add_and_sum_last_axis<HALF_NY, true>(
            FLOAT_ONE / dt,
            output,
            FreeEnergy_Functor{fields_now, FLOAT_ONE},
            FreeEnergy_Functor{fields_prev, -FLOAT_ONE}
        );

}

// W forcing
__global__
void dWdt_forcing_kzkx(
    const FLUCS_COMPLEX* fields,
    FLUCS_FLOAT* output
){

    add_and_sum_last_axis<HALF_NY, true>(
            FLOAT_ONE,
            output,
            FreeEnergyForcing_Functor{fields, FLOAT_ONE}
        );

}

__global__
void dWpdt_kzkx(
    const FLUCS_COMPLEX* fields_now,
    const FLUCS_COMPLEX* fields_prev,
    const FLUCS_FLOAT dt, 
    FLUCS_FLOAT* output
){

    add_and_sum_last_axis<HALF_NY, true>(
            FLOAT_ONE / dt,
            output,
            FreeEnergyThetap_Functor{fields_now, FLOAT_ONE},
            FreeEnergyThetap_Functor{fields_prev, -FLOAT_ONE}
        );

}

__global__
void dWmdt_kzkx(
    const FLUCS_COMPLEX* fields_now,
    const FLUCS_COMPLEX* fields_prev,
    const FLUCS_FLOAT dt, 
    FLUCS_FLOAT* output
){

    add_and_sum_last_axis<HALF_NY, true>(
            FLOAT_ONE / dt,
            output,
            FreeEnergyThetam_Functor{fields_now, FLOAT_ONE},
            FreeEnergyThetam_Functor{fields_prev, -FLOAT_ONE}
        );

}

// dWdt_hyperdissipation_kx
__global__
void dWdt_hyperdissipation_kx_kzkx(
    const FLUCS_COMPLEX* fields,
    const FLUCS_FLOAT dt,
    FLUCS_FLOAT* output
) {
    add_and_sum_last_axis<HALF_NY, true>(
        FLOAT_ONE,
        output,
        HyperdissipationKx_Functor<FreeEnergy_Functor>{
            FreeEnergy_Functor{fields, (FLUCS_FLOAT)2.0}, dt
        }
    );
}

__global__
void dWpdt_hyperdissipation_kx_kzkx(
    const FLUCS_COMPLEX* fields,
    const FLUCS_FLOAT dt,
    FLUCS_FLOAT* output
) {
    add_and_sum_last_axis<HALF_NY, true>(
        FLOAT_ONE,
        output,
        HyperdissipationKx_Functor<FreeEnergyThetap_Functor>{
            FreeEnergyThetap_Functor{fields, (FLUCS_FLOAT)2.0}, dt
        }
    );
}

__global__
void dWmdt_hyperdissipation_kx_kzkx(
    const FLUCS_COMPLEX* fields,
    const FLUCS_FLOAT dt,
    FLUCS_FLOAT* output
) {
    add_and_sum_last_axis<HALF_NY, true>(
        FLOAT_ONE,
        output,
        HyperdissipationKx_Functor<FreeEnergyThetam_Functor>{
            FreeEnergyThetam_Functor{fields, (FLUCS_FLOAT)2.0}, dt
        }
    );
}


// dWdt_hyperdissipation_ky
__global__
void dWdt_hyperdissipation_ky_kzkx(
    const FLUCS_COMPLEX* fields,
    const FLUCS_FLOAT dt,
    FLUCS_FLOAT* output
) {
    add_and_sum_last_axis<HALF_NY, true>(
        FLOAT_ONE,
        output,
        HyperdissipationKy_Functor<FreeEnergy_Functor>{
            FreeEnergy_Functor{fields, (FLUCS_FLOAT)2.0}, dt
        }
    );
}

__global__
void dWpdt_hyperdissipation_ky_kzkx(
    const FLUCS_COMPLEX* fields,
    const FLUCS_FLOAT dt,
    FLUCS_FLOAT* output
) {
    add_and_sum_last_axis<HALF_NY, true>(
        FLOAT_ONE,
        output,
        HyperdissipationKy_Functor<FreeEnergyThetap_Functor>{
            FreeEnergyThetap_Functor{fields, (FLUCS_FLOAT)2.0}, dt
        }
    );
}

__global__
void dWmdt_hyperdissipation_ky_kzkx(
    const FLUCS_COMPLEX* fields,
    const FLUCS_FLOAT dt,
    FLUCS_FLOAT* output
) {
    add_and_sum_last_axis<HALF_NY, true>(
        FLOAT_ONE,
        output,
        HyperdissipationKy_Functor<FreeEnergyThetam_Functor>{
            FreeEnergyThetam_Functor{fields, (FLUCS_FLOAT)2.0}, dt
        }
    );
}


// dWdt_hyperdissipation_kz
__global__
void dWdt_hyperdissipation_kz_kzkx(
    const FLUCS_COMPLEX* fields,
    const FLUCS_FLOAT dt,
    FLUCS_FLOAT* output
) {
    add_and_sum_last_axis<HALF_NY, true>(
        FLOAT_ONE,
        output,
        HyperdissipationKz_Functor<FreeEnergy_Functor>{
            FreeEnergy_Functor{fields, (FLUCS_FLOAT)2.0}, dt
        }
    );
}

__global__
void dWpdt_hyperdissipation_kz_kzkx(
    const FLUCS_COMPLEX* fields,
    const FLUCS_FLOAT dt,
    FLUCS_FLOAT* output
) {
    add_and_sum_last_axis<HALF_NY, true>(
        FLOAT_ONE,
        output,
        HyperdissipationKz_Functor<FreeEnergyThetap_Functor>{
            FreeEnergyThetap_Functor{fields, (FLUCS_FLOAT)2.0}, dt
        }
    );
}

__global__
void dWmdt_hyperdissipation_kz_kzkx(
    const FLUCS_COMPLEX* fields,
    const FLUCS_FLOAT dt,
    FLUCS_FLOAT* output
) {
    add_and_sum_last_axis<HALF_NY, true>(
        FLOAT_ONE,
        output,
        HyperdissipationKz_Functor<FreeEnergyThetam_Functor>{
            FreeEnergyThetam_Functor{fields, (FLUCS_FLOAT)2.0}, dt
        }
    );
}

// dWdt_hyperdissipation_perp
__global__
void dWdt_hyperdissipation_perp_kzkx(
    const FLUCS_COMPLEX* fields,
    const FLUCS_FLOAT dt,
    FLUCS_FLOAT* output
) {
    add_and_sum_last_axis<HALF_NY, true>(
        FLOAT_ONE,
        output,
        HyperdissipationPerp_Functor<FreeEnergy_Functor>{
            FreeEnergy_Functor{fields, (FLUCS_FLOAT)2.0}, dt
        }
    );
}

__global__
void dWpdt_hyperdissipation_perp_kzkx(
    const FLUCS_COMPLEX* fields,
    const FLUCS_FLOAT dt,
    FLUCS_FLOAT* output
) {
    add_and_sum_last_axis<HALF_NY, true>(
        FLOAT_ONE,
        output,
        HyperdissipationPerp_Functor<FreeEnergyThetap_Functor>{
            FreeEnergyThetap_Functor{fields, (FLUCS_FLOAT)2.0}, dt
        }
    );
}

__global__
void dWmdt_hyperdissipation_perp_kzkx(
    const FLUCS_COMPLEX* fields,
    const FLUCS_FLOAT dt,
    FLUCS_FLOAT* output
) {
    add_and_sum_last_axis<HALF_NY, true>(
        FLOAT_ONE,
        output,
        HyperdissipationPerp_Functor<FreeEnergyThetam_Functor>{
            FreeEnergyThetam_Functor{fields, (FLUCS_FLOAT)2.0}, dt
        }
    );
}

////////////////////////////////////////////////////////////////////////////////
// Diagnostics: Helicity (H)
////////////////////////////////////////////////////////////////////////////////

struct Helicity_Functor {
    const FLUCS_COMPLEX* fields;
    const FLUCS_FLOAT multiplier;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        const FLUCS_COMPLEX phi = fields[index];
        const FLUCS_COMPLEX apar = fields[index + HALFUNPADDEDSIZE];

        indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
        const size_t ikx = indices.ikx;
        const size_t iky = indices.iky;

        const FLUCS_FLOAT kx = kx_from_ikx(ikx);
        const FLUCS_FLOAT ky = ky_from_iky(iky);
        const FLUCS_FLOAT kperp2 = kx*kx + ky*ky;

        const FLUCS_FLOAT cross_term = (
            phi.real() * apar.real() + phi.imag() * apar.imag()
        );

        const FLUCS_FLOAT helicity = - ((FLUCS_FLOAT)2.0) * (
            one_minus_gamma0_over_alpha(kperp2) * kperp2 
            * (FLOAT_ONE + DE2 * kperp2) * cross_term
        );

        return multiplier * helicity;
    }
};

struct HelicityForcing_Functor {
    const FLUCS_COMPLEX* fields;
    const FLUCS_FLOAT multiplier;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        const FLUCS_COMPLEX phi = fields[index];
        const FLUCS_COMPLEX apar = fields[index + HALFUNPADDEDSIZE];
        FLUCS_COMPLEX forcing_terms[NUMBER_OF_FIELDS] = {0};

        add_forcing_elsasser(index, (FLUCS_FLOAT)0, 0, fields, forcing_terms);

        indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
        const size_t ikx = indices.ikx;
        const size_t iky = indices.iky;

        const FLUCS_FLOAT kx = kx_from_ikx(ikx);
        const FLUCS_FLOAT ky = ky_from_iky(iky);
        const FLUCS_FLOAT kperp2 = kx*kx + ky*ky;

        const FLUCS_FLOAT cross_term = -(
            phi.real() * forcing_terms[1].real() + phi.imag() * forcing_terms[1].imag() +
            apar.real() * forcing_terms[0].real() + apar.imag() * forcing_terms[0].imag()
        );

        const FLUCS_FLOAT helicity = - ((FLUCS_FLOAT)2.0) * (
            one_minus_gamma0_over_alpha(kperp2) * kperp2 
            * (FLOAT_ONE + DE2 * kperp2) * cross_term
        );

        return multiplier * helicity;
    }
};

struct HelicityThetap_Functor {
    const FLUCS_COMPLEX* fields;
    const FLUCS_FLOAT multiplier;

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        // Indices
        indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
        const FLUCS_FLOAT kx = kx_from_ikx(indices.ikx);
        const FLUCS_FLOAT ky = ky_from_iky(indices.iky);
        const FLUCS_FLOAT kperp2 = kx*kx + ky*ky;

        // Thetas
        FLUCS_COMPLEX thetap, thetam;
        FLUCS_FLOAT vphase;
        get_thetas_from_fields(index, fields, thetap, thetam, vphase);

        return multiplier * ((FLUCS_FLOAT)0.5) * kperp2
            * (thetap.real()*thetap.real() + thetap.imag()*thetap.imag())
            / vphase;
    }
};

struct HelicityThetam_Functor {
    const FLUCS_COMPLEX* fields;
    const FLUCS_FLOAT multiplier;

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        // Indices
        indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
        const FLUCS_FLOAT kx = kx_from_ikx(indices.ikx);
        const FLUCS_FLOAT ky = ky_from_iky(indices.iky);
        const FLUCS_FLOAT kperp2 = kx*kx + ky*ky;

        // Thetas
        FLUCS_COMPLEX thetap, thetam;
        FLUCS_FLOAT vphase;
        get_thetas_from_fields(index, fields, thetap, thetam, vphase);

        return multiplier * ((FLUCS_FLOAT)0.5) * kperp2
            * (thetam.real()*thetam.real() + thetam.imag()*thetam.imag())
            / vphase;
    }
};

// H
__global__
void H_kzkx(
    const FLUCS_COMPLEX* fields,
    FLUCS_FLOAT* output
) {
    add_and_sum_last_axis<HALF_NY, true>(
        FLOAT_ONE,
        output,
        Helicity_Functor{fields, FLOAT_ONE}
    );
}

// H forcing
__global__
void dHdt_forcing_kzkx(
    const FLUCS_COMPLEX* fields,
    FLUCS_FLOAT* output
){

    add_and_sum_last_axis<HALF_NY, true>(
            FLOAT_ONE,
            output,
            HelicityForcing_Functor{fields, FLOAT_ONE}
        );

}

__global__
void Hp_kzkx(
    const FLUCS_COMPLEX* fields,
    FLUCS_FLOAT* output
) {
    add_and_sum_last_axis<HALF_NY, true>(
        FLOAT_ONE,
        output,
        HelicityThetap_Functor{fields, FLOAT_ONE}
    );
}

__global__
void Hm_kzkx(
    const FLUCS_COMPLEX* fields,
    FLUCS_FLOAT* output
) {
    add_and_sum_last_axis<HALF_NY, true>(
        FLOAT_ONE,
        output,
        HelicityThetam_Functor{fields, FLOAT_ONE}
    );
}

// dHdt
__global__
void dHdt_kzkx(
    const FLUCS_COMPLEX* fields_now,
    const FLUCS_COMPLEX* fields_prev,
    const FLUCS_FLOAT dt,
    FLUCS_FLOAT* output
) {
    add_and_sum_last_axis<HALF_NY, true>(
        FLOAT_ONE / dt,
        output,
        Helicity_Functor{fields_now, FLOAT_ONE},
        Helicity_Functor{fields_prev, -FLOAT_ONE}
    );
}

__global__
void dHpdt_kzkx(
    const FLUCS_COMPLEX* fields_now,
    const FLUCS_COMPLEX* fields_prev,
    const FLUCS_FLOAT dt,
    FLUCS_FLOAT* output
) {
    add_and_sum_last_axis<HALF_NY, true>(
        FLOAT_ONE / dt,
        output,
        HelicityThetap_Functor{fields_now, FLOAT_ONE},
        HelicityThetap_Functor{fields_prev, -FLOAT_ONE}
    );
}

__global__
void dHmdt_kzkx(
    const FLUCS_COMPLEX* fields_now,
    const FLUCS_COMPLEX* fields_prev,
    const FLUCS_FLOAT dt,
    FLUCS_FLOAT* output
) {
    add_and_sum_last_axis<HALF_NY, true>(
        FLOAT_ONE / dt,
        output,
        HelicityThetam_Functor{fields_now, FLOAT_ONE},
        HelicityThetam_Functor{fields_prev, -FLOAT_ONE}
    );
}

// dHdt_hyperdissipation_kx
__global__
void dHdt_hyperdissipation_kx_kzkx(
    const FLUCS_COMPLEX* fields,
    const FLUCS_FLOAT dt,
    FLUCS_FLOAT* output
) {
    add_and_sum_last_axis<HALF_NY, true>(
        FLOAT_ONE,
        output,
        HyperdissipationKx_Functor<Helicity_Functor>{
            Helicity_Functor{fields, (FLUCS_FLOAT)2.0}, dt
        }
    );
}

__global__
void dHpdt_hyperdissipation_kx_kzkx(
    const FLUCS_COMPLEX* fields,
    const FLUCS_FLOAT dt,
    FLUCS_FLOAT* output
) {
    add_and_sum_last_axis<HALF_NY, true>(
        FLOAT_ONE,
        output,
        HyperdissipationKx_Functor<HelicityThetap_Functor>{
            HelicityThetap_Functor{fields, (FLUCS_FLOAT)2.0}, dt
        }
    );
}

__global__
void dHmdt_hyperdissipation_kx_kzkx(
    const FLUCS_COMPLEX* fields,
    const FLUCS_FLOAT dt,
    FLUCS_FLOAT* output
) {
    add_and_sum_last_axis<HALF_NY, true>(
        FLOAT_ONE,
        output,
        HyperdissipationKx_Functor<HelicityThetam_Functor>{
            HelicityThetam_Functor{fields, (FLUCS_FLOAT)2.0}, dt
        }
    );
}

// dHdt_hyperdissipation_ky
__global__
void dHdt_hyperdissipation_ky_kzkx(
    const FLUCS_COMPLEX* fields,
    const FLUCS_FLOAT dt,
    FLUCS_FLOAT* output
) {
    add_and_sum_last_axis<HALF_NY, true>(
        FLOAT_ONE,
        output,
        HyperdissipationKy_Functor<Helicity_Functor>{
            Helicity_Functor{fields, (FLUCS_FLOAT)2.0}, dt
        }
    );
}

__global__
void dHpdt_hyperdissipation_ky_kzkx(
    const FLUCS_COMPLEX* fields,
    const FLUCS_FLOAT dt,
    FLUCS_FLOAT* output
) {
    add_and_sum_last_axis<HALF_NY, true>(
        FLOAT_ONE,
        output,
        HyperdissipationKy_Functor<HelicityThetap_Functor>{
            HelicityThetap_Functor{fields, (FLUCS_FLOAT)2.0}, dt
        }
    );
}

__global__
void dHmdt_hyperdissipation_ky_kzkx(
    const FLUCS_COMPLEX* fields,
    const FLUCS_FLOAT dt,
    FLUCS_FLOAT* output
) {
    add_and_sum_last_axis<HALF_NY, true>(
        FLOAT_ONE,
        output,
        HyperdissipationKy_Functor<HelicityThetam_Functor>{
            HelicityThetam_Functor{fields, (FLUCS_FLOAT)2.0}, dt
        }
    );
}

// dHdt_hyperdissipation_kz
__global__
void dHdt_hyperdissipation_kz_kzkx(
    const FLUCS_COMPLEX* fields,
    const FLUCS_FLOAT dt,
    FLUCS_FLOAT* output
) {
    add_and_sum_last_axis<HALF_NY, true>(
        FLOAT_ONE,
        output,
        HyperdissipationKz_Functor<Helicity_Functor>{
            Helicity_Functor{fields, (FLUCS_FLOAT)2.0}, dt
        }
    );
}

__global__
void dHpdt_hyperdissipation_kz_kzkx(
    const FLUCS_COMPLEX* fields,
    const FLUCS_FLOAT dt,
    FLUCS_FLOAT* output
) {
    add_and_sum_last_axis<HALF_NY, true>(
        FLOAT_ONE,
        output,
        HyperdissipationKz_Functor<HelicityThetap_Functor>{
            HelicityThetap_Functor{fields, (FLUCS_FLOAT)2.0}, dt
        }
    );
}

__global__
void dHmdt_hyperdissipation_kz_kzkx(
    const FLUCS_COMPLEX* fields,
    const FLUCS_FLOAT dt,
    FLUCS_FLOAT* output
) {
    add_and_sum_last_axis<HALF_NY, true>(
        FLOAT_ONE,
        output,
        HyperdissipationKz_Functor<HelicityThetam_Functor>{
            HelicityThetam_Functor{fields, (FLUCS_FLOAT)2.0}, dt
        }
    );
}

// dHdt_hyperdissipation_perp
__global__
void dHdt_hyperdissipation_perp_kzkx(
    const FLUCS_COMPLEX* fields,
    const FLUCS_FLOAT dt,
    FLUCS_FLOAT* output
) {
    add_and_sum_last_axis<HALF_NY, true>(
        FLOAT_ONE,
        output,
        HyperdissipationPerp_Functor<Helicity_Functor>{
            Helicity_Functor{fields, (FLUCS_FLOAT)2.0}, dt
        }
    );
}

__global__
void dHpdt_hyperdissipation_perp_kzkx(
    const FLUCS_COMPLEX* fields,
    const FLUCS_FLOAT dt,
    FLUCS_FLOAT* output
) {
    add_and_sum_last_axis<HALF_NY, true>(
        FLOAT_ONE,
        output,
        HyperdissipationPerp_Functor<HelicityThetap_Functor>{
            HelicityThetap_Functor{fields, (FLUCS_FLOAT)2.0}, dt
        }
    );
}

__global__
void dHmdt_hyperdissipation_perp_kzkx(
    const FLUCS_COMPLEX* fields,
    const FLUCS_FLOAT dt,
    FLUCS_FLOAT* output
) {
    add_and_sum_last_axis<HALF_NY, true>(
        FLOAT_ONE,
        output,
        HyperdissipationPerp_Functor<HelicityThetam_Functor>{
            HelicityThetam_Functor{fields, (FLUCS_FLOAT)2.0}, dt
        }
    );
}

} // extern "C"
