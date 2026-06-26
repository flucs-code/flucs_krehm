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

    const FLUCS_FLOAT dxapar = in_bounds
        ? real_derivatives_and_bits[real_index + 2*PADDEDSIZE]
        : (FLUCS_FLOAT)0;
    const FLUCS_FLOAT dyapar = in_bounds
        ? real_derivatives_and_bits[real_index + 3*PADDEDSIZE]
        : (FLUCS_FLOAT)0;

    const FLUCS_FLOAT cfl_phi = flucs_fabs(dxphi) * (NY / LY)
        + flucs_fabs(dyphi) * (NX / LX);

    const FLUCS_FLOAT cfl_apar = flucs_fabs(dxapar) * (NY / LY)
        + flucs_fabs(dyapar) * (NX / LX);

    // This works fine for de = 0, but we might need to 
    // include a higher-order perp derivative in CFL
    // when running with finite de
    const FLUCS_FLOAT cfl = cfl_phi + cfl_apar;

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
// Model helper functions
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
void get_thetas_from_components(
    const size_t index,
    const FLUCS_COMPLEX phi,
    const FLUCS_COMPLEX apar,
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

    // Useful intermediate quantities
    const FLUCS_FLOAT gamma_factor = one_minus_gamma0_over_alpha(kperp2);
    vphase = get_phase_velocity(kperp2, gamma_factor);

    // Construct Elsasser potentials
    const FLUCS_FLOAT phi_factor = vphase * gamma_factor;
    const FLUCS_FLOAT prefactor = sqrt(FLOAT_ONE + kperp2 * DE2);

    thetap = prefactor * (phi_factor * phi + apar);
    thetam = prefactor * (phi_factor * phi - apar);
}

__device__ __forceinline__
void get_thetas_from_fields(
    const size_t index,
    const FLUCS_COMPLEX* fields,
    FLUCS_COMPLEX& thetap,
    FLUCS_COMPLEX& thetam,
    FLUCS_FLOAT& vphase
){
    // Fields
    const FLUCS_COMPLEX phi = fields[index];
    const FLUCS_COMPLEX apar = fields[index + HALFUNPADDEDSIZE];

    get_thetas_from_components(index, phi, apar, thetap, thetam, vphase);
}

__device__ __forceinline__
FLUCS_FLOAT get_thetas_free_energy_rate(
    const size_t index,
    const FLUCS_COMPLEX* fields,
    const FLUCS_COMPLEX rates[2],
    const int theta_sign
){
    // Indices
    indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
    const size_t ikx = indices.ikx;
    const size_t iky = indices.iky;

    // Wavenumbers
    const FLUCS_FLOAT kx = kx_from_ikx(ikx);
    const FLUCS_FLOAT ky = ky_from_iky(iky);

    const FLUCS_FLOAT kperp2 = kx*kx + ky*ky;

    // Thetas
    FLUCS_COMPLEX thetap, thetam;
    FLUCS_FLOAT vphase;
    get_thetas_from_fields(index, fields, thetap, thetam, vphase);

    // Rates
    const FLUCS_COMPLEX phi_rate = rates[0];
    const FLUCS_COMPLEX apar_rate = rates[1];

    // Useful intermediate quantities
    const FLUCS_FLOAT gamma_factor = one_minus_gamma0_over_alpha(kperp2);
    const FLUCS_FLOAT phi_factor = vphase * gamma_factor;
    const FLUCS_FLOAT prefactor = sqrt(FLOAT_ONE + kperp2 * DE2);

    const FLUCS_COMPLEX theta = theta_sign > 0 ? thetap : thetam;
    const FLUCS_COMPLEX theta_rate = prefactor * (
        phi_factor * phi_rate + ((FLUCS_FLOAT)theta_sign) * apar_rate
    );

    return kperp2 * (
        theta.real() * theta_rate.real()
        + theta.imag() * theta_rate.imag()
    );
}

////////////////////////////////////////////////////////////////////////////////
// Forcing
////////////////////////////////////////////////////////////////////////////////

#ifdef FORCING

// The ky=0 modes are stored together with their conjugate partners. Construct
// forcing from their physical, conjugate-symmetric component so negative
// damping does not amplify roundoff.
__device__ __forceinline__
void get_forcing_fields(
    const size_t index,
    const FLUCS_COMPLEX* fields,
    FLUCS_COMPLEX& phi,
    FLUCS_COMPLEX& apar
){
    const indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);

    phi = fields[index];
    apar = fields[index + HALFUNPADDEDSIZE];

    if (indices.iky != 0)
        return;

    const size_t conjugate_ikz = indices.ikz == 0 ? 0 : NZ - indices.ikz;
    const size_t conjugate_ikx = indices.ikx == 0 ? 0 : NX - indices.ikx;
    const size_t conjugate_index = index_from_3d<NZ, NX, HALF_NY>(
        conjugate_ikz, conjugate_ikx, 0
    );

    phi = ((FLUCS_FLOAT)0.5) * (
        phi
        + conj(fields[conjugate_index])
    );
    apar = ((FLUCS_FLOAT)0.5) * (
        apar
        + conj(fields[conjugate_index + HALFUNPADDEDSIZE])
    );
}

#if defined(FORCING_METHOD_ELSASSER)
__device__ __forceinline__
void add_forcing_elsasser(
    const size_t index,
    const FLUCS_FLOAT dt,
    const long long current_step,
    const FLUCS_COMPLEX* previous_fields,
    FLUCS_COMPLEX explicit_terms[2]
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
    const FLUCS_FLOAT kz_abs = flucs_fabs(kz);

    if (kperp2 == ((FLUCS_FLOAT)0.0))
        return;
    
    if (!(kperp2 > FORCING_KPERP2_MIN &&
          kperp2 < FORCING_KPERP2_MAX &&
          kz_abs > FORCING_KZ_MIN &&
          kz_abs < FORCING_KZ_MAX))
        return;

    // Fields
    FLUCS_COMPLEX phi, apar;
    get_forcing_fields(index, previous_fields, phi, apar);

    // Matrices
    FLUCS_COMPLEX thetap, thetam;
    FLUCS_FLOAT vphase;
    get_thetas_from_components(index, phi, apar, thetap, thetam, vphase);

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
#endif // FORCING_METHOD_ELSASSER

#if defined(FORCING_METHOD_MEYRAND)
__device__ __forceinline__
void add_forcing_meyrand(
    const size_t index,
    const FLUCS_FLOAT dt,
    const long long current_step,
    const FLUCS_COMPLEX* previous_fields,
    FLUCS_COMPLEX explicit_terms[2]
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
    const FLUCS_FLOAT one_plus_kperp2de2 = FLOAT_ONE + kperp2 * DE2;
    const FLUCS_FLOAT kz_abs = flucs_fabs(kz);

    if (kperp2 == ((FLUCS_FLOAT)0.0))
        return;
    
    if (!(kperp2 > FORCING_KPERP2_MIN &&
          kperp2 < FORCING_KPERP2_MAX &&
          kz_abs > FORCING_KZ_MIN &&
          kz_abs < FORCING_KZ_MAX))
        return;

    // Various factors that appear in the terms below
    const FLUCS_FLOAT gamma_factor = one_minus_gamma0_over_alpha(kperp2);
    const FLUCS_FLOAT one_plus_taubarinv = FLOAT_ONE + taubarinv(kperp2);
    const FLUCS_FLOAT taubarinv_over_kperp2de2_factor = one_plus_taubarinv / one_plus_kperp2de2;
    const FLUCS_FLOAT helicity_prefactor = 2 * gamma_factor * kperp2 * one_plus_kperp2de2;

    // Fields
    FLUCS_COMPLEX phi, apar;
    get_forcing_fields(index, previous_fields, phi, apar);

    // Useful combinations of fields
    const FLUCS_FLOAT phi2 = (
        phi.real()*phi.real() + phi.imag()*phi.imag()
    );
    const FLUCS_FLOAT apar2 = (
        apar.real()*apar.real() + apar.imag()*apar.imag()
    );
    const FLUCS_FLOAT real_phi_conj_apar = (
        phi.real() * apar.real() + phi.imag() * apar.imag()
    );

    const FLUCS_FLOAT a00 = helicity_prefactor * apar2;
    const FLUCS_FLOAT a10 = -helicity_prefactor * real_phi_conj_apar;
    const FLUCS_FLOAT a01 = taubarinv_over_kperp2de2_factor * a10;
    const FLUCS_FLOAT a11 = helicity_prefactor * taubarinv_over_kperp2de2_factor * phi2;

    const FLUCS_FLOAT det = a00 * a11 - a10 * a01;

    if (flucs_fabs(det) < FLUCS_EPSILON * (
            a00*a00 + a01*a01 + a10*a10 + a11*a11
        )
    ) {
        // Don't do anything if det is too small
        // __trap();  // useful for debugging
        return;
    }

    const FLUCS_FLOAT inv_det = FLOAT_ONE / det;

    // Forcing matrix
    const FLUCS_FLOAT m00 = (a00 + FORCING_IMBALANCE * a01) * inv_det;
    const FLUCS_FLOAT m01 = (a10 + FORCING_IMBALANCE * a11) * inv_det;

    const FLUCS_FLOAT m10 = (FORCING_IMBALANCE * a00 * taubarinv_over_kperp2de2_factor + a01 * gamma_factor) * inv_det;
    const FLUCS_FLOAT m11 = (FORCING_IMBALANCE * a10 * taubarinv_over_kperp2de2_factor + a11 * gamma_factor) * inv_det;

    // Construct forcing
    explicit_terms[0] -= FORCING_EPSILON_PHI  * (m00 * phi + m01 * apar);
    explicit_terms[1] -= FORCING_EPSILON_APAR * (m10 * phi + m11 * apar);
}
#endif // FORCING_METHOD_MEYRAND

__device__ void add_forcing_explicit(
    const size_t index,
    const FLUCS_FLOAT dt, 
    const long long current_step,
    const FLUCS_COMPLEX* previous_fields,
    FLUCS_COMPLEX explicit_terms[2] 
){
    #if defined(FORCING_METHOD_ELSASSER)
        add_forcing_elsasser(
            index, dt, current_step, previous_fields, explicit_terms
        );
    #endif

    #if defined(FORCING_METHOD_MEYRAND)
        add_forcing_meyrand(
            index, dt, current_step, previous_fields, explicit_terms
        );
    #endif
}

#endif // FORCING


////////////////////////////////////////////////////////////////////////////////
// Diagnostics: Free Energy
////////////////////////////////////////////////////////////////////////////////

struct FreeEnergy_Functor {
    const FLUCS_COMPLEX* fields;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        // Fields
        const FLUCS_COMPLEX phi = fields[index];
        const FLUCS_COMPLEX apar = fields[index + HALFUNPADDEDSIZE];

        // Indices and wavenumbers
        indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
        const size_t ikx = indices.ikx;
        const size_t iky = indices.iky;

        const FLUCS_FLOAT kx = kx_from_ikx(ikx);
        const FLUCS_FLOAT ky = ky_from_iky(iky);
        const FLUCS_FLOAT kperp2 = kx*kx + ky*ky;

        // Contributions
        const FLUCS_FLOAT phi_contribution = (
            (1 + taubarinv(kperp2)) * one_minus_gamma0_over_alpha(kperp2) * kperp2 
        ) * (phi.real() * phi.real() + phi.imag() * phi.imag());

        const FLUCS_FLOAT apar_contribution = (
            kperp2 * (1 + DE2 * kperp2)
        ) * (apar.real() * apar.real() + apar.imag() * apar.imag());

        return phi_contribution + apar_contribution;
    }
};

struct FreeEnergyUperp_Functor {
    const FLUCS_COMPLEX* fields;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        // Field
        const FLUCS_COMPLEX phi = fields[index];

        // Indices and wavenumbers
        indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);

        const FLUCS_FLOAT kx = kx_from_ikx(indices.ikx);
        const FLUCS_FLOAT ky = ky_from_iky(indices.iky);
        const FLUCS_FLOAT kperp2 = kx*kx + ky*ky;

        // Result
        return one_minus_gamma0_over_alpha(kperp2) * kperp2 * (
            phi.real()*phi.real() + phi.imag()*phi.imag()
        );
    }
};

struct FreeEnergyDens_Functor {
    const FLUCS_COMPLEX* fields;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        // Field
        const FLUCS_COMPLEX phi = fields[index];

        // Indices and wavenumbers
        indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
        const FLUCS_FLOAT kx = kx_from_ikx(indices.ikx);
        const FLUCS_FLOAT ky = ky_from_iky(indices.iky);
        const FLUCS_FLOAT kperp2 = kx*kx + ky*ky;

        // Result
        return taubarinv(kperp2) * one_minus_gamma0_over_alpha(kperp2) 
            * kperp2 * (phi.real()*phi.real() + phi.imag()*phi.imag());
    }
};

struct FreeEnergyBperp_Functor {
    const FLUCS_COMPLEX* fields;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        // Field
        const FLUCS_COMPLEX apar = fields[index + HALFUNPADDEDSIZE];

        // Indices and wavenumbers
        indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);

        const FLUCS_FLOAT kx = kx_from_ikx(indices.ikx);
        const FLUCS_FLOAT ky = ky_from_iky(indices.iky);
        const FLUCS_FLOAT kperp2 = kx*kx + ky*ky;

        // Result
        return kperp2 * (apar.real()*apar.real() + apar.imag()*apar.imag());
    }
};

struct FreeEnergyUpar_Functor {
    const FLUCS_COMPLEX* fields;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        // Field
        const FLUCS_COMPLEX apar = fields[index + HALFUNPADDEDSIZE];

        // Indices and wavenumbers
        indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);

        const FLUCS_FLOAT kx = kx_from_ikx(indices.ikx);
        const FLUCS_FLOAT ky = ky_from_iky(indices.iky);
        const FLUCS_FLOAT kperp2 = kx*kx + ky*ky;
        
        // Result
        return DE2 * kperp2 * kperp2 * (
            apar.real()*apar.real() + apar.imag()*apar.imag()
        );
    }
};

struct FreeEnergyForcing_Functor {
    const FLUCS_COMPLEX* fields;
    const FLUCS_FLOAT dt;
    const long long current_step;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        // Fields
        const FLUCS_COMPLEX phi = fields[index];
        const FLUCS_COMPLEX apar = fields[index + HALFUNPADDEDSIZE];

        // Forcing terms
        FLUCS_COMPLEX forcing_terms[2] = {0};
#ifdef FORCING_EXPLICIT
        add_forcing_explicit(
            index, dt, current_step, fields, forcing_terms
        );
#endif

        // Indices and wavenumbers
        indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
        const size_t ikx = indices.ikx;
        const size_t iky = indices.iky;

        const FLUCS_FLOAT kx = kx_from_ikx(ikx);
        const FLUCS_FLOAT ky = ky_from_iky(iky);
        const FLUCS_FLOAT kperp2 = kx*kx + ky*ky;

        // Contributions
        const FLUCS_FLOAT phi_contribution = -2.0 * (
            (1 + taubarinv(kperp2)) * one_minus_gamma0_over_alpha(kperp2) * kperp2 
        ) * (phi.real() * forcing_terms[0].real() + phi.imag() * forcing_terms[0].imag());

        const FLUCS_FLOAT apar_contribution = -2.0 * (
            kperp2 * (1 + DE2 * kperp2)
        ) * (apar.real() * forcing_terms[1].real() + apar.imag() * forcing_terms[1].imag());

        return phi_contribution + apar_contribution;
    }
};

struct FreeEnergyNonlinear_Functor {
    const FLUCS_COMPLEX* fields;
    const FLUCS_COMPLEX* dft_bits;

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        // Fields
        const FLUCS_COMPLEX phi = fields[index];
        const FLUCS_COMPLEX apar = fields[index + HALFUNPADDEDSIZE];

        // Nonlinear terms
        FLUCS_COMPLEX nonlinear_terms[2] = {0};
#ifdef NONLINEAR
        add_nonlinear_terms(index, dft_bits, nonlinear_terms);
#endif

        // Indices and wavenumbers
        indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
        const size_t ikx = indices.ikx;
        const size_t iky = indices.iky;

        const FLUCS_FLOAT kx = kx_from_ikx(ikx);
        const FLUCS_FLOAT ky = ky_from_iky(iky);
        const FLUCS_FLOAT kperp2 = kx*kx + ky*ky;

        // Contributions
        const FLUCS_FLOAT phi_contribution = -2.0 * (
            (1 + taubarinv(kperp2)) * one_minus_gamma0_over_alpha(kperp2) * kperp2
        ) * (
            phi.real() * nonlinear_terms[0].real()
          + phi.imag() * nonlinear_terms[0].imag()
        );

        const FLUCS_FLOAT apar_contribution = -2.0 * (
            kperp2 * (1 + DE2 * kperp2)
        ) * (
            apar.real() * nonlinear_terms[1].real()
          + apar.imag() * nonlinear_terms[1].imag()
        );

        return phi_contribution + apar_contribution;
    }
};

struct FreeEnergyHyperdissipation_Functor {
    const FLUCS_COMPLEX* fields;
    const FLUCS_FLOAT adaptive_rate;

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        return (FLUCS_FLOAT)2.0
            * Hyperdissipation_Functor<FreeEnergy_Functor>{
                FreeEnergy_Functor{fields},
                adaptive_rate
            }(index);
    }
};

struct FreeEnergyHyperdissipationComponent_Functor {
    const FLUCS_COMPLEX* fields;
    const FLUCS_FLOAT adaptive_rate;
    const int hyperdissipation_type;

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        return (FLUCS_FLOAT)2.0
            * HyperdissipationSelector_Functor<FreeEnergy_Functor>{
                FreeEnergy_Functor{fields},
                adaptive_rate,
                hyperdissipation_type
            }(index);
    }
};

struct FreeEnergyThetap_Functor {
    const FLUCS_COMPLEX* fields;
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

        return ((FLUCS_FLOAT)0.5) * kperp2
            * (thetap.real()*thetap.real() + thetap.imag()*thetap.imag());
    }
};

struct FreeEnergyThetapForcing_Functor {
    const FLUCS_COMPLEX* fields;
    const FLUCS_FLOAT dt;
    const long long current_step;

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        // Forcing terms
        FLUCS_COMPLEX forcing_terms[2] = {0};
#ifdef FORCING_EXPLICIT
        add_forcing_explicit(
            index, dt, current_step, fields, forcing_terms
        );
#endif

        // Physical rates due to forcing
        FLUCS_COMPLEX rates[2] = {0};
        rates[0] = -forcing_terms[0];
        rates[1] = -forcing_terms[1];

        return get_thetas_free_energy_rate(
            index, fields, rates, +1
        );
    }
};

struct FreeEnergyThetapNonlinear_Functor {
    const FLUCS_COMPLEX* fields;
    const FLUCS_COMPLEX* dft_bits;

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        // Nonlinear terms
        FLUCS_COMPLEX nonlinear_terms[2] = {0};
#ifdef NONLINEAR
        add_nonlinear_terms(index, dft_bits, nonlinear_terms);
#endif

        // Physical rates due to nonlinear terms
        FLUCS_COMPLEX rates[2] = {0};
        rates[0] = -nonlinear_terms[0];
        rates[1] = -nonlinear_terms[1];

        return get_thetas_free_energy_rate(
            index, fields, rates, +1
        );
    }
};

struct FreeEnergyThetapHyperdissipation_Functor {
    const FLUCS_COMPLEX* fields;
    const FLUCS_FLOAT adaptive_rate;

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        return (FLUCS_FLOAT)2.0
            * Hyperdissipation_Functor<FreeEnergyThetap_Functor>{
                FreeEnergyThetap_Functor{fields},
                adaptive_rate
            }(index);
    }
};

struct FreeEnergyThetapHyperdissipationComponent_Functor {
    const FLUCS_COMPLEX* fields;
    const FLUCS_FLOAT adaptive_rate;
    const int hyperdissipation_type;

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        return (FLUCS_FLOAT)2.0
            * HyperdissipationSelector_Functor<FreeEnergyThetap_Functor>{
                FreeEnergyThetap_Functor{fields},
                adaptive_rate,
                hyperdissipation_type
            }(index);
    }
};

struct FreeEnergyThetam_Functor {
    const FLUCS_COMPLEX* fields;
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

        return ((FLUCS_FLOAT)0.5) * kperp2
            * (thetam.real()*thetam.real() + thetam.imag()*thetam.imag());
    }
};

struct FreeEnergyThetamForcing_Functor {
    const FLUCS_COMPLEX* fields;
    const FLUCS_FLOAT dt;
    const long long current_step;

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        // Forcing terms
        FLUCS_COMPLEX forcing_terms[2] = {0};
#ifdef FORCING_EXPLICIT
        add_forcing_explicit(
            index, dt, current_step, fields, forcing_terms
        );
#endif

        // Physical rates due to forcing
        FLUCS_COMPLEX rates[2] = {0};
        rates[0] = -forcing_terms[0];
        rates[1] = -forcing_terms[1];

        return get_thetas_free_energy_rate(
            index, fields, rates, -1
        );
    }
};

struct FreeEnergyThetamNonlinear_Functor {
    const FLUCS_COMPLEX* fields;
    const FLUCS_COMPLEX* dft_bits;

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        // Nonlinear terms
        FLUCS_COMPLEX nonlinear_terms[2] = {0};
#ifdef NONLINEAR
        add_nonlinear_terms(index, dft_bits, nonlinear_terms);
#endif

        // Physical rates due to nonlinear terms
        FLUCS_COMPLEX rates[2] = {0};
        rates[0] = -nonlinear_terms[0];
        rates[1] = -nonlinear_terms[1];

        return get_thetas_free_energy_rate(
            index, fields, rates, -1
        );
    }
};

struct FreeEnergyThetamHyperdissipation_Functor {
    const FLUCS_COMPLEX* fields;
    const FLUCS_FLOAT adaptive_rate;

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        return (FLUCS_FLOAT)2.0
            * Hyperdissipation_Functor<FreeEnergyThetam_Functor>{
                FreeEnergyThetam_Functor{fields},
                adaptive_rate
            }(index);
    }
};

struct FreeEnergyThetamHyperdissipationComponent_Functor {
    const FLUCS_COMPLEX* fields;
    const FLUCS_FLOAT adaptive_rate;
    const int hyperdissipation_type;

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        return (FLUCS_FLOAT)2.0
            * HyperdissipationSelector_Functor<FreeEnergyThetam_Functor>{
                FreeEnergyThetam_Functor{fields},
                adaptive_rate,
                hyperdissipation_type
            }(index);
    }
};

////////////////////////////////////////////////////////////////////////////////
// Diagnostics: Helicity
////////////////////////////////////////////////////////////////////////////////

struct Helicity_Functor {
    const FLUCS_COMPLEX* fields;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        // Fields
        const FLUCS_COMPLEX phi = fields[index];
        const FLUCS_COMPLEX apar = fields[index + HALFUNPADDEDSIZE];

        // Indices and wavenumbers
        indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
        const size_t ikx = indices.ikx;
        const size_t iky = indices.iky;

        const FLUCS_FLOAT kx = kx_from_ikx(ikx);
        const FLUCS_FLOAT ky = ky_from_iky(iky);
        const FLUCS_FLOAT kperp2 = kx*kx + ky*ky;

        // Helicity
        const FLUCS_FLOAT cross_term = (
            phi.real() * apar.real() + phi.imag() * apar.imag()
        );

        const FLUCS_FLOAT helicity = + ((FLUCS_FLOAT)2.0) * (
            one_minus_gamma0_over_alpha(kperp2) * kperp2 
            * (FLOAT_ONE + DE2 * kperp2) * cross_term
        );

        return helicity;
    }
};

struct HelicityApar_Functor {
    const FLUCS_COMPLEX* fields;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        // Fields
        const FLUCS_COMPLEX phi = fields[index];
        const FLUCS_COMPLEX apar = fields[index + HALFUNPADDEDSIZE];

        // Indices and wavenumbers
        indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
        const size_t ikx = indices.ikx;
        const size_t iky = indices.iky;

        const FLUCS_FLOAT kx = kx_from_ikx(ikx);
        const FLUCS_FLOAT ky = ky_from_iky(iky);
        const FLUCS_FLOAT kperp2 = kx*kx + ky*ky;

        // Helicity
        const FLUCS_FLOAT cross_term = (
            phi.real() * apar.real() + phi.imag() * apar.imag()
        );

        const FLUCS_FLOAT helicity = + ((FLUCS_FLOAT)2.0) * (
            one_minus_gamma0_over_alpha(kperp2) * kperp2 
            * FLOAT_ONE * cross_term
        );

        return helicity;
    }
};

struct HelicityUpar_Functor {
    const FLUCS_COMPLEX* fields;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        // Fields
        const FLUCS_COMPLEX phi = fields[index];
        const FLUCS_COMPLEX apar = fields[index + HALFUNPADDEDSIZE];

        // Indices and wavenumbers
        indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
        const size_t ikx = indices.ikx;
        const size_t iky = indices.iky;

        const FLUCS_FLOAT kx = kx_from_ikx(ikx);
        const FLUCS_FLOAT ky = ky_from_iky(iky);
        const FLUCS_FLOAT kperp2 = kx*kx + ky*ky;

        // Helicity
        const FLUCS_FLOAT cross_term = (
            phi.real() * apar.real() + phi.imag() * apar.imag()
        );

        const FLUCS_FLOAT helicity = + ((FLUCS_FLOAT)2.0) * (
            one_minus_gamma0_over_alpha(kperp2) * kperp2 
            * DE2 * kperp2 * cross_term
        );

        return helicity;
    }
};

struct HelicityForcing_Functor {
    const FLUCS_COMPLEX* fields;
    const FLUCS_FLOAT dt;
    const long long current_step;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        // Fields
        const FLUCS_COMPLEX phi = fields[index];
        const FLUCS_COMPLEX apar = fields[index + HALFUNPADDEDSIZE];

        // Forcing terms
        FLUCS_COMPLEX forcing_terms[2] = {0};
#ifdef FORCING_EXPLICIT
        add_forcing_explicit(
            index, dt, current_step, fields, forcing_terms
        );
#endif

        // Indices and wavenumbers
        indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
        const size_t ikx = indices.ikx;
        const size_t iky = indices.iky;

        const FLUCS_FLOAT kx = kx_from_ikx(ikx);
        const FLUCS_FLOAT ky = ky_from_iky(iky);
        const FLUCS_FLOAT kperp2 = kx*kx + ky*ky;

        // Helicity forcing
        const FLUCS_FLOAT cross_term = -(
            phi.real() * forcing_terms[1].real() + phi.imag() * forcing_terms[1].imag() +
            apar.real() * forcing_terms[0].real() + apar.imag() * forcing_terms[0].imag()
        );

        const FLUCS_FLOAT helicity = + ((FLUCS_FLOAT)2.0) * (
            one_minus_gamma0_over_alpha(kperp2) * kperp2 
            * (FLOAT_ONE + DE2 * kperp2) * cross_term
        );

        return helicity;
    }
};

struct HelicityNonlinear_Functor {
    const FLUCS_COMPLEX* fields;
    const FLUCS_COMPLEX* dft_bits;

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        // Fields
        const FLUCS_COMPLEX phi = fields[index];
        const FLUCS_COMPLEX apar = fields[index + HALFUNPADDEDSIZE];

        // Nonlinear terms
        FLUCS_COMPLEX nonlinear_terms[2] = {0};
#ifdef NONLINEAR
        add_nonlinear_terms(index, dft_bits, nonlinear_terms);
#endif

        // Indices and wavenumbers
        indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
        const size_t ikx = indices.ikx;
        const size_t iky = indices.iky;

        const FLUCS_FLOAT kx = kx_from_ikx(ikx);
        const FLUCS_FLOAT ky = ky_from_iky(iky);
        const FLUCS_FLOAT kperp2 = kx*kx + ky*ky;

        // Helicity nonlinear rate
        const FLUCS_FLOAT cross_term = -(
              phi.real()  * nonlinear_terms[1].real()
            + phi.imag()  * nonlinear_terms[1].imag()
            + apar.real() * nonlinear_terms[0].real()
            + apar.imag() * nonlinear_terms[0].imag()
        );

        return ((FLUCS_FLOAT)2.0)
            * one_minus_gamma0_over_alpha(kperp2) * kperp2
            * (FLOAT_ONE + DE2 * kperp2) * cross_term;
    }
};

struct HelicityHyperdissipation_Functor {
    const FLUCS_COMPLEX* fields;
    const FLUCS_FLOAT adaptive_rate;

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        return (FLUCS_FLOAT)2.0
            * Hyperdissipation_Functor<Helicity_Functor>{
                Helicity_Functor{fields},
                adaptive_rate
            }(index);
    }
};

struct HelicityHyperdissipationComponent_Functor {
    const FLUCS_COMPLEX* fields;
    const FLUCS_FLOAT adaptive_rate;
    const int hyperdissipation_type;

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        return (FLUCS_FLOAT)2.0
            * HyperdissipationSelector_Functor<Helicity_Functor>{
                Helicity_Functor{fields},
                adaptive_rate,
                hyperdissipation_type
            }(index);
    }
};

struct HelicityThetap_Functor {
    const FLUCS_COMPLEX* fields;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        // Indices
        indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
        const FLUCS_FLOAT kx = kx_from_ikx(indices.ikx);
        const FLUCS_FLOAT ky = ky_from_iky(indices.iky);
        const FLUCS_FLOAT kperp2 = kx*kx + ky*ky;

        // Phase velocity
        const FLUCS_FLOAT gamma_factor =
            one_minus_gamma0_over_alpha(kperp2);
        const FLUCS_FLOAT vphase =
            get_phase_velocity(kperp2, gamma_factor);

        return FreeEnergyThetap_Functor{fields}(index) / vphase;
    }
};

struct HelicityThetapForcing_Functor {
    const FLUCS_COMPLEX* fields;
    const FLUCS_FLOAT dt;
    const long long current_step;

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        // Indices
        indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
        const FLUCS_FLOAT kx = kx_from_ikx(indices.ikx);
        const FLUCS_FLOAT ky = ky_from_iky(indices.iky);
        const FLUCS_FLOAT kperp2 = kx*kx + ky*ky;

        // Phase velocity
        const FLUCS_FLOAT gamma_factor =
            one_minus_gamma0_over_alpha(kperp2);
        const FLUCS_FLOAT vphase =
            get_phase_velocity(kperp2, gamma_factor);

        return FreeEnergyThetapForcing_Functor{
            fields, dt, current_step
        }(index) / vphase;
    }
};

struct HelicityThetapNonlinear_Functor {
    const FLUCS_COMPLEX* fields;
    const FLUCS_COMPLEX* dft_bits;

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        // Indices
        indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
        const FLUCS_FLOAT kx = kx_from_ikx(indices.ikx);
        const FLUCS_FLOAT ky = ky_from_iky(indices.iky);
        const FLUCS_FLOAT kperp2 = kx*kx + ky*ky;

        // Phase velocity
        const FLUCS_FLOAT gamma_factor =
            one_minus_gamma0_over_alpha(kperp2);
        const FLUCS_FLOAT vphase =
            get_phase_velocity(kperp2, gamma_factor);

        return FreeEnergyThetapNonlinear_Functor{
            fields, dft_bits
        }(index) / vphase;
    }
};

struct HelicityThetapHyperdissipation_Functor {
    const FLUCS_COMPLEX* fields;
    const FLUCS_FLOAT adaptive_rate;

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        return (FLUCS_FLOAT)2.0
            * Hyperdissipation_Functor<HelicityThetap_Functor>{
                HelicityThetap_Functor{fields},
                adaptive_rate
            }(index);
    }
};

struct HelicityThetapHyperdissipationComponent_Functor {
    const FLUCS_COMPLEX* fields;
    const FLUCS_FLOAT adaptive_rate;
    const int hyperdissipation_type;

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        return (FLUCS_FLOAT)2.0
            * HyperdissipationSelector_Functor<HelicityThetap_Functor>{
                HelicityThetap_Functor{fields},
                adaptive_rate,
                hyperdissipation_type
            }(index);
    }
};

struct HelicityThetam_Functor {
    const FLUCS_COMPLEX* fields;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        // Indices
        indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
        const FLUCS_FLOAT kx = kx_from_ikx(indices.ikx);
        const FLUCS_FLOAT ky = ky_from_iky(indices.iky);
        const FLUCS_FLOAT kperp2 = kx*kx + ky*ky;

        // Phase velocity
        const FLUCS_FLOAT gamma_factor =
            one_minus_gamma0_over_alpha(kperp2);
        const FLUCS_FLOAT vphase =
            get_phase_velocity(kperp2, gamma_factor);

        return FreeEnergyThetam_Functor{fields}(index) / vphase;
    }
};

struct HelicityThetamForcing_Functor {
    const FLUCS_COMPLEX* fields;
    const FLUCS_FLOAT dt;
    const long long current_step;

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        // Indices
        indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
        const FLUCS_FLOAT kx = kx_from_ikx(indices.ikx);
        const FLUCS_FLOAT ky = ky_from_iky(indices.iky);
        const FLUCS_FLOAT kperp2 = kx*kx + ky*ky;

        // Phase velocity
        const FLUCS_FLOAT gamma_factor =
            one_minus_gamma0_over_alpha(kperp2);
        const FLUCS_FLOAT vphase =
            get_phase_velocity(kperp2, gamma_factor);

        return FreeEnergyThetamForcing_Functor{
            fields, dt, current_step
        }(index) / vphase;
    }
};

struct HelicityThetamNonlinear_Functor {
    const FLUCS_COMPLEX* fields;
    const FLUCS_COMPLEX* dft_bits;

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        // Indices
        indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
        const FLUCS_FLOAT kx = kx_from_ikx(indices.ikx);
        const FLUCS_FLOAT ky = ky_from_iky(indices.iky);
        const FLUCS_FLOAT kperp2 = kx*kx + ky*ky;

        // Phase velocity
        const FLUCS_FLOAT gamma_factor =
            one_minus_gamma0_over_alpha(kperp2);
        const FLUCS_FLOAT vphase =
            get_phase_velocity(kperp2, gamma_factor);

        return FreeEnergyThetamNonlinear_Functor{
            fields, dft_bits
        }(index) / vphase;
    }
};

struct HelicityThetamHyperdissipation_Functor {
    const FLUCS_COMPLEX* fields;
    const FLUCS_FLOAT adaptive_rate;

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        return (FLUCS_FLOAT)2.0
            * Hyperdissipation_Functor<HelicityThetam_Functor>{
                HelicityThetam_Functor{fields},
                adaptive_rate
            }(index);
    }
};

struct HelicityThetamHyperdissipationComponent_Functor {
    const FLUCS_COMPLEX* fields;
    const FLUCS_FLOAT adaptive_rate;
    const int hyperdissipation_type;

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        return (FLUCS_FLOAT)2.0
            * HyperdissipationSelector_Functor<HelicityThetam_Functor>{
                HelicityThetam_Functor{fields},
                adaptive_rate,
                hyperdissipation_type
            }(index);
    }
};

} // extern "C"
