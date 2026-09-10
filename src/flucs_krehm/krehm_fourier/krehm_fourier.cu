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

__device__ __forceinline__
FLUCS_FLOAT one_minus_gamma0_over_alpha(FLUCS_FLOAT kperp2) {

// ERMHD variant that sets gamma0 to zero
#ifdef ERMHD
    const FLUCS_FLOAT alpha = ((FLUCS_FLOAT)0.5) * RHOI2 * kperp2;
    return alpha < FLUCS_EPSILON ? FLOAT_ONE : FLOAT_ONE / alpha;
#else
    return one_minus_gamma0_over_alpha_operator(kperp2);
#endif
}

__device__ __forceinline__
FLUCS_FLOAT taubarinv(FLUCS_FLOAT kperp2) {

// ERMHD variant that sets gamma0 to zero
#ifdef ERMHD
    return ZTE_OVER_TI;
#else
    return taubarinv_operator(kperp2);
#endif
}

// Fetches the linear matrix for a given mode
__device__ void get_linear_matrix(
    const size_t index, 
    const FLUCS_FLOAT dt,
    const FLUCS_FLOAT current_time,
    const long long current_step, 
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
    const FLUCS_COMPLEX fields_global[NUMBER_OF_FIELDS][HALFSIZE],
    FLUCS_COMPLEX dft_derivatives_global[NUMBER_OF_DFT_DERIVATIVES][HALFSIZE]
){
    const size_t index = blockDim.x * blockIdx.x + threadIdx.x;

    // Check if we are within bounds
    if (!(index < HALFSIZE))
        return;

    indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
    const size_t ikx = indices.ikx;
    const size_t iky = indices.iky;
    const size_t ikz = indices.ikz;

    // Check if mode should be zeroed
    if (is_mode_padded(ikz, ikx, iky)){

        dft_derivatives_global[0][index] = 0;
        dft_derivatives_global[1][index] = 0;
        dft_derivatives_global[2][index] = 0;
        dft_derivatives_global[3][index] = 0;
        dft_derivatives_global[4][index] = 0;
        dft_derivatives_global[5][index] = 0;
        return;
    }

    // Wavenumbers and derivatives
    const FLUCS_FLOAT kx = kx_from_ikx(ikx);
    const FLUCS_FLOAT ky = ky_from_iky(iky);

    const FLUCS_COMPLEX dx = dx_from_ikx(ikx);
    const FLUCS_COMPLEX dy = dy_from_iky(iky);

    const FLUCS_FLOAT kperp2 = kx*kx + ky*ky;

    // Fields
    const FLUCS_COMPLEX phi = fields_global[0][index];
    const FLUCS_COMPLEX apar = fields_global[1][index];

    dft_derivatives_global[0][index] = dx * phi;
    dft_derivatives_global[1][index] = dy * phi;
    dft_derivatives_global[2][index] = dx * apar;
    dft_derivatives_global[3][index] = dy * apar;
    dft_derivatives_global[4][index] = (
        one_minus_gamma0_over_alpha(kperp2) * kperp2 * phi
    );
    dft_derivatives_global[5][index] = kperp2 * apar;

}

// Finds the nonlinear combinations of (real-space) derivatives required to 
// construct the nonlinear terms
__global__ void find_nonlinear_bits(
    const FLUCS_FLOAT real_derivatives_global
        [NUMBER_OF_DFT_DERIVATIVES][FULLSIZE],
    FLUCS_FLOAT real_bits_global[NUMBER_OF_DFT_BITS][FULLSIZE],
    const bool calculate_cfl,
    FLUCS_FLOAT* cfl_rate_global
){
    const size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    const bool in_bounds = index < FULLSIZE;

    // Inactive threads contribute zero but must participate in the
    // block-wide CFL reduction.
    const FLUCS_FLOAT dxphi = in_bounds
        ? real_derivatives_global[0][index]
        : (FLUCS_FLOAT)0;

    const FLUCS_FLOAT dyphi = in_bounds
        ? real_derivatives_global[1][index]
        : (FLUCS_FLOAT)0;

    const FLUCS_FLOAT dxapar = in_bounds
        ? real_derivatives_global[2][index]
        : (FLUCS_FLOAT)0;

    const FLUCS_FLOAT dyapar = in_bounds
        ? real_derivatives_global[3][index]
        : (FLUCS_FLOAT)0;

    if (calculate_cfl) {
        const FLUCS_FLOAT cfl_phi =
              flucs_fabs(dxphi) * (NY_UNPADDED / LY)
            + flucs_fabs(dyphi) * (NX_UNPADDED / LX);

        const FLUCS_FLOAT cfl_apar =
              flucs_fabs(dxapar) * (NY_UNPADDED / LY)
            + flucs_fabs(dyapar) * (NX_UNPADDED / LX);

        // This is sufficient for de = 0, but a higher-order perpendicular
        // derivative may be required for finite de.
        update_cfl(cfl_phi + cfl_apar, cfl_rate_global);
    }

    if (!in_bounds)
        return;

    // These arrays may alias, so load every remaining input before writing
    // any nonlinear product.
    const FLUCS_FLOAT one_minus_gamma0_over_alpha_kperp2phi =
        real_derivatives_global[4][index];
    const FLUCS_FLOAT kperp2apar = real_derivatives_global[5][index];

    real_bits_global[0][index] = (
        dxphi * one_minus_gamma0_over_alpha_kperp2phi  - dxapar * kperp2apar
    );
    real_bits_global[1][index] = (
        dyphi * one_minus_gamma0_over_alpha_kperp2phi  - dyapar * kperp2apar
    );
    real_bits_global[2][index] = (
        dxphi * (DE2 * kperp2apar)
        - (((FLUCS_FLOAT)0.5) * RHOI2 * ZTE_OVER_TI) * dxapar * one_minus_gamma0_over_alpha_kperp2phi
    );
    real_bits_global[3][index] = (
        dyphi * (DE2 * kperp2apar)
        - (((FLUCS_FLOAT)0.5) * RHOI2 * ZTE_OVER_TI) * dyapar * one_minus_gamma0_over_alpha_kperp2phi
    );
    real_bits_global[4][index] = (
        dxphi * dyapar - dyphi * dxapar
    );
}

// Returns the nonlinear terms for a given mode
__device__ void add_nonlinear_terms(
    const size_t index,
    const FLUCS_FLOAT dt,
    const FLUCS_FLOAT current_time,
    const long long current_step,
    const FLUCS_COMPLEX dft_bits_global[NUMBER_OF_DFT_BITS][HALFSIZE],
    FLUCS_COMPLEX explicit_terms[NUMBER_OF_FIELDS]
){
    // Indices
    indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
    const size_t ikx = indices.ikx;
    const size_t iky = indices.iky;

    // Ignore kperp2 = 0 modes
    if (ikx == 0 && iky == 0)
        return;

    // Wavenumbers and indices 
    const FLUCS_FLOAT kx = kx_from_ikx(ikx);
    const FLUCS_FLOAT ky = ky_from_iky(iky);
    const FLUCS_COMPLEX dx = dx_from_ikx(ikx);
    const FLUCS_COMPLEX dy = dy_from_iky(iky);

    const FLUCS_FLOAT kperp2 = kx*kx + ky*ky;

    const FLUCS_COMPLEX dxphi_gamma_phi = dft_bits_global[0][index];
    const FLUCS_COMPLEX dyphi_gamma_phi = dft_bits_global[1][index];
    const FLUCS_COMPLEX dxphi_de2_apar = dft_bits_global[2][index];
    const FLUCS_COMPLEX dyphi_de2_apar = dft_bits_global[3][index];
    const FLUCS_COMPLEX poisson_phi_apar = dft_bits_global[4][index];
    
    // Calculate nonnlinear terms
    explicit_terms[0] += DFT_FULLSIZE_FACTOR * (
        dy * dxphi_gamma_phi - dx * dyphi_gamma_phi
    ) / (one_minus_gamma0_over_alpha(kperp2) * kperp2);

    explicit_terms[1] += DFT_FULLSIZE_FACTOR * (
        poisson_phi_apar
        + dy * dxphi_de2_apar - dx * dyphi_de2_apar
    ) / (FLOAT_ONE + kperp2*DE2);

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
    const FLUCS_COMPLEX* fields_global,
    FLUCS_COMPLEX& thetap,
    FLUCS_COMPLEX& thetam,
    FLUCS_FLOAT& vphase
){
    // Fields
    const FLUCS_COMPLEX phi = fields_global[index];
    const FLUCS_COMPLEX apar = fields_global[index + HALFSIZE];

    get_thetas_from_components(index, phi, apar, thetap, thetam, vphase);
}

__device__ __forceinline__
FLUCS_FLOAT get_thetas_free_energy_rate(
    const size_t index,
    const FLUCS_COMPLEX* fields_global,
    const FLUCS_COMPLEX rates[NUMBER_OF_FIELDS],
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
    get_thetas_from_fields(index, fields_global, thetap, thetam, vphase);

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

#if defined(FORCING_METHOD_ELSASSER)
__device__ __forceinline__
void add_forcing_elsasser(
    const size_t index,
    const FLUCS_FLOAT dt,
    const FLUCS_FLOAT current_time,
    const long long current_step,
    const FLUCS_COMPLEX previous_fields_forcing[NUMBER_OF_FIELDS],
    FLUCS_COMPLEX explicit_terms[NUMBER_OF_FIELDS]
)
{
    // Unused variables
    (void)dt;
    (void)current_step;

    if (!forcing_range_mask(index))
        return;

    // Indices
    indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
    const size_t ikx = indices.ikx;
    const size_t iky = indices.iky;

    // Wavenumbers
    const FLUCS_FLOAT kx = kx_from_ikx(ikx);
    const FLUCS_FLOAT ky = ky_from_iky(iky);

    const FLUCS_FLOAT kperp2 = kx*kx + ky*ky;

    if (kperp2 == ((FLUCS_FLOAT)0.0))
        return;

    // Fields
    const FLUCS_COMPLEX phi = previous_fields_forcing[0];
    const FLUCS_COMPLEX apar = previous_fields_forcing[1];

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
    const FLUCS_FLOAT current_time,
    const long long current_step,
    const FLUCS_COMPLEX previous_fields_forcing[NUMBER_OF_FIELDS],
    FLUCS_COMPLEX explicit_terms[NUMBER_OF_FIELDS]
)
{
    // Unused variables
    (void)dt;
    (void)current_step;

    if (!forcing_range_mask(index))
        return;

    // Indices
    indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
    const size_t ikx = indices.ikx;
    const size_t iky = indices.iky;

    // Wavenumbers
    const FLUCS_FLOAT kx = kx_from_ikx(ikx);
    const FLUCS_FLOAT ky = ky_from_iky(iky);

    const FLUCS_FLOAT kperp2 = kx*kx + ky*ky;
    const FLUCS_FLOAT one_plus_kperp2de2 = FLOAT_ONE + kperp2 * DE2;

    if (kperp2 == ((FLUCS_FLOAT)0.0))
        return;

    // Various factors that appear in the terms below
    const FLUCS_FLOAT gamma_factor = one_minus_gamma0_over_alpha(kperp2);
    const FLUCS_FLOAT one_plus_taubarinv = FLOAT_ONE + taubarinv(kperp2);
    const FLUCS_FLOAT taubarinv_over_kperp2de2_factor = one_plus_taubarinv / one_plus_kperp2de2;
    const FLUCS_FLOAT helicity_prefactor = 2 * gamma_factor * kperp2 * one_plus_kperp2de2;

    // Fields
    const FLUCS_COMPLEX phi = previous_fields_forcing[0];
    const FLUCS_COMPLEX apar = previous_fields_forcing[1];

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

#ifndef FORCING_FROM_SOLVER
#if defined(FORCING_METHOD_PHASE)
__device__ __forceinline__
void add_forcing_phase(
    const size_t index,
    const FLUCS_FLOAT dt,
    const FLUCS_FLOAT current_time,
    const long long current_step,
    const FLUCS_COMPLEX previous_fields_forcing[NUMBER_OF_FIELDS],
    FLUCS_COMPLEX explicit_terms[NUMBER_OF_FIELDS]
)
{
    // Unused variables
    (void)current_step;

    if (!forcing_range_mask(index))
        return;

    // Indices
    indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
    const size_t ikx = indices.ikx;
    const size_t iky = indices.iky;

    // Wavenumbers
    const FLUCS_FLOAT kx = kx_from_ikx(ikx);
    const FLUCS_FLOAT ky = ky_from_iky(iky);

    const FLUCS_FLOAT kperp2 = kx*kx + ky*ky;

    if (kperp2 == ((FLUCS_FLOAT)0.0))
        return;

    // Various factors that appear in the terms below
    const FLUCS_FLOAT gamma_factor = one_minus_gamma0_over_alpha(kperp2);
    const FLUCS_FLOAT one_plus_taubarinv = FLOAT_ONE + taubarinv(kperp2);
    const FLUCS_FLOAT one_plus_kperp2de2 = FLOAT_ONE + kperp2 * DE2;
    const FLUCS_FLOAT helicity_prefactor = 2 * gamma_factor * kperp2 * one_plus_kperp2de2;

    // Fields
    const FLUCS_COMPLEX phi = previous_fields_forcing[0];
    const FLUCS_COMPLEX apar = previous_fields_forcing[1];

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
    const FLUCS_FLOAT imag_phi_conj_apar = (
        phi.imag() * apar.real() - phi.real() * apar.imag()
    );

    // Energies
    const FLUCS_FLOAT W_phi = one_plus_taubarinv * gamma_factor * kperp2 * phi2;
    const FLUCS_FLOAT W_apar = one_plus_kperp2de2 * kperp2 * apar2;

    // Helicity injection rate
    const FLUCS_FLOAT helicity_injection_rate = (
        FORCING_IMBALANCE
        * (FORCING_EPSILON_PHI + FORCING_EPSILON_APAR)
    );

    // Amplitude and relative-phase rates
    const FLUCS_FLOAT phi_growth_rate = (
        FORCING_EPSILON_PHI * ((FLUCS_FLOAT)0.5) / W_phi
    );
    const FLUCS_FLOAT apar_growth_rate = (
        FORCING_EPSILON_APAR * ((FLUCS_FLOAT)0.5) / W_apar
    );

    // Regularise expressions to avoid singularities from aligned fields
    FLUCS_FLOAT regularisation  = ((FLUCS_FLOAT)0.0);
    FLUCS_FLOAT apar_phase_rate = ((FLUCS_FLOAT)0.0);

    if (imag_phi_conj_apar * imag_phi_conj_apar > FLUCS_EPSILON * phi2 * apar2) 
    {
        apar_phase_rate = (
            + helicity_injection_rate
            - helicity_prefactor
                * (phi_growth_rate + apar_growth_rate)
                * real_phi_conj_apar
        ) / (helicity_prefactor * imag_phi_conj_apar);

        const FLUCS_FLOAT apar_phase_rate_max = (
            ((FLUCS_FLOAT)0.1) / dt 
            // Factor of 0.1 is chosen to avoid transients
        );
        const FLUCS_FLOAT ratio = (
            apar_phase_rate/ (((FLUCS_FLOAT)2.0) * apar_phase_rate_max)
        );

        regularisation = FLOAT_ONE / (FLOAT_ONE + ratio * ratio);
    }

    // Construct forcing
    explicit_terms[0] -= regularisation * phi_growth_rate * phi;
    explicit_terms[1] -= FLUCS_COMPLEX(
        regularisation * apar_growth_rate,
        regularisation * apar_phase_rate
    ) * apar;
}
#endif // FORCING_METHOD_PHASE

__device__ void add_forcing_explicit(
    const size_t index,
    const FLUCS_FLOAT dt,
    const FLUCS_FLOAT current_time, 
    const long long current_step,
    const FLUCS_COMPLEX previous_fields_forcing[NUMBER_OF_FIELDS],
    FLUCS_COMPLEX explicit_terms[NUMBER_OF_FIELDS]
){
    #if defined(FORCING_METHOD_ELSASSER)
        add_forcing_elsasser(
            index, dt, current_time, current_step,
            previous_fields_forcing, explicit_terms
        );
    #endif

    #if defined(FORCING_METHOD_MEYRAND)
        add_forcing_meyrand(
            index, dt, current_time, current_step,
            previous_fields_forcing, explicit_terms
        );
    #endif

    #if defined(FORCING_METHOD_PHASE)
        add_forcing_phase(
            index, dt, current_time, current_step,
            previous_fields_forcing, explicit_terms
        );
    #endif
}
#endif // not FORCING_FROM_SOLVER

#endif // FORCING


////////////////////////////////////////////////////////////////////////////////
// Diagnostics: Free Energy
////////////////////////////////////////////////////////////////////////////////

struct FreeEnergy_Functor {
    const FLUCS_COMPLEX* __restrict__ fields_global;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        // Fields
        const FLUCS_COMPLEX phi = fields_global[index];
        const FLUCS_COMPLEX apar = fields_global[index + HALFSIZE];

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
    const FLUCS_COMPLEX* __restrict__ fields_global;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        // Field
        const FLUCS_COMPLEX phi = fields_global[index];

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
    const FLUCS_COMPLEX* __restrict__ fields_global;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        // Field
        const FLUCS_COMPLEX phi = fields_global[index];

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
    const FLUCS_COMPLEX* __restrict__ fields_global;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        // Field
        const FLUCS_COMPLEX apar = fields_global[index + HALFSIZE];

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
    const FLUCS_COMPLEX* __restrict__ fields_global;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        // Field
        const FLUCS_COMPLEX apar = fields_global[index + HALFSIZE];

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
    const FLUCS_COMPLEX (* __restrict__ fields_global)[HALFSIZE];
    const FLUCS_FLOAT dt;
    const FLUCS_FLOAT current_time;
    const long long current_step;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        // Fields
        const FLUCS_COMPLEX phi = fields_global[0][index];
        const FLUCS_COMPLEX apar = fields_global[1][index];

        // Forcing terms
        FLUCS_COMPLEX forcing_terms[NUMBER_OF_FIELDS] = {0};
#ifdef FORCING_EXPLICIT
        FLUCS_COMPLEX fields_forcing[NUMBER_OF_FIELDS];
        get_forcing_fields(index, fields_global, fields_forcing);
        add_forcing_explicit(
            index,
            dt,
            current_time,
            current_step,
            fields_forcing,
            forcing_terms
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
    const FLUCS_COMPLEX* __restrict__ fields_global;
    const FLUCS_FLOAT dt;
    const FLUCS_FLOAT current_time;
    const long long current_step;
    const FLUCS_COMPLEX (* __restrict__ dft_bits_global)[HALFSIZE];

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        // Fields
        const FLUCS_COMPLEX phi = fields_global[index];
        const FLUCS_COMPLEX apar = fields_global[index + HALFSIZE];

        // Nonlinear terms
        FLUCS_COMPLEX nonlinear_terms[NUMBER_OF_FIELDS] = {0};
#ifdef NONLINEAR
        add_nonlinear_terms(
            index,
            dt,
            current_time,
            current_step,
            dft_bits_global,
            nonlinear_terms
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
    const FLUCS_COMPLEX* __restrict__ fields_global;
    const FLUCS_FLOAT adaptive_rate;

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        return (FLUCS_FLOAT)2.0
            * Hyperdissipation_Functor<FreeEnergy_Functor>{
                FreeEnergy_Functor{fields_global},
                adaptive_rate
            }(index);
    }
};

struct FreeEnergyHyperdissipationComponent_Functor {
    const FLUCS_COMPLEX* __restrict__ fields_global;
    const FLUCS_FLOAT adaptive_rate;
    const int hyperdissipation_type;

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        return (FLUCS_FLOAT)2.0
            * HyperdissipationSelector_Functor<FreeEnergy_Functor>{
                FreeEnergy_Functor{fields_global},
                adaptive_rate,
                hyperdissipation_type
            }(index);
    }
};

struct FreeEnergyThetap_Functor {
    const FLUCS_COMPLEX* __restrict__ fields_global;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        
        // Indices
        indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
        const FLUCS_FLOAT kx = kx_from_ikx(indices.ikx);
        const FLUCS_FLOAT ky = ky_from_iky(indices.iky);
        const FLUCS_FLOAT kperp2 = kx*kx + ky*ky;

        // Thetas
        FLUCS_COMPLEX thetap, thetam;
        FLUCS_FLOAT vphase;
        get_thetas_from_fields(index, fields_global, thetap, thetam, vphase);

        return ((FLUCS_FLOAT)0.5) * kperp2
            * (thetap.real()*thetap.real() + thetap.imag()*thetap.imag());
    }
};

struct FreeEnergyThetapForcing_Functor {
    const FLUCS_COMPLEX (* __restrict__ fields_global)[HALFSIZE];
    const FLUCS_FLOAT dt;
    const FLUCS_FLOAT current_time;
    const long long current_step;

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        // Forcing terms
        FLUCS_COMPLEX forcing_terms[NUMBER_OF_FIELDS] = {0};
#ifdef FORCING_EXPLICIT
        FLUCS_COMPLEX fields_forcing[NUMBER_OF_FIELDS];
        get_forcing_fields(index, fields_global, fields_forcing);
        add_forcing_explicit(
            index,
            dt,
            current_time,
            current_step,
            fields_forcing,
            forcing_terms
        );
#endif

        // Physical rates due to forcing
        FLUCS_COMPLEX rates[NUMBER_OF_FIELDS] = {0};
        rates[0] = -forcing_terms[0];
        rates[1] = -forcing_terms[1];

        return get_thetas_free_energy_rate(
            index, fields_global[0], rates, +1
        );
    }
};

struct FreeEnergyThetapNonlinear_Functor {
    const FLUCS_COMPLEX* __restrict__ fields_global;
    const FLUCS_FLOAT dt;
    const FLUCS_FLOAT current_time;
    const long long current_step;
    const FLUCS_COMPLEX (* __restrict__ dft_bits_global)[HALFSIZE];

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        // Nonlinear terms
        FLUCS_COMPLEX nonlinear_terms[NUMBER_OF_FIELDS] = {0};
#ifdef NONLINEAR
        add_nonlinear_terms(
            index,
            dt,
            current_time,
            current_step,
            dft_bits_global,
            nonlinear_terms
        );
#endif

        // Physical rates due to nonlinear terms
        FLUCS_COMPLEX rates[NUMBER_OF_FIELDS] = {0};
        rates[0] = -nonlinear_terms[0];
        rates[1] = -nonlinear_terms[1];

        return get_thetas_free_energy_rate(
            index, fields_global, rates, +1
        );
    }
};

struct FreeEnergyThetapHyperdissipation_Functor {
    const FLUCS_COMPLEX* __restrict__ fields_global;
    const FLUCS_FLOAT adaptive_rate;

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        return (FLUCS_FLOAT)2.0
            * Hyperdissipation_Functor<FreeEnergyThetap_Functor>{
                FreeEnergyThetap_Functor{fields_global},
                adaptive_rate
            }(index);
    }
};

struct FreeEnergyThetapHyperdissipationComponent_Functor {
    const FLUCS_COMPLEX* __restrict__ fields_global;
    const FLUCS_FLOAT adaptive_rate;
    const int hyperdissipation_type;

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        return (FLUCS_FLOAT)2.0
            * HyperdissipationSelector_Functor<FreeEnergyThetap_Functor>{
                FreeEnergyThetap_Functor{fields_global},
                adaptive_rate,
                hyperdissipation_type
            }(index);
    }
};

struct FreeEnergyThetam_Functor {
    const FLUCS_COMPLEX* __restrict__ fields_global;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        
        // Indices
        indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
        const FLUCS_FLOAT kx = kx_from_ikx(indices.ikx);
        const FLUCS_FLOAT ky = ky_from_iky(indices.iky);
        const FLUCS_FLOAT kperp2 = kx*kx + ky*ky;

        // Thetas
        FLUCS_COMPLEX thetap, thetam;
        FLUCS_FLOAT vphase;
        get_thetas_from_fields(index, fields_global, thetap, thetam, vphase);

        return ((FLUCS_FLOAT)0.5) * kperp2
            * (thetam.real()*thetam.real() + thetam.imag()*thetam.imag());
    }
};

struct FreeEnergyThetamForcing_Functor {
    const FLUCS_COMPLEX (* __restrict__ fields_global)[HALFSIZE];
    const FLUCS_FLOAT dt;
    const FLUCS_FLOAT current_time;
    const long long current_step;

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        // Forcing terms
        FLUCS_COMPLEX forcing_terms[NUMBER_OF_FIELDS] = {0};
#ifdef FORCING_EXPLICIT
        FLUCS_COMPLEX fields_forcing[NUMBER_OF_FIELDS];
        get_forcing_fields(index, fields_global, fields_forcing);
        add_forcing_explicit(
            index,
            dt,
            current_time,
            current_step,
            fields_forcing,
            forcing_terms
        );
#endif

        // Physical rates due to forcing
        FLUCS_COMPLEX rates[NUMBER_OF_FIELDS] = {0};
        rates[0] = -forcing_terms[0];
        rates[1] = -forcing_terms[1];

        return get_thetas_free_energy_rate(
            index, fields_global[0], rates, -1
        );
    }
};

struct FreeEnergyThetamNonlinear_Functor {
    const FLUCS_COMPLEX* __restrict__ fields_global;
    const FLUCS_FLOAT dt;
    const FLUCS_FLOAT current_time;
    const long long current_step;
    const FLUCS_COMPLEX (* __restrict__ dft_bits_global)[HALFSIZE];

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        // Nonlinear terms
        FLUCS_COMPLEX nonlinear_terms[NUMBER_OF_FIELDS] = {0};
#ifdef NONLINEAR
        add_nonlinear_terms(
            index,
            dt,
            current_time,
            current_step,
            dft_bits_global,
            nonlinear_terms
        );
#endif

        // Physical rates due to nonlinear terms
        FLUCS_COMPLEX rates[NUMBER_OF_FIELDS] = {0};
        rates[0] = -nonlinear_terms[0];
        rates[1] = -nonlinear_terms[1];

        return get_thetas_free_energy_rate(
            index, fields_global, rates, -1
        );
    }
};

struct FreeEnergyThetamHyperdissipation_Functor {
    const FLUCS_COMPLEX* __restrict__ fields_global;
    const FLUCS_FLOAT adaptive_rate;

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        return (FLUCS_FLOAT)2.0
            * Hyperdissipation_Functor<FreeEnergyThetam_Functor>{
                FreeEnergyThetam_Functor{fields_global},
                adaptive_rate
            }(index);
    }
};

struct FreeEnergyThetamHyperdissipationComponent_Functor {
    const FLUCS_COMPLEX* __restrict__ fields_global;
    const FLUCS_FLOAT adaptive_rate;
    const int hyperdissipation_type;

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        return (FLUCS_FLOAT)2.0
            * HyperdissipationSelector_Functor<FreeEnergyThetam_Functor>{
                FreeEnergyThetam_Functor{fields_global},
                adaptive_rate,
                hyperdissipation_type
            }(index);
    }
};

////////////////////////////////////////////////////////////////////////////////
// Diagnostics: Helicity
////////////////////////////////////////////////////////////////////////////////

struct Helicity_Functor {
    const FLUCS_COMPLEX* __restrict__ fields_global;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        // Fields
        const FLUCS_COMPLEX phi = fields_global[index];
        const FLUCS_COMPLEX apar = fields_global[index + HALFSIZE];

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
    const FLUCS_COMPLEX* __restrict__ fields_global;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        // Fields
        const FLUCS_COMPLEX phi = fields_global[index];
        const FLUCS_COMPLEX apar = fields_global[index + HALFSIZE];

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
    const FLUCS_COMPLEX* __restrict__ fields_global;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        // Fields
        const FLUCS_COMPLEX phi = fields_global[index];
        const FLUCS_COMPLEX apar = fields_global[index + HALFSIZE];

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
    const FLUCS_COMPLEX (* __restrict__ fields_global)[HALFSIZE];
    const FLUCS_FLOAT dt;
    const FLUCS_FLOAT current_time;
    const long long current_step;

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        // Fields
        const FLUCS_COMPLEX phi = fields_global[0][index];
        const FLUCS_COMPLEX apar = fields_global[1][index];

        // Forcing terms
        FLUCS_COMPLEX forcing_terms[NUMBER_OF_FIELDS] = {0};
#ifdef FORCING_EXPLICIT
        FLUCS_COMPLEX fields_forcing[NUMBER_OF_FIELDS];
        get_forcing_fields(index, fields_global, fields_forcing);
        add_forcing_explicit(
            index,
            dt,
            current_time,
            current_step,
            fields_forcing,
            forcing_terms
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
    const FLUCS_COMPLEX* __restrict__ fields_global;
    const FLUCS_FLOAT dt;
    const FLUCS_FLOAT current_time;
    const long long current_step;
    const FLUCS_COMPLEX (* __restrict__ dft_bits_global)[HALFSIZE];

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        // Fields
        const FLUCS_COMPLEX phi = fields_global[index];
        const FLUCS_COMPLEX apar = fields_global[index + HALFSIZE];

        // Nonlinear terms
        FLUCS_COMPLEX nonlinear_terms[NUMBER_OF_FIELDS] = {0};
#ifdef NONLINEAR
        add_nonlinear_terms(
            index,
            dt,
            current_time,
            current_step,
            dft_bits_global,
            nonlinear_terms
        );
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
    const FLUCS_COMPLEX* __restrict__ fields_global;
    const FLUCS_FLOAT adaptive_rate;

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        return (FLUCS_FLOAT)2.0
            * Hyperdissipation_Functor<Helicity_Functor>{
                Helicity_Functor{fields_global},
                adaptive_rate
            }(index);
    }
};

struct HelicityHyperdissipationComponent_Functor {
    const FLUCS_COMPLEX* __restrict__ fields_global;
    const FLUCS_FLOAT adaptive_rate;
    const int hyperdissipation_type;

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        return (FLUCS_FLOAT)2.0
            * HyperdissipationSelector_Functor<Helicity_Functor>{
                Helicity_Functor{fields_global},
                adaptive_rate,
                hyperdissipation_type
            }(index);
    }
};

struct HelicityThetap_Functor {
    const FLUCS_COMPLEX* __restrict__ fields_global;
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

        return FreeEnergyThetap_Functor{fields_global}(index) / vphase;
    }
};

struct HelicityThetapForcing_Functor {
    const FLUCS_COMPLEX (* __restrict__ fields_global)[HALFSIZE];
    const FLUCS_FLOAT dt;
    const FLUCS_FLOAT current_time;
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
            fields_global, dt, current_time, current_step
        }(index) / vphase;
    }
};

struct HelicityThetapNonlinear_Functor {
    const FLUCS_COMPLEX* __restrict__ fields_global;
    const FLUCS_FLOAT dt;
    const FLUCS_FLOAT current_time;
    const long long current_step;
    const FLUCS_COMPLEX (* __restrict__ dft_bits_global)[HALFSIZE];

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
            fields_global, dt, current_time, current_step, dft_bits_global
        }(index) / vphase;
    }
};

struct HelicityThetapHyperdissipation_Functor {
    const FLUCS_COMPLEX* __restrict__ fields_global;
    const FLUCS_FLOAT adaptive_rate;

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        return (FLUCS_FLOAT)2.0
            * Hyperdissipation_Functor<HelicityThetap_Functor>{
                HelicityThetap_Functor{fields_global},
                adaptive_rate
            }(index);
    }
};

struct HelicityThetapHyperdissipationComponent_Functor {
    const FLUCS_COMPLEX* __restrict__ fields_global;
    const FLUCS_FLOAT adaptive_rate;
    const int hyperdissipation_type;

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        return (FLUCS_FLOAT)2.0
            * HyperdissipationSelector_Functor<HelicityThetap_Functor>{
                HelicityThetap_Functor{fields_global},
                adaptive_rate,
                hyperdissipation_type
            }(index);
    }
};

struct HelicityThetam_Functor {
    const FLUCS_COMPLEX* __restrict__ fields_global;
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

        return FreeEnergyThetam_Functor{fields_global}(index) / vphase;
    }
};

struct HelicityThetamForcing_Functor {
    const FLUCS_COMPLEX (* __restrict__ fields_global)[HALFSIZE];
    const FLUCS_FLOAT dt;
    const FLUCS_FLOAT current_time;
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
            fields_global, dt, current_time, current_step
        }(index) / vphase;
    }
};

struct HelicityThetamNonlinear_Functor {
    const FLUCS_COMPLEX* __restrict__ fields_global;
    const FLUCS_FLOAT dt;
    const FLUCS_FLOAT current_time;
    const long long current_step;
    const FLUCS_COMPLEX (* __restrict__ dft_bits_global)[HALFSIZE];

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
            fields_global, dt, current_time, current_step, dft_bits_global
        }(index) / vphase;
    }
};

struct HelicityThetamHyperdissipation_Functor {
    const FLUCS_COMPLEX* __restrict__ fields_global;
    const FLUCS_FLOAT adaptive_rate;

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        return (FLUCS_FLOAT)2.0
            * Hyperdissipation_Functor<HelicityThetam_Functor>{
                HelicityThetam_Functor{fields_global},
                adaptive_rate
            }(index);
    }
};

struct HelicityThetamHyperdissipationComponent_Functor {
    const FLUCS_COMPLEX* __restrict__ fields_global;
    const FLUCS_FLOAT adaptive_rate;
    const int hyperdissipation_type;

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        return (FLUCS_FLOAT)2.0
            * HyperdissipationSelector_Functor<HelicityThetam_Functor>{
                HelicityThetam_Functor{fields_global},
                adaptive_rate,
                hyperdissipation_type
            }(index);
    }
};

} // extern "C"
