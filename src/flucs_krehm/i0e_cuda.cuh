/*
 * CUDA implementation of the i0e(x) = exp(-|x|) * I_0(x) function
 * where I_0 is the modified Bessel function.
 *
 * Code adapted from the Cephes Math Library.
 * Original copyright header:
 *
 * Cephes Math Library Release 2.8:  June, 2000
 * Copyright 1984, 1987, 2000 by Stephen L. Moshier
*/


#ifdef DOUBLE_PRECISION
    #ifndef flucs_rsqrt
        #define flucs_rsqrt rsqrt
    #endif
    #ifndef flucs_fma
        #define flucs_fma fma
    #endif
    
    #ifndef GAMMA0_ALPHA_CUTOFF
        #define GAMMA0_ALPHA_CUTOFF ((FLUCS_FLOAT)5e-3)
    #endif
#else
    #ifndef flucs_rsqrt
        #define flucs_rsqrt rsqrtf
    #endif
    #ifndef flucs_fma
        #define flucs_fma fmaf
    #endif

    #ifndef GAMMA0_ALPHA_CUTOFF
        #define GAMMA0_ALPHA_CUTOFF ((FLUCS_FLOAT)0.15)
    #endif
#endif

extern "C" {

__constant__ FLUCS_FLOAT i0e_A[] = {
-4.41534164647933937950E-18,
 3.33079451882223809783E-17,
-2.43127984654795469359E-16,
 1.71539128555513303061E-15,
-1.16853328779934516808E-14,
 7.67618549860493561688E-14,
-4.85644678311192946090E-13,
 2.95505266312963983461E-12,
-1.72682629144155570723E-11,
 9.67580903537323691224E-11,
-5.18979560163526290666E-10,
 2.65982372468238665035E-9,
-1.30002500998624804212E-8,
 6.04699502254191894932E-8,
-2.67079385394061173391E-7,
 1.11738753912010371815E-6,
-4.41673835845875056359E-6,
 1.64484480707288970893E-5,
-5.75419501008210370398E-5,
 1.88502885095841655729E-4,
-5.76375574538582365885E-4,
 1.63947561694133579842E-3,
-4.32430999505057594430E-3,
 1.05464603945949983183E-2,
-2.37374148058994688156E-2,
 4.93052842396707084878E-2,
-9.49010970480476444210E-2,
 1.71620901522208775349E-1,
-3.04682672343198398683E-1,
 6.76795274409476084995E-1
};

__constant__ FLUCS_FLOAT i0e_B[] =
{
-7.23318048787475395456E-18,
-4.83050448594418207126E-18,
 4.46562142029675999901E-17,
 3.46122286769746109310E-17,
-2.82762398051658348494E-16,
-3.42548561967721913462E-16,
 1.77256013305652638360E-15,
 3.81168066935262242075E-15,
-9.55484669882830764870E-15,
-4.15056934728722208663E-14,
 1.54008621752140982691E-14,
 3.85277838274214270114E-13,
 7.18012445138366623367E-13,
-1.79417853150680611778E-12,
-1.32158118404477131188E-11,
-3.14991652796324136454E-11,
 1.18891471078464383424E-11,
 4.94060238822496958910E-10,
 3.39623202570838634515E-9,
 2.26666899049817806459E-8,
 2.04891858946906374183E-7,
 2.89137052083475648297E-6,
 6.88975834691682398426E-5,
 3.36911647825569408990E-3,
 8.04490411014108831608E-1
};

__device__ __forceinline__ FLUCS_FLOAT chbevl_i0e_A(const FLUCS_FLOAT x) {
    FLUCS_FLOAT b0 = i0e_A[0];
    FLUCS_FLOAT b1 = (FLUCS_FLOAT)0.0;
    FLUCS_FLOAT b2;

    #pragma unroll
    for (int i = 1; i < 30; i++) {

        b2 = b1;
        b1 = b0;
        b0 = flucs_fma(x, b1, i0e_A[i] - b2);
    }

    return (FLUCS_FLOAT)0.5 * (b0 - b2);
}

__device__ __forceinline__ FLUCS_FLOAT chbevl_i0e_B(const FLUCS_FLOAT x) {
    FLUCS_FLOAT b0 = i0e_B[0];
    FLUCS_FLOAT b1 = (FLUCS_FLOAT)0.0;
    FLUCS_FLOAT b2;

    #pragma unroll
    for (int i = 1; i < 25; i++) {

        b2 = b1;
        b1 = b0;
        b0 = flucs_fma(x, b1, i0e_B[i] - b2);
    }

    return (FLUCS_FLOAT)0.5 * (b0 - b2);
}

__device__ __forceinline__ FLUCS_FLOAT flucs_i0e(FLUCS_FLOAT x) {
    x = flucs_fabs(x);

    if( x < 8.0 ) {
        return chbevl_i0e_A( (FLUCS_FLOAT)0.5 * x - (FLUCS_FLOAT)2.0);
    }

    return chbevl_i0e_B( (FLUCS_FLOAT)32.0/x - (FLUCS_FLOAT)2.0) * flucs_rsqrt(x);
}

//General KREHM variant
__device__ __forceinline__
FLUCS_FLOAT one_minus_gamma0_over_alpha_operator(FLUCS_FLOAT kperp2) {

    const FLUCS_FLOAT alpha = 0.5 * RHOI2 * kperp2;

    // For small alpha, use Taylor
    if (alpha < GAMMA0_ALPHA_CUTOFF)
        return FLOAT_ONE - alpha * (
            (FLUCS_FLOAT)(3.0 / 4) - alpha * (
                (FLUCS_FLOAT)(5.0 / 12) - alpha * (
                    (FLUCS_FLOAT)(35.0 / 192) - alpha * (
                        (FLUCS_FLOAT)(21.0 / 320)
                    )
                )
            )
        );

    return (FLOAT_ONE - flucs_i0e(alpha)) / alpha;
}

//General KREHM variant
__device__ __forceinline__
FLUCS_FLOAT taubarinv_operator(FLUCS_FLOAT kperp2) {

    const FLUCS_FLOAT alpha = 0.5 * RHOI2 * kperp2;
    return (ZTE_OVER_TI) * (FLOAT_ONE - flucs_i0e(alpha));
}

}
