#pragma once

#ifndef ZIMG_LIBM_WRAPPER_H_
#define ZIMG_LIBM_WRAPPER_H_

/**
 * @file
 *
 * To ensure reproducable results during testing, the use of inexact math
 * library functions is avoided in the library. Instead, all calls to such
 * functions must be dispatched through one of the function pointers below.
 */

#ifdef __cplusplus
extern "C" {
#endif

extern float (*zimg_x_expf)(float x);
extern float (*zimg_x_logf)(float x);

extern float (*zimg_x_powf)(float x, float y);

/* Prevent MSVC from executing the legacy x87 FSQRT instruction if possible. */
#if defined(_M_IX86_FP) && _M_IX86_FP > 0
  #include <emmintrin.h>
  #define zimg_x_sqrtf(x) (_mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(x))))
#else
  #define zimg_x_sqrtf sqrtf
#endif

extern double (*zimg_x_sin)(double x);
extern double (*zimg_x_cos)(double x);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ZIMG_LIBM_WRAPPER_H_ */
