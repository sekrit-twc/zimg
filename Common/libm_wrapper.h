#pragma once

#ifndef ZIMG_LIBM_WRAPPER_H_
#define ZIMG_LIBM_WRAPPER_H_

#ifdef __cplusplus
extern "C" {
#endif

extern double (*_zimg_sin)(double x);
extern double (*_zimg_cos)(double x);
extern double (*_zimg_pow)(double x, double y);
extern float (*_zimg_powf)(float x, float y);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ZIMG_LIBM_WRAPPER_H_ */
