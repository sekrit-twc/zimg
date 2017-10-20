#ifndef MUSL_MYMATH_H_
#define MUSL_MYMATH_H_

#ifdef __cplusplus
extern "C" {
#endif

float myexpf(float x);
float mylogf(float x);
float mylog10f(float x);

float mypowf(float x, float y);

double mycos(double x);
double mysin(double x);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* MUSL_MYMATH_H_ */
