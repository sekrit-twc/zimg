#ifndef MUSL_MYMATH_H_
#define MUSL_MYMATH_H_

#ifdef __cplusplus
extern "C" {
#endif

double mycos(double x);
double mysin(double x);
double mypow(double x, double y);
float mypowf(float x, float y);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* MUSL_MYMATH_H_ */
