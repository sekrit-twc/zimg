#include <math.h>
#include "libm_wrapper.h"

float (*zimg_x_expf)(float) = expf;
float (*zimg_x_logf)(float) = logf;
float (*zimg_x_log10f)(float) = log10f;

float (*zimg_x_powf)(float, float) = powf;

double (*zimg_x_sin)(double) = sin;
double (*zimg_x_cos)(double) = cos;
