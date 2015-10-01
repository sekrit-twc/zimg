#include <cmath>
#include "libm_wrapper.h"

double (*_zimg_sin)(double) = std::sin;
double (*_zimg_cos)(double) = std::cos;
double (*_zimg_pow)(double, double) = std::pow;
float (*_zimg_powf)(float, float) = std::pow;
