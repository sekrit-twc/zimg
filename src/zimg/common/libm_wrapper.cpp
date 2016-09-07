#include <cmath>
#include "libm_wrapper.h"

double (*zimg_x_sin)(double) = std::sin;
double (*zimg_x_cos)(double) = std::cos;
float (*zimg_x_powf)(float, float) = std::pow;
