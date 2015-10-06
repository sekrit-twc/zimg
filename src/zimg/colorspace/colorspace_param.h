#pragma once

#ifndef ZIMG_COLORSPACE_COLORSPACE_PARAM_H_
#define ZIMG_COLORSPACE_COLORSPACE_PARAM_H_

#include "colorspace.h"
#include "matrix3.h"

namespace zimg {;
namespace colorspace {;

// Kr and Kb YUV<-->RGB constants.
const double REC_601_KR = 0.299;
const double REC_601_KB = 0.114;
const double REC_709_KR = 0.2126;
const double REC_709_KB = 0.0722;
const double REC_2020_KR = 0.2627;
const double REC_2020_KB = 0.0593;

// R, G, B primaries in XY.
const double SMPTE_C_PRIMARIES[3][2] = { { 0.630, 0.340 }, { 0.310, 0.595 }, { 0.155, 0.070 } };
const double REC_709_PRIMARIES[3][2] = { { 0.640, 0.330 }, { 0.300, 0.600 }, { 0.150, 0.060 } };
const double REC_2020_PRIMARIES[3][2] = { { 0.708, 0.292 }, { 0.170, 0.797 }, { 0.131, 0.046 } };

// D65 white point in XY.
const double ILLUMINANT_D65[2] = { 0.3127f, 0.3290f };

/**
 * Obtain 3x3 matrix for converting from YUV to RGB.
 *
 * @param matrix matrix coefficients
 * @return conversion function as matrix
 * @throws IllegalArgument on invalid matrix
 */
Matrix3x3 ncl_yuv_to_rgb_matrix(MatrixCoefficients matrix);

/**
 * Obtain 3x3 matrix for converting from RGB toYUV.
 *
 * @see ncl_yuv_to_rgb_matrix
 */
Matrix3x3 ncl_rgb_to_yuv_matrix(MatrixCoefficients matrix);

/**
 * Obtain 3x3 matrix for converting from RGB to XYZ.
 *
 * @param primaries primaries
 * @param conversion function as matrix
 * @throws IllegalArgument on invalid primaries
 */
Matrix3x3 gamut_rgb_to_xyz_matrix(ColorPrimaries primaries);

/**
 * Obtain 3x3 matrix for converting from XYZ to RGB.
 *
 * @see gamut_rgb_to_xyz_matrix
 */
Matrix3x3 gamut_xyz_to_rgb_matrix(ColorPrimaries primaries);

} // namespace colorspace
} // namespace zimg

#endif // ZIMG_COLORSPACE_COLORSPACE_PARAM_H_
