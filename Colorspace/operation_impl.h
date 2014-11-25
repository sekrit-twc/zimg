#pragma once

#ifndef ZIMG_COLORSPACE_OPERATION_IMPL_H_
#define ZIMG_COLORSPACE_OPERATION_IMPL_H_

#include <cmath>
#include "Common/cpuinfo.h"
#include "matrix3.h"

namespace zimg {;
namespace colorspace {;

class Operation;

const float TRANSFER_ALPHA = 1.09929682680944f;
const float TRANSFER_BETA = 0.018053968510807f;

inline float rec_709_gamma(float x)
{
	if (x < TRANSFER_BETA)
		x = x * 4.5f;
	else
		x = TRANSFER_ALPHA * std::pow(x, 0.45f) - (TRANSFER_ALPHA - 1.0f);

	return x;
}

inline float rec_709_inverse_gamma(float x)
{
	if (x < 4.5f * TRANSFER_BETA)
		x = x / 4.5f;
	else
		x = std::pow((x + (TRANSFER_ALPHA - 1.0f)) / TRANSFER_ALPHA, 1.0f / 0.45f);

	return x;
}

/**
 * Base class for matrix operation implementations.
 */
class MatrixOperationImpl : public Operation {
protected:
	/**
	 * Transformation matrix.
	 */
	float m_matrix[3][3];

	/**
	 * Initialize the implementation with the given matrix.
	 *
	 * @param m transformation matrix
	 */
	MatrixOperationImpl(const Matrix3x3 &matrix);
};

/**
 * Create operation consisting of applying a 3x3 matrix to each pixel triplet.
 *
 * @param m matrix
 * @param cpu create operation optimized for given cpu
 * @return concrete operation
 */
Operation *create_matrix_operation(const Matrix3x3 &m, CPUClass cpu);

/**
 * Create operation consisting of applying Rec.709 transfer function.
 *
 * @param cpu create operation optimized for given cpu
 * @return concrete operation
 */
Operation *create_rec709_gamma_operation(CPUClass cpu);

/**
 * Create operation consisting of inverting Rec.709 transfer function.
 *
 * @see create_rec709_gamma_operation
 */
Operation *create_rec709_inverse_gamma_operation(CPUClass cpu);

} // namespace colorspace
} // namespace zimg

#endif // ZIMG_COLORSPACE_OPERATION_IMPL_H_

