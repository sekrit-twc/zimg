#pragma once

#ifndef ZIMG_COLORSPACE_OPERATION_IMPL_H_
#define ZIMG_COLORSPACE_OPERATION_IMPL_H_

#include "common/libm_wrapper.h"
#include "operation.h"

namespace zimg {
enum class CPUClass;
}

namespace zimg::colorspace {

struct Matrix3x3;
struct TransferFunction;

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
	explicit MatrixOperationImpl(const Matrix3x3 &matrix);
};

/**
 * Create operation consisting of applying a 3x3 matrix to each pixel triplet.
 *
 * @param m matrix
 * @param cpu create operation optimized for given cpu
 * @return concrete operation
 */
std::unique_ptr<Operation> create_matrix_operation(const Matrix3x3 &m, CPUClass cpu);

/**
 * Create operation consisting of converting linear light to non-linear ("gamma") encoding.
 *
 * @param transfer transfer functions
 * @param params parameters
 * @param cpu create operation optimized for given cpu
 * @return concrete operation
 */
std::unique_ptr<Operation> create_gamma_operation(const TransferFunction &func, const OperationParams &params, CPUClass cpu);

/**
 * Create operation consisting of converting non-linear ("gamma") encoding to linear light.
 *
 * @see create_gamma_operation
 */
std::unique_ptr<Operation> create_inverse_gamma_operation(const TransferFunction &func, const OperationParams &params, CPUClass cpu);

/**
 * Create operation consisting of converting linear light to ARIB STD-B67 using display-referred EOTF.
 *
 * @param m RGB to YUV conversion matrix for color primaries
 * @param params parameters
 * @return concrete operation
 */
std::unique_ptr<Operation> create_arib_b67_operation(const Matrix3x3 &m, const OperationParams &params);

/**
 * Create operation consisting of converting ARIB STD-B67 to linear light using display-referred EOTF.
 *
 * @param m RGB to YUV conversion matrix for color primaries
 * @param params parameters
 * @return concrete operation
 */
std::unique_ptr<Operation> create_inverse_arib_b67_operation(const Matrix3x3 &m, const OperationParams &params);

} // namespace zimg::colorspace

#endif // ZIMG_COLORSPACE_OPERATION_IMPL_H_
