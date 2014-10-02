#pragma once

#ifndef ZIMG_COLORSPACE_OPERATION_IMPL_H_
#define ZIMG_COLORSPACE_OPERATION_IMPL_H_

#include "Common/cpuinfo.h"
#include "matrix3.h"

namespace zimg {;
namespace colorspace {;

class Operation;

/**
 * Create operation consisting of applying a 3x3 matrix to each pixel triplet.
 *
 * @param m matrix
 * @param cpu create operation optimized for given cpu
 */
Operation *create_matrix_operation(const Matrix3x3 &m, CPUClass cpu);

/**
 * Create operation consisting of applying Rec.709 transfer function.
 *
 * @param cpu create operation optimized for given cpu
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
	