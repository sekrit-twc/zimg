#pragma once

#ifndef ZIMG_UNRESIZE_BILINEAR_H_
#define ZIMG_UNRESIZE_BILINEAR_H_

#include "common/alloc.h"

namespace zimg {
namespace unresize {

/**
 * Execution context for unresize algorithm.
 *
 * See unresize.h for description of algorithm.
 *
 * Note: Although the struct comments below use one-based indices,
 *       all arrays are stored with conventional zero-based indexing.
 */
struct BilinearContext {
	/**
	 * Dimension of upsampled image (M).
	 */
	unsigned input_width;

	/**
	 * Dimension of unresized image (N).
	 */
	unsigned output_width;

	/**
	 * Packed storage of (A') as row + offset.
	 * The matrix is stored as a 2-D array of matrix_row_size rows
	 * and dst_width columns.
	 *
	 * Each row is a contiguous portion of a row in the full matrix (A').
	 * matrix_row_offsets stores the original column index of row band in (A').
	 *
	 * The relationship to the original matrix (A') is given by the following.
	 *
	 * matrix_coefficients(i, j) = A'(i, matrix_row_offsets(i) + j)
	 *
	 */
	AlignedVector<float> matrix_coefficients;
	AlignedVector<unsigned> matrix_row_offsets;
	unsigned matrix_row_size;
	unsigned matrix_row_stride;

	/**
	 * LU decomposition of (A' A) stored as three arrays of dimension (N).
	 *
	 * The relationship to L and U is given by the following.
	 *
	 * lu_c(i) = L(i, i - 1)
	 * lu_l(i) = 1 / L(i, i)
	 * lu_u(i) = U(i, i + 1)
	 *
	 * lu_c(1) and lu_u(N) are set to 0 to simplify the execution loop.
	 * lu_l is stored inverted as it is used in forward substitution as a divisor.
	 */
	AlignedVector<float> lu_c;
	AlignedVector<float> lu_l;
	AlignedVector<float> lu_u;
};

/**
 * Initialize a BilinearContext for a given scaling factor.
 *
 * @param in dimension of original vector
 * @param out dimension of upscaled vector
 * @param shift center shift relative to upscaled vector
 * @return an initialized context
 */
BilinearContext create_bilinear_context(unsigned in, unsigned out, double shift);

} // namespace unresize
} // namespace zimg

#endif // ZIMG_UNRESIZE_BILINEAR_H_
