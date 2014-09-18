#ifndef ZIMG_UNRESIZE_H_
#define ZIMG_UNRESIZE_H_

#include <cstdint>
#include <memory>
#include "Common/pixel.h"

namespace zimg {;
namespace unresize {;

class UnresizeImpl;

/**
 * Unresize: reverses the effect of the bilinear scaling method.
 *
 * Linear interpolation in one dimension from an input dimension N to an
 * output dimension M can be represented as the matrix product:
 *
 * A x = y
 *
 * A is the interpolation function
 * x is the original vector
 * y is the resized vector
 *
 *
 * Unresize attempts to recover x given the resized vector y.
 * This is done by the method of least squares.
 *
 * A' A x = A' y
 *
 * A' is the transpose of A
 *
 *
 * The problem resolves to solving a linear system.
 *
 * P x = y'
 *
 * P is (A' A)
 * y' is (A' y)
 *
 *
 * Given the width of the bilinear filter, P is a tridiagonal matrix of
 * dimension N, and so the system can be solved by simple substitution after
 * LU factorization.
 *
 * Using a convention that U has a main diagonal of ones, the factoization is
 * given by the following.
 *
 *
 * The following names will be given to relevant diagonals.
 *
 * a(i) = P(i, i)
 * b(i) = P(i, i + 1)
 * c(i) = P(i, i - 1)
 * l(i) = L(i, i)
 * u(i) = U(i, i + 1)
 *
 * The computation of l and u can be described by the following procedure.
 *
 * l(1) = a(1)
 * u(1) = b(1) / a(1)
 *
 * FOR (i = 1 : N - 1)
 *   l(i) = a(i) - c(i) * u(i - 1)
 *   u(i) = b(i) / l(i)
 *
 * l(N) = a(N) - c(N) * u(N - 1)
 *
 *
 * The solution to the system can be described by the procedure.
 *
 * L U x = y'
 *
 * z(1) = y'(1) / l(1)
 * FOR (i = 2 : N)
 *   z(i) = (y'(i) - c(i) * z(i - 1)) / l(i)
 *
 * x(N) = z(N)
 * FOR (i = N - 1 : 1)
 *   x(i) = z(i) - u(i) * x'(i + 1)
 *
 *
 * The implementation of Unresize caches the values of P, l, u, and c for given
 * dimensions N and M. Execution is done by first computing y' and then
 * performing the tridiagonal algorithm to obtain x.
 *
 * Generalization to two dimensions is done by processing each dimension.
 *
 *
 * In the class comments below, "input" refers to the upsampled image
 * and "output" refers to the unresized image.
 */
class Unresize {
	int m_src_width;
	int m_src_height;
	int m_dst_width;
	int m_dst_height;

	std::shared_ptr<UnresizeImpl> m_impl;
public:
	/**
	 * Initialize a null context. Cannot be used for execution.
	 */
	Unresize() = default;

	/**
	 * Initialize a context to unresize a given bilinear resampling.
	 *
	 * @param dst_width output image width
	 * @param dst_height output image height
	 * @param src_width input image width
	 * @param src_height input image height
	 * @param shift_w horizontal center shift relative to upsampled image
	 * @param shift_h vertical center shift relative to upsampled image
	 * @param x86 whether to create an x86-optimized kernel
	 * @throws ZimgIllegalArgument on invalid dimensions
	 * @throws ZimgUnsupportedError if not supported
	 * @throws ZimgOutOfMemory if out of memory
	 */
	Unresize(int src_width, int src_height, int dst_width, int dst_height, float shift_w, float shift_h, bool x86);

	/**
	 * Destroy context.
	 */
	~Unresize();

	/**
	 * @param dst output format
	 * @param src input format
	 * @return the size of the required temporary buffer in floats
	 */
	size_t tmp_size(PixelType src, PixelType dst) const;

	/**
	 * Process an image. All pointers must be 32-byte aligned.
	 *
	 * @param dst pointer to output buffer
	 * @param src pointer to input buffer
	 * @param dst_stride stride of output buffer
	 * @param src_stride stride of input buffer
	 * @param dst_type format of output
	 * @param src_type format of input
	 * @param tmp pointer to temporay buffer of sufficient size (@see Unresize::tmp_size)
	 */
	void process(const uint8_t *src, uint8_t *dst, float *tmp, int src_stride, int dst_stride, PixelType src_type, PixelType dst_type) const;

	/**
	 * @see Unresize::process(uint8_t*, const uint8_t*, int, int, PixelType, PixelType, float*) const
	 */
	void process(const float *src, float *dst, float *tmp, int src_stride, int dst_stride) const;
};

} // namespace unresize
} // namespace zimg

#endif // ZIMG_UNRESIZE_H_
