#pragma once

#ifndef ZIMG_UNRESIZE_UNRESIZE_H_
#define ZIMG_UNRESIZE_UNRESIZE_H_

#include <utility>

namespace zimg {;

enum class CPUClass;
enum class PixelType;

namespace graph {;

class ImageFilter;

} // namespace graph


namespace unresize {;

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
 */
std::pair<graph::ImageFilter *, graph::ImageFilter *> create_unresize(
	PixelType type, unsigned src_Width, unsigned src_height, unsigned dst_width, unsigned dst_height,
	double shift_w, double shift_h, CPUClass cpu);

} // namespace unresize
} // namespace zimg

#endif // ZIMG_UNRESIZE_UNRESIZE_H_
