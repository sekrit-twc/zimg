#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include "common/except.h"
#include "common/matrix.h"
#include "common/zassert.h"
#include "bilinear.h"

namespace zimg {
namespace unresize {

namespace {

template <class T>
T epsilon()
{
	return std::numeric_limits<T>::epsilon();
}

template <class T>
struct TridiagonalLU {
	std::vector<T> l;
	std::vector<T> u;
	std::vector<T> c;

	explicit TridiagonalLU(size_t n) : l(n), u(n), c(n)
	{}
};

template <class T>
TridiagonalLU<T> tridiagonal_decompose(const RowMatrix<T> &m)
{
	size_t n = m.rows();
	TridiagonalLU<T> lu{ n };
	T zero = static_cast<T>(0);
	T eps = epsilon<T>();

	lu.c[0] = zero;
	lu.l[0] = m[0][0];
	lu.u[0] = m[0][1] / (m[0][0] + eps);

	for (size_t i = 1; i < n - 1; ++i) {
		lu.c[i] = m[i][i - 1];
		lu.l[i] = m[i][i] - lu.c[i] * lu.u[i - 1];
		lu.u[i] = m[i][i + 1] / (lu.l[i] + eps);
	}

	lu.c[n - 1] = m[n - 1][n - 2];
	lu.l[n - 1] = m[n - 1][n - 1] - lu.c[n - 1] * lu.u[n - 2];
	lu.u[n - 1] = zero;

	return lu;
}

/**
 * Compute the coefficients for a bilinear scaling matrix.
 *
 * @param in input (unscaled) dimension
 * @param out output (scaled) dimension
 * @param shift center shift applied to input
 * @return the scaling matrix
 */
RowMatrix<double> bilinear_weights(unsigned in, unsigned out, double shift)
{
	RowMatrix<double> m{ out, in };

	// Position of outermost samples on input grid.
	double leftmost = 0.5 + shift;
	double rightmost = in - 0.5 + shift;

	// Indices corresponding to the samples stored at leftmost and rightmost.
	unsigned leftmost_idx = static_cast<unsigned>(std::min(std::max(std::floor(leftmost), 0.0), in - 1.0));
	unsigned rightmost_idx = static_cast<unsigned>(std::min(std::max(std::floor(rightmost), 0.0), in - 1.0));

	for (unsigned i = 0; i < out; ++i) {
		// Position of output sample on input grid.
		double position = (i + 0.5) * in / out;

		// For samples outside the input range, mirror the nearest input pixel.
		if (position <= leftmost) {
			m[i][leftmost_idx] = 1.0;
		} else if (position >= rightmost) {
			m[i][rightmost_idx] = 1.0;
		} else {
			// Index of nearest input pixels to output position.
			unsigned left_idx = static_cast<unsigned>(std::floor(position - leftmost));
			unsigned right_idx = left_idx + 1;

			// Distance between output position and left input.
			double distance = position - left_idx - leftmost;

			double left_weight = 1.0 - distance;
			double right_weight = distance;

			m[i][left_idx] = left_weight;
			m[i][right_idx] = right_weight;
		}
	}

	return m;
}

} // namespace


BilinearContext create_bilinear_context(unsigned in, unsigned out, double shift)
{
	BilinearContext ctx;

	try {
		// Map output shift to input shift.
		RowMatrix<double> m = bilinear_weights(in, out, -shift * in / out);
		RowMatrix<double> transpose_m = ~m;
		RowMatrix<double> pinv_m = transpose_m * m;
		TridiagonalLU<double> lu = tridiagonal_decompose(pinv_m);

		size_t rows = transpose_m.rows();
		size_t cols = transpose_m.cols();

		ctx.input_width = out;
		ctx.output_width = in;

		size_t rowsize = 0;
		for (size_t i = 0; i < rows; ++i) {
			rowsize = std::max(transpose_m.row_right(i) - transpose_m.row_left(i), rowsize);
		}
		zassert_d(rowsize, "empty matrix");

		if (rowsize > floor_n(SIZE_MAX, AlignmentOf<float>::value))
			throw error::OutOfMemory{};

		size_t rowstride = ceil_n(rowsize, AlignmentOf<float>::value);
		if (rows > SIZE_MAX / rowstride)
			throw error::OutOfMemory{};

		ctx.matrix_coefficients.resize(rowstride * rows);
		ctx.matrix_row_offsets.resize(rows);
		ctx.matrix_row_size = static_cast<unsigned>(rowsize);
		ctx.matrix_row_stride = static_cast<unsigned>(rowstride);
		for (size_t i = 0; i < rows; ++i) {
			size_t left = std::min(transpose_m.row_left(i), cols - rowsize);

			for (size_t j = 0; j < transpose_m.row_right(i) - left; ++j) {
				ctx.matrix_coefficients[i * rowstride + j] = static_cast<float>(transpose_m[i][left + j]);
			}
			ctx.matrix_row_offsets[i] = static_cast<unsigned>(left);
		}

		ctx.lu_c.resize(rows);
		ctx.lu_l.resize(rows);
		ctx.lu_u.resize(rows);
		for (size_t i = 0; i < rows; ++i) {
			ctx.lu_c[i] = static_cast<float>(lu.c[i]);
			ctx.lu_l[i] = static_cast<float>((1.0 / (lu.l[i] + epsilon<float>()))); // Pre-invert this value, as it is used in division.
			ctx.lu_u[i] = static_cast<float>(lu.u[i]);
		}
	} catch (const std::length_error &) {
		throw error::OutOfMemory{};
	}

	return ctx;
}

} // namespace unresize
} // namespace zimg
