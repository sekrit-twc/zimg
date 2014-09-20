#include <algorithm>
#include "bilinear.h"
#include "matrix.h"

namespace zimg {;
namespace unresize {;

namespace {;

/**
 * Compute the coefficients for a bilinear scaling matrix.
 *
 * @param in input (unscaled) dimension
 * @param out output (scaled) dimension
 * @param shift center shift applied to input
 * @return the scaling matrix
 */
RowMatrix bilinear_weights(int in, int out, float shift)
{
	RowMatrix m(out, in);

	// Position of outermost samples on input grid.
	float leftmost = 0.5f + shift;
	float rightmost = in - 0.5f + shift;

	// Indices corresponding to the samples stored at leftmost and rightmost.
	int leftmost_idx = std::max((int)leftmost, 0);
	int rightmost_idx = std::min((int)rightmost, in - 1);

	for (int i = 0; i < out; ++i) {
		// Position of output sample on input grid.
		float position = (i + 0.5f) * (float)in / (float)out;

		// For samples outside the input range, mirror the nearest input pixel.
		if (position <= leftmost) {
			m[i][leftmost_idx] = 1.f;
		} else if (position >= rightmost) {
			m[i][rightmost_idx] = 1.f;
		} else {
			// Index of nearest input pixels to output position.
			int left_idx = (int)(position - leftmost);
			int right_idx = left_idx + 1;

			// Distance between output position and left input.
			float distance = position - left_idx - leftmost;

			float left_weight = 1.f - distance;
			float right_weight = distance;

			m[i][left_idx] = left_weight;
			m[i][right_idx] = right_weight;
		}
	}

	return m;
}

} // namespace


BilinearContext create_bilinear_context(int in, int out, float shift)
{
	BilinearContext ctx;

	// Map output shift to input shift.
	RowMatrix m = bilinear_weights(in, out, -shift * (float)in / (float)out);
	RowMatrix transpose_m = transpose(m);
	RowMatrix pinv_m = matrix_matrix_product(transpose_m, m);
	TridiagonalLU lu = tridiagonal_decompose(pinv_m);

	int rows = transpose_m.rows();
	int cols = transpose_m.cols();

	ctx.dst_width = in;

	int rowsize = 0;
	for (int i = 0; i < rows; ++i) {
		rowsize = std::max(transpose_m.row_right(i) - transpose_m.row_left(i), rowsize);
	}
	rowsize;
	int rowstride = align(rowsize, 8);

	ctx.matrix_coefficients.resize(rowstride * rows);
	ctx.matrix_row_offsets.resize(rows);
	ctx.matrix_row_size = rowsize;
	ctx.matrix_row_stride = rowstride;
	for (int i = 0; i < rows; ++i) {
		int left = std::max(std::min(transpose_m.row_left(i), cols - rowsize), 0);

		for (int j = 0; j < transpose_m.row_right(i) - left; ++j) {
			ctx.matrix_coefficients[i * rowstride + j] = transpose_m[i][left + j];
		}
		ctx.matrix_row_offsets[i] = left;
	}

	ctx.lu_c.resize(rows);
	ctx.lu_l.resize(rows);
	ctx.lu_u.resize(rows);
	for (int i = 0; i < rows; ++i) {
		ctx.lu_c[i] = lu.c(i);
		ctx.lu_l[i] = 1.f / (lu.l(i) + 0.001f); // Pre-invert this value, as it is used in division.
		ctx.lu_u[i] = lu.u(i);
	}

	return ctx;
}

} // namespace unresize
} // namespace zimg
