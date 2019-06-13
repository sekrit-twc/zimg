#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#include "common/except.h"
#include "common/libm_wrapper.h"
#include "common/matrix.h"
#include "common/zassert.h"
#include "filter.h"

namespace zimg {
namespace resize {

namespace {

constexpr double PI = 3.14159265358979323846;

double sinc(double x) noexcept
{
	// Guaranteed to not yield division by zero on IEEE machine with accurate sin(x).
	return x == 0.0 ? 1.0 : zimg_x_sin(x * PI) / (x * PI);
}

double poly3(double x, double c0, double c1, double c2, double c3) noexcept
{
	return c0 + x * (c1 + x * (c2 + x * c3));
}

double round_halfup(double x) noexcept
{
	/* When rounding on the pixel grid, the invariant
	 *   round(x - 1) == round(x) - 1
	 * must be preserved. This precludes the use of modes such as
	 * half-to-even and half-away-from-zero.
	 */
	bool sign = std::signbit(x);

	x = std::round(std::abs(x));
	return sign ? -x : x;
}


FilterContext matrix_to_filter(const RowMatrix<double> &m)
{
	size_t width = 0;

	for (size_t i = 0; i < m.rows(); ++i) {
		width = std::max(width, m.row_right(i) - m.row_left(i));
	}
	zassert_d(width, "empty matrix");

	if (width > floor_n(UINT_MAX, AlignmentOf<uint16_t>::value))
		error::throw_<error::OutOfMemory>();
	if (width > floor_n(UINT_MAX, AlignmentOf<float>::value))
		error::throw_<error::OutOfMemory>();

	FilterContext e{};

	try {
		e.filter_width = static_cast<unsigned>(width);
		e.filter_rows = static_cast<unsigned>(m.rows());
		e.input_width = static_cast<unsigned>(m.cols());
		e.stride = static_cast<unsigned>(ceil_n(width, AlignmentOf<float>::value));
		e.stride_i16 = static_cast<unsigned>(ceil_n(width, AlignmentOf<uint16_t>::value));

		if (e.filter_rows > UINT_MAX / e.stride || e.filter_rows > UINT_MAX / e.stride_i16)
			error::throw_<error::OutOfMemory>();

		e.data.resize(static_cast<size_t>(e.stride) * e.filter_rows);
		e.data_i16.resize(static_cast<size_t>(e.stride_i16) * e.filter_rows);
		e.left.resize(e.filter_rows);
	} catch (const std::length_error &) {
		error::throw_<error::OutOfMemory>();
	}

	for (size_t i = 0; i < m.rows(); ++i) {
		unsigned left = static_cast<unsigned>(std::min(m.row_left(i), m.cols() - width));
		double f32_err = 0.0f;
		double i16_err = 0;

		double f32_sum = 0.0;
		int16_t i16_sum = 0;
		int16_t i16_greatest = 0;
		size_t i16_greatest_idx = 0;

		/* Dither filter coefficients when rounding them to their storage format.
		 * This minimizes accumulation of error and ensures that the filter
		 * continues to sum as close to 1.0 as possible after rounding.
		 */
		for (size_t j = 0; j < width; ++j) {
			double coeff = m[i][left + j];

			double coeff_expected_f32 = coeff - f32_err;
			double coeff_expected_i16 = coeff * (1 << 14) - i16_err;

			float coeff_f32 = static_cast<float>(coeff_expected_f32);
			int16_t coeff_i16 = static_cast<int16_t>(std::lrint(coeff_expected_i16));

			f32_err = static_cast<double>(coeff_f32) - coeff_expected_f32;
			i16_err = static_cast<double>(coeff_i16) - coeff_expected_i16;

			if (std::abs(coeff_i16) > i16_greatest) {
				i16_greatest = coeff_i16;
				i16_greatest_idx = j;
			}

			f32_sum += coeff_f32;
			i16_sum += coeff_i16;

			e.data[i * e.stride + j] = coeff_f32;
			e.data_i16[i * e.stride_i16 + j] = coeff_i16;
		}

		/* The final sum may still be off by a few ULP. This can not be fixed for
		 * floating point data, since the error is dependent on summation order,
		 * but for integer data, the error can be added to the greatest coefficient.
		 */
		zassert_d(1.0 - f32_sum <= FLT_EPSILON, "error too great");
		zassert_d(std::abs((1 << 14) - i16_sum) <= 1, "error too great");

		e.data_i16[i * e.stride_i16 + i16_greatest_idx] += (1 << 14) - i16_sum;

		e.left[i] = left;
	}

	return e;
}

} // namespace


Filter::~Filter() = default;

unsigned PointFilter::support() const { return 0; }

double PointFilter::operator()(double x) const { return 1.0; }


unsigned BilinearFilter::support() const { return 1; }

double BilinearFilter::operator()(double x) const
{
	return std::max(1.0 - std::abs(x), 0.0);
}


BicubicFilter::BicubicFilter(double b, double c) :
	p0{ (  6.0 -  2.0 * b           ) / 6.0 },
	p2{ (-18.0 + 12.0 * b +  6.0 * c) / 6.0 },
	p3{ ( 12.0 -  9.0 * b -  6.0 * c) / 6.0 },
	q0{ (         8.0 * b + 24.0 * c) / 6.0 },
	q1{ (       -12.0 * b - 48.0 * c) / 6.0 },
	q2{ (         6.0 * b + 30.0 * c) / 6.0 },
	q3{ (              -b -  6.0 * c) / 6.0 }
{}

unsigned BicubicFilter::support() const { return 2; }

double BicubicFilter::operator()(double x) const
{
	x = std::abs(x);

	if (x < 1.0)
		return poly3(x, p0, 0.0, p2, p3);
	else if (x < 2.0)
		return poly3(x, q0, q1, q2, q3);
	else
		return 0.0;
}


unsigned Spline16Filter::support() const { return 2; }

double Spline16Filter::operator()(double x) const
{
	x = std::abs(x);

	if (x < 1.0) {
		return poly3(x, 1.0,  -1.0 / 5.0, -9.0 / 5.0, 1.0);
	} else if (x < 2.0) {
		x -= 1.0;
		return poly3(x, 0.0, -7.0 / 15.0,  4.0 / 5.0, -1.0 / 3.0);
	} else {
		return 0.0;
	}
}


unsigned Spline36Filter::support() const { return 3; }

double Spline36Filter::operator()(double x) const
{
	x = std::abs(x);

	if (x < 1.0) {
		return poly3(x, 1.0,   -3.0 / 209.0, -453.0 / 209.0, 13.0 / 11.0);
	} else if (x < 2.0) {
		x -= 1.0;
		return poly3(x, 0.0, -156.0 / 209.0,  270.0 / 209.0, -6.0 / 11.0);
	} else if (x < 3.0) {
		x -= 2.0;
		return poly3(x, 0.0,   26.0 / 209.0,  -45.0 / 209.0,  1.0 / 11.0);
	} else {
		return 0.0;
	}
}


LanczosFilter::LanczosFilter(unsigned taps) : taps{ taps }
{
	if (taps <= 0)
		error::throw_<error::IllegalArgument>("lanczos tap count must be positive");
}

unsigned LanczosFilter::support() const { return taps; }

double LanczosFilter::operator()(double x) const
{
	x = std::abs(x);
	return x < taps ? sinc(x) * sinc(x / taps) : 0.0;
}


FilterContext compute_filter(const Filter &f, unsigned src_dim, unsigned dst_dim, double shift, double width)
{
	double scale = static_cast<double>(dst_dim) / width;
	double step = std::min(scale, 1.0);
	double support = static_cast<double>(f.support()) / step;
	unsigned filter_size = std::max(static_cast<unsigned>(std::ceil(support)) * 2U, 1U);

	if (support > static_cast<unsigned>(UINT_MAX / 2))
		error::throw_<error::ResamplingNotAvailable>("filter width too great");

	try {
		RowMatrix<double> m{ dst_dim, src_dim };

		for (unsigned i = 0; i < dst_dim; ++i) {
			// Position of output sample on input grid.
			double pos = (i + 0.5) / scale + shift;
			double begin_pos = round_halfup(pos - filter_size / 2.0) + 0.5;

			double total = 0.0;
			for (unsigned j = 0; j < filter_size; ++j) {
				double xpos = begin_pos + j;
				total += f((xpos - pos) * step);
			}

			size_t left = SIZE_MAX;

			for (unsigned j = 0; j < filter_size; ++j) {
				double xpos = begin_pos + j;
				double real_pos;

				// Mirror the position if it goes beyond image bounds.
				if (xpos < 0.0)
					real_pos = -xpos;
				else if (xpos >= src_dim)
					real_pos = 2.0 * src_dim - xpos;
				else
					real_pos = xpos;

				// Clamp the position if it is still out of bounds.
				real_pos = std::min(std::max(real_pos, 0.0), std::nextafter(src_dim, -INFINITY));

				size_t idx = static_cast<size_t>(std::floor(real_pos));
				m[i][idx] += f((xpos - pos) * step) / total;
				left = std::min(left, idx);
			}

			// Force allocating an entry to keep the left offset table sorted.
			if (m[i][left] == 0.0) {
				m[i][left] = DBL_EPSILON;
				m[i][left] = 0.0;
			}
		}

		return matrix_to_filter(m);
	} catch (const std::length_error &) {
		error::throw_<error::OutOfMemory>();
	}
}

} // namespace resize
} // namespace zimg
