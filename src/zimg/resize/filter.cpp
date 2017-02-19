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

double sq(double x) noexcept { return x * x; }

double cube(double x) noexcept { return x * x * x; }

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
		throw error::OutOfMemory{};
	if (width > floor_n(UINT_MAX, AlignmentOf<float>::value))
		throw error::OutOfMemory{};

	FilterContext e{};

	try {
		e.filter_width = static_cast<unsigned>(width);
		e.filter_rows = static_cast<unsigned>(m.rows());
		e.input_width = static_cast<unsigned>(m.cols());
		e.stride = static_cast<unsigned>(ceil_n(width, AlignmentOf<float>::value));
		e.stride_i16 = static_cast<unsigned>(ceil_n(width, AlignmentOf<uint16_t>::value));

		if (e.filter_rows > UINT_MAX / e.stride || e.filter_rows > UINT_MAX / e.stride_i16)
			throw error::OutOfMemory{};

		e.data.resize(static_cast<size_t>(e.stride) * e.filter_rows);
		e.data_i16.resize(static_cast<size_t>(e.stride_i16) * e.filter_rows);
		e.left.resize(e.filter_rows);
	} catch (const std::length_error &) {
		throw error::OutOfMemory{};
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

int PointFilter::support() const { return 0; }

double PointFilter::operator()(double x) const { return 1.0; }


int BilinearFilter::support() const { return 1; }

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

int BicubicFilter::support() const { return 2; }

double BicubicFilter::operator()(double x) const
{
	x = std::abs(x);

	if (x < 1.0)
		return p0 +          p2 * sq(x) + p3 * cube(x);
	else if (x < 2.0)
		return q0 + q1 * x + q2 * sq(x) + q3 * cube(x);
	else
		return 0.0;
}


int Spline16Filter::support() const { return 2; }

double Spline16Filter::operator()(double x) const
{
	x = std::abs(x);

	if (x < 1.0) {
		return 1.0 - (1.0 / 5.0 * x)   - (9.0 / 5.0 * sq(x)) + cube(x);
	} else if (x < 2.0) {
		x -= 1.0;
		return       (-7.0 / 15.0 * x) + (4.0 / 5.0 * sq(x)) - (1.0 / 3.0 * cube(x));
	} else {
		return 0.0;
	}
}


int Spline36Filter::support() const { return 3; }

double Spline36Filter::operator()(double x) const
{
	x = std::abs(x);

	if (x < 1.0) {
		return 1.0 - (3.0 / 209.0 * x)    - (453.0 / 209.0 * sq(x)) + (13.0 / 11.0 * cube(x));
	} else if (x < 2.0) {
		x -= 1.0;
		return       (-156.0 / 209.0 * x) + (270.0 / 209.0 * sq(x)) - (6.0 / 11.0 * cube(x));
	} else if (x < 3.0) {
		x -= 2.0;
		return       (26.0 / 209.0 * x)   - (45.0 / 209.0 * sq(x))  + (1.0 / 11.0 * cube(x));
	} else {
		return 0.0;
	}
}


LanczosFilter::LanczosFilter(int taps) : taps{ taps }
{
	if (taps <= 0)
		throw error::IllegalArgument{ "lanczos tap count must be positive" };
}

int LanczosFilter::support() const { return taps; }

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
	int filter_size = std::max(static_cast<int>(std::ceil(support)) * 2, 1);

	if (std::abs(shift) >= src_dim || shift + width >= 2 * src_dim)
		throw error::ResamplingNotAvailable{ "image shift or subwindow too great" };
	if (src_dim <= support || width <= support)
		throw error::ResamplingNotAvailable{ "filter width too great for image dimensions" };

	try {
		RowMatrix<double> m{ dst_dim, src_dim };

		for (unsigned i = 0; i < dst_dim; ++i) {
			// Position of output sample on input grid.
			double pos = (i + 0.5) / scale + shift;
			double begin_pos = round_halfup(pos - filter_size / 2.0) + 0.5;

			double total = 0.0;
			for (int j = 0; j < filter_size; ++j) {
				double xpos = begin_pos + j;
				total += f((xpos - pos) * step);
			}

			for (int j = 0; j < filter_size; ++j) {
				double xpos = begin_pos + j;
				double real_pos;

				// Mirror the position if it goes beyond image bounds.
				if (xpos < 0.0)
					real_pos = -xpos;
				else if (xpos >= src_dim)
					real_pos = std::min(2.0 * src_dim - xpos, src_dim - 0.5);
				else
					real_pos = xpos;

				m[i][static_cast<size_t>(std::floor(real_pos))] += f((xpos - pos) * step) / total;
			}
		}

		return matrix_to_filter(m);
	} catch (const std::length_error &) {
		throw error::OutOfMemory{};
	}
}

} // namespace resize
} // namespace zimg
