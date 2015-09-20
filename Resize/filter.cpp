#include <algorithm>
#include <cstddef>
#include <cmath>
#include <vector>
#include "Common/except.h"
#include "Common/matrix.h"
#include "filter.h"

namespace zimg {;
namespace resize {;

namespace {;

const double PI = 3.14159265358979323846;

double sinc(double x)
{
	// Guaranteed to not yield division by zero on IEEE machine with accurate sin(x).
	return x == 0.0 ? 1.0 : std::sin(x * PI) / (x * PI);
}

double sq(double x)
{
	return x * x;
}

double cube(double x)
{
	return x * x * x;
}

FilterContext matrix_to_filter(const RowMatrix<double> &m)
{
	size_t width = 0;

	for (size_t i = 0; i < m.rows(); ++i) {
		width = std::max(width, m.row_right(i) - m.row_left(i));
	}

	FilterContext e{};

	e.filter_width = (unsigned)width;
	e.filter_rows = (unsigned)m.rows();
	e.input_width = (unsigned)m.cols();
	e.stride = (unsigned)align(width, AlignmentOf<float>::value);
	e.stride_i16 = (unsigned)align(width, AlignmentOf<uint16_t>::value);
	e.data.resize((size_t)e.stride * e.filter_rows);
	e.data_i16.resize((size_t)e.stride_i16 * e.filter_rows);
	e.left.resize(e.filter_rows);

	for (size_t i = 0; i < m.rows(); ++i) {
		unsigned left = (unsigned)std::min(m.row_left(i), m.cols() - width);

		for (size_t j = 0; j < width; ++j) {
			float coeff = (float)m[i][left + j];
			int16_t coeff_i16 = (int16_t)std::round(coeff * (float)(1 << 14));

			e.data[i * e.stride + j] = coeff;
			e.data_i16[i * e.stride_i16 + j] = coeff_i16;
		}
		e.left[i] = left;
	}

	return e;
}

} // namespace


Filter::~Filter()
{
}

int PointFilter::support() const
{
	return 0;
}

double PointFilter::operator()(double x) const
{
	return 1.0;
}

int BilinearFilter::support() const
{
	return 1;
}

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
{
}

int BicubicFilter::support() const
{
	return 2;
}

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

int Spline16Filter::support() const
{
	return 2;
}

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

int Spline36Filter::support() const
{
	return 3;
}

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

LanczosFilter::LanczosFilter(int taps) : taps(taps)
{
}

int LanczosFilter::support() const
{
	return taps;
}

double LanczosFilter::operator()(double x) const
{
	x = std::abs(x);
	return x < taps ? sinc(x) * sinc(x / taps) : 0.0;
}

FilterContext compute_filter(const Filter &f, int src_dim, int dst_dim, double shift, double width)
{
	double scale = (double)dst_dim / width;
	double step = std::min(scale, 1.0);
	double support = (double)f.support() / step;
	int filter_size = std::max((int)std::ceil(support * 2), 1);

	if (std::abs(shift) >= src_dim || shift + width >= 2 * src_dim)
		throw zimg::error::ResamplingNotAvailable{ "image shift or subwindow too great" };
	if (src_dim <= support || width <= support)
		throw zimg::error::ResamplingNotAvailable{ "filter width too great for image dimensions" };

	// Preserving center position with point upsampling filter is impossible.
	// Instead, the top-left position is preserved to avoid mirroring artifacts.
	if (filter_size == 1 && scale >= 1.0)
		shift += 0.5;

	RowMatrix<double> m{ (size_t)dst_dim, (size_t)src_dim };
	for (int i = 0; i < dst_dim; ++i) {
		// Position of output sample on input grid.
		double pos = (i + 0.5) / scale + shift;
		double begin_pos = std::floor(pos + support - filter_size + 0.5) + 0.5;

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

			m[i][(size_t)std::floor(real_pos)] += f((xpos - pos) * step) / total;
		}
	}

	return matrix_to_filter(m);
}

} // namespace resize
} // namespace zimg
