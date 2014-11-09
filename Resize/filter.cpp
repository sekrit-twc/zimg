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
	return std::abs(x) < 0.0001 ? 1.0 : std::sin(x * PI) / (x * PI);
}

double sq(double x)
{
	return x * x;
}

double cube(double x)
{
	return x * x * x;
}

EvaluatedFilter matrix_to_filter(const RowMatrix<double> &m)
{
	size_t width = 0;

	for (size_t i = 0; i < m.rows(); ++i) {
		width = std::max(width, m.row_right(i) - m.row_left(i));
	}

	EvaluatedFilter e{ (int)width, (int)m.rows() };

	for (size_t i = 0; i < m.rows(); ++i) {
		int left = (int)std::min(m.row_left(i), m.cols() - width);

		for (size_t j = 0; j < width; ++j) {
			float coeff = (float)m[i][left + j];
			int16_t coeff_i16 = (int16_t)std::round(coeff * (float)(1 << 14));

			e.data()[(ptrdiff_t)i * e.stride() + j] = coeff;
			e.data_i16()[(ptrdiff_t)i * e.stride_i16() + j] = coeff_i16;
		}
		e.left()[i] = left;
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


EvaluatedFilter::EvaluatedFilter(int width, int height) :
	m_width{ width },
	m_height{ height },
	m_stride{ align(width, AlignmentOf<float>::value) },
	m_stride_i16{ align(width, AlignmentOf<int16_t>::value) },
	m_data((size_t)m_stride * height),
	m_data_i16((size_t)m_stride_i16 * height),
	m_left(height)
{
}

int EvaluatedFilter::width() const
{
	return m_width;
}

int EvaluatedFilter::height() const
{
	return m_height;
}

ptrdiff_t EvaluatedFilter::stride() const
{
	return m_stride;
}

ptrdiff_t EvaluatedFilter::stride_i16() const
{
	return m_stride_i16;
}

float *EvaluatedFilter::data()
{
	return m_data.data();
}

const float *EvaluatedFilter::data() const
{
	return m_data.data();
}

int16_t *EvaluatedFilter::data_i16()
{
	return m_data_i16.data();
}

const int16_t *EvaluatedFilter::data_i16() const
{
	return m_data_i16.data();
}

int *EvaluatedFilter::left()
{
	return m_left.data();
}

const int *EvaluatedFilter::left() const
{
	return m_left.data();
}


EvaluatedFilter compute_filter(const Filter &f, int src_dim, int dst_dim, double shift, double width)
{
	double scale = (double)dst_dim / width;
	double step = std::min(scale, 1.0);
	double support = (double)f.support() / step;
	int filter_size = std::max((int)std::ceil(support * 2), 1);

	if (std::abs(shift) >= src_dim || shift + width >= 2 * src_dim)
		throw ZimgIllegalArgument{ "window too far" };
	if (src_dim <= support)
		throw ZimgIllegalArgument{ "filter too wide" };
	if (width <= support)
		throw ZimgIllegalArgument{ "subwindow too small" };

	double minpos = 0.5;
	double maxpos = (double)src_dim - 0.5;

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
			if (xpos < minpos)
				real_pos = 2.0 * minpos - xpos;
			else if (xpos > maxpos)
				real_pos = 2.0 * maxpos - xpos;
			else
				real_pos = xpos;

			m[i][(size_t)std::floor(real_pos)] += f((xpos - pos) * step) / total;
		}
	}

	return matrix_to_filter(m);
}

} // namespace resize
} // namespace zimg
