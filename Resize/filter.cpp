#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include "filter.h"

#define M_PI 3.14159265358979323846

namespace {;

using namespace resize;

double sinc(double x)
{
	return std::abs(x) < 0.0001 ? 1.0 : std::sin(x * (double)M_PI) / (x * (double)M_PI);
}

double sq(double x)
{
	return x * x;
}

double cube(double x)
{
	return x * x * x;
}

class Matrix {
	int m_width;
	int m_height;
	std::vector<double> m_matrix;
public:
	Matrix() = default;

	Matrix(int width, int height) : m_width{ width }, m_height{ height }, m_matrix(width * height)
	{}

	int width() const { return m_width; }

	int height() const { return m_height; }

	double &at(int i, int j)
	{
		if (i > m_height || j > m_width || i < 0 || j < 0)
			throw std::out_of_range{ "matrix index out of bounds" };

		return m_matrix.at(i * m_width + j);
	}

	double at(int i, int j) const
	{
		return const_cast<Matrix *>(this)->at(i, j);
	}
};

EvaluatedFilter compress_matrix(const Matrix &m)
{
	int width = 0;

	for (int i = 0; i < m.height(); ++i) {
		int left;
		int right;

		for (left = 0; left < m.width(); ++left) {
			if (m.at(i, left))
				break;
		}
		for (right = m.width(); right > left; --right) {
			if (m.at(i, right - 1))
				break;
		}

		width = std::max(width, right - left);
	}

	EvaluatedFilter e{ width, m.height() };

	for (int i = 0; i < m.height(); ++i) {
		int left;

		for (left = 0; left < m.width() - width; ++left) {
			if (m.at(i, left))
				break;
		}

		for (int j = 0; j < width; ++j) {
			e.data()[i * e.stride() + j] = (float)m.at(i, left + j);
		}
		e.left()[i] = left;
	}

	return e;
}

}; // namespace


namespace resize {;

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
	m_stride{ align(width, 8) },
	m_data(m_stride * height),
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

int EvaluatedFilter::stride() const
{
	return m_stride;
}

float *EvaluatedFilter::data()
{
	return m_data.data();
}

const float *EvaluatedFilter::data() const
{
	return m_data.data();
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

	if (-shift >= src_dim || shift + width >= 2 * src_dim)
		throw std::domain_error{ "window too far" };
	if (src_dim <= support)
		throw std::domain_error{ "filter too wide" };
	if (width <= support)
		throw std::domain_error{ "subwindow too small" };

	double minpos = 0.5f;
	double maxpos = (double)src_dim - 0.5f;

	Matrix m{ src_dim, dst_dim };
	for (int i = 0; i < dst_dim; ++i) {
		// Position of output sample on input grid.
		double pos = (i + 0.5f) / scale + shift;
		double begin_pos = std::floor(pos + support - filter_size + 0.5f) + 0.5f;

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

			m.at(i, (int)real_pos) += f((xpos - pos) * step) / total;
		}
	}

	return compress_matrix(m);
}

} // namespace resize
