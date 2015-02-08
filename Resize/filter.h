#pragma once

#ifndef ZIMG_RESIZE_FILTER_H_
#define ZIMG_RESIZE_FILTER_H_

#include <cstddef>
#include <cstdint>
#include "Common/align.h"

namespace zimg {;
namespace resize {;

/**
 * Functor to compute filter taps.
 */
class Filter {
public:
	/**
	 * Destroy filter.
	 */
	virtual ~Filter() = 0;

	/**
	 * @return filter support
	 */
	virtual int support() const = 0;

	/**
	 * @param x position to evaluate
	 * @return filter coefficient at position
	 */
	virtual double operator()(double x) const = 0;
};

/**
 * Point (a.k.a. nearest neighbor) filter.
 */
class PointFilter : public Filter {
public:
	int support() const override;

	double operator()(double x) const override;
};

/**
 * Bilinear (a.k.a. triangle) filter.
 */
class BilinearFilter : public Filter {
public:
	int support() const override;

	double operator()(double x) const override;
};

/**
 * Bicubic (a.k.a. Mitchell-Netravali) filter.
 */
class BicubicFilter : public Filter {
	double p0, p2, p3;
	double q0, q1, q2, q3;
public:
	/**
	 * Initialize a BicubicFilter with given parameters.
	 *
	 * @param b "b" parameter to bicubic filter
	 * @param c "c" parameter to bicubic filter
	 */
	BicubicFilter(double b, double c);

	int support() const override;

	double operator()(double x) const override;
};

/**
 * Spline16 filter from Avisynth.
 */
class Spline16Filter : public Filter {
public:
	int support() const override;

	double operator()(double x) const override;
};

/**
 * Spline36 filter from Avisynth.
 */
class Spline36Filter : public Filter {
public:
	int support() const override;

	double operator()(double x) const override;
};

/**
 * Lanczos filter.
 */
class LanczosFilter : public Filter {
	int taps;
public:
	/**
	 * Initialize a LanczosFilter for a given number of taps.
	 *
	 * @param taps number of taps
	 */
	LanczosFilter(int taps);

	int support() const override;

	double operator()(double x) const override;
};

/**
 * Computed filter taps for a given scale and shift.
 */
struct FilterContext {
	/**
	 * Number of coefficients in a filter row.
	 */
	int filter_width;

	/**
	 * Total number of filter rows.
	 */
	int filter_rows;

	/**
	 * Distance between filter rows in units of coefficients.
	 */
	ptrdiff_t stride;
	ptrdiff_t stride_i16;

	/**
	 * Filter data.
	 */
	AlignedVector<float> data;
	AlignedVector<int16_t> data_i16;

	/**
	 * Indices of leftmost non-zero coefficients.
	 */
	AlignedVector<int> left;
};

/**
 * Compute the resizing function (matrix) for a filter, scale, and shift.
 * The destination buffer should be allocated in accordance with get_filter_size
 *
 * @param f filter
 * @param src_dim source dimension in pixels
 * @param dst_dim target dimension in pixels
 * @param shift shift to apply in units of source pixels
 * @param width active subwindow in units of source pixels
 * @return the computed filter
 * @throws ZimgIllegalArgument on unsupported parameter combinations
 */
FilterContext compute_filter(const Filter &f, int src_dim, int dst_dim, double shift, double width);

} // namespace resize
} // namespace zimg

#endif // ZIMG_RESIZE_FILTER_H_
