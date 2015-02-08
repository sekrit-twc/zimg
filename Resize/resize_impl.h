#pragma once

#ifndef ZIMG_RESIZE_RESIZE_IMPL_H_
#define ZIMG_RESIZE_RESIZE_IMPL_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include "Common/osdep.h"
#include "Common/plane.h"
#include "filter.h"

namespace zimg {;

enum class CPUClass;

template <class T>
class ImagePlane;

namespace resize {;

struct ScalarPolicy_U16 {
	typedef int32_t num_type;

	FORCE_INLINE int32_t coeff(const FilterContext &filter, ptrdiff_t row, ptrdiff_t k)
	{
		return filter.data_i16[row * filter.stride_i16 + k];
	}

	FORCE_INLINE int32_t load(const uint16_t *src)
	{
		uint16_t x = *src;
		return (int32_t)x + (int32_t)INT16_MIN; // Make signed.
	}

	FORCE_INLINE void store(uint16_t *dst, int32_t x)
	{
		// Convert from 16.14 to 16.0.
		x = ((x + (1 << 13)) >> 14) - (int32_t)INT16_MIN;

		// Clamp out of range values.
		x = std::max(std::min(x, (int32_t)UINT16_MAX), (int32_t)0);

		*dst = (uint16_t)x;
	}
};

struct ScalarPolicy_F32 {
	typedef float num_type;

	FORCE_INLINE float coeff(const FilterContext &filter, ptrdiff_t row, ptrdiff_t k)
	{
		return filter.data[row * filter.stride + k];
	}

	FORCE_INLINE float load(const float *src) { return *src; }

	FORCE_INLINE void store(float *dst, float x) { *dst = x; }
};

template <class T, class Policy>
inline FORCE_INLINE void filter_plane_h_scalar(const FilterContext &filter, const ImagePlane<const T> &src, const ImagePlane<T> &dst,
                                               ptrdiff_t i_begin, ptrdiff_t i_end, ptrdiff_t j_begin, ptrdiff_t j_end, Policy policy)
{
	for (ptrdiff_t i = i_begin; i < i_end; ++i) {
		for (ptrdiff_t j = j_begin; j < j_end; ++j) {
			ptrdiff_t left = filter.left[j];
			typename Policy::num_type accum = 0;

			for (int k = 0; k < filter.filter_width; ++k) {
				typename Policy::num_type coeff = policy.coeff(filter, j, k);
				typename Policy::num_type x = policy.load(&src[i][left + k]);

				accum += coeff * x;
			}

			policy.store(&dst[i][j], accum);
		}
	}
}

template <class T, class Policy>
inline FORCE_INLINE void filter_plane_v_scalar(const FilterContext &filter, const ImagePlane<const T> &src, const ImagePlane<T> &dst,
                                               ptrdiff_t i_begin, ptrdiff_t i_end, ptrdiff_t j_begin, ptrdiff_t j_end, Policy policy)
{
	for (ptrdiff_t i = i_begin; i < i_end; ++i) {
		for (ptrdiff_t j = j_begin; j < j_end; ++j) {
			ptrdiff_t top = filter.left[i];
			typename Policy::num_type accum = 0;

			for (ptrdiff_t k = 0; k < filter.filter_width; ++k) {
				typename Policy::num_type coeff = policy.coeff(filter, i, k);
				typename Policy::num_type x = policy.load(&src[top + k][j]);

				accum += coeff * x;
			}

			policy.store(&dst[i][j], accum);
		}
	}
}

/**
 * Base class for implementations of resizing filter.
 */
class ResizeImpl {
protected:
	/**
	 * Filter coefficients.
	 */
	FilterContext m_filter;

	/**
	 * Initialize the implementation with the given coefficients.
	 *
	 * @param filter coefficients
	 */
	ResizeImpl(const FilterContext &filter);
public:
	/**
	 * Destroy implementation.
	 */
	virtual ~ResizeImpl() = 0;

	/**
	 * Execute filter pass on an unsigned 16-bit image.
	 *
	 * @param src input plane
	 * @param dst output plane
	 * @param tmp temporary buffer (implementation defined size)
	 * @throws ZimgUnsupportedError if not supported
	 */
	virtual void process_u16(const ImagePlane<const uint16_t> &src, const ImagePlane<uint16_t> &dst, uint16_t *tmp) const = 0;

	/**
	 * Execute filter pass on a half precision 16-bit image.
	 *
	 * @see ResizeImpl::process_u16
	 */
	virtual void process_f16(const ImagePlane<const uint16_t> &src, const ImagePlane<uint16_t> &dst, uint16_t *tmp) const = 0;

	/**
	 * Execute filter pass on a single precision 32-bit image.
	 *
	 * @see ResizeImpl::process_u16
	 */
	virtual void process_f32(const ImagePlane<const float> &src, const ImagePlane<float> &dst, float *tmp) const = 0;
};

/**
 * Create a concrete ResizeImpl.
 *
 * @param f filter
 * @param horizontal whether filter is horizontal
 * @param src_dim input dimension
 * @param dst_dim output dimension
 * @param shift input shift
 * @param subwidth input window size
 */
ResizeImpl *create_resize_impl(const Filter &f, bool horizontal, int src_dim, int dst_dim, double shift, double subwidth, CPUClass cpu);

} // namespace resize
} // namespace zimg

#endif // ZIMG_RESIZE_RESIZE_IMPL_H_
