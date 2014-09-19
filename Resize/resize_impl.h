#pragma once

#ifndef ZIMG_RESIZE_RESIZE_IMPL_H_
#define ZIMG_RESIZE_RESIZE_IMPL_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include "Common/cpuinfo.h"
#include "Common/osdep.h"
#include "filter.h"

namespace zimg {;
namespace resize {;

struct ScalarPolicy_U16 {
	typedef int32_t num_type;

	int32_t coeff(const EvaluatedFilter &filter, ptrdiff_t row, ptrdiff_t k)
	{
		return filter.data_i16()[row * filter.stride_i16() + k];
	}

	int32_t load(const uint16_t *src)
	{
		uint16_t x = *src;
		return (int32_t)x + (int32_t)INT16_MIN; // Make signed.
	}

	void store(uint16_t *dst, int32_t x)
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

	float coeff(const EvaluatedFilter &filter, ptrdiff_t row, ptrdiff_t k)
	{
		return filter.data()[row * filter.stride() + k];
	}

	float load(const float *src) { return *src; }

	void store(float *dst, float x) { *dst = x; }
};

template <class T, class Policy>
inline FORCE_INLINE void filter_plane_h_scalar(const EvaluatedFilter &filter, const T * RESTRICT src, T * RESTRICT dst,
                                               ptrdiff_t i_begin, ptrdiff_t i_end, ptrdiff_t j_begin, ptrdiff_t j_end, ptrdiff_t src_stride, ptrdiff_t dst_stride, Policy policy)
{
	for (ptrdiff_t i = i_begin; i < i_end; ++i) {
		for (ptrdiff_t j = j_begin; j < j_end; ++j) {
			ptrdiff_t left = filter.left()[j];
			Policy::num_type accum = 0;

			for (int k = 0; k < filter.width(); ++k) {
				Policy::num_type coeff = policy.coeff(filter, j, k);
				Policy::num_type x = policy.load(src + i * src_stride + left + k);

				accum += coeff * x;
			}

			policy.store(dst + i * dst_stride + j, accum);
		}
	}
}

template <class T, class Policy>
inline FORCE_INLINE void filter_plane_v_scalar(const EvaluatedFilter &filter, const T * RESTRICT src, T * RESTRICT dst,
                                               ptrdiff_t i_begin, ptrdiff_t i_end, ptrdiff_t j_begin, ptrdiff_t j_end, ptrdiff_t src_stride, ptrdiff_t dst_stride, Policy policy)
{
	for (ptrdiff_t i = i_begin; i < i_end; ++i) {
		for (ptrdiff_t j = j_begin; j < j_end; ++j) {
			ptrdiff_t top = filter.left()[i];
			Policy::num_type accum = 0;

			for (ptrdiff_t k = 0; k < filter.width(); ++k) {
				Policy::num_type coeff = policy.coeff(filter, i, k);
				Policy::num_type x = policy.load(src + (top + k) * src_stride + j);

				accum += coeff * x;
			}

			policy.store(dst + i * dst_stride + j, accum);
		}
	}
}

/**
 * Base class for implementations of resizing filter.
 */
class ResizeImpl {
protected:
	/**
	 * Filter coefficients for horizontal pass.
	 */
	EvaluatedFilter m_filter_h;

	/**
	 * Filter coefficients for vertical pass.
	 */
	EvaluatedFilter m_filter_v;

	/**
	 * Initialize the implementation with the given coefficients.
	 *
	 * @param filter_h horizontal coefficients
	 * @param filter_v vertical coefficients
	 */
	ResizeImpl(const EvaluatedFilter &filter_h, const EvaluatedFilter &filter_v);
public:
	/**
	 * Destroy implementation.
	 */
	virtual ~ResizeImpl() = 0;

	/**
	 * Execute horizontal filter pass on an unsigned 16-bit image.
	 *
	 * @param src input image
	 * @param dst output image
	 * @param tmp temporary buffer (implementation defined size)
	 * @param src_width width of input image (must match filter)
	 * @param src_height height of input image (must match output image)
	 * @param src_stride stride of input image
	 * @param dst_stride stride of output image
	 * @throws ZimgUnsupportedError if not supported
	 */
	virtual void process_u16_h(const uint16_t *src, uint16_t *dst, uint16_t *tmp,
	                           int src_width, int src_height, int src_stride, int dst_stride) const = 0;

	/**
	 * Execute vertical filter pass on an unsigned 16-bit image.
	 *
	 * @param src input image
	 * @param dst output image
	 * @param tmp temporary buffer (implementation defined size)
	 * @param src_width width of input image (must match output image)
	 * @param src_height height of input image (must match filter)
	 * @param src_stride stride of input image
	 * @param dst_stride stride of output image
	 * @throws ZimgUnsupportedError if not supported
	 */
	virtual void process_u16_v(const uint16_t *src, uint16_t *dst, uint16_t *tmp,
	                           int src_width, int src_height, int src_stride, int dst_stride) const = 0;

	/**
	 * Execute horizontal filter pass on a half precision 16-bit image.
	 *
	 * @see ResizeImpl::process_u16_h
	 */
	virtual void process_f16_h(const uint16_t *src, uint16_t *dst, uint16_t *tmp,
	                           int src_width, int src_height, int src_stride, int dst_stride) const = 0;

	/**
	 * Execute vertical filter pass on a half precision 16-bit image.
	 *
	 *  @see ResizeImpl::process_u16_v
	 */
	virtual void process_f16_v(const uint16_t *src, uint16_t *dst, uint16_t *tmp,
	                           int src_width, int src_height, int src_stride, int dst_stride) const = 0;

	/**
	 * Execute horizontal filter pass on a single precision 32-bit image.
	 *
	 * @see ResizeImpl::process_u16_h
	 */
	virtual void process_f32_h(const float *src, float *dst, float *tmp,
	                           int src_width, int src_height, int src_stride, int dst_stride) const = 0;

	/**
	 * Execute vertical filter pass on a single precision 32-bit image.
	 *
	 * @see ResizeImpl::process_u16_v
	 */
	virtual void process_f32_v(const float *src, float *dst, float *tmp,
	                           int src_width, int src_height, int src_stride, int dst_stride) const = 0;
};

/**
 * Create a concrete ResizeImpl.

 * @see Resize::Resize
 */
ResizeImpl *create_resize_impl(const Filter &f, int src_width, int src_height, int dst_width, int dst_height,
                               double shift_w, double shift_h, double subwidth, double subheight, CPUClass cpu);

} // namespace resize
} // namespace zimg

#endif // ZIMG_RESIZE_RESIZE_IMPL_H_
