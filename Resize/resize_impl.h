#ifndef RESIZE_IMPL_H_
#define RESIZE_IMPL_H_

#include "filter.h"
#include "osdep.h"

namespace resize {;

/**
 * Convert 16.0 unsigned to 16.0 signed and store in 32-bit.
 */
inline FORCE_INLINE int32_t unpack_u16(uint16_t x)
{
	return (int32_t)x + (int32_t)INT16_MIN;
}

/**
 * Arighmetic right shift of x by n with rounding.
 */
inline FORCE_INLINE int32_t round_shift(int32_t x, int32_t n)
{
	return (x + (1 << (n - 1))) >> n;
}

/**
 * Convert 16.14 signed fixed point to 16.0 unsigned.
 */
inline FORCE_INLINE uint16_t pack_i30(int32_t x)
{
	// Reduce 16.14 fixed point to 16.0 and convert to unsigned.
	x = round_shift(x, 14) - (int32_t)INT16_MIN;
	x = x < 0 ? 0 : x;
	x = x > UINT16_MAX ? UINT16_MAX : x;
	return (uint16_t)x;
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
	 */
	virtual void process_u16_h(const uint16_t * RESTRICT src, uint16_t * RESTRICT dst, uint16_t * RESTRICT tmp,
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
	 */
	virtual void process_u16_v(const uint16_t * RESTRICT src, uint16_t * RESTRICT dst, uint16_t * RESTRICT tmp,
	                           int src_width, int src_height, int src_stride, int dst_stride) const = 0;

	/**
	 * Execute horizontal filter pass on a single precision 32-bit image.
	 *
	 * @see ResizeImpl::process_u16_h
	 */
	virtual void process_f32_h(const float * RESTRICT src, float * RESTRICT dst, float * RESTRICT tmp,
	                           int src_width, int src_height, int src_stride, int dst_stride) const = 0;

	/**
	 * Execute vertical filter pass on a single precision 32-bit image.
	 *
	 * @see ResizeImpl::process_u16_v
	 */
	virtual void process_f32_v(const float * RESTRICT src, float * RESTRICT dst, float * RESTRICT tmp,
	                           int src_width, int src_height, int src_stride, int dst_stride) const = 0;
};

ResizeImpl *create_resize_impl(const Filter &f, int src_width, int src_height, int dst_width, int dst_height,
                               double shift_w, double shift_h, double subwidth, double subheight, bool x86);

} // namespace resize

#endif // RESIZE_IMPL_H_
