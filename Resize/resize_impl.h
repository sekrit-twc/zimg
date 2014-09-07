#ifndef RESIZE_IMPL_H_
#define RESIZE_IMPL_H_

#include "filter.h"
#include "osdep.h"

namespace resize {;

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
