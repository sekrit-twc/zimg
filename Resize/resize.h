#ifndef RESIZE_H_
#define RESIZE_H_

#include <memory>
#include "filter.h"
#include "osdep.h"

namespace resize {;

/**
 * Enum for supported input and output formats.
 *
 * BYTE = 8 bits
 * WORD = 16 bits
 * HALF = 16 bits
 * FLOAT = 32 bits
 */
enum class PixelType {
	BYTE,
	WORD,
	HALF,
	FLOAT
};

class ResizeImpl;

/**
 * Resize: applies a resizing filter.
 *
 * Each instance is applicable only for its given set of resizing parameters.
 */
class Resize {
	int m_src_width;
	int m_src_height;
	int m_dst_width;
	int m_dst_height;
	bool m_skip_h;
	bool m_skip_v;
	std::shared_ptr<ResizeImpl> m_impl;

	size_t max_frame_size(PixelType type) const;

	void copy_plane(const void *src, void *dst, int src_stride_bytes, int dst_stride_bytes) const;
public:
	/**
	 * Initialize a null context. Cannot be used for execution.
	 */
	Resize() = default;

	/**
	 * Initialize a context to apply a given resizing filter.
	 *
	 * @param f filter
	 * @param src_width width of input image
	 * @param src_height height of input image
	 * @param dst_width width of output image
	 * @param dst_height height of output image
	 * @param shift_w horizontal shift in units of source pixels
	 * @param shift_h vertical shift in units of source pixels
	 * @param subwidth active horizontal subwindow in units of source pixels
	 * @param subheight active vertical subwindow in units of source pixels
	 * @param x86 whether to create an x86-optimiezd kernel 
	 */
	Resize(const Filter &f, int src_width, int src_height, int dst_width, int dst_height,
	       double shift_w, double shift_h, double subwidth, double subheight, bool x86);

	/**
	 * Get the size of the temporary buffer required by the filter.
	 *
	 * @param type pixel type to process
	 * @return the size of temporary buffer in units of pixels
	 */
	size_t tmp_size(PixelType type) const;

	/**
	 * Process an unsigned 16-bit image.
	 *
	 * @param src input image
	 * @param dst output image
	 * @param tmp temporary buffer (@see Resize::tmp_size)
	 * @param src_stride stride of input image
	 * @param dst_stride stride of output image
	 */
	void process_u16(const uint16_t * RESTRICT src, uint16_t * RESTRICT dst, uint16_t * RESTRICT tmp, int src_stride, int dst_stride) const;

	/**
	 * Process an half precision 16-bit image.
	 *
	 * @param src input image
	 * @param dst output image
	 * @param tmp temporary buffer (@see Resize::tmp_size)
	 * @param src_stride stride of input image
	 * @param dst_stride stride of output image
	 */
	void process_f16(const uint16_t * RESTRICT src, uint16_t * RESTRICT dst, uint16_t * RESTRICT tmp, int src_stride, int dst_stride) const;

	/**
	 * Process a single precision 32-bit image.
	 *
	 * @see Resize::process_u16
	 */
	void process_f32(const float * RESTRICT src, float * RESTRICT dst, float * RESTRICT tmp, int src_stride, int dst_stride) const;
};

} // namespace resize

#endif // RESIZE_H_
