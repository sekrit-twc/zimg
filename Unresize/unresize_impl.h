#ifndef ZIMG_UNRESIZE_IMPL_H_
#define ZING_UNRESIZE_IMPL_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include "bilinear.h"

namespace zimg {;
namespace unresize {;

/**
 * Convert numeric types with saturation.
 *
 * @param T output type
 * @param x value
 * @return x as T
 */
template <class T>
T clamp_float(float x)
{
	return static_cast<T>(std::min(std::max(x, 0.0f), 1.0f) * static_cast<float>(std::numeric_limits<T>::max()));
}

/**
 * Base class for unresize kernel implementations.
 * All pointers passed to member functions must be 32-byte aligned.
 */
class UnresizeImpl {
protected:
	/**
	 * Coefficients for the horizontal pass.
	 */
	BilinearContext m_hcontext;

	/**
	 * Coefficients for the vertical pass.
	 */
	BilinearContext m_vcontext;

	/**
	 * Initialize the implementation with the given coefficients.
	 *
	 * @param hcontext horizontal coefficients
	 * @param vcontext vertical coefficients
	 */
	UnresizeImpl(const BilinearContext &hcontext, const BilinearContext &vcontext);
public:
	/**
	 * Destroy implementation
	 */
	virtual ~UnresizeImpl() = 0;

	/**
	 * Process a single scanline using scalar code.
	 *
	 * @param src input buffer
	 * @param dst output buffer
	 * @param tmp temporary buffer with sufficient space for one line/column
	 * @param horizontal use horizontal context if true, else vertical context
	 */
	void unresize_scanline(const float *src, float *dst, float *tmp, bool horizontal) const;

	/**
	 * Process 4 scanlines using horizontal context.
	 *
	 * @param src input buffer containing 4 scanlines
	 * @param dst output buffer to receive 4 scanlines
	 * @param tmp temporary buffer with sufficient space for 4 scanlines
	 * @param src_stride stride of input buffer
	 * @param dst_stride stride of output buffer
	 */
	virtual void unresize_scanline4_h(const float *src, float *dst, float *tmp, int src_stride, int dst_stride) const = 0;

	/**
	 * Process 4 columns using vertical context.
	 * The image columns must first be transposed as scanlines.
	 *
	 * @see UnresizeImpl::unresize_scanline4_h
	 */
	virtual void unresize_scanline4_v(const float *src, float *dst, float *tmp, int src_stride, int dst_stride) const = 0;

	/**
	 * Transpose an image plane.
	 *
	 * @param src input buffer
	 * @param dst output buffer with sufficient space for transposed image
	 * @param src_width width of input plane
	 * @param src_height height of input plane
	 * @param src_stride stride of input plane
	 * @param dst_stride stride to use in output buffer
	 */
	virtual void transpose_plane(const float *src, float *dst, int src_width, int src_height, int src_stride, int dst_stride) const = 0;

	/**
	 * Convert a scanline from byte to float.
	 *
	 * @param src input buffer
	 * @param dst output buffer
	 * @param width scanline width
	 */
	virtual void load_scanline_u8(const uint8_t *src, float *dst, int width) const = 0;

	/**
	 * Convert a scanline from word to float.
	 *
	 * @see UnresizeImpl::load_scanline_u8
	 */
	virtual void load_scanline_u16(const uint16_t *src, float *dst, int width) const = 0;

	/**
	 * Convert a scanline from float to byte.
	 *
	 * @param dst output buffer
	 * @param src input buffer
	 * @param width scanline width
	 */
	virtual void store_scanline_u8(const float *src, uint8_t *dst, int width) const = 0;

	/**
	 * Convert a scanline from float to word.
	 *
	 * @see UnresizeImpl::store_scanline_u8
	 */
	virtual void store_scanline_u16(const float *src, uint16_t *dst, int width) const = 0;
};

/**
 * Concrete implementation of UnresizeImpl using scalar processing.
 */
class UnresizeImplC final : public UnresizeImpl {
public:
	/**
	 * @see: UnresizeImpl::UnresizeImpl
	 */
	UnresizeImplC(const BilinearContext &hcontext, const BilinearContext &vcontext);

	void unresize_scanline4_h(const float *src, float *dst, float *tmp, int src_stride, int dst_stride) const override;

	void unresize_scanline4_v(const float *src, float *dst, float *tmp, int src_stride, int dst_stride) const override;

	void transpose_plane(const float *src, float *dst, int src_width, int src_height, int src_stride, int dst_stride) const override;

	void load_scanline_u8(const uint8_t *src, float *dst, int width) const override;

	void load_scanline_u16(const uint16_t *src, float *dst, int width) const override;

	void store_scanline_u8(const float *src, uint8_t *dst, int width) const override;

	void store_scanline_u16(const float *src, uint16_t *dst, int width) const override;
};

#ifdef ZIMG_X86
/**
 * Concrete implementation of UnresizeImpl using SSE2 intrinsics.
 *
 * @param HWIDTH matrix_row_width of horizontal context
 * @param VWIDTH matrix_row_width of vertical context
 */
template <int HWIDTH, int VWIDTH>
class UnresizeImplX86 final : public UnresizeImpl {
private:
	/**
	 * Template for implementation of unresize_scanline4_h and unresize_scanline4_v.
	 *
	 * @param WIDTH matrix_row_width of context
	 * @param ctx context
	 * @see UnresizeImpl::unresize_scanline4
	 */
	template <int WIDTH>
	void unresize_scanline4(const BilinearContext &ctx, const float *src, float *dst, float *tmp, int src_stride, int dst_stride) const;
public:
	/**
	* @see: UnresizeImpl::UnresizeImpl
	*/
	UnresizeImplX86(const BilinearContext &hcontext, const BilinearContext &vcontext);

	void unresize_scanline4_h(const float *src, float *dst, float *tmp, int src_stride, int dst_stride) const override;

	void unresize_scanline4_v(const float *src, float *dst, float *tmp, int src_stride, int dst_stride) const override;

	void transpose_plane(const float *src, float *dst, int src_width, int src_height, int src_stride, int dst_stride) const override;

	void load_scanline_u8(const uint8_t *src, float *dst, int width) const override;

	void load_scanline_u16(const uint16_t *src, float *dst, int width) const override;

	void store_scanline_u8(const float *src, uint8_t *dst, int width) const override;

	void store_scanline_u16(const float *src, uint16_t *dst, int width) const override;
};
#endif // ZIMG_X86

/**
 * Create and allocate a execution kernel.
 *
 * @param src_width upsampled image width
 * @param src_height upsampled image height
 * @param dst_width unresized image width
 * @param dst_height unresized image height
 * @param shift_w horizontal center shift relative to upsampled image
 * @param shift_h vertical center shift relative to upsampled image
 * @param x86 whether to create an x86-optimized kernel
 * @return a pointer to the allocated kernel
 * @throws ZimgIllegalArgument on invalid dimensions
 * @throws ZimgUnsupportedError if not supported
 */
UnresizeImpl *create_unresize_impl(int src_width, int src_height, int dst_width, int dst_height, float shift_w, float shift_h, bool x86);

} // namespace unresize
} // namespace zimg

#endif // UNRESIZE_IMPL_H
