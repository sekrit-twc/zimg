#pragma once

#ifndef ZIMG_UNRESIZE_IMPL_H_
#define ZING_UNRESIZE_IMPL_H_

#include <cstddef>
#include "Common/osdep.h"
#include "Common/plane.h"
#include "bilinear.h"

namespace zimg {;

enum class CPUClass;

namespace unresize {;

inline FORCE_INLINE void filter_scanline_h_forward(const BilinearContext &ctx, const ImagePlane<float> &src, float * RESTRICT tmp,
                                                   ptrdiff_t i, ptrdiff_t j_begin, ptrdiff_t j_end)
{
	const float * RESTRICT src_p = src.data();
	int src_stride = src.stride();

	const float *c = ctx.lu_c.data();
	const float *l = ctx.lu_l.data();

	float z = j_begin ? src_p[i * src_stride + j_begin - 1] : 0;

	// Matrix-vector product, and forward substitution loop.
	for (ptrdiff_t j = j_begin; j < j_end; ++j) {
		const float *row = ctx.matrix_coefficients.data() + j * ctx.matrix_row_stride;
		ptrdiff_t left = ctx.matrix_row_offsets[j];

		float accum = 0;
		for (ptrdiff_t k = 0; k < ctx.matrix_row_size; ++k) {
			accum += row[k] * src_p[i * src_stride + left + k];
		}

		z = (accum - c[j] * z) * l[j];
		tmp[j] = z;
	}
}

inline FORCE_INLINE void filter_scanline_h_back(const BilinearContext &ctx, const float * RESTRICT tmp, ImagePlane<float> &dst,
                                                ptrdiff_t i, ptrdiff_t j_begin, ptrdiff_t j_end)
{
	float * RESTRICT dst_p = dst.data();
	int dst_stride = dst.stride();

	const float *u = ctx.lu_u.data();
	float w = j_begin < ctx.dst_width ? dst_p[i * dst_stride + j_begin] : 0;

	// Backward substitution.
	for (ptrdiff_t j = j_begin; j > j_end; --j) {
		w = tmp[j - 1] - u[j - 1] * w;
		dst_p[i * dst_stride + j - 1] = w;
	}
}

inline FORCE_INLINE void filter_scanline_v_forward(const BilinearContext &ctx, const ImagePlane<float> &src, ImagePlane<float> &dst,
                                                   ptrdiff_t i, ptrdiff_t j_begin, ptrdiff_t j_end)
{
	const float * RESTRICT src_p = src.data();
	float * RESTRICT dst_p = dst.data();
	int src_stride = src.stride();
	int dst_stride = dst.stride();

	const float *c = ctx.lu_c.data();
	const float *l = ctx.lu_l.data();

	const float *row = ctx.matrix_coefficients.data() + i * ctx.matrix_row_stride;
	ptrdiff_t top = ctx.matrix_row_offsets[i];

	for (ptrdiff_t j = j_begin; j < j_end; ++j) {
		float z = i ? dst_p[(i - 1) * dst_stride + j] : 0;

		float accum = 0;
		for (ptrdiff_t k = 0; k < ctx.matrix_row_size; ++k) {
			accum += row[k] * src_p[(top + k) * src_stride + j];
		}

		z = (accum - c[i] * z) * l[i];
		dst_p[i * dst_stride + j] = z;
	}
}

inline FORCE_INLINE void filter_scanline_v_back(const BilinearContext &ctx, ImagePlane<float> &dst, ptrdiff_t i, ptrdiff_t j_begin, ptrdiff_t j_end)
{
	float * RESTRICT dst_p = dst.data();
	int dst_stride = dst.stride();

	const float *u = ctx.lu_u.data();

	for (ptrdiff_t j = j_begin; j < j_end; ++j) {
		float w = i < ctx.dst_width ? dst_p[i * dst_stride + j] : 0;

		w = dst_p[(i - 1) * dst_stride + j] - u[i - 1] * w;
		dst_p[(i - 1) * dst_stride + j] = w;
	}
}


/**
 * Base class for implementations of unresizing filter.
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

	virtual void process_f32_h(const ImagePlane<float> &src, ImagePlane<float> &dst, float *tmp) const = 0;

	virtual void process_f32_v(const ImagePlane<float> &src, ImagePlane<float> &dst, float *tmp) const = 0;
};

/**
 * Create and allocate a execution kernel.
 *
 * @see Unresize::Unresize
 */
UnresizeImpl *create_unresize_impl(int src_width, int src_height, int dst_width, int dst_height, float shift_w, float shift_h, CPUClass cpu);

} // namespace unresize
} // namespace zimg

#endif // UNRESIZE_IMPL_H
