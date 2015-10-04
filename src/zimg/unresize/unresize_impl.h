#pragma once

#ifndef ZIMG_UNRESIZE_UNRESIZE_IMPL_H_
#define ZIMG_UNRESIZE_UNRESIZE_IMPL_H_

#include <cstddef>
#include "common/ccdep.h"
#include "bilinear.h"
#include "plane.h"

namespace zimg {;

enum class CPUClass;

namespace unresize {;

struct ScalarPolicy_F32 {
	FORCE_INLINE float load(const float *src) { return *src; }

	FORCE_INLINE void store(float *dst, float x) { *dst = x; }
};

template <class T, class Policy>
inline FORCE_INLINE void filter_scanline_h_forward(const BilinearContext &ctx, const ImagePlane<const T> &src, T * RESTRICT tmp,
                                                   ptrdiff_t i, ptrdiff_t j_begin, ptrdiff_t j_end, Policy policy)
{
	const float *c = ctx.lu_c.data();
	const float *l = ctx.lu_l.data();

	float z = j_begin ? policy.load(&src[i][j_begin - 1]) : 0;

	// Matrix-vector product, and forward substitution loop.
	for (ptrdiff_t j = j_begin; j < j_end; ++j) {
		const float *row = ctx.matrix_coefficients.data() + j * ctx.matrix_row_stride;
		ptrdiff_t left = ctx.matrix_row_offsets[j];

		float accum = 0;
		for (ptrdiff_t k = 0; k < ctx.matrix_row_size; ++k) {
			float coeff = row[k];
			float x = policy.load(&src[i][left + k]);
			accum += coeff * x;
		}

		z = (accum - c[j] * z) * l[j];
		policy.store(&tmp[j], z);
	}
}

template <class T, class Policy>
inline FORCE_INLINE void filter_scanline_h_back(const BilinearContext &ctx, const T * RESTRICT tmp, const ImagePlane<T> &dst,
                                                ptrdiff_t i, ptrdiff_t j_begin, ptrdiff_t j_end, Policy policy)
{
	const float *u = ctx.lu_u.data();
	float w = j_begin < ctx.dst_width ? policy.load(&dst[i][j_begin]) : 0;

	// Backward substitution.
	for (ptrdiff_t j = j_begin; j > j_end; --j) {
		w = policy.load(&tmp[j - 1]) - u[j - 1] * w;
		policy.store(&dst[i][j - 1], w);
	}
}

template <class T, class Policy>
inline FORCE_INLINE void filter_scanline_v_forward(const BilinearContext &ctx, const ImagePlane<const T> &src, const ImagePlane<T> &dst,
                                                   ptrdiff_t i, ptrdiff_t j_begin, ptrdiff_t j_end, Policy policy)
{
	const float *c = ctx.lu_c.data();
	const float *l = ctx.lu_l.data();

	const float *row = ctx.matrix_coefficients.data() + i * ctx.matrix_row_stride;
	ptrdiff_t top = ctx.matrix_row_offsets[i];

	for (ptrdiff_t j = j_begin; j < j_end; ++j) {
		float z = i ? policy.load(&dst[i - 1][j]) : 0;

		float accum = 0;
		for (ptrdiff_t k = 0; k < ctx.matrix_row_size; ++k) {
			float coeff = row[k];
			float x = policy.load(&src[top + k][j]);
			accum += coeff * x;
		}

		z = (accum - c[i] * z) * l[i];
		policy.store(&dst[i][j], z);
	}
}

template <class T, class Policy>
inline FORCE_INLINE void filter_scanline_v_back(const BilinearContext &ctx, const ImagePlane<T> &dst, ptrdiff_t i, ptrdiff_t j_begin, ptrdiff_t j_end, Policy policy)
{
	const float *u = ctx.lu_u.data();

	for (ptrdiff_t j = j_begin; j < j_end; ++j) {
		float w = i < ctx.dst_width ? policy.load(&dst[i][j]) : 0;

		w = policy.load(&dst[i - 1][j]) - u[i - 1] * w;
		policy.store(&dst[i - 1][j], w);
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

	virtual void process_f16_h(const ImagePlane<const uint16_t> &src, const ImagePlane<uint16_t> &dst, uint16_t *tmp) const = 0;

	virtual void process_f16_v(const ImagePlane<const uint16_t> &src, const ImagePlane<uint16_t> &dst, uint16_t *tmp) const = 0;

	virtual void process_f32_h(const ImagePlane<const float> &src, const ImagePlane<float> &dst, float *tmp) const = 0;

	virtual void process_f32_v(const ImagePlane<const float> &src, const ImagePlane<float> &dst, float *tmp) const = 0;
};

/**
 * Create and allocate a execution kernel.
 *
 * @see Unresize::Unresize
 */
UnresizeImpl *create_unresize_impl(int src_width, int src_height, int dst_width, int dst_height, float shift_w, float shift_h, CPUClass cpu);

} // namespace unresize
} // namespace zimg

#endif // ZIMG_UNRESIZE_UNRESIZE_IMPL_H_
