#ifdef ZIMG_LOONGARCH

#include <cstddef>

#include <lsxintrin.h>
#include "common/align.h"
#include "common/ccdep.h"
#include "common/checked_int.h"
#include "common/except.h"
#include "common/pixel.h"
#include "common/zassert.h"
#include "unresize/bilinear.h"
#include "unresize/unresize_impl.h"
#include "unresize_impl_loongarch.h"

#include "common/loongarch/lsx_util.h"

namespace zimg::unresize {

namespace {

// Process the forward substitution for one row of the vertical unresize,
// vectorizing across 4 adjacent output columns at a time.
void unresize_line_forward_v_f32_lsx(unsigned filter_offset, const float *RESTRICT filter_data, unsigned filter_width,
                                     float c_, float l_, const float *RESTRICT src, ptrdiff_t src_stride, unsigned src_mask,
                                     const float *RESTRICT above, float *RESTRICT dst, unsigned width)
{
	float cf = c_, lf = l_;
	const __m128 c_vec = (__m128)__lsx_vldrepl_w(&cf, 0);
	const __m128 l_vec = (__m128)__lsx_vldrepl_w(&lf, 0);

	unsigned vec_left = ceil_n(0, 4);
	unsigned vec_right = floor_n(width, 4);

	for (unsigned j = vec_left; j < vec_right; j += 4) {
		__m128 z = above ? (__m128)__lsx_vld((void *)(above + j), 0) : (__m128)__lsx_vldi(0);
		__m128 accum = (__m128)__lsx_vldi(0);

		for (unsigned k = 0; k < filter_width; ++k) {
			float cv = filter_data[k];
			const __m128 c_k = (__m128)__lsx_vldrepl_w(&cv, 0);
			__m128 x = (__m128)__lsx_vld((void *)(src + (static_cast<ptrdiff_t>((filter_offset + k) & src_mask) * src_stride) / sizeof(float) + j), 0);
			accum = __lsx_vfmadd_s(c_k, x, accum);
		}

		z = __lsx_vfmul_s(__lsx_vfnmsub_s(c_vec, z, accum), l_vec); // (accum - c*z) * l
		__lsx_vst((__m128i)z, dst + j, 0);
	}

	// Tail: process remaining columns scalar to match the reference exactly.
	for (unsigned j = vec_right; j < width; ++j) {
		unsigned offset = vec_right;
		float z = above ? above[offset + (j - vec_right)] : 0.0f;
		float accum = 0.0f;

		for (unsigned k = 0; k < filter_width; ++k) {
			accum += filter_data[k] * src[(static_cast<ptrdiff_t>((filter_offset + k) & src_mask) * src_stride) / sizeof(float) + offset + (j - vec_right)];
		}

		z = (accum - c_ * z) * l_;
		dst[offset + (j - vec_right)] = z;
	}
}

// Backward substitution for the vertical unresize, vectorized across columns.
void unresize_line_back_v_f32_lsx(float u_, const float *RESTRICT below, float *RESTRICT dst, unsigned width)
{
	float uf = u_;
	const __m128 u_vec = (__m128)__lsx_vldrepl_w(&uf, 0);

	unsigned vec_right = floor_n(width, 4);

	for (unsigned j = 0; j < vec_right; j += 4) {
		__m128 w = below ? (__m128)__lsx_vld((void *)(below + j), 0) : (__m128)__lsx_vldi(0);
		__m128 d = (__m128)__lsx_vld((void *)(dst + j), 0);

		w = __lsx_vfnmsub_s(u_vec, w, d); // dst - u * w
		__lsx_vst((__m128i)w, dst + j, 0);
	}

	for (unsigned j = vec_right; j < width; ++j) {
		float w = below ? below[j] : 0.0f;
		w = dst[j] - u_ * w;
		dst[j] = w;
	}
}


// Horizontal unresize: process 4 rows at a time with a 4x4 transpose, so that
// the width direction is uniform across 4 rows and can be vectorized.
void transpose_line_4x4_ps(float *RESTRICT dst, const float * const * RESTRICT src, unsigned left, unsigned right)
{       
        __m128 x0, x1, x2, x3;

	for (unsigned j = left; j < right; j += 4) {

		x0 = (__m128)__lsx_vld((void *)(src[0] + j), 0);
		x1 = (__m128)__lsx_vld((void *)(src[1] + j), 0);
		x2 = (__m128)__lsx_vld((void *)(src[2] + j), 0);
		x3 = (__m128)__lsx_vld((void *)(src[3] + j), 0);

		lsx_transpose4_f32(x0, x1, x2, x3);

		__lsx_vst((__m128i)x0, dst + 0, 0);
		__lsx_vst((__m128i)x1, dst + 4, 0);
		__lsx_vst((__m128i)x2, dst + 8, 0);
		__lsx_vst((__m128i)x3, dst + 12, 0);

		dst += 16;
	}
}

// Forward pass of the horizontal unresize across 4 rows (4-wide vector).
void unresize_line4_h_f32_lsx(const unsigned * RESTRICT filter_left, const float * RESTRICT filter_data, unsigned filter_stride, unsigned filter_width,
                              const float *lu_c, const float *lu_l, const float *lu_u, const float * RESTRICT src, float * const * RESTRICT dst, float *tmp, unsigned width)
{
	__m128 z = (__m128)__lsx_vldi(0);
	__m128 w = (__m128)__lsx_vldi(0);

	for (size_t j = 0; j < width; ++j) {
		__m128 accum = (__m128)__lsx_vldi(0);
		const float *coeffs = filter_data + j * filter_stride;
		const float *src_p = src + filter_left[j] * 4;

		for (size_t k = 0; k < filter_width; ++k) {
			float cv = coeffs[k];
			const __m128 c = (__m128)__lsx_vldrepl_w(&cv, 0);
			__m128 x = (__m128)__lsx_vld((void *)(src_p + k * 4), 0);
			accum = __lsx_vfmadd_s(c, x, accum);
		}

		float cv = lu_c[j], lv = lu_l[j];
		const __m128 c = (__m128)__lsx_vldrepl_w(&cv, 0);
		const __m128 l = (__m128)__lsx_vldrepl_w(&lv, 0);
		z = __lsx_vfmul_s(__lsx_vfnmsub_s(c, z, accum), l); // (accum - c*z) * l
		__lsx_vst((__m128i)z, tmp + j * 4, 0);
	}

	for (size_t j = width; j != 0; --j) {
		float uv = lu_u[j - 1];
		const __m128 u = (__m128)__lsx_vldrepl_w(&uv, 0);
		__m128 val = (__m128)__lsx_vld((void *)(tmp + (j - 1) * 4), 0);
		w = __lsx_vfnmsub_s(u, w, val); // dst[j-1] - u[j-1]*w
		__lsx_vst((__m128i)w, tmp + (j - 1) * 4, 0);
	}

	// Scatter the 4-row results back (row-major per output column).
	for (size_t j = 0; j < width; ++j) {
		__m128 v = (__m128)__lsx_vld((void *)(tmp + j * 4), 0);
		lsx_scatter_f32(dst[0] + j, dst[1] + j, dst[2] + j, dst[3] + j, v);
	}
}


class UnresizeImplH_F32_LSX final : public UnresizeImplH {
public:
	UnresizeImplH_F32_LSX(const BilinearContext &context, unsigned height) :
		UnresizeImplH(context, context.output_width, height, PixelType::FLOAT)
	{
		m_desc.step = 4;
		m_desc.scratchpad_size = ((ceil_n(checked_size_t{ m_context.input_width }, 4) + m_context.output_width) * 4 * sizeof(float)).get();
	}

	void process(const graphengine::BufferDescriptor *in, const graphengine::BufferDescriptor *out,
	             unsigned i, unsigned left, unsigned right, void *, void *tmp) const noexcept override
	{
		const float *src_ptr[4] = { 0 };
		float *dst_ptr[4] = { 0 };
		float *transpose_buf = static_cast<float *>(tmp);
		float *transpose_buf2 = transpose_buf + ceil_n(static_cast<size_t>(m_context.input_width), 4) * 4;
		unsigned height = m_desc.format.height;

		src_ptr[0] = in->get_line<float>(std::min(i + 0, height - 1));
		src_ptr[1] = in->get_line<float>(std::min(i + 1, height - 1));
		src_ptr[2] = in->get_line<float>(std::min(i + 2, height - 1));
		src_ptr[3] = in->get_line<float>(std::min(i + 3, height - 1));

		transpose_line_4x4_ps(transpose_buf, src_ptr, 0, m_context.input_width);

		dst_ptr[0] = out->get_line<float>(std::min(i + 0, height - 1));
		dst_ptr[1] = out->get_line<float>(std::min(i + 1, height - 1));
		dst_ptr[2] = out->get_line<float>(std::min(i + 2, height - 1));
		dst_ptr[3] = out->get_line<float>(std::min(i + 3, height - 1));

		unresize_line4_h_f32_lsx(m_context.matrix_row_offsets.data(), m_context.matrix_coefficients.data(), m_context.matrix_row_stride, m_context.matrix_row_size,
		                         m_context.lu_c.data(), m_context.lu_l.data(), m_context.lu_u.data(), transpose_buf, dst_ptr, transpose_buf2, m_context.output_width);
	}
};


class UnresizeImplV_F32_LSX final : public UnresizeImplV {
public:
	UnresizeImplV_F32_LSX(const BilinearContext &context, unsigned width) :
		UnresizeImplV(context, width, context.output_width, PixelType::FLOAT)
	{
		m_desc.alignment_mask = 3;
	}

	void process(const graphengine::BufferDescriptor *in, const graphengine::BufferDescriptor *out,
	             unsigned, unsigned left, unsigned right, void *, void *) const noexcept override
	{
		unsigned height = m_desc.format.height;

		const float *above = nullptr;
		for (unsigned i = 0; i < height; ++i) {
			float *cur = out->get_line<float>(i);
			unresize_line_forward_v_f32_lsx(m_context.matrix_row_offsets[i], m_context.matrix_coefficients.data() + i * m_context.matrix_row_stride, m_context.matrix_row_size,
			                                m_context.lu_c[i], m_context.lu_l[i], static_cast<const float *>(in->ptr) + left, in->stride, in->mask, above, cur, right - left);
			above = cur;
		}

		const float *below = nullptr;
		for (unsigned i = height; i != 0; --i) {
			float *cur = out->get_line<float>(i - 1) + left;
			unresize_line_back_v_f32_lsx(m_context.lu_u[i - 1], below, cur, right - left);
			below = cur;
		}
	}
};

} // namespace


std::unique_ptr<graphengine::Filter> create_unresize_impl_h_lsx(const BilinearContext &context, unsigned height, PixelType type)
{
	if (type != PixelType::FLOAT)
		return nullptr;

	return std::make_unique<UnresizeImplH_F32_LSX>(context, height);
}

std::unique_ptr<graphengine::Filter> create_unresize_impl_v_lsx(const BilinearContext &context, unsigned width, PixelType type)
{
	if (type != PixelType::FLOAT)
		return nullptr;

	return std::make_unique<UnresizeImplV_F32_LSX>(context, width);
}

} // namespace zimg::unresize

#endif // ZIMG_LOONGARCH
