#ifdef ZIMG_X86

#include <cstddef>
#include <immintrin.h>
#include "common/align.h"
#include "common/ccdep.h"
#include "common/checked_int.h"
#include "common/except.h"
#include "common/pixel.h"
#include "common/zassert.h"
#include "unresize/bilinear.h"
#include "unresize/unresize_impl.h"
#include "unresize_impl_x86.h"

#include "common/x86/avx2_util.h"
#include "common/x86/sse2_util.h"

namespace zimg::unresize {

namespace {

void transpose_line_8x8_ps(float *RESTRICT dst, const float * const * RESTRICT src, unsigned left, unsigned right)
{
	for (unsigned j = left; j < right; j += 8) {
		__m256 x0, x1, x2, x3, x4, x5, x6, x7;

		x0 = _mm256_load_ps(src[0] + j);
		x1 = _mm256_load_ps(src[1] + j);
		x2 = _mm256_load_ps(src[2] + j);
		x3 = _mm256_load_ps(src[3] + j);
		x4 = _mm256_load_ps(src[4] + j);
		x5 = _mm256_load_ps(src[5] + j);
		x6 = _mm256_load_ps(src[6] + j);
		x7 = _mm256_load_ps(src[7] + j);

		mm256_transpose8_ps(x0, x1, x2, x3, x4, x5, x6, x7);

		_mm256_store_ps(dst + 0, x0);
		_mm256_store_ps(dst + 8, x1);
		_mm256_store_ps(dst + 16, x2);
		_mm256_store_ps(dst + 24, x3);
		_mm256_store_ps(dst + 32, x4);
		_mm256_store_ps(dst + 40, x5);
		_mm256_store_ps(dst + 48, x6);
		_mm256_store_ps(dst + 56, x7);

		dst += 64;
	}
}

void unresize_line8_h_f32_avx2(const unsigned * RESTRICT filter_left, const float * RESTRICT filter_data, unsigned filter_stride, unsigned filter_width,
                               const float *lu_c, const float *lu_l, const float *lu_u, const float * RESTRICT src, float * const * RESTRICT dst, float *tmp, unsigned width)
{
	__m256 z = _mm256_setzero_ps();
	__m256 w = _mm256_setzero_ps();

	for (size_t j = 0; j < width; ++j) {
		__m256 accum = _mm256_setzero_ps();
		const float *coeffs = filter_data + j * filter_stride;
		const float *src_p = src + filter_left[j] * 8;

		for (size_t k = 0; k < filter_width; ++k) {
			__m256 c = _mm256_broadcast_ss(coeffs + k);
			__m256 x = _mm256_load_ps(src_p + k * 8);
			accum = _mm256_fmadd_ps(c, x, accum);
		}

		__m256 c = _mm256_broadcast_ss(lu_c + j);
		__m256 l = _mm256_broadcast_ss(lu_l + j);
		z = _mm256_mul_ps(_mm256_fnmadd_ps(c, z, accum), l); // (accum - c * z) + l
		_mm256_store_ps(tmp + j * 8, z);
	}


	for (size_t j = width; j > floor_n(width, 8); --j) {
		__m256 val = _mm256_load_ps(tmp + (j - 1) * 8);
		w = _mm256_fnmadd_ps(_mm256_broadcast_ss(lu_u + j - 1), w, val); // dst[j - 1] - u[j - 1] * w
		mm_scatter_ps(dst[0] + j - 1, dst[1] + j - 1, dst[2] + j - 1, dst[3] + j - 1, _mm256_castps256_ps128(w));
		mm_scatter_ps(dst[4] + j - 1, dst[5] + j - 1, dst[6] + j - 1, dst[7] + j - 1, _mm256_extractf128_ps(w, 1));
	}

	for (size_t j = floor_n(width, 8); j != 0; j -= 8) {
		__m256 val7 = _mm256_load_ps(tmp + (j - 1) * 8);
		__m256 val6 = _mm256_load_ps(tmp + (j - 2) * 8);
		__m256 val5 = _mm256_load_ps(tmp + (j - 3) * 8);
		__m256 val4 = _mm256_load_ps(tmp + (j - 4) * 8);
		__m256 val3 = _mm256_load_ps(tmp + (j - 5) * 8);
		__m256 val2 = _mm256_load_ps(tmp + (j - 6) * 8);
		__m256 val1 = _mm256_load_ps(tmp + (j - 7) * 8);
		__m256 val0 = _mm256_load_ps(tmp + (j - 8) * 8);

		w = _mm256_fnmadd_ps(_mm256_broadcast_ss(lu_u + j - 1), w, val7);
		val7 = w;

		w = _mm256_fnmadd_ps(_mm256_broadcast_ss(lu_u + j - 2), w, val6);
		val6 = w;

		w = _mm256_fnmadd_ps(_mm256_broadcast_ss(lu_u + j - 3), w, val5);
		val5 = w;

		w = _mm256_fnmadd_ps(_mm256_broadcast_ss(lu_u + j - 4), w, val4);
		val4 = w;

		w = _mm256_fnmadd_ps(_mm256_broadcast_ss(lu_u + j - 5), w, val3);
		val3 = w;

		w = _mm256_fnmadd_ps(_mm256_broadcast_ss(lu_u + j - 6), w, val2);
		val2 = w;

		w = _mm256_fnmadd_ps(_mm256_broadcast_ss(lu_u + j - 7), w, val1);
		val1 = w;

		w = _mm256_fnmadd_ps(_mm256_broadcast_ss(lu_u + j - 8), w, val0);
		val0 = w;

		mm256_transpose8_ps(val0, val1, val2, val3, val4, val5, val6, val7);

		_mm256_store_ps(dst[0] + j - 8, val0);
		_mm256_store_ps(dst[1] + j - 8, val1);
		_mm256_store_ps(dst[2] + j - 8, val2);
		_mm256_store_ps(dst[3] + j - 8, val3);
		_mm256_store_ps(dst[4] + j - 8, val4);
		_mm256_store_ps(dst[5] + j - 8, val5);
		_mm256_store_ps(dst[6] + j - 8, val6);
		_mm256_store_ps(dst[7] + j - 8, val7);
	}
}


void unresize_line_forward_v_f32_avx2(unsigned filter_offset, const float * RESTRICT filter_data, unsigned filter_width,
                                      float c_, float l_, const float * RESTRICT src, ptrdiff_t src_stride, unsigned src_mask,
                                      const float * RESTRICT above, float * RESTRICT dst, unsigned width)
{
	__m256 c = _mm256_set1_ps(c_);
	__m256 l = _mm256_set1_ps(l_);

	for (unsigned j = 0; j < floor_n(width, 8); j += 8) {
		__m256 z = above ? _mm256_load_ps(above + j) : _mm256_setzero_ps();
		__m256 accum = _mm256_setzero_ps();

		for (unsigned k = 0; k < filter_width; ++k) {
			__m256 c = _mm256_broadcast_ss(filter_data + k);
			__m256 x = _mm256_load_ps(src + (static_cast<ptrdiff_t>((filter_offset + k) & src_mask) * src_stride) / sizeof(float) + j);
			accum = _mm256_fmadd_ps(c, x, accum);
		}

		z = _mm256_mul_ps(_mm256_fnmadd_ps(c, z, accum), l); // (accum - c * z) + l
		_mm256_store_ps(dst + j, z);
	}

	for (unsigned j = floor_n(width, 8); j < width; j += 8) {
		__m256 z = above ? _mm256_load_ps(above + j) : _mm256_setzero_ps();
		__m256 accum = _mm256_setzero_ps();

		for (unsigned k = 0; k < filter_width; ++k) {
			__m256 c = _mm256_broadcast_ss(filter_data + k);
			__m256 x = _mm256_load_ps(src + (static_cast<ptrdiff_t>((filter_offset + k) & src_mask) * src_stride) / sizeof(float) + j);
			accum = _mm256_fmadd_ps(c, x, accum);
		}

		z = _mm256_mul_ps(_mm256_fnmadd_ps(c, z, accum), l); // (accum - c * z) + l
		mm256_store_idxlo_ps(dst + j, z, width % 8);
	}
}

void unresize_line_back_v_f32_avx2(float u_, const float * RESTRICT below, float * RESTRICT dst, unsigned width)
{
	__m256 u = _mm256_set1_ps(u_);

	for (unsigned j = 0; j < floor_n(width, 8); j += 8) {
		__m256 w = below ? _mm256_load_ps(below + j) : _mm256_setzero_ps();
		w = _mm256_fnmadd_ps(u, w, _mm256_load_ps(dst + j)); // dst[i] - u[i] * w
		_mm256_store_ps(dst + j, w);
	}
	for (unsigned j = floor_n(width, 8); j < width; j += 8) {
		__m256 w = below ? _mm256_load_ps(below + j) : _mm256_setzero_ps();
		w = _mm256_fnmadd_ps(u, w, _mm256_load_ps(dst + j)); // dst[i] - u[i] * w
		mm256_store_idxlo_ps(dst + j, w, width % 8);
	}
}


class UnresizeImplH_F32_AVX2 final : public UnresizeImplH {
public:
	UnresizeImplH_F32_AVX2(const BilinearContext &context, unsigned height) :
		UnresizeImplH(context, context.output_width, height, PixelType::FLOAT)
	{
		m_desc.step = 8;
		m_desc.scratchpad_size = ((static_cast<checked_size_t>(m_context.input_width) + m_context.output_width) * 8 * sizeof(float)).get();
	}

	void process(const graphengine::BufferDescriptor *in, const graphengine::BufferDescriptor *out,
	             unsigned i, unsigned left, unsigned right, void *, void *tmp) const noexcept override
	{
		const float *src_ptr[8] = { 0 };
		float *dst_ptr[8] = { 0 };
		float *transpose_buf = static_cast<float *>(tmp);
		float *transpose_buf2 = transpose_buf + m_context.input_width * 8;
		unsigned height = m_desc.format.height;

		src_ptr[0] = in->get_line<float>(std::min(i + 0, height - 1));
		src_ptr[1] = in->get_line<float>(std::min(i + 1, height - 1));
		src_ptr[2] = in->get_line<float>(std::min(i + 2, height - 1));
		src_ptr[3] = in->get_line<float>(std::min(i + 3, height - 1));
		src_ptr[4] = in->get_line<float>(std::min(i + 4, height - 1));
		src_ptr[5] = in->get_line<float>(std::min(i + 5, height - 1));
		src_ptr[6] = in->get_line<float>(std::min(i + 6, height - 1));
		src_ptr[7] = in->get_line<float>(std::min(i + 7, height - 1));

		transpose_line_8x8_ps(transpose_buf, src_ptr, 0, m_context.input_width);

		dst_ptr[0] = out->get_line<float>(std::min(i + 0, height - 1));
		dst_ptr[1] = out->get_line<float>(std::min(i + 1, height - 1));
		dst_ptr[2] = out->get_line<float>(std::min(i + 2, height - 1));
		dst_ptr[3] = out->get_line<float>(std::min(i + 3, height - 1));
		dst_ptr[4] = out->get_line<float>(std::min(i + 4, height - 1));
		dst_ptr[5] = out->get_line<float>(std::min(i + 5, height - 1));
		dst_ptr[6] = out->get_line<float>(std::min(i + 6, height - 1));
		dst_ptr[7] = out->get_line<float>(std::min(i + 7, height - 1));

		unresize_line8_h_f32_avx2(m_context.matrix_row_offsets.data(), m_context.matrix_coefficients.data(), m_context.matrix_row_stride, m_context.matrix_row_size,
		                          m_context.lu_c.data(), m_context.lu_l.data(), m_context.lu_u.data(), transpose_buf, dst_ptr, transpose_buf2, m_context.output_width);
	}
};


class UnresizeImplV_F32_AVX2 final : public UnresizeImplV {
public:
	UnresizeImplV_F32_AVX2(const BilinearContext &context, unsigned width) :
		UnresizeImplV(context, width, context.output_width, PixelType::FLOAT)
	{
		m_desc.alignment_mask = 7;
	}

	void process(const graphengine::BufferDescriptor *in, const graphengine::BufferDescriptor *out,
	             unsigned, unsigned left, unsigned right, void *, void *) const noexcept override
	{
		unsigned height = m_desc.format.height;

		const float *above = nullptr;
		for (unsigned i = 0; i < height; ++i) {
			float *cur = out->get_line<float>(i);
			unresize_line_forward_v_f32_avx2(m_context.matrix_row_offsets[i], m_context.matrix_coefficients.data() + i * m_context.matrix_row_stride, m_context.matrix_row_size,
			                                 m_context.lu_c[i], m_context.lu_l[i], static_cast<const float *>(in->ptr) + left, in->stride, in->mask, above, cur, right - left);
			above = cur;
		}

		const float *below = nullptr;
		for (unsigned i = height; i != 0; --i) {
			float *cur = out->get_line<float>(i - 1) + left;
			unresize_line_back_v_f32_avx2(m_context.lu_u[i - 1], below, cur, right - left);
			below = cur;
		}
	}
};

} // namespace


std::unique_ptr<graphengine::Filter> create_unresize_impl_h_avx2(const BilinearContext &context, unsigned height, PixelType type)
{
	if (type != PixelType::FLOAT)
		return nullptr;

	return std::make_unique<UnresizeImplH_F32_AVX2>(context, height);
}

std::unique_ptr<graphengine::Filter> create_unresize_impl_v_avx2(const BilinearContext &context, unsigned width, PixelType type)
{
	if (type != PixelType::FLOAT)
		return nullptr;

	return std::make_unique<UnresizeImplV_F32_AVX2>(context, width);
}

} // namespace zimg::unresize

#endif
