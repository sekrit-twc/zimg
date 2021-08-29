#ifdef ZIMG_X86

#include <cstddef>
#include <xmmintrin.h>
#include "common/align.h"
#include "common/ccdep.h"
#include "common/checked_int.h"
#include "common/except.h"
#include "common/pixel.h"
#include "common/zassert.h"
#include "graph/image_filter.h"
#include "unresize/bilinear.h"
#include "unresize/unresize_impl.h"
#include "unresize_impl_x86.h"

#include "common/x86/sse_util.h"

namespace zimg {
namespace unresize {

namespace {

void transpose_line_4x4_ps(float * RESTRICT dst, const float *src_p0, const float *src_p1, const float *src_p2, const float *src_p3, unsigned width)
{
	for (unsigned j = 0; j < width; j += 4) {
		__m128 x0, x1, x2, x3;

		x0 = _mm_load_ps(src_p0 + j);
		x1 = _mm_load_ps(src_p1 + j);
		x2 = _mm_load_ps(src_p2 + j);
		x3 = _mm_load_ps(src_p3 + j);

		_MM_TRANSPOSE4_PS(x0, x1, x2, x3);

		_mm_store_ps(dst + 0, x0);
		_mm_store_ps(dst + 4, x1);
		_mm_store_ps(dst + 8, x2);
		_mm_store_ps(dst + 12, x3);

		dst += 16;
	}
}


void unresize_line4_h_f32_sse(const unsigned * RESTRICT filter_left, const float * RESTRICT filter_data, unsigned filter_stride, unsigned filter_width,
                              const float *lu_c, const float *lu_l, const float *lu_u, const float * RESTRICT src, float * const * RESTRICT dst, float *tmp, unsigned width)
{
	__m128 z = _mm_setzero_ps();
	__m128 w = _mm_setzero_ps();

	for (size_t j = 0; j < width; ++j) {
		__m128 accum = _mm_setzero_ps();
		const float *coeffs = filter_data + j * filter_stride;
		const float *src_p = src + filter_left[j] * 4;

		for (size_t k = 0; k < filter_width; ++k) {
			__m128 c = _mm_set_ps1(coeffs[k]);
			__m128 x = _mm_load_ps(src_p + k * 4);
			accum = _mm_add_ps(accum, _mm_mul_ps(c, x));
		}

		__m128 c = _mm_set_ps1(lu_c[j]);
		__m128 l = _mm_set_ps1(lu_l[j]);
		z = _mm_mul_ps(_mm_sub_ps(accum, _mm_mul_ps(c, z)), l); // (accum - c * z) + l
		_mm_store_ps(tmp + j * 4, z);
	}


	for (size_t j = width; j > floor_n(width, 4); --j) {
		w = _mm_sub_ps(_mm_load_ps(tmp + (j - 1) * 4), _mm_mul_ps(_mm_set_ps1(lu_u[j - 1]), w)); // dst[j - 1] - u[j - 1] * w
		mm_scatter_ps(dst[0] + j - 1, dst[1] + j - 1, dst[2] + j - 1, dst[3] + j - 1, w);
	}

	for (size_t j = floor_n(width, 4); j != 0; j -= 4) {
		__m128 val3 = _mm_load_ps(tmp + (j - 1) * 4);
		__m128 val2 = _mm_load_ps(tmp + (j - 2) * 4);
		__m128 val1 = _mm_load_ps(tmp + (j - 3) * 4);
		__m128 val0 = _mm_load_ps(tmp + (j - 4) * 4);

		w = _mm_sub_ps(val3, _mm_mul_ps(_mm_set_ps1(lu_u[j - 1]), w));
		val3 = w;

		w = _mm_sub_ps(val2, _mm_mul_ps(_mm_set_ps1(lu_u[j - 2]), w));
		val2 = w;

		w = _mm_sub_ps(val1, _mm_mul_ps(_mm_set_ps1(lu_u[j - 3]), w));
		val1 = w;

		w = _mm_sub_ps(val0, _mm_mul_ps(_mm_set_ps1(lu_u[j - 4]), w));
		val0 = w;

		_MM_TRANSPOSE4_PS(val0, val1, val2, val3);
		_mm_store_ps(dst[0] + j - 4, val0);
		_mm_store_ps(dst[1] + j - 4, val1);
		_mm_store_ps(dst[2] + j - 4, val2);
		_mm_store_ps(dst[3] + j - 4, val3);
	}
}


void unresize_line_forward_v_f32_sse(unsigned filter_offset, const float * RESTRICT filter_data, unsigned filter_width,
                                     float c_, float l_, const float * RESTRICT src, ptrdiff_t src_stride, unsigned src_mask,
                                     const float * RESTRICT above, float * RESTRICT dst, unsigned width)
{
	__m128 c = _mm_set_ps1(c_);
	__m128 l = _mm_set_ps1(l_);

	for (unsigned j = 0; j < floor_n(width, 4); j += 4) {
		__m128 z = above ? _mm_load_ps(above + j) : _mm_setzero_ps();
		__m128 accum = _mm_setzero_ps();

		for (unsigned k = 0; k < filter_width; ++k) {
			__m128 c = _mm_set_ps1(filter_data[k]);
			__m128 x = _mm_load_ps(src + (static_cast<ptrdiff_t>((filter_offset + k) & src_mask) * src_stride) / sizeof(float) + j);
			accum = _mm_add_ps(accum, _mm_mul_ps(c, x));
		}

		z = _mm_mul_ps(_mm_sub_ps(accum, _mm_mul_ps(c, z)), l); // (accum - c * z) + l
		_mm_store_ps(dst + j, z);
	}

	for (unsigned j = floor_n(width, 4); j < width; ++j) {
		__m128 z = above ? _mm_load_ps(above + j) : _mm_setzero_ps();
		__m128 accum = _mm_setzero_ps();

		for (unsigned k = 0; k < filter_width; ++k) {
			__m128 c = _mm_set_ps1(filter_data[k]);
			__m128 x = _mm_load_ps(src + (static_cast<ptrdiff_t>((filter_offset + k) & src_mask) * src_stride) / sizeof(float) + j);
			accum = _mm_add_ps(accum, _mm_mul_ps(c, x));
		}

		z = _mm_mul_ps(_mm_sub_ps(accum, _mm_mul_ps(c, z)), l); // (accum - c * z) + l
		mm_store_idxlo_ps(dst + j, z, width % 4);
	}
}

void unresize_line_back_v_f32_sse(float u_, const float * RESTRICT below, float * RESTRICT dst, unsigned width)
{
	__m128 u = _mm_set_ps1(u_);

	for (unsigned j = 0; j < floor_n(width, 4); j += 4) {
		__m128 w = below ? _mm_load_ps(below + j) : _mm_setzero_ps();
		w = _mm_sub_ps(_mm_load_ps(dst + j), _mm_mul_ps(u, w)); // dst[i] - u[i] * w
		_mm_store_ps(dst + j, w);
	}
	for (unsigned j = floor_n(width, 4); j < width; ++j) {
		__m128 w = below ? _mm_load_ps(below + j) : _mm_setzero_ps();
		w = _mm_sub_ps(_mm_load_ps(dst + j), _mm_mul_ps(u, w)); // dst[i] - u[i] * w
		mm_store_idxlo_ps(dst + j, w, width % 4);
	}
}


class UnresizeImplH_F32_SSE final : public UnresizeImplH {
public:
	UnresizeImplH_F32_SSE(const BilinearContext &context, unsigned height) :
		UnresizeImplH(context, image_attributes{ context.output_width, height, PixelType::FLOAT })
	{}

	unsigned get_simultaneous_lines() const override { return 4; }

	size_t get_tmp_size(unsigned, unsigned) const override
	{
		try {
			checked_size_t size = checked_size_t{ m_context.input_width } * 4 * sizeof(float);
			size += checked_size_t{ m_context.output_width } * 4 * sizeof(float);
			return size.get();
		} catch (const std::overflow_error &) {
			error::throw_<error::OutOfMemory>();
		}
	}

	void process(void *, const graph::ImageBuffer<const void> *src, const graph::ImageBuffer<void> *dst, void *tmp, unsigned i, unsigned, unsigned) const override
	{
		const auto &src_buf = graph::static_buffer_cast<const float>(*src);
		const auto &dst_buf = graph::static_buffer_cast<float>(*dst);

		const float *src_ptr[4] = { 0 };
		float *dst_ptr[4] = { 0 };
		float *transpose_buf = static_cast<float *>(tmp);
		float *transpose_buf2 = transpose_buf + m_context.input_width * 4;
		unsigned height = get_image_attributes().height;

		src_ptr[0] = src_buf[std::min(i + 0, height - 1)];
		src_ptr[1] = src_buf[std::min(i + 1, height - 1)];
		src_ptr[2] = src_buf[std::min(i + 2, height - 1)];
		src_ptr[3] = src_buf[std::min(i + 3, height - 1)];

		transpose_line_4x4_ps(transpose_buf, src_ptr[0], src_ptr[1], src_ptr[2], src_ptr[3], m_context.input_width);

		dst_ptr[0] = dst_buf[std::min(i + 0, height - 1)];
		dst_ptr[1] = dst_buf[std::min(i + 1, height - 1)];
		dst_ptr[2] = dst_buf[std::min(i + 2, height - 1)];
		dst_ptr[3] = dst_buf[std::min(i + 3, height - 1)];

		unresize_line4_h_f32_sse(m_context.matrix_row_offsets.data(), m_context.matrix_coefficients.data(), m_context.matrix_row_stride, m_context.matrix_row_size,
		                         m_context.lu_c.data(), m_context.lu_l.data(), m_context.lu_u.data(), transpose_buf, dst_ptr, transpose_buf2, m_context.output_width);
	}
};


class UnresizeImplV_F32_SSE final : public UnresizeImplV {
public:
	UnresizeImplV_F32_SSE(const BilinearContext &context, unsigned width) :
		UnresizeImplV(context, image_attributes{ width, context.output_width, PixelType::FLOAT })
	{}

	void process(void *, const graph::ImageBuffer<const void> *src, const graph::ImageBuffer<void> *dst, void *, unsigned, unsigned, unsigned) const override
	{
		const auto &src_buf = graph::static_buffer_cast<const float>(*src);
		const auto &dst_buf = graph::static_buffer_cast<float>(*dst);

		unsigned width = get_image_attributes().width;
		unsigned height = get_image_attributes().height;

		const float *above = nullptr;

		for (unsigned i = 0; i < height; ++i) {
			float *cur = dst_buf[i];
			unresize_line_forward_v_f32_sse(m_context.matrix_row_offsets[i], m_context.matrix_coefficients.data() + i * m_context.matrix_row_stride, m_context.matrix_row_size,
			                                m_context.lu_c[i], m_context.lu_l[i], src_buf.data(), src_buf.stride(), src_buf.mask(), above, cur, width);
			above = cur;
		}

		const float *below = nullptr;

		for (unsigned i = height; i != 0; --i) {
			float *cur = dst_buf[i - 1];
			unresize_line_back_v_f32_sse(m_context.lu_u[i - 1], below, cur, width);
			below = cur;
		}
	}
};

} // namespace


std::unique_ptr<graph::ImageFilter> create_unresize_impl_h_sse(const BilinearContext &context, unsigned height, PixelType type)
{
	if (type != PixelType::FLOAT)
		return nullptr;

	return std::make_unique<UnresizeImplH_F32_SSE>(context, height);
}

std::unique_ptr<graph::ImageFilter> create_unresize_impl_v_sse(const BilinearContext &context, unsigned width, PixelType type)
{
	if (type != PixelType::FLOAT)
		return nullptr;

	return std::make_unique<UnresizeImplV_F32_SSE>(context, width);
}

} // namespace unresize
} // namespace zimg

#endif
