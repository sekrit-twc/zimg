#ifdef ZIMG_X86

#include <algorithm>
#include <stdexcept>
#include <immintrin.h>
#include "common/align.h"
#include "common/ccdep.h"
#include "common/checked_int.h"
#include "common/except.h"
#include "common/make_array.h"
#include "common/pixel.h"
#include "graph/image_filter.h"
#include "resize/filter.h"
#include "resize/resize_impl.h"
#include "resize_impl_x86.h"

#include "common/x86/sse_util.h"
#include "common/x86/avx_util.h"

namespace zimg {
namespace resize {

namespace {

void transpose_line_8x8_ps(float * RESTRICT dst, const float * const * RESTRICT src, unsigned left, unsigned right)
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


template <int Taps>
inline FORCE_INLINE __m256 resize_line8_h_f32_avx_xiter(unsigned j,
                                                        const unsigned * RESTRICT filter_left, const float * RESTRICT filter_data, unsigned filter_stride, unsigned filter_width,
                                                        const float * RESTRICT src_ptr, unsigned src_base)
{
	static_assert(Taps <= 8, "only up to 8 taps can be unrolled");
	static_assert(Taps >= -3, "only up to 3 taps in epilogue");
	constexpr int Tail = Taps >= 4 ? Taps - 4 : Taps > 0 ? Taps : -Taps;

	const float *filter_coeffs = filter_data + j * filter_stride;
	const float *src_p = src_ptr + (filter_left[j] - src_base) * 8;

	__m256 accum0 = _mm256_setzero_ps();
	__m256 accum1 = _mm256_setzero_ps();
	__m256 x, c, coeffs;

	unsigned k_end = Taps >= 4 ? 4 : Taps > 0 ? 0 : floor_n(filter_width, 4);

	for (unsigned k = 0; k < k_end; k += 4) {
		coeffs = _mm256_broadcast_ps((const __m128 *)(filter_coeffs + k));

		c = _mm256_shuffle_ps(coeffs, coeffs, _MM_SHUFFLE(0, 0, 0, 0));
		x = _mm256_load_ps(src_p + 0);
		x = _mm256_mul_ps(c, x);
		accum0 = _mm256_add_ps(accum0, x);

		c = _mm256_shuffle_ps(coeffs, coeffs, _MM_SHUFFLE(1, 1, 1, 1));
		x = _mm256_load_ps(src_p + 8);
		x = _mm256_mul_ps(c, x);
		accum1 = _mm256_add_ps(accum1, x);

		c = _mm256_shuffle_ps(coeffs, coeffs, _MM_SHUFFLE(2, 2, 2, 2));
		x = _mm256_load_ps(src_p + 16);
		x = _mm256_mul_ps(c, x);
		accum0 = _mm256_add_ps(accum0, x);

		c = _mm256_shuffle_ps(coeffs, coeffs, _MM_SHUFFLE(3, 3, 3, 3));
		x = _mm256_load_ps(src_p + 24);
		x = _mm256_mul_ps(c, x);
		accum1 = _mm256_add_ps(accum1, x);

		src_p += 32;
	}

	if (Tail >= 1) {
		coeffs = _mm256_broadcast_ps((const __m128 *)(filter_coeffs + k_end));

		c = _mm256_shuffle_ps(coeffs, coeffs, _MM_SHUFFLE(0, 0, 0, 0));
		x = _mm256_load_ps(src_p + 0);
		x = _mm256_mul_ps(c, x);
		accum0 = _mm256_add_ps(accum0, x);
	}
	if (Tail >= 2) {
		c = _mm256_shuffle_ps(coeffs, coeffs, _MM_SHUFFLE(1, 1, 1, 1));
		x = _mm256_load_ps(src_p + 8);
		x = _mm256_mul_ps(c, x);
		accum1 = _mm256_add_ps(accum1, x);
	}
	if (Tail >= 3) {
		c = _mm256_shuffle_ps(coeffs, coeffs, _MM_SHUFFLE(2, 2, 2, 2));
		x = _mm256_load_ps(src_p + 16);
		x = _mm256_mul_ps(c, x);
		accum0 = _mm256_add_ps(accum0, x);
	}
	if (Tail >= 4) {
		c = _mm256_shuffle_ps(coeffs, coeffs, _MM_SHUFFLE(3, 3, 3, 3));
		x = _mm256_load_ps(src_p + 24);
		x = _mm256_mul_ps(c, x);
		accum1 = _mm256_add_ps(accum1, x);
	}

	if (Taps <= 0 || Taps >= 2)
		accum0 = _mm256_add_ps(accum0, accum1);

	return accum0;
}

template <int Taps>
void resize_line8_h_f32_avx(const unsigned *filter_left, const float * RESTRICT filter_data, unsigned filter_stride, unsigned filter_width,
							const float * RESTRICT src_ptr, float * const *dst_ptr, unsigned src_base, unsigned left, unsigned right)
{
	unsigned vec_left = ceil_n(left, 8);
	unsigned vec_right = floor_n(right, 8);

	float * RESTRICT dst_p0 = dst_ptr[0];
	float * RESTRICT dst_p1 = dst_ptr[1];
	float * RESTRICT dst_p2 = dst_ptr[2];
	float * RESTRICT dst_p3 = dst_ptr[3];
	float * RESTRICT dst_p4 = dst_ptr[4];
	float * RESTRICT dst_p5 = dst_ptr[5];
	float * RESTRICT dst_p6 = dst_ptr[6];
	float * RESTRICT dst_p7 = dst_ptr[7];
#define XITER resize_line8_h_f32_avx_xiter<Taps>
#define XARGS filter_left, filter_data, filter_stride, filter_width, src_ptr, src_base
	for (unsigned j = left; j < vec_left; ++j) {
		__m256 x = XITER(j, XARGS);
		mm_scatter_ps(dst_p0 + j, dst_p1 + j, dst_p2 + j, dst_p3 + j, _mm256_castps256_ps128(x));
		mm_scatter_ps(dst_p4 + j, dst_p5 + j, dst_p6 + j, dst_p7 + j, _mm256_extractf128_ps(x, 1));
	}

	for (unsigned j = vec_left; j < vec_right; j += 8) {
		__m256 x0, x1, x2, x3, x4, x5, x6, x7;

		x0 = XITER(j + 0, XARGS);
		x1 = XITER(j + 1, XARGS);
		x2 = XITER(j + 2, XARGS);
		x3 = XITER(j + 3, XARGS);
		x4 = XITER(j + 4, XARGS);
		x5 = XITER(j + 5, XARGS);
		x6 = XITER(j + 6, XARGS);
		x7 = XITER(j + 7, XARGS);

		mm256_transpose8_ps(x0, x1, x2, x3, x4, x5, x6, x7);

		_mm256_store_ps(dst_p0 + j, x0);
		_mm256_store_ps(dst_p1 + j, x1);
		_mm256_store_ps(dst_p2 + j, x2);
		_mm256_store_ps(dst_p3 + j, x3);
		_mm256_store_ps(dst_p4 + j, x4);
		_mm256_store_ps(dst_p5 + j, x5);
		_mm256_store_ps(dst_p6 + j, x6);
		_mm256_store_ps(dst_p7 + j, x7);
	}

	for (unsigned j = vec_right; j < right; ++j) {
		__m256 x = XITER(j, XARGS);
		mm_scatter_ps(dst_p0 + j, dst_p1 + j, dst_p2 + j, dst_p3 + j, _mm256_castps256_ps128(x));
		mm_scatter_ps(dst_p4 + j, dst_p5 + j, dst_p6 + j, dst_p7 + j, _mm256_extractf128_ps(x, 1));
	}
#undef XITER
#undef XARGS
}

constexpr auto resize_line8_h_f32_avx_jt_small = make_array(
	resize_line8_h_f32_avx<1>,
	resize_line8_h_f32_avx<2>,
	resize_line8_h_f32_avx<3>,
	resize_line8_h_f32_avx<4>,
	resize_line8_h_f32_avx<5>,
	resize_line8_h_f32_avx<6>,
	resize_line8_h_f32_avx<7>,
	resize_line8_h_f32_avx<8>);

constexpr auto resize_line8_h_f32_avx_jt_large = make_array(
	resize_line8_h_f32_avx<0>,
	resize_line8_h_f32_avx<-1>,
	resize_line8_h_f32_avx<-2>,
	resize_line8_h_f32_avx<-3>);


template <unsigned Taps, bool Continue>
inline FORCE_INLINE __m256 resize_line_v_f32_avx_xiter(unsigned j,
                                                       const float * RESTRICT src_p0, const float * RESTRICT src_p1,
                                                       const float * RESTRICT src_p2, const float * RESTRICT src_p3,
                                                       const float * RESTRICT src_p4, const float * RESTRICT src_p5,
                                                       const float * RESTRICT src_p6, const float * RESTRICT src_p7, const float * RESTRICT accum_p,
                                                       const __m256 &c0, const __m256 &c1, const __m256 &c2, const __m256 &c3,
                                                       const __m256 &c4, const __m256 &c5, const __m256 &c6, const __m256 &c7)
{
	static_assert(Taps >= 1 && Taps <= 8, "must have between 1-8 taps");

	__m256 accum0 = _mm256_setzero_ps();
	__m256 accum1 = _mm256_setzero_ps();
	__m256 accum2 = _mm256_setzero_ps();
	__m256 accum3 = _mm256_setzero_ps();
	__m256 x;

	if (Taps >= 1) {
		x = _mm256_load_ps(src_p0 + j);
		x = _mm256_mul_ps(c0, x);
		accum0 = Continue ? _mm256_add_ps(_mm256_load_ps(accum_p + j), x) : x;
	}
	if (Taps >= 2) {
		x = _mm256_load_ps(src_p1 + j);
		x = _mm256_mul_ps(c1, x);
		accum1 = x;
	}
	if (Taps >= 3) {
		x = _mm256_load_ps(src_p2 + j);
		x = _mm256_mul_ps(c2, x);
		accum0 = _mm256_add_ps(accum0, x);
	}
	if (Taps >= 4) {
		x = _mm256_load_ps(src_p3 + j);
		x = _mm256_mul_ps(c3, x);
		accum1 = _mm256_add_ps(accum1, x);
	}

	if (Taps >= 5) {
		x = _mm256_load_ps(src_p4 + j);
		x = _mm256_mul_ps(c4, x);
		accum2 = x;
	}
	if (Taps >= 6) {
		x = _mm256_load_ps(src_p5 + j);
		x = _mm256_mul_ps(c5, x);
		accum3 = x;
	}
	if (Taps >= 7) {
		x = _mm256_load_ps(src_p6 + j);
		x = _mm256_mul_ps(c6, x);
		accum2 = _mm256_add_ps(accum2, x);
	}
	if (Taps >= 8) {
		x = _mm256_load_ps(src_p7 + j);
		x = _mm256_mul_ps(c7, x);
		accum3 = _mm256_add_ps(accum3, x);
	}

	accum0 = (Taps >= 2) ? _mm256_add_ps(accum0, accum1) : accum0;
	accum2 = (Taps >= 6) ? _mm256_add_ps(accum2, accum3) : accum2;
	accum0 = (Taps >= 4) ? _mm256_add_ps(accum0, accum2) : accum0;
	return accum0;
}

template <unsigned Taps, bool Continue>
void resize_line_v_f32_avx(const float *filter_data, const float * const *src_lines, float * RESTRICT dst, unsigned left, unsigned right)
{
	const float * RESTRICT src_p0 = src_lines[0];
	const float * RESTRICT src_p1 = src_lines[1];
	const float * RESTRICT src_p2 = src_lines[2];
	const float * RESTRICT src_p3 = src_lines[3];
	const float * RESTRICT src_p4 = src_lines[4];
	const float * RESTRICT src_p5 = src_lines[5];
	const float * RESTRICT src_p6 = src_lines[6];
	const float * RESTRICT src_p7 = src_lines[7];

	unsigned vec_left = ceil_n(left, 8);
	unsigned vec_right = floor_n(right, 8);

	const __m256 c0 = _mm256_broadcast_ss(filter_data + 0);
	const __m256 c1 = _mm256_broadcast_ss(filter_data + 1);
	const __m256 c2 = _mm256_broadcast_ss(filter_data + 2);
	const __m256 c3 = _mm256_broadcast_ss(filter_data + 3);
	const __m256 c4 = _mm256_broadcast_ss(filter_data + 4);
	const __m256 c5 = _mm256_broadcast_ss(filter_data + 5);
	const __m256 c6 = _mm256_broadcast_ss(filter_data + 6);
	const __m256 c7 = _mm256_broadcast_ss(filter_data + 7);

	__m256 accum;

#define XITER resize_line_v_f32_avx_xiter<Taps, Continue>
#define XARGS src_p0, src_p1, src_p2, src_p3, src_p4, src_p5, src_p6, src_p7, dst, c0, c1, c2, c3, c4, c5, c6, c7
	if (left != vec_left) {
		accum = XITER(vec_left - 8, XARGS);
		mm256_store_idxhi_ps(dst + vec_left - 8, accum, left % 8);
	}

	for (unsigned j = vec_left; j < vec_right; j += 8) {
		accum = XITER(j, XARGS);
		_mm256_store_ps(dst + j, accum);
	}

	if (right != vec_right) {
		accum = XITER(vec_right, XARGS);
		mm256_store_idxlo_ps(dst + vec_right, accum, right % 8);
	}
#undef XITER
#undef XARGS
}

constexpr auto resize_line_v_f32_avx_jt_init = make_array(
	resize_line_v_f32_avx<1, false>,
	resize_line_v_f32_avx<2, false>,
	resize_line_v_f32_avx<3, false>,
	resize_line_v_f32_avx<4, false>,
	resize_line_v_f32_avx<5, false>,
	resize_line_v_f32_avx<6, false>,
	resize_line_v_f32_avx<7, false>,
	resize_line_v_f32_avx<8, false>);

constexpr auto resize_line_v_f32_avx_jt_cont = make_array(
	resize_line_v_f32_avx<1, true>,
	resize_line_v_f32_avx<2, true>,
	resize_line_v_f32_avx<3, true>,
	resize_line_v_f32_avx<4, true>,
	resize_line_v_f32_avx<5, true>,
	resize_line_v_f32_avx<6, true>,
	resize_line_v_f32_avx<7, true>,
	resize_line_v_f32_avx<8, true>);


class ResizeImplH_F32_AVX final : public ResizeImplH {
	decltype(resize_line8_h_f32_avx_jt_small)::value_type m_func;
public:
	ResizeImplH_F32_AVX(const FilterContext &filter, unsigned height) :
		ResizeImplH(filter, image_attributes{ filter.filter_rows, height, PixelType::FLOAT }),
		m_func{}
	{
		if (filter.filter_width <= 8)
			m_func = resize_line8_h_f32_avx_jt_small[filter.filter_width - 1];
		else
			m_func = resize_line8_h_f32_avx_jt_large[filter.filter_width % 4];
	}

	unsigned get_simultaneous_lines() const override { return 8; }

	size_t get_tmp_size(unsigned left, unsigned right) const override
	{
		auto range = get_required_col_range(left, right);

		try {
			checked_size_t size = (static_cast<checked_size_t>(range.second) - floor_n(range.first, 8) + 8) * sizeof(float) * 8;
			return size.get();
		} catch (const std::overflow_error &) {
			error::throw_<error::OutOfMemory>();
		}
	}

	void process(void *, const graph::ImageBuffer<const void> *src, const graph::ImageBuffer<void> *dst, void *tmp, unsigned i, unsigned left, unsigned right) const override
	{
		const auto &src_buf = graph::static_buffer_cast<const float>(*src);
		const auto &dst_buf = graph::static_buffer_cast<float>(*dst);
		auto range = get_required_col_range(left, right);

		const float *src_ptr[8] = { 0 };
		float *dst_ptr[8] = { 0 };
		float *transpose_buf = static_cast<float *>(tmp);
		unsigned height = get_image_attributes().height;

		src_ptr[0] = src_buf[std::min(i + 0, height - 1)];
		src_ptr[1] = src_buf[std::min(i + 1, height - 1)];
		src_ptr[2] = src_buf[std::min(i + 2, height - 1)];
		src_ptr[3] = src_buf[std::min(i + 3, height - 1)];
		src_ptr[4] = src_buf[std::min(i + 4, height - 1)];
		src_ptr[5] = src_buf[std::min(i + 5, height - 1)];
		src_ptr[6] = src_buf[std::min(i + 6, height - 1)];
		src_ptr[7] = src_buf[std::min(i + 7, height - 1)];

		transpose_line_8x8_ps(transpose_buf, src_ptr, floor_n(range.first, 8), ceil_n(range.second, 8));

		dst_ptr[0] = dst_buf[std::min(i + 0, height - 1)];
		dst_ptr[1] = dst_buf[std::min(i + 1, height - 1)];
		dst_ptr[2] = dst_buf[std::min(i + 2, height - 1)];
		dst_ptr[3] = dst_buf[std::min(i + 3, height - 1)];
		dst_ptr[4] = dst_buf[std::min(i + 4, height - 1)];
		dst_ptr[5] = dst_buf[std::min(i + 5, height - 1)];
		dst_ptr[6] = dst_buf[std::min(i + 6, height - 1)];
		dst_ptr[7] = dst_buf[std::min(i + 7, height - 1)];

		m_func(m_filter.left.data(), m_filter.data.data(), m_filter.stride, m_filter.filter_width,
			   transpose_buf, dst_ptr, floor_n(range.first, 8), left, right);
	}
};


class ResizeImplV_F32_AVX final : public ResizeImplV {
public:
	ResizeImplV_F32_AVX(const FilterContext &filter, unsigned width) :
		ResizeImplV(filter, image_attributes{ width, filter.filter_rows, zimg::PixelType::FLOAT })
	{}

	void process(void *, const graph::ImageBuffer<const void> *src, const graph::ImageBuffer<void> *dst, void *, unsigned i, unsigned left, unsigned right) const override
	{
		const auto &src_buf = graph::static_buffer_cast<const float>(*src);
		const auto &dst_buf = graph::static_buffer_cast<float>(*dst);

		const float *filter_data = m_filter.data.data() + i * m_filter.stride;
		unsigned filter_width = m_filter.filter_width;
		unsigned src_height = m_filter.input_width;

		const float *src_lines[8] = { 0 };
		float *dst_line = dst_buf[i];

		{
			unsigned taps_remain = std::min(filter_width - 0, 8U);
			unsigned top = m_filter.left[i] + 0;

			src_lines[0] = src_buf[std::min(top + 0, src_height - 1)];
			src_lines[1] = src_buf[std::min(top + 1, src_height - 1)];
			src_lines[2] = src_buf[std::min(top + 2, src_height - 1)];
			src_lines[3] = src_buf[std::min(top + 3, src_height - 1)];
			src_lines[4] = src_buf[std::min(top + 4, src_height - 1)];
			src_lines[5] = src_buf[std::min(top + 5, src_height - 1)];
			src_lines[6] = src_buf[std::min(top + 6, src_height - 1)];
			src_lines[7] = src_buf[std::min(top + 7, src_height - 1)];

			resize_line_v_f32_avx_jt_init[taps_remain - 1](filter_data + 0, src_lines, dst_line, left, right);
		}

		for (unsigned k = 8; k < filter_width; k += 8) {
			unsigned taps_remain = std::min(filter_width - k, 8U);
			unsigned top = m_filter.left[i] + k;

			src_lines[0] = src_buf[std::min(top + 0, src_height - 1)];
			src_lines[1] = src_buf[std::min(top + 1, src_height - 1)];
			src_lines[2] = src_buf[std::min(top + 2, src_height - 1)];
			src_lines[3] = src_buf[std::min(top + 3, src_height - 1)];
			src_lines[4] = src_buf[std::min(top + 4, src_height - 1)];
			src_lines[5] = src_buf[std::min(top + 5, src_height - 1)];
			src_lines[6] = src_buf[std::min(top + 6, src_height - 1)];
			src_lines[7] = src_buf[std::min(top + 7, src_height - 1)];

			resize_line_v_f32_avx_jt_cont[taps_remain - 1](filter_data + k, src_lines, dst_line, left, right);
		}
	}
};

} // namespace


std::unique_ptr<graph::ImageFilter> create_resize_impl_h_avx(const FilterContext &context, unsigned height, PixelType type, unsigned depth)
{
	std::unique_ptr<graph::ImageFilter> ret;

	if (type == PixelType::FLOAT)
		ret = std::make_unique<ResizeImplH_F32_AVX>(context, height);

	return ret;
}

std::unique_ptr<graph::ImageFilter> create_resize_impl_v_avx(const FilterContext &context, unsigned width, PixelType type, unsigned depth)
{
	std::unique_ptr<graph::ImageFilter> ret;

	if (type == zimg::PixelType::FLOAT)
		ret = std::make_unique<ResizeImplV_F32_AVX>(context, width);

	return ret;
}

} // namespace resize
} // namespace zimg

#endif // ZIMG_X86
