#ifdef ZIMG_X86

#include <algorithm>
#include <stdexcept>
#include <xmmintrin.h>
#include "common/align.h"
#include "common/ccdep.h"
#include "common/checked_int.h"
#include "common/except.h"
#include "common/make_array.h"
#include "common/pixel.h"
#include "common/unroll.h"
#include "resize/filter.h"
#include "resize/resize_impl.h"
#include "resize_impl_x86.h"

#include "common/x86/sse_util.h"

namespace zimg::resize {

namespace {

void transpose_line_4x4_ps(float * RESTRICT dst, const float *src_p0, const float *src_p1, const float *src_p2, const float *src_p3, unsigned left, unsigned right)
{
	for (unsigned j = left; j < right; j += 4) {
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


template <int Taps>
inline FORCE_INLINE __m128 resize_line4_h_f32_sse_xiter(unsigned j, const unsigned *filter_left, const float *filter_data, unsigned filter_stride, unsigned filter_width,
                                                        const float *src, unsigned src_base)
{
	static_assert(Taps <= 8, "only up to 8 taps can be unrolled");
	static_assert(Taps >= -3, "only up to 3 taps in epilogue");
	constexpr int Tail = Taps >= 4 ? Taps - 4 : Taps > 0 ? Taps : -Taps;

	const float *filter_coeffs = filter_data + j * filter_stride;
	const float *src_p = src + (filter_left[j] - src_base) * 4;

	__m128 accum0 = _mm_setzero_ps();
	__m128 accum1 = _mm_setzero_ps();
	__m128 coeffs;

	auto f = ZIMG_UNROLL_FUNC(kk)
	{
		__m128 &acc = kk % 2 ? accum1 : accum0;
		__m128 c, x;

		c = _mm_shuffle_ps(coeffs, coeffs, static_cast<unsigned>(_MM_SHUFFLE(kk, kk, kk, kk)));
		x = _mm_load_ps(src_p + kk * 4);
		x = _mm_mul_ps(c, x);

		acc = _mm_add_ps(acc, x);
	};

	unsigned k_end = Taps >= 4 ? 4 : Taps > 0 ? 0 : floor_n(filter_width, 4);

	for (unsigned k = 0; k < k_end; k += 4) {
		coeffs = _mm_load_ps(filter_coeffs + k);
		unroll<4>(f);
		src_p += 16;
	}

	if constexpr (Tail) {
		coeffs = _mm_load_ps(filter_coeffs + k_end);
		unroll<4>(f);
	}

	if constexpr (Taps <= 0 || Taps >= 2)
		accum0 = _mm_add_ps(accum0, accum1);

	return accum0;
}

template <int Taps>
void resize_line4_h_f32_sse(const unsigned * RESTRICT filter_left, const float * RESTRICT filter_data, unsigned filter_stride, unsigned filter_width,
                            const float * RESTRICT src, float * const * RESTRICT dst, unsigned src_base, unsigned left, unsigned right)
{
	unsigned vec_left = ceil_n(left, 4);
	unsigned vec_right = floor_n(right, 4);

#define XITER resize_line4_h_f32_sse_xiter<Taps>
#define XARGS filter_left, filter_data, filter_stride, filter_width, src, src_base
	for (unsigned j = left; j < vec_left; ++j) {
		__m128 x = XITER(j, XARGS);
		mm_scatter_ps(dst[0] + j, dst[1] + j, dst[2] + j, dst[3] + j, x);
	}

	for (unsigned j = vec_left; j < vec_right; j += 4) {
		__m128 x0, x1, x2, x3;

		x0 = XITER(j + 0, XARGS);
		x1 = XITER(j + 1, XARGS);
		x2 = XITER(j + 2, XARGS);
		x3 = XITER(j + 3, XARGS);

		_MM_TRANSPOSE4_PS(x0, x1, x2, x3);

		_mm_store_ps(dst[0] + j, x0);
		_mm_store_ps(dst[1] + j, x1);
		_mm_store_ps(dst[2] + j, x2);
		_mm_store_ps(dst[3] + j, x3);
	}

	for (unsigned j = vec_right; j < right; ++j) {
		__m128 x = XITER(j, XARGS);
		mm_scatter_ps(dst[0] + j, dst[1] + j, dst[2] + j, dst[3] + j, x);
	}
#undef XITER
#undef XARGS
}

constexpr auto resize_line4_h_f32_sse_jt_small = make_array(
	resize_line4_h_f32_sse<1>,
	resize_line4_h_f32_sse<2>,
	resize_line4_h_f32_sse<3>,
	resize_line4_h_f32_sse<4>,
	resize_line4_h_f32_sse<5>,
	resize_line4_h_f32_sse<6>,
	resize_line4_h_f32_sse<7>,
	resize_line4_h_f32_sse<8>);

constexpr auto resize_line4_h_f32_sse_jt_large = make_array(
	resize_line4_h_f32_sse<0>,
	resize_line4_h_f32_sse<-1>,
	resize_line4_h_f32_sse<-2>,
	resize_line4_h_f32_sse<-3>);


template <unsigned Taps, bool Continue>
inline FORCE_INLINE __m128 resize_line_v_f32_sse_xiter(unsigned j, const float * const srcp[4], const float *accum_p, const __m128 c[4])
{
	static_assert(Taps >= 1 && Taps <= 4, "must have between 1-4 taps");

	__m128 accum0, accum1;

	unroll<Taps>(ZIMG_UNROLL_FUNC(k)
	{
		__m128 &acc = k % 2 ? accum1 : accum0;
		__m128 x;

		x = _mm_load_ps(srcp[k] + j);
		x = _mm_mul_ps(c[k], x);

		if constexpr (k == 0 && Continue)
			acc = _mm_add_ps(_mm_load_ps(accum_p + j), x);
		else if constexpr (k == 0 || k == 1)
			acc = x;
		else
			acc = _mm_add_ps(acc, x);
	});

	if constexpr (Taps >= 2) accum0 = _mm_add_ps(accum0, accum1);
	return accum0;
}

template <unsigned Taps, bool Continue>
void resize_line_v_f32_sse(const float * RESTRICT filter_data, const float * const * RESTRICT src, float * RESTRICT dst, unsigned left, unsigned right)
{
	const float *srcp[4] = { src[0], src[1], src[2], src[3] };
	unsigned vec_left = ceil_n(left, 4);
	unsigned vec_right = floor_n(right, 4);

	const __m128 c[4] = { _mm_set_ps1(filter_data[0]), _mm_set_ps1(filter_data[1]), _mm_set_ps1(filter_data[2]), _mm_set_ps1(filter_data[3]) };
	__m128 accum;

#define XITER resize_line_v_f32_sse_xiter<Taps, Continue>
#define XARGS srcp, dst, c
	if (left != vec_left) {
		accum = XITER(vec_left - 4, XARGS);
		mm_store_idxhi_ps(dst + vec_left - 4, accum, left % 4);
	}

	for (unsigned j = vec_left; j < vec_right; j += 4) {
		accum = XITER(j, XARGS);
		_mm_store_ps(dst + j, accum);
	}

	if (right != vec_right) {
		accum = XITER(vec_right, XARGS);
		mm_store_idxlo_ps(dst + vec_right, accum, right % 4);
	}
#undef XITER
#undef XARGS
}

constexpr auto resize_line_v_f32_sse_jt_init = make_array(
	resize_line_v_f32_sse<1, false>,
	resize_line_v_f32_sse<2, false>,
	resize_line_v_f32_sse<3, false>,
	resize_line_v_f32_sse<4, false>);

constexpr auto resize_line_v_f32_sse_jt_cont = make_array(
	resize_line_v_f32_sse<1, true>,
	resize_line_v_f32_sse<2, true>,
	resize_line_v_f32_sse<3, true>,
	resize_line_v_f32_sse<4, true>);


class ResizeImplH_F32_SSE : public ResizeImplH {
	decltype(resize_line4_h_f32_sse_jt_small)::value_type m_func;
public:
	ResizeImplH_F32_SSE(const FilterContext &filter, unsigned height) try :
		ResizeImplH(filter, height, PixelType::FLOAT),
		m_func{}
	{
		m_desc.step = 4;
		m_desc.scratchpad_size = (ceil_n(checked_size_t{ filter.input_width }, 4) * sizeof(float) * 4).get();

		if (filter.filter_width <= 8)
			m_func = resize_line4_h_f32_sse_jt_small[filter.filter_width - 1];
		else
			m_func = resize_line4_h_f32_sse_jt_large[filter.filter_width % 4];
	} catch (const std::overflow_error &) {
		error::throw_<error::OutOfMemory>();
	}

	void process(const graphengine::BufferDescriptor *in, const graphengine::BufferDescriptor *out,
	             unsigned i, unsigned left, unsigned right, void *, void *tmp) const noexcept override
	{
		auto range = get_col_deps(left, right);

		const float *src_ptr[4] = { 0 };
		float *dst_ptr[4] = { 0 };
		float *transpose_buf = static_cast<float *>(tmp);
		unsigned height = m_desc.format.height;

		src_ptr[0] = in->get_line<float>(std::min(i + 0, height - 1));
		src_ptr[1] = in->get_line<float>(std::min(i + 1, height - 1));
		src_ptr[2] = in->get_line<float>(std::min(i + 2, height - 1));
		src_ptr[3] = in->get_line<float>(std::min(i + 3, height - 1));

		transpose_line_4x4_ps(transpose_buf, src_ptr[0], src_ptr[1], src_ptr[2], src_ptr[3], floor_n(range.first, 4), ceil_n(range.second, 4));

		dst_ptr[0] = out->get_line<float>(std::min(i + 0, height - 1));
		dst_ptr[1] = out->get_line<float>(std::min(i + 1, height - 1));
		dst_ptr[2] = out->get_line<float>(std::min(i + 2, height - 1));
		dst_ptr[3] = out->get_line<float>(std::min(i + 3, height - 1));

		m_func(m_filter.left.data(), m_filter.data.data(), m_filter.stride, m_filter.filter_width,
		       transpose_buf, dst_ptr, floor_n(range.first, 4), left, right);
	}
};


class ResizeImplV_F32_SSE : public ResizeImplV {
public:
	ResizeImplV_F32_SSE(const FilterContext &filter, unsigned width) :
		ResizeImplV(filter, width, PixelType::FLOAT)
	{}

	void process(const graphengine::BufferDescriptor *in, const graphengine::BufferDescriptor *out,
	             unsigned i, unsigned left, unsigned right, void *, void *) const noexcept override
	{
		const float *filter_data = m_filter.data.data() + i * m_filter.stride;
		unsigned filter_width = m_filter.filter_width;
		unsigned src_height = m_filter.input_width;

		const float *src_lines[4] = { 0 };
		float *dst_line = out->get_line<float>(i);

		{
			unsigned taps_remain = std::min(filter_width - 0, 4U);
			unsigned top = m_filter.left[i] + 0;

			src_lines[0] = in->get_line<float>(std::min(top + 0, src_height - 1));
			src_lines[1] = in->get_line<float>(std::min(top + 1, src_height - 1));
			src_lines[2] = in->get_line<float>(std::min(top + 2, src_height - 1));
			src_lines[3] = in->get_line<float>(std::min(top + 3, src_height - 1));

			resize_line_v_f32_sse_jt_init[taps_remain - 1](filter_data + 0, src_lines, dst_line, left, right);
		}

		for (unsigned k = 4; k < filter_width; k += 4) {
			unsigned taps_remain = std::min(filter_width - k, 4U);
			unsigned top = m_filter.left[i] + k;

			src_lines[0] = in->get_line<float>(std::min(top + 0, src_height - 1));
			src_lines[1] = in->get_line<float>(std::min(top + 1, src_height - 1));
			src_lines[2] = in->get_line<float>(std::min(top + 2, src_height - 1));
			src_lines[3] = in->get_line<float>(std::min(top + 3, src_height - 1));

			resize_line_v_f32_sse_jt_cont[taps_remain - 1](filter_data + k, src_lines, dst_line, left, right);
		}
	}
};

} // namespace


std::unique_ptr<graphengine::Filter> create_resize_impl_h_sse(const FilterContext &context, unsigned height, PixelType type, unsigned depth)
{
	std::unique_ptr<graphengine::Filter> ret;

	if (type == PixelType::FLOAT)
		ret = std::make_unique<ResizeImplH_F32_SSE>(context, height);

	return ret;
}

std::unique_ptr<graphengine::Filter> create_resize_impl_v_sse(const FilterContext &context, unsigned width, PixelType type, unsigned depth)
{
	std::unique_ptr<graphengine::Filter> ret;

	if (type == PixelType::FLOAT)
		ret = std::make_unique<ResizeImplV_F32_SSE>(context, width);

	return ret;
}

} // namespace zimg::resize

#endif // ZIMG_X86
