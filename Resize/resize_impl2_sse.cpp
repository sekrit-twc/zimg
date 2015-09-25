#ifdef ZIMG_X86

#include <algorithm>
#include <xmmintrin.h>
#include "Common/align.h"
#include "Common/osdep.h"
#include "Common/pixel.h"
#include "Common/linebuffer.h"
#include "Common/zfilter.h"
#include "filter.h"
#include "resize_impl2.h"
#include "resize_impl2_x86.h"

namespace zimg {;
namespace resize {;

namespace {;

inline FORCE_INLINE void mm_store_left(float *dst, __m128 x, unsigned count)
{
	switch (count - 1) {
	case 2:
		x = _mm_shuffle_ps(x, x, _MM_SHUFFLE(3, 2, 1, 1));
		_mm_store_ss(dst + 1, x);
	case 1:
		x = _mm_shuffle_ps(x, x, _MM_SHUFFLE(3, 2, 1, 2));
		_mm_store_ss(dst + 2, x);
	case 0:
		x = _mm_shuffle_ps(x, x, _MM_SHUFFLE(3, 2, 1, 3));
		_mm_store_ss(dst + 3, x);
	}
}

inline FORCE_INLINE void mm_store_right(float *dst, __m128 x, unsigned count)
{
	__m128 y;

	switch (count - 1) {
	case 2:
		y = _mm_shuffle_ps(x, x, _MM_SHUFFLE(3, 2, 1, 2));
		_mm_store_ss(dst + 2, y);
	case 1:
		y = _mm_shuffle_ps(x, x, _MM_SHUFFLE(3, 2, 1, 1));
		_mm_store_ss(dst + 1, y);
	case 0:
		y = _mm_shuffle_ps(x, x, _MM_SHUFFLE(3, 2, 1, 0));
		_mm_store_ss(dst + 0, y);
	}
}

template <unsigned N, bool UpdateAccum>
inline FORCE_INLINE __m128 resize_line_v_f32_sse_xiter(unsigned j,
                                                       const float * RESTRICT src_p0, const float * RESTRICT src_p1,
                                                       const float * RESTRICT src_p2, const float * RESTRICT src_p3, const float * RESTRICT dst_p,
                                                       const __m128 &c0, const __m128 &c1, const __m128 &c2, const __m128 &c3)
{
	__m128 accum0 = _mm_setzero_ps();
	__m128 accum1 = _mm_setzero_ps();
	__m128 x;

	if (N >= 0) {
		x = _mm_load_ps(src_p0 + j);
		x = _mm_mul_ps(c0, x);
		accum0 = UpdateAccum ? _mm_add_ps(_mm_load_ps(dst_p + j), x) : x;
	}
	if (N >= 1) {
		x = _mm_load_ps(src_p1 + j);
		x = _mm_mul_ps(c1, x);
		accum1 = x;
	}
	if (N >= 2) {
		x = _mm_load_ps(src_p2 + j);
		x = _mm_mul_ps(c2, x);
		accum0 = _mm_add_ps(accum0, x);
	}
	if (N >= 3) {
		x = _mm_load_ps(src_p3 + j);
		x = _mm_mul_ps(c3, x);
		accum1 = _mm_add_ps(accum1, x);
	}

	accum0 = (N >= 1) ? _mm_add_ps(accum0, accum1) : accum0;
	return accum0;
}

template <unsigned N, bool UpdateAccum>
void resize_line_v_f32_sse(const float *filter_data, const float * const *src_lines, float *dst, unsigned left, unsigned right)
{
	const float * RESTRICT src_p0 = src_lines[0];
	const float * RESTRICT src_p1 = src_lines[1];
	const float * RESTRICT src_p2 = src_lines[2];
	const float * RESTRICT src_p3 = src_lines[3];
	float * RESTRICT dst_p = dst;

	unsigned vec_begin = align(left, 4);
	unsigned vec_end = mod(right, 4);

	const __m128 c0 = _mm_set_ps1(filter_data[0]);
	const __m128 c1 = _mm_set_ps1(filter_data[1]);
	const __m128 c2 = _mm_set_ps1(filter_data[2]);
	const __m128 c3 = _mm_set_ps1(filter_data[3]);

	__m128 accum;

#define XITER resize_line_v_f32_sse_xiter<N, UpdateAccum>
#define XARGS src_p0, src_p1, src_p2, src_p3, dst_p, c0, c1, c2, c3
	if (left != vec_begin) {
		accum = XITER(vec_begin - 4, XARGS);
		mm_store_left(dst_p + vec_begin - 4, accum, vec_begin - left);
	}

	for (unsigned j = vec_begin; j < vec_end; j += 4) {
		accum = XITER(j, XARGS);
		_mm_store_ps(dst_p + j, accum);
	}

	if (right != vec_end) {
		accum = XITER(vec_end, XARGS);
		mm_store_right(dst_p + vec_end, accum, right - vec_end);
	}
#undef XITER
#undef XARGS
}

const decltype(&resize_line_v_f32_sse<0, false>) resize_line_v_f32_sse_jt_a[] = {
	resize_line_v_f32_sse<0, false>,
	resize_line_v_f32_sse<1, false>,
	resize_line_v_f32_sse<2, false>,
	resize_line_v_f32_sse<3, false>,
};

const decltype(&resize_line_v_f32_sse<0, false>) resize_line_v_f32_sse_jt_b[] = {
	resize_line_v_f32_sse<0, true>,
	resize_line_v_f32_sse<1, true>,
	resize_line_v_f32_sse<2, true>,
	resize_line_v_f32_sse<3, true>,
};

class ResizeImplV_F32_SSE final : public ResizeImplV {
public:
	ResizeImplV_F32_SSE(const FilterContext &filter, unsigned width) :
		ResizeImplV(filter, image_attributes{ width, filter.filter_rows, zimg::PixelType::FLOAT })
	{
	}

	void process(void *, const ZimgImageBufferConst &src, const ZimgImageBuffer &dst, void *, unsigned i, unsigned left, unsigned right) const override
	{
		LineBuffer<const float> src_buf{ src };
		LineBuffer<float> dst_buf{ dst };

		const float *filter_data = m_filter.data.data() + i * m_filter.stride;
		unsigned filter_width = m_filter.filter_width;
		unsigned src_height = m_filter.input_width;

		const float *src_lines[4] = { 0 };
		float *dst_line = dst_buf[i];

		for (unsigned k = 0; k < filter_width; k += 4) {
			unsigned taps_remain = std::min(filter_width - k, 4U);
			unsigned top = m_filter.left[i] + k;

			src_lines[0] = src_buf[std::min(top + 0, src_height - 1)];
			src_lines[1] = src_buf[std::min(top + 1, src_height - 1)];
			src_lines[2] = src_buf[std::min(top + 2, src_height - 1)];
			src_lines[3] = src_buf[std::min(top + 3, src_height - 1)];

			if (k == 0)
				resize_line_v_f32_sse_jt_a[taps_remain - 1](filter_data + k, src_lines, dst_line, left, right);
			else
				resize_line_v_f32_sse_jt_b[taps_remain - 1](filter_data + k, src_lines, dst_line, left, right);
		}
	}
};

} // namespace


IZimgFilter *create_resize_impl2_v_sse(const FilterContext &context, unsigned width, PixelType type, unsigned depth)
{
	IZimgFilter *ret = nullptr;

	if (type == zimg::PixelType::FLOAT)
		ret = new ResizeImplV_F32_SSE{ context, width };

	return ret;
}

} // namespace resize
} // namespace zimg

#endif // ZIMG_X86
