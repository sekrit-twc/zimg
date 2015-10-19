#ifdef ZIMG_X86

#include <algorithm>
#include <immintrin.h>
#include "common/align.h"
#include "common/ccdep.h"

#define HAVE_CPU_AVX
  #include "common/x86util.h"
#undef HAVE_CPU_AVX

#include "common/make_unique.h"
#include "common/pixel.h"
#include "graph/image_filter.h"
#include "filter.h"
#include "resize_impl.h"
#include "resize_impl_x86.h"

namespace zimg {;
namespace resize {;

namespace {;

inline FORCE_INLINE void mm256_store_left(float *dst, __m256 x, unsigned count)
{
	mm256_store_left_ps(dst, x, count * 4);
}

inline FORCE_INLINE void mm256_store_right(float *dst, __m256 x, unsigned count)
{
	mm256_store_right_ps(dst, x, count * 4);
}

template <unsigned N, bool UpdateAccum>
inline FORCE_INLINE __m256 resize_line_v_f32_avx_xiter(unsigned j,
                                                       const float * RESTRICT src_p0, const float * RESTRICT src_p1,
                                                       const float * RESTRICT src_p2, const float * RESTRICT src_p3,
                                                       const float * RESTRICT src_p4, const float * RESTRICT src_p5,
                                                       const float * RESTRICT src_p6, const float * RESTRICT src_p7, const float * RESTRICT dst_p,
                                                       const __m256 &c0, const __m256 &c1, const __m256 &c2, const __m256 &c3,
                                                       const __m256 &c4, const __m256 &c5, const __m256 &c6, const __m256 &c7)
{
	__m256 accum0 = _mm256_setzero_ps();
	__m256 accum1 = _mm256_setzero_ps();
	__m256 accum2 = _mm256_setzero_ps();
	__m256 accum3 = _mm256_setzero_ps();
	__m256 x;

	if (N >= 0) {
		x = _mm256_load_ps(src_p0 + j);
		x = _mm256_mul_ps(c0, x);
		accum0 = UpdateAccum ? _mm256_add_ps(_mm256_load_ps(dst_p + j), x) : x;
	}
	if (N >= 1) {
		x = _mm256_load_ps(src_p1 + j);
		x = _mm256_mul_ps(c1, x);
		accum1 = x;
	}
	if (N >= 2) {
		x = _mm256_load_ps(src_p2 + j);
		x = _mm256_mul_ps(c2, x);
		accum0 = _mm256_add_ps(accum0, x);
	}
	if (N >= 3) {
		x = _mm256_load_ps(src_p3 + j);
		x = _mm256_mul_ps(c3, x);
		accum1 = _mm256_add_ps(accum1, x);
	}

	if (N >= 4) {
		x = _mm256_load_ps(src_p4 + j);
		x = _mm256_mul_ps(c4, x);
		accum2 = x;
	}
	if (N >= 5) {
		x = _mm256_load_ps(src_p5 + j);
		x = _mm256_mul_ps(c5, x);
		accum3 = x;
	}
	if (N >= 6) {
		x = _mm256_load_ps(src_p6 + j);
		x = _mm256_mul_ps(c6, x);
		accum2 = _mm256_add_ps(accum2, x);
	}
	if (N >= 7) {
		x = _mm256_load_ps(src_p7 + j);
		x = _mm256_mul_ps(c7, x);
		accum3 = _mm256_add_ps(accum3, x);
	}

	accum0 = (N >= 1) ? _mm256_add_ps(accum0, accum1) : accum0;
	accum2 = (N >= 5) ? _mm256_add_ps(accum2, accum3) : accum2;
	accum0 = (N >= 4) ? _mm256_add_ps(accum0, accum2) : accum0;
	return accum0;
}

template <unsigned N, bool UpdateAccum>
void resize_line_v_f32_avx(const float *filter_data, const float * const *src_lines, float *dst, unsigned left, unsigned right)
{
	const float * RESTRICT src_p0 = src_lines[0];
	const float * RESTRICT src_p1 = src_lines[1];
	const float * RESTRICT src_p2 = src_lines[2];
	const float * RESTRICT src_p3 = src_lines[3];
	const float * RESTRICT src_p4 = src_lines[4];
	const float * RESTRICT src_p5 = src_lines[5];
	const float * RESTRICT src_p6 = src_lines[6];
	const float * RESTRICT src_p7 = src_lines[7];
	float * RESTRICT dst_p = dst;

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

#define XITER resize_line_v_f32_avx_xiter<N, UpdateAccum>
#define XARGS src_p0, src_p1, src_p2, src_p3, src_p4, src_p5, src_p6, src_p7, dst_p, c0, c1, c2, c3, c4, c5, c6, c7
	if (left != vec_left) {
		accum = XITER(vec_left - 8, XARGS);
		mm256_store_left(dst_p + vec_left - 8, accum, vec_left - left);
	}

	for (unsigned j = vec_left; j < vec_right; j += 8) {
		accum = XITER(j, XARGS);
		_mm256_store_ps(dst_p + j, accum);
	}

	if (right != vec_right) {
		accum = XITER(vec_right, XARGS);
		mm256_store_right(dst_p + vec_right, accum, right - vec_right);
	}
#undef XITER
#undef XARGS
}

const decltype(&resize_line_v_f32_avx<0, false>) resize_line_v_f32_avx_jt_a[] = {
	resize_line_v_f32_avx<0, false>,
	resize_line_v_f32_avx<1, false>,
	resize_line_v_f32_avx<2, false>,
	resize_line_v_f32_avx<3, false>,
	resize_line_v_f32_avx<4, false>,
	resize_line_v_f32_avx<5, false>,
	resize_line_v_f32_avx<6, false>,
	resize_line_v_f32_avx<7, false>,
};

const decltype(&resize_line_v_f32_avx<0, false>) resize_line_v_f32_avx_jt_b[] = {
	resize_line_v_f32_avx<0, true>,
	resize_line_v_f32_avx<1, true>,
	resize_line_v_f32_avx<2, true>,
	resize_line_v_f32_avx<3, true>,
	resize_line_v_f32_avx<4, true>,
	resize_line_v_f32_avx<5, true>,
	resize_line_v_f32_avx<6, true>,
	resize_line_v_f32_avx<7, true>,
};


class ResizeImplV_F32_AVX final : public ResizeImplV {
public:
	ResizeImplV_F32_AVX(const FilterContext &filter, unsigned width) :
		ResizeImplV(filter, image_attributes{ width, filter.filter_rows, zimg::PixelType::FLOAT })
	{
	}

	void process(void *, const graph::ImageBuffer<const void> *src, const graph::ImageBuffer<void> *dst, void *, unsigned i, unsigned left, unsigned right) const override
	{
		auto src_buf = graph::static_buffer_cast<const float>(*src);
		auto dst_buf = graph::static_buffer_cast<float>(*dst);

		const float *filter_data = m_filter.data.data() + i * m_filter.stride;
		unsigned filter_width = m_filter.filter_width;
		unsigned src_height = m_filter.input_width;

		const float *src_lines[8] = { 0 };
		float *dst_line = dst_buf[i];

		for (unsigned k = 0; k < filter_width; k += 8) {
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

			if (k == 0)
				resize_line_v_f32_avx_jt_a[taps_remain - 1](filter_data + k, src_lines, dst_line, left, right);
			else
				resize_line_v_f32_avx_jt_b[taps_remain - 1](filter_data + k, src_lines, dst_line, left, right);
		}
	}
};

} // namespace


std::unique_ptr<graph::ImageFilter> create_resize_impl_v_avx(const FilterContext &context, unsigned width, PixelType type, unsigned depth)
{
	std::unique_ptr<graph::ImageFilter> ret;

	if (type == zimg::PixelType::FLOAT)
		ret = ztd::make_unique<ResizeImplV_F32_AVX>(context, width);

	return ret;
}

} // namespace resize
} // namespace zimg

#endif // ZIMG_X86
