#ifdef ZIMG_X86

#include <cstdint>
#include <emmintrin.h> // SSE2
#include "except.h"
#include "osdep.h"
#include "resize_impl.h"
#include "resize_impl_x86.h"

namespace zimg {;
namespace resize {;

namespace {;

void transpose4_ps(__m128 &x0, __m128 &x1, __m128 &x2, __m128 &x3)
{
	__m128d t0 = _mm_castps_pd(_mm_unpacklo_ps(x0, x1));
	__m128d t1 = _mm_castps_pd(_mm_unpacklo_ps(x2, x3));
	__m128d t2 = _mm_castps_pd(_mm_unpackhi_ps(x0, x1));
	__m128d t3 = _mm_castps_pd(_mm_unpackhi_ps(x2, x3));
	__m128d o0 = _mm_unpacklo_pd(t0, t1);
	__m128d o1 = _mm_unpackhi_pd(t0, t1);
	__m128d o2 = _mm_unpacklo_pd(t2, t3);
	__m128d o3 = _mm_unpackhi_pd(t2, t3);
	x0 = _mm_castpd_ps(o0);
	x1 = _mm_castpd_ps(o1);
	x2 = _mm_castpd_ps(o2);
	x3 = _mm_castpd_ps(o3);
}

void transpose4_epi32(__m128i &x0, __m128i &x1, __m128i &x2, __m128i &x3)
{
	__m128 tmp0 = _mm_castsi128_ps(x0);
	__m128 tmp1 = _mm_castsi128_ps(x1);
	__m128 tmp2 = _mm_castsi128_ps(x2);
	__m128 tmp3 = _mm_castsi128_ps(x3);

	transpose4_ps(tmp0, tmp1, tmp2, tmp3);

	x0 = _mm_castps_si128(tmp0);
	x1 = _mm_castps_si128(tmp1);
	x2 = _mm_castps_si128(tmp2);
	x3 = _mm_castps_si128(tmp3);
}

template <bool DoLoop>
void filter_plane_fp_h(const EvaluatedFilter &filter, const float * RESTRICT src, float * RESTRICT dst,
                       int src_width, int src_height, int src_stride, int dst_stride)
{
	for (int i = 0; i < mod(src_height, 4); i += 4) {
		int j;

		for (j = 0; j < mod(filter.height(), 4); ++j) {
			__m128 x0, x1, x2, x3;
			__m128 accum = _mm_setzero_ps();
			__m128 cached[4];

			const float *filter_row = filter.data() + j * filter.stride();
			int left = filter.left()[j];

			if (left + filter.stride() > src_width)
				break;

			for (int k = 0; k < (DoLoop ? filter.width() : 4); k += 4) {
				__m128 coeff = _mm_load_ps(filter_row + k);
				
				x0 = _mm_loadu_ps(src + (i + 0) * src_stride + left + k);
				x0 = _mm_mul_ps(coeff, x0);

				x1 = _mm_loadu_ps(src + (i + 1) * src_stride + left + k);
				x1 = _mm_mul_ps(coeff, x1);

				x2 = _mm_loadu_ps(src + (i + 2) * src_stride + left + k);
				x2 = _mm_mul_ps(coeff, x2);

				x3 = _mm_loadu_ps(src + (i + 3) * src_stride + left + k);
				x3 = _mm_mul_ps(coeff, x3);

				transpose4_ps(x0, x1, x2, x3);

				x0 = _mm_add_ps(x0, x2);
				x1 = _mm_add_ps(x1, x3);
				
				accum = _mm_add_ps(accum, x0);
				accum = _mm_add_ps(accum, x1);
			}
			cached[j % 4] = accum;

			if (j % 4 == 3) {
				int dst_j = mod(j, 4);

				transpose4_ps(cached[0], cached[1], cached[2], cached[3]);

				_mm_store_ps(dst + (i + 0) * dst_stride + dst_j, cached[0]);
				_mm_store_ps(dst + (i + 1) * dst_stride + dst_j, cached[1]);
				_mm_store_ps(dst + (i + 2) * dst_stride + dst_j, cached[2]);
				_mm_store_ps(dst + (i + 3) * dst_stride + dst_j, cached[3]);
			}
		}
		filter_plane_h_scalar(filter, src, dst, i, i + 4, mod(j, 4), filter.height(), src_stride, dst_stride, ScalarPolicy_F32{});
	}
	filter_plane_h_scalar(filter, src, dst, mod(src_height, 4), src_height, 0, filter.height(), src_stride, dst_stride, ScalarPolicy_F32{});
}

void filter_plane_fp_v(const EvaluatedFilter &filter, const float * RESTRICT src, float * RESTRICT dst,
                       int src_width, int src_height, int src_stride, int dst_stride)
{
	for (int i = 0; i < filter.height(); ++i) {
		__m128 coeff0, coeff1, coeff2, coeff3;
		__m128 x0, x1, x2, x3;
		__m128 accum0, accum1;

		const float *src_ptr0, *src_ptr1, *src_ptr2, *src_ptr3;
		float *dst_ptr = dst + i * dst_stride;

		for (int k = 0; k < mod(filter.width(), 4); k += 4) {
			src_ptr0 = src + (filter.left()[i] + k + 0) * src_stride;
			src_ptr1 = src + (filter.left()[i] + k + 1) * src_stride;
			src_ptr2 = src + (filter.left()[i] + k + 2) * src_stride;
			src_ptr3 = src + (filter.left()[i] + k + 3) * src_stride;

			coeff0 = _mm_set_ps1(filter.data()[i * filter.stride() + k + 0]);
			coeff1 = _mm_set_ps1(filter.data()[i * filter.stride() + k + 1]);
			coeff2 = _mm_set_ps1(filter.data()[i * filter.stride() + k + 2]);
			coeff3 = _mm_set_ps1(filter.data()[i * filter.stride() + k + 3]);

			for (int j = 0; j < mod(src_width, 4); j += 4) {
				x0 = _mm_load_ps(src_ptr0 + j);
				accum0 = _mm_mul_ps(coeff0, x0);

				x1 = _mm_load_ps(src_ptr1 + j);
				accum1 = _mm_mul_ps(coeff1, x1);

				x2 = _mm_load_ps(src_ptr2 + j);
				x2 = _mm_mul_ps(coeff2, x2);
				accum0 = _mm_add_ps(accum0, x2);

				x3 = _mm_load_ps(src_ptr3 + j);
				x3 = _mm_mul_ps(coeff3, x3);
				accum1 = _mm_add_ps(accum1, x3);

				accum0 = _mm_add_ps(accum0, accum1);

				if (k)
					accum0 = _mm_add_ps(accum0, _mm_load_ps(dst_ptr + j));

				_mm_store_ps(dst_ptr + j, accum0);
			}
		}
		if (filter.width() % 4) {
			int m = filter.width() % 4;
			int k = filter.width() - m;

			coeff2 = _mm_set_ps1(filter.data()[i * filter.stride() + k + 2]);
			coeff1 = _mm_set_ps1(filter.data()[i * filter.stride() + k + 1]);
			coeff0 = _mm_set_ps1(filter.data()[i * filter.stride() + k + 0]);

			src_ptr2 = src + (filter.left()[i] + k + 2) * src_stride;
			src_ptr1 = src + (filter.left()[i] + k + 1) * src_stride;
			src_ptr0 = src + (filter.left()[i] + k + 0) * src_stride;

			for (int j = 0; j < mod(src_width, 4); j += 4) {
				accum0 = _mm_setzero_ps();
				accum1 = _mm_setzero_ps();

				switch (m) {
				case 3:
					x2 = _mm_load_ps(src_ptr2 + j);
					accum0 = _mm_mul_ps(coeff2, x2);
				case 2:
					x1 = _mm_load_ps(src_ptr1 + j);
					accum1 = _mm_mul_ps(coeff1, x1);
				case 1:
					x0 = _mm_load_ps(src_ptr0 + j);
					x0 = _mm_mul_ps(coeff0, x0);
					accum0 = _mm_add_ps(accum0, x0);
				}

				accum0 = _mm_add_ps(accum0, accum1);

				if (k)
					accum0 = _mm_add_ps(accum0, _mm_load_ps(dst_ptr + j));

				_mm_store_ps(dst_ptr + j, accum0);
			}
		}
		filter_plane_v_scalar(filter, src, dst, i, i + 1, mod(src_width, 4), src_width, src_stride, dst_stride, ScalarPolicy_F32{});
	}
}

class ResizeImplSSE2 : public ResizeImpl {
public:
	ResizeImplSSE2(const EvaluatedFilter &filter_h, const EvaluatedFilter &filter_v) : ResizeImpl(filter_h, filter_v)
	{}

	void process_u16_h(const uint16_t * RESTRICT src, uint16_t * RESTRICT dst, uint16_t * RESTRICT tmp,
		int src_width, int src_height, int src_stride, int dst_stride) const override
	{
		throw ZimgUnsupportedError{ "not implemented yet" };
	}

	void process_u16_v(const uint16_t * RESTRICT src, uint16_t * RESTRICT dst, uint16_t * RESTRICT tmp,
		int src_width, int src_height, int src_stride, int dst_stride) const override
	{
		throw ZimgUnsupportedError{ "not implemented yet" };
	}

	void process_f16_h(const uint16_t * RESTRICT src, uint16_t * RESTRICT dst, uint16_t * RESTRICT tmp,
		int src_width, int src_height, int src_stride, int dst_stride) const override
	{
		throw ZimgUnsupportedError{ "not implemented yet" };
	}

	void process_f16_v(const uint16_t * RESTRICT src, uint16_t * RESTRICT dst, uint16_t * RESTRICT tmp,
		int src_width, int src_height, int src_stride, int dst_stride) const override
	{
		throw ZimgUnsupportedError{ "not implemented yet" };
	}

	void process_f32_h(const float * RESTRICT src, float * RESTRICT dst, float * RESTRICT tmp,
		int src_width, int src_height, int src_stride, int dst_stride) const override
	{
		if (m_filter_h.width() >= 4)
			filter_plane_fp_h<true>(m_filter_h, src, dst, src_width, src_height, src_stride, dst_stride);
		else
			filter_plane_fp_h<false>(m_filter_h, src, dst, src_width, src_height, src_stride, dst_stride);
	}

	void process_f32_v(const float * RESTRICT src, float * RESTRICT dst, float * RESTRICT tmp,
		int src_width, int src_height, int src_stride, int dst_stride) const override
	{
		filter_plane_fp_v(m_filter_v, src, dst, src_width, src_height, src_stride, dst_stride);
	}
};

} // namespace


ResizeImpl *create_resize_impl_sse2(const EvaluatedFilter &filter_h, const EvaluatedFilter &filter_v)
{
	return new ResizeImplSSE2{ filter_h, filter_v };
}

} // namespace resize
} // namespace zimg

#endif // ZIMG_X86
