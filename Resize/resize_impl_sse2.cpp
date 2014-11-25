#ifdef ZIMG_X86

#include <cstddef>
#include <cstdint>
#include <emmintrin.h>
#include "Common/align.h"
#include "Common/except.h"
#include "Common/osdep.h"
#include "Common/plane.h"
#include "resize_impl.h"
#include "resize_impl_x86.h"

namespace zimg {;
namespace resize {;

namespace {;

inline FORCE_INLINE void transpose4_ps(__m128 &x0, __m128 &x1, __m128 &x2, __m128 &x3)
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

inline FORCE_INLINE void transpose4_epi32(__m128i &x0, __m128i &x1, __m128i &x2, __m128i &x3)
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

inline FORCE_INLINE __m128i mhadd_epi16_epi32(__m128i a, __m128i b)
{
	__m128i lo, hi, uplo, uphi;

	lo = _mm_mullo_epi16(a, b);
	hi = _mm_mulhi_epi16(a, b);

	uplo = _mm_unpacklo_epi16(lo, hi);
	uphi = _mm_unpackhi_epi16(lo, hi);

	return _mm_add_epi32(uplo, uphi);
}

inline FORCE_INLINE void fmadd_epi16_epi32(__m128i a, __m128i b, __m128i &accum0, __m128i &accum1)
{
	__m128i lo, hi, uplo, uphi;

	lo = _mm_mullo_epi16(a, b);
	hi = _mm_mulhi_epi16(a, b);

	uplo = _mm_unpacklo_epi16(lo, hi);
	uphi = _mm_unpackhi_epi16(lo, hi);

	accum0 = _mm_add_epi32(accum0, uplo);
	accum1 = _mm_add_epi32(accum1, uphi);
}

inline FORCE_INLINE __m128i pack_i30_epi32(__m128i lo, __m128i hi)
{
	__m128i offset = _mm_set1_epi32(1 << 13);

	lo = _mm_add_epi32(lo, offset);
	hi = _mm_add_epi32(hi, offset);

	lo = _mm_srai_epi32(lo, 14);
	hi = _mm_srai_epi32(hi, 14);

	return  _mm_packs_epi32(lo, hi);
}

template <bool DoLoop>
void filter_plane_u16_h(const EvaluatedFilter &filter, const ImagePlane<const uint16_t> &src, const ImagePlane<uint16_t> &dst)
{
	__m128i INT16_MIN_EPI16 = _mm_set1_epi16(INT16_MIN);

	const int16_t *filter_data = filter.data_i16();
	const int *filter_left = filter.left();
	ptrdiff_t filter_stride = filter.stride_i16();

	int src_width = src.width();
	int src_height = src.height();

	for (ptrdiff_t i = 0; i < mod(src_height, 4); i += 4) {
		const uint16_t *src_p0 = src[i + 0];
		const uint16_t *src_p1 = src[i + 1];
		const uint16_t *src_p2 = src[i + 2];
		const uint16_t *src_p3 = src[i + 3];

		uint16_t *dst_p0 = dst[i + 0];
		uint16_t *dst_p1 = dst[i + 1];
		uint16_t *dst_p2 = dst[i + 2];
		uint16_t *dst_p3 = dst[i + 3];

		ptrdiff_t j;

		for (j = 0; j < filter.height(); ++j) {
			__m128i accum = _mm_setzero_si128();
			__m128i cached[8];

			const int16_t *filter_row = &filter_data[j * filter_stride];
			ptrdiff_t left = filter_left[j];

			if (left + filter_stride > src_width)
				break;

			for (ptrdiff_t k = 0; k < (DoLoop ? filter.width() : 8); k += 8) {
				__m128i coeff = _mm_load_si128((const __m128i *)&filter_row[k]);
				__m128i x0, x1, x2, x3;

				x0 = _mm_loadu_si128((const __m128i *)&src_p0[left + k]);
				x0 = _mm_add_epi16(x0, INT16_MIN_EPI16);
				x0 = mhadd_epi16_epi32(coeff, x0);

				x1 = _mm_loadu_si128((const __m128i *)&src_p1[left + k]);
				x1 = _mm_add_epi16(x1, INT16_MIN_EPI16);
				x1 = mhadd_epi16_epi32(coeff, x1);

				x2 = _mm_loadu_si128((const __m128i *)&src_p2[left + k]);
				x2 = _mm_add_epi16(x2, INT16_MIN_EPI16);
				x2 = mhadd_epi16_epi32(coeff, x2);

				x3 = _mm_loadu_si128((const __m128i *)&src_p3[left + k]);
				x3 = _mm_add_epi16(x3, INT16_MIN_EPI16);
				x3 = mhadd_epi16_epi32(coeff, x3);

				transpose4_epi32(x0, x1, x2, x3);

				x0 = _mm_add_epi32(x0, x2);
				x1 = _mm_add_epi32(x1, x3);

				accum = _mm_add_epi32(accum, x0);
				accum = _mm_add_epi32(accum, x1);
			}
			cached[j % 8] = accum;

			if (j % 8 == 7) {
				ptrdiff_t dst_j = mod(j, 8);
				__m128i packed;

				transpose4_epi32(cached[0], cached[1], cached[2], cached[3]);
				transpose4_epi32(cached[4], cached[5], cached[6], cached[7]);

				packed = pack_i30_epi32(cached[0], cached[4]);
				packed = _mm_sub_epi16(packed, INT16_MIN_EPI16);
				_mm_store_si128((__m128i *)&dst_p0[dst_j], packed);

				packed = pack_i30_epi32(cached[1], cached[5]);
				packed = _mm_sub_epi16(packed, INT16_MIN_EPI16);
				_mm_store_si128((__m128i *)&dst_p1[dst_j], packed);

				packed = pack_i30_epi32(cached[2], cached[6]);
				packed = _mm_sub_epi16(packed, INT16_MIN_EPI16);
				_mm_store_si128((__m128i *)&dst_p2[dst_j], packed);

				packed = pack_i30_epi32(cached[3], cached[7]);
				packed = _mm_sub_epi16(packed, INT16_MIN_EPI16);
				_mm_store_si128((__m128i *)&dst_p3[dst_j], packed);
			}
		}
		filter_plane_h_scalar(filter, src, dst, i, i + 4, mod(j, 8), filter.height(), ScalarPolicy_U16{});
	}
	filter_plane_h_scalar(filter, src, dst, mod(src_height, 4), src_height, 0, filter.height(), ScalarPolicy_U16{});
}

template <bool DoLoop>
void filter_plane_fp_h(const EvaluatedFilter &filter, const ImagePlane<const float> &src, const ImagePlane<float> &dst)
{
	const float *filter_data = filter.data();
	const int *filter_left = filter.left();
	ptrdiff_t filter_stride = filter.stride();

	int src_width = src.width();
	int src_height = src.height();

	for (ptrdiff_t i = 0; i < mod(src_height, 4); i += 4) {
		const float *src_p0 = src[i + 0];
		const float *src_p1 = src[i + 1];
		const float *src_p2 = src[i + 2];
		const float *src_p3 = src[i + 3];

		float *dst_p0 = dst[i + 0];
		float *dst_p1 = dst[i + 1];
		float *dst_p2 = dst[i + 2];
		float *dst_p3 = dst[i + 3];

		ptrdiff_t j;

		for (j = 0; j < filter.height(); ++j) {
			__m128 accum = _mm_setzero_ps();
			__m128 cached[4];

			const float *filter_row = &filter_data[j * filter_stride];
			ptrdiff_t left = filter_left[j];

			if (left + filter_stride > src_width)
				break;

			for (ptrdiff_t k = 0; k < (DoLoop ? filter.width() : 4); k += 4) {
				__m128 coeff = _mm_load_ps(filter_row + k);
				__m128 x0, x1, x2, x3;

				x0 = _mm_loadu_ps(&src_p0[left + k]);
				x0 = _mm_mul_ps(coeff, x0);

				x1 = _mm_loadu_ps(&src_p1[left + k]);
				x1 = _mm_mul_ps(coeff, x1);

				x2 = _mm_loadu_ps(&src_p2[left + k]);
				x2 = _mm_mul_ps(coeff, x2);

				x3 = _mm_loadu_ps(&src_p3[left + k]);
				x3 = _mm_mul_ps(coeff, x3);

				transpose4_ps(x0, x1, x2, x3);

				x0 = _mm_add_ps(x0, x2);
				x1 = _mm_add_ps(x1, x3);
				
				accum = _mm_add_ps(accum, x0);
				accum = _mm_add_ps(accum, x1);
			}
			cached[j % 4] = accum;

			if (j % 4 == 3) {
				ptrdiff_t dst_j = mod(j, 4);

				transpose4_ps(cached[0], cached[1], cached[2], cached[3]);

				_mm_store_ps(&dst_p0[dst_j], cached[0]);
				_mm_store_ps(&dst_p1[dst_j], cached[1]);
				_mm_store_ps(&dst_p2[dst_j], cached[2]);
				_mm_store_ps(&dst_p3[dst_j], cached[3]);
			}
		}
		filter_plane_h_scalar(filter, src, dst, i, i + 4, mod(j, 4), filter.height(), ScalarPolicy_F32{});
	}
	filter_plane_h_scalar(filter, src, dst, mod(src_height, 4), src_height, 0, filter.height(), ScalarPolicy_F32{});
}

void filter_plane_u16_v(const EvaluatedFilter &filter, const ImagePlane<const uint16_t> &src, const ImagePlane<uint16_t> &dst, uint16_t *tmp)
{
	__m128i INT16_MIN_EPI16 = _mm_set1_epi16(INT16_MIN);

	const int16_t *filter_data = filter.data_i16();
	const int *filter_left = filter.left();
	ptrdiff_t filter_stride = filter.stride_i16();

	int src_width = src.width();

	for (ptrdiff_t i = 0; i < filter.height(); ++i) {
		const int16_t *filter_row = &filter_data[i * filter_stride];
		int top = filter_left[i];
		uint16_t *dst_ptr = dst[i];

		for (ptrdiff_t k = 0; k < mod(filter.width(), 4); k += 4) {
			const uint16_t *src_ptr0 = src[top + k + 0];
			const uint16_t *src_ptr1 = src[top + k + 1];
			const uint16_t *src_ptr2 = src[top + k + 2];
			const uint16_t *src_ptr3 = src[top + k + 3];

			__m128i coeff0 = _mm_set1_epi16(filter_row[k + 0]);
			__m128i coeff1 = _mm_set1_epi16(filter_row[k + 1]);
			__m128i coeff2 = _mm_set1_epi16(filter_row[k + 2]);
			__m128i coeff3 = _mm_set1_epi16(filter_row[k + 3]);

			for (ptrdiff_t j = 0; j < mod(src_width, 8); j += 8) {
				__m128i x0, x1, x2, x3;
				__m128i packed;

				__m128i accum0l = _mm_setzero_si128();
				__m128i accum0h = _mm_setzero_si128();
				__m128i accum1l = _mm_setzero_si128();
				__m128i accum1h = _mm_setzero_si128();

				x0 = _mm_load_si128((const __m128i *)&src_ptr0[j]);
				x0 = _mm_add_epi16(x0, INT16_MIN_EPI16);
				fmadd_epi16_epi32(coeff0, x0, accum0l, accum0h);

				x1 = _mm_load_si128((const __m128i *)&src_ptr1[j]);
				x1 = _mm_add_epi16(x1, INT16_MIN_EPI16);
				fmadd_epi16_epi32(coeff1, x1, accum1l, accum1h);

				x2 = _mm_load_si128((const __m128i *)&src_ptr2[j]);
				x2 = _mm_add_epi16(x2, INT16_MIN_EPI16);
				fmadd_epi16_epi32(coeff2, x2, accum0l, accum0h);

				x3 = _mm_load_si128((const __m128i *)&src_ptr3[j]);
				x3 = _mm_add_epi16(x3, INT16_MIN_EPI16);
				fmadd_epi16_epi32(coeff3, x3, accum1l, accum1h);

				accum0l = _mm_add_epi32(accum0l, accum1l);
				accum0h = _mm_add_epi32(accum0h, accum1h);

				if (k) {
					accum0l = _mm_add_epi32(accum0l, _mm_load_si128((const __m128i *)&tmp[j * 2 + 0]));
					accum0h = _mm_add_epi32(accum0h, _mm_load_si128((const __m128i *)&tmp[j * 2 + 8]));
				}

				if (k == filter.width() - 4) {
					packed = pack_i30_epi32(accum0l, accum0h);
					packed = _mm_sub_epi16(packed, INT16_MIN_EPI16);
					_mm_store_si128((__m128i *)&dst_ptr[j], packed);
				} else {
					_mm_store_si128((__m128i *)&tmp[j * 2 + 0], accum0l);
					_mm_store_si128((__m128i *)&tmp[j * 2 + 8], accum0h);
				}
			}
		}
		if (filter.width() % 4) {
			ptrdiff_t m = filter.width() % 4;
			ptrdiff_t k = filter.width() - m;

			const uint16_t *src_ptr0 = src[top + k + 0];
			const uint16_t *src_ptr1 = src[top + k + 1];
			const uint16_t *src_ptr2 = src[top + k + 2];

			__m128i coeff0 = _mm_set1_epi16(filter_row[k + 0]);
			__m128i coeff1 = _mm_set1_epi16(filter_row[k + 1]);
			__m128i coeff2 = _mm_set1_epi16(filter_row[k + 2]);

			for (ptrdiff_t j = 0; j < mod(src_width, 8); j += 8) {
				__m128i x0, x1, x2;
				__m128i packed;

				__m128i accum0l = _mm_setzero_si128();
				__m128i accum0h = _mm_setzero_si128();
				__m128i accum1l = _mm_setzero_si128();
				__m128i accum1h = _mm_setzero_si128();

				switch (m) {
				case 3:
					x2 = _mm_load_si128((const __m128i *)&src_ptr2[j]);
					x2 = _mm_add_epi16(x2, INT16_MIN_EPI16);
					fmadd_epi16_epi32(coeff2, x2, accum0l, accum0h);
				case 2:
					x1 = _mm_load_si128((const __m128i *)&src_ptr1[j]);
					x1 = _mm_add_epi16(x1, INT16_MIN_EPI16);
					fmadd_epi16_epi32(coeff1, x1, accum1l, accum1h);
				case 1:
					x0 = _mm_load_si128((const __m128i *)&src_ptr0[j]);
					x0 = _mm_add_epi16(x0, INT16_MIN_EPI16);
					fmadd_epi16_epi32(coeff0, x0, accum0l, accum0h);
				}

				accum0l = _mm_add_epi32(accum0l, accum1l);
				accum0h = _mm_add_epi32(accum0h, accum1h);

				if (k) {
					accum0l = _mm_add_epi32(accum0l, _mm_load_si128((const __m128i *)&tmp[j * 2 + 0]));
					accum0h = _mm_add_epi32(accum0h, _mm_load_si128((const __m128i *)&tmp[j * 2 + 8]));
				}

				packed = pack_i30_epi32(accum0l, accum0h);
				packed = _mm_sub_epi16(packed, INT16_MIN_EPI16);

				_mm_store_si128((__m128i *)&dst_ptr[j], packed);
			}
		}
		filter_plane_v_scalar(filter, src, dst, i, i + 1, mod(src_width, 8), src_width, ScalarPolicy_U16{});
	}
}

void filter_plane_fp_v(const EvaluatedFilter &filter, const ImagePlane<const float> &src, const ImagePlane<float> &dst)
{
	const float *filter_data = filter.data();
	const int *filter_left = filter.left();
	ptrdiff_t filter_stride = filter.stride();

	int src_width = src.width();

	for (ptrdiff_t i = 0; i < filter.height(); ++i) {
		const float *filter_row = &filter_data[i * filter_stride];
		int top = filter_left[i];
		float *dst_ptr = dst[i];

		for (ptrdiff_t k = 0; k < mod(filter.width(), 4); k += 4) {
			const float *src_ptr0 = src[top + k + 0];
			const float *src_ptr1 = src[top + k + 1];
			const float *src_ptr2 = src[top + k + 2];
			const float *src_ptr3 = src[top + k + 3];

			__m128 coeff0 = _mm_set_ps1(filter_row[k + 0]);
			__m128 coeff1 = _mm_set_ps1(filter_row[k + 1]);
			__m128 coeff2 = _mm_set_ps1(filter_row[k + 2]);
			__m128 coeff3 = _mm_set_ps1(filter_row[k + 3]);

			for (ptrdiff_t j = 0; j < mod(src_width, 4); j += 4) {
				__m128 x0, x1, x2, x3;
				__m128 accum0, accum1;

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
					accum0 = _mm_add_ps(accum0, _mm_load_ps(&dst_ptr[j]));

				_mm_store_ps(&dst_ptr[j], accum0);
			}
		}
		if (filter.width() % 4) {
			ptrdiff_t m = filter.width() % 4;
			ptrdiff_t k = filter.width() - m;

			const float *src_ptr0 = src[top + k + 0];
			const float *src_ptr1 = src[top + k + 1];
			const float *src_ptr2 = src[top + k + 2];

			__m128 coeff0 = _mm_set_ps1(filter_row[k + 0]);
			__m128 coeff1 = _mm_set_ps1(filter_row[k + 1]);
			__m128 coeff2 = _mm_set_ps1(filter_row[k + 2]);

			for (ptrdiff_t j = 0; j < mod(src_width, 4); j += 4) {
				__m128 x0, x1, x2;

				__m128 accum0 = _mm_setzero_ps();
				__m128 accum1 = _mm_setzero_ps();

				switch (m) {
				case 3:
					x2 = _mm_load_ps(&src_ptr2[j]);
					accum0 = _mm_mul_ps(coeff2, x2);
				case 2:
					x1 = _mm_load_ps(&src_ptr1[j]);
					accum1 = _mm_mul_ps(coeff1, x1);
				case 1:
					x0 = _mm_load_ps(&src_ptr0[j]);
					x0 = _mm_mul_ps(coeff0, x0);
					accum0 = _mm_add_ps(accum0, x0);
				}

				accum0 = _mm_add_ps(accum0, accum1);

				if (k)
					accum0 = _mm_add_ps(accum0, _mm_load_ps(&dst_ptr[j]));

				_mm_store_ps(&dst_ptr[j], accum0);
			}
		}
		filter_plane_v_scalar(filter, src, dst, i, i + 1, mod(src_width, 4), src_width, ScalarPolicy_F32{});
	}
}

class ResizeImplSSE2 : public ResizeImpl {
public:
	ResizeImplSSE2(const EvaluatedFilter &filter_h, const EvaluatedFilter &filter_v) : ResizeImpl(filter_h, filter_v)
	{}

	void process_u16_h(const ImagePlane<const uint16_t> &src, const ImagePlane<uint16_t> &dst, uint16_t *tmp) const override
	{
		if (m_filter_h.width() > 8)
			filter_plane_u16_h<true>(m_filter_h, src, dst);
		else
			filter_plane_u16_h<false>(m_filter_h, src, dst);
	}

	void process_u16_v(const ImagePlane<const uint16_t> &src, const ImagePlane<uint16_t> &dst, uint16_t *tmp) const override
	{
		filter_plane_u16_v(m_filter_v, src, dst, tmp);
	}

	void process_f16_h(const ImagePlane<const uint16_t> &src, const ImagePlane<uint16_t> &dst, uint16_t *tmp) const override
	{
		throw ZimgUnsupportedError{ "f16 not supported in SSE2 impl" };
	}

	void process_f16_v(const ImagePlane<const uint16_t> &src, const ImagePlane<uint16_t> &dst, uint16_t *tmp) const override
	{
		throw ZimgUnsupportedError{ "f16 not supported in SSE2 impl" };
	}

	void process_f32_h(const ImagePlane<const float> &src, const ImagePlane<float> &dst, float *tmp) const override
	{
		if (m_filter_h.width() > 4)
			filter_plane_fp_h<true>(m_filter_h, src, dst);
		else
			filter_plane_fp_h<false>(m_filter_h, src, dst);
	}

	void process_f32_v(const ImagePlane<const float> &src, const ImagePlane<float> &dst, float *tmp) const override
	{
		filter_plane_fp_v(m_filter_v, src, dst);
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
