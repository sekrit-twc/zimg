#ifdef ZIMG_X86

#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include "Common/align.h"
#include "Common/osdep.h"
#include "Common/plane.h"
#include "filter.h"
#include "resize_impl.h"
#include "resize_impl_x86.h"

namespace zimg {;
namespace resize {;

namespace {;

struct ScalarPolicy_F16 {
	typedef float num_type;

	FORCE_INLINE float coeff(const EvaluatedFilter &filter, ptrdiff_t row, ptrdiff_t k)
	{
		return filter.data()[row * filter.stride() + k];
	}

	FORCE_INLINE float load(const uint16_t *src) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16(*src))); }

	FORCE_INLINE void store(uint16_t *dst, float x) { *dst = _mm_extract_epi16(_mm_cvtps_ph(_mm_set_ps1(x), 0), 0); }
};

struct VectorPolicy_F16 : public ScalarPolicy_F16 {
	FORCE_INLINE __m256 load_8(const uint16_t *p) { return _mm256_cvtph_ps(_mm_load_si128((const __m128i *)p)); }
	FORCE_INLINE __m256 loadu_8(const uint16_t *p) { return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)p)); }

	FORCE_INLINE void store_8(uint16_t *p, __m256 x) { _mm_store_si128((__m128i *)p, _mm256_cvtps_ph(x, 0)); }
};

struct VectorPolicy_F32 : public ScalarPolicy_F32 {
	FORCE_INLINE __m256 load_8(const float *p) { return _mm256_load_ps(p); }
	FORCE_INLINE __m256 loadu_8(const float *p) { return _mm256_loadu_ps(p); }

	FORCE_INLINE void store_8(float *p, __m256 x) { _mm256_store_ps(p, x); }
};

inline FORCE_INLINE void transpose8_ps(__m256 &row0, __m256 &row1, __m256 &row2, __m256 &row3, __m256 &row4, __m256 &row5, __m256 &row6, __m256 &row7)
{
	__m256 t0, t1, t2, t3, t4, t5, t6, t7;
	__m256 tt0, tt1, tt2, tt3, tt4, tt5, tt6, tt7;

	t0 = _mm256_unpacklo_ps(row0, row1);
	t1 = _mm256_unpackhi_ps(row0, row1);
	t2 = _mm256_unpacklo_ps(row2, row3);
	t3 = _mm256_unpackhi_ps(row2, row3);
	t4 = _mm256_unpacklo_ps(row4, row5);
	t5 = _mm256_unpackhi_ps(row4, row5);
	t6 = _mm256_unpacklo_ps(row6, row7);
	t7 = _mm256_unpackhi_ps(row6, row7);

	tt0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
	tt1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
	tt2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
	tt3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
	tt4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
	tt5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
	tt6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
	tt7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));

	row0 = _mm256_permute2f128_ps(tt0, tt4, 0x20);
	row1 = _mm256_permute2f128_ps(tt1, tt5, 0x20);
	row2 = _mm256_permute2f128_ps(tt2, tt6, 0x20);
	row3 = _mm256_permute2f128_ps(tt3, tt7, 0x20);
	row4 = _mm256_permute2f128_ps(tt0, tt4, 0x31);
	row5 = _mm256_permute2f128_ps(tt1, tt5, 0x31);
	row6 = _mm256_permute2f128_ps(tt2, tt6, 0x31);
	row7 = _mm256_permute2f128_ps(tt3, tt7, 0x31);
}

inline FORCE_INLINE void transpose8_epi32(__m256i &row0, __m256i &row1, __m256i &row2, __m256i &row3, __m256i &row4, __m256i &row5, __m256i &row6, __m256i &row7)
{
	__m256 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;

	tmp0 = _mm256_castsi256_ps(row0);
	tmp1 = _mm256_castsi256_ps(row1);
	tmp2 = _mm256_castsi256_ps(row2);
	tmp3 = _mm256_castsi256_ps(row3);
	tmp4 = _mm256_castsi256_ps(row4);
	tmp5 = _mm256_castsi256_ps(row5);
	tmp6 = _mm256_castsi256_ps(row6);
	tmp7 = _mm256_castsi256_ps(row7);

	transpose8_ps(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7);

	row0 = _mm256_castps_si256(tmp0);
	row1 = _mm256_castps_si256(tmp1);
	row2 = _mm256_castps_si256(tmp2);
	row3 = _mm256_castps_si256(tmp3);
	row4 = _mm256_castps_si256(tmp4);
	row5 = _mm256_castps_si256(tmp5);
	row6 = _mm256_castps_si256(tmp6);
	row7 = _mm256_castps_si256(tmp7);
}

// Performs accum[0,1] += x * unpack[lo,hi](a,b)
inline FORCE_INLINE void fmadd_epi16_2(__m256i x, __m256i a, __m256i b, __m256i &accum0, __m256i &accum1)
{
	__m256i tmplo, tmphi;

	tmplo = _mm256_unpacklo_epi16(a, b);
	tmphi = _mm256_unpackhi_epi16(a, b);
	tmplo = _mm256_madd_epi16(x, tmplo);
	tmphi = _mm256_madd_epi16(x, tmphi);
	accum0 = _mm256_add_epi32(accum0, tmplo);
	accum1 = _mm256_add_epi32(accum1, tmphi);
}

inline FORCE_INLINE __m256i pack_i30_epi32(__m256i lo, __m256i hi)
{
	__m256i offset = _mm256_set1_epi32(1 << 13);

	lo = _mm256_add_epi32(lo, offset);
	hi = _mm256_add_epi32(hi, offset);

	lo = _mm256_srai_epi32(lo, 14);
	hi = _mm256_srai_epi32(hi, 14);

	return  _mm256_packs_epi32(lo, hi);
}

template <bool DoLoop>
void filter_plane_u16_h(const EvaluatedFilter &filter, const ImagePlane<const uint16_t> &src, const ImagePlane<uint16_t> &dst)
{
	__m256i INT16_MIN_EPI16 = _mm256_set1_epi16(INT16_MIN);

	const int16_t *filter_data = filter.data_i16();
	const int *filter_left = filter.left();
	ptrdiff_t filter_stride = filter.stride_i16();

	int src_width = src.width();
	int src_height = src.height();

	for (ptrdiff_t i = 0; i < mod(src_height, 8); i += 8) {
		const uint16_t *src_ptr0 = src[i + 0];
		const uint16_t *src_ptr1 = src[i + 1];
		const uint16_t *src_ptr2 = src[i + 2];
		const uint16_t *src_ptr3 = src[i + 3];
		const uint16_t *src_ptr4 = src[i + 4];
		const uint16_t *src_ptr5 = src[i + 5];
		const uint16_t *src_ptr6 = src[i + 6];
		const uint16_t *src_ptr7 = src[i + 7];

		uint16_t *dst_ptr0 = dst[i + 0];
		uint16_t *dst_ptr1 = dst[i + 1];
		uint16_t *dst_ptr2 = dst[i + 2]; 
		uint16_t *dst_ptr3 = dst[i + 3];
		uint16_t *dst_ptr4 = dst[i + 4];
		uint16_t *dst_ptr5 = dst[i + 5];
		uint16_t *dst_ptr6 = dst[i + 6];
		uint16_t *dst_ptr7 = dst[i + 7];

		ptrdiff_t j;

		for (j = 0; j < filter.height(); ++j) {
			__m256i accum = _mm256_setzero_si256();
			__m256i cached[16];

			const int16_t *filter_row = &filter_data[j * filter_stride];
			ptrdiff_t left = filter_left[j];

			if (left + filter_stride > src_width)
				break;

			for (ptrdiff_t k = 0; k < (DoLoop ? filter.width() : 16); k += 16) {
				__m256i coeff = _mm256_load_si256((const __m256i *)&filter_row[k]);
				__m256i x0, x1, x2, x3, x4, x5, x6, x7;

				x0 = _mm256_loadu_si256((const __m256i *)&src_ptr0[left + k]);
				x0 = _mm256_add_epi16(x0, INT16_MIN_EPI16);
				x0 = _mm256_madd_epi16(coeff, x0);

				x1 = _mm256_loadu_si256((const __m256i *)&src_ptr1[left + k]);
				x1 = _mm256_add_epi16(x1, INT16_MIN_EPI16);
				x1 = _mm256_madd_epi16(coeff, x1);

				x2 = _mm256_loadu_si256((const __m256i *)&src_ptr2[left + k]);
				x2 = _mm256_add_epi16(x2, INT16_MIN_EPI16);
				x2 = _mm256_madd_epi16(coeff, x2);

				x3 = _mm256_loadu_si256((const __m256i *)&src_ptr3[left + k]);
				x3 = _mm256_add_epi16(x3, INT16_MIN_EPI16);
				x3 = _mm256_madd_epi16(coeff, x3);

				x4 = _mm256_loadu_si256((const __m256i *)&src_ptr4[left + k]);
				x4 = _mm256_add_epi16(x4, INT16_MIN_EPI16);
				x4 = _mm256_madd_epi16(coeff, x4);

				x5 = _mm256_loadu_si256((const __m256i *)&src_ptr5[left + k]);
				x5 = _mm256_add_epi16(x5, INT16_MIN_EPI16);
				x5 = _mm256_madd_epi16(coeff, x5);

				x6 = _mm256_loadu_si256((const __m256i *)&src_ptr6[left + k]);
				x6 = _mm256_add_epi16(x6, INT16_MIN_EPI16);
				x6 = _mm256_madd_epi16(coeff, x6);

				x7 = _mm256_loadu_si256((const __m256i *)&src_ptr7[left + k]);
				x7 = _mm256_add_epi16(x7, INT16_MIN_EPI16);
				x7 = _mm256_madd_epi16(coeff, x7);

				transpose8_epi32(x0, x1, x2, x3, x4, x5, x6, x7);

				x0 = _mm256_add_epi32(x0, x4);
				x1 = _mm256_add_epi32(x1, x5);
				x2 = _mm256_add_epi32(x2, x6);
				x3 = _mm256_add_epi32(x3, x7);

				x0 = _mm256_add_epi32(x0, x2);
				x1 = _mm256_add_epi32(x1, x3);

				accum = _mm256_add_epi32(accum, x0);
				accum = _mm256_add_epi32(accum, x1);
			}
			cached[j % 16] = accum;

			if (j % 16 == 15) {
				ptrdiff_t dst_j = mod(j, 16);
				__m256i packed;

				transpose8_epi32(cached[0], cached[1], cached[2], cached[3], cached[8], cached[9], cached[10], cached[11]);
				transpose8_epi32(cached[4], cached[5], cached[6], cached[7], cached[12], cached[13], cached[14], cached[15]);

				packed = pack_i30_epi32(cached[0], cached[4]);
				packed = _mm256_sub_epi16(packed, INT16_MIN_EPI16);
				_mm256_store_si256((__m256i *)&dst_ptr0[dst_j], packed);

				packed = pack_i30_epi32(cached[1], cached[5]);
				packed = _mm256_sub_epi16(packed, INT16_MIN_EPI16);
				_mm256_store_si256((__m256i *)&dst_ptr1[dst_j], packed);

				packed = pack_i30_epi32(cached[2], cached[6]);
				packed = _mm256_sub_epi16(packed, INT16_MIN_EPI16);
				_mm256_store_si256((__m256i *)&dst_ptr2[dst_j], packed);

				packed = pack_i30_epi32(cached[3], cached[7]);
				packed = _mm256_sub_epi16(packed, INT16_MIN_EPI16);
				_mm256_store_si256((__m256i *)&dst_ptr3[dst_j], packed);

				packed = pack_i30_epi32(cached[8], cached[12]);
				packed = _mm256_sub_epi16(packed, INT16_MIN_EPI16);
				_mm256_store_si256((__m256i *)&dst_ptr4[dst_j], packed);

				packed = pack_i30_epi32(cached[9], cached[13]);
				packed = _mm256_sub_epi16(packed, INT16_MIN_EPI16);
				_mm256_store_si256((__m256i *)&dst_ptr5[dst_j], packed);

				packed = pack_i30_epi32(cached[10], cached[14]);
				packed = _mm256_sub_epi16(packed, INT16_MIN_EPI16);
				_mm256_store_si256((__m256i *)&dst_ptr6[dst_j], packed);

				packed = pack_i30_epi32(cached[11], cached[15]);
				packed = _mm256_sub_epi16(packed, INT16_MIN_EPI16);
				_mm256_store_si256((__m256i *)&dst_ptr7[dst_j], packed);
			}
		}
		filter_plane_h_scalar(filter, src, dst, i, i + 8, mod(j, 16), filter.height(), ScalarPolicy_U16{});
	}
	filter_plane_h_scalar(filter, src, dst, mod(src_height, 8), src_height, 0, filter.height(), ScalarPolicy_U16{});
}

template <bool DoLoop, class T, class Policy>
void filter_plane_fp_h(const EvaluatedFilter &filter, const ImagePlane<const T> &src, const ImagePlane<T> &dst, Policy policy)
{
	const float *filter_data = filter.data();
	const int *filter_left = filter.left();
	ptrdiff_t filter_stride = filter.stride();

	int src_width = src.width();
	int src_height = src.height();

	for (ptrdiff_t i = 0; i < mod(src_height, 8); i += 8) {
		const T *src_ptr0 = src[i + 0];
		const T *src_ptr1 = src[i + 1];
		const T *src_ptr2 = src[i + 2];
		const T *src_ptr3 = src[i + 3];
		const T *src_ptr4 = src[i + 4];
		const T *src_ptr5 = src[i + 5];
		const T *src_ptr6 = src[i + 6];
		const T *src_ptr7 = src[i + 7];

		T *dst_ptr0 = dst[i + 0];
		T *dst_ptr1 = dst[i + 1];
		T *dst_ptr2 = dst[i + 2];
		T *dst_ptr3 = dst[i + 3];
		T *dst_ptr4 = dst[i + 4];
		T *dst_ptr5 = dst[i + 5];
		T *dst_ptr6 = dst[i + 6];
		T *dst_ptr7 = dst[i + 7];

		ptrdiff_t j;

		for (j = 0; j < filter.height(); ++j) {
			__m256 accum = _mm256_setzero_ps();
			__m256 cached[8];

			const float *filter_row = &filter_data[j * filter_stride];
			ptrdiff_t left = filter_left[j];

			if (left + filter_stride > src_width)
				break;

			for (ptrdiff_t k = 0; k < (DoLoop ? filter.width() : 8); k += 8) {
				__m256 coeff = _mm256_load_ps(filter_row + k);
				__m256 x0, x1, x2, x3, x4, x5, x6, x7;

				x0 = policy.loadu_8(&src_ptr0[left + k]);
				x0 = _mm256_mul_ps(coeff, x0);
				
				x1 = policy.loadu_8(&src_ptr1[left + k]);
				x1 = _mm256_mul_ps(coeff, x1);

				x2 = policy.loadu_8(&src_ptr2[left + k]);
				x2 = _mm256_mul_ps(coeff, x2);

				x3 = policy.loadu_8(&src_ptr3[left + k]);
				x3 = _mm256_mul_ps(coeff, x3);

				x4 = policy.loadu_8(&src_ptr4[left + k]);
				x4 = _mm256_mul_ps(coeff, x4);

				x5 = policy.loadu_8(&src_ptr5[left + k]);
				x5 = _mm256_mul_ps(coeff, x5);

				x6 = policy.loadu_8(&src_ptr6[left + k]);
				x6 = _mm256_mul_ps(coeff, x6);

				x7 = policy.loadu_8(&src_ptr7[left + k]);
				x7 = _mm256_mul_ps(coeff, x7);

				transpose8_ps(x0, x1, x2, x3, x4, x5, x6, x7);

				x0 = _mm256_add_ps(x0, x4);
				x1 = _mm256_add_ps(x1, x5);
				x2 = _mm256_add_ps(x2, x6);
				x3 = _mm256_add_ps(x3, x7);

				x0 = _mm256_add_ps(x0, x2);
				x1 = _mm256_add_ps(x1, x3);

				accum = _mm256_add_ps(accum, x0);
				accum = _mm256_add_ps(accum, x1);
			}
			cached[j % 8] = accum;

			if (j % 8 == 7) {
				ptrdiff_t dst_j = mod(j, 8);

				transpose8_ps(cached[0], cached[1], cached[2], cached[3], cached[4], cached[5], cached[6], cached[7]);

				policy.store_8(&dst_ptr0[dst_j], cached[0]);
				policy.store_8(&dst_ptr1[dst_j], cached[1]);
				policy.store_8(&dst_ptr2[dst_j], cached[2]);
				policy.store_8(&dst_ptr3[dst_j], cached[3]);
				policy.store_8(&dst_ptr4[dst_j], cached[4]);
				policy.store_8(&dst_ptr5[dst_j], cached[5]);
				policy.store_8(&dst_ptr6[dst_j], cached[6]);
				policy.store_8(&dst_ptr7[dst_j], cached[7]);
			}
		}
		filter_plane_h_scalar(filter, src, dst, i, i + 8, mod(j, 8), filter.height(), policy);
	}
	filter_plane_h_scalar(filter, src, dst, mod(src_height, 8), src_height, 0, filter.height(), policy);
}

void filter_plane_u16_v(const EvaluatedFilter &filter, const ImagePlane<const uint16_t> &src, const ImagePlane<uint16_t> &dst, uint16_t *tmp)
{
	__m256i INT16_MIN_EPI16 = _mm256_set1_epi16(INT16_MIN);

	const int16_t *filter_data = filter.data_i16();
	const int *filter_left = filter.left();
	ptrdiff_t filter_stride = filter.stride_i16();

	int src_width = src.width();

	for (ptrdiff_t i = 0; i < filter.height(); ++i) {
		const int16_t *filter_row = &filter_data[i * filter_stride];
		int top = filter_left[i];
		uint16_t *dst_ptr = dst[i];

		for (ptrdiff_t k = 0; k < mod(filter.width(), 8); k += 8) {
			const uint16_t *src_ptr0 = src[top + k + 0];
			const uint16_t *src_ptr1 = src[top + k + 1];
			const uint16_t *src_ptr2 = src[top + k + 2];
			const uint16_t *src_ptr3 = src[top + k + 3];
			const uint16_t *src_ptr4 = src[top + k + 4];
			const uint16_t *src_ptr5 = src[top + k + 5];
			const uint16_t *src_ptr6 = src[top + k + 6];
			const uint16_t *src_ptr7 = src[top + k + 7];

			__m256i coeff0 = _mm256_set1_epi16(filter_row[k + 0]);
			__m256i coeff1 = _mm256_set1_epi16(filter_row[k + 1]);
			__m256i coeff2 = _mm256_set1_epi16(filter_row[k + 2]);
			__m256i coeff3 = _mm256_set1_epi16(filter_row[k + 3]);
			__m256i coeff4 = _mm256_set1_epi16(filter_row[k + 4]);
			__m256i coeff5 = _mm256_set1_epi16(filter_row[k + 5]);
			__m256i coeff6 = _mm256_set1_epi16(filter_row[k + 6]);
			__m256i coeff7 = _mm256_set1_epi16(filter_row[k + 7]);

			__m256i coeff01 = _mm256_unpacklo_epi16(coeff0, coeff1);
			__m256i coeff23 = _mm256_unpacklo_epi16(coeff2, coeff3);
			__m256i coeff45 = _mm256_unpacklo_epi16(coeff4, coeff5);
			__m256i coeff67 = _mm256_unpacklo_epi16(coeff6, coeff7);

			for (ptrdiff_t j = 0; j < mod(src_width, 16); j += 16) {
				__m256i x0, x1, x2, x3, x4, x5, x6, x7;
				__m256i packed;

				__m256i accum0l = _mm256_setzero_si256();
				__m256i accum0h = _mm256_setzero_si256();
				__m256i accum1l = _mm256_setzero_si256();
				__m256i accum1h = _mm256_setzero_si256();

				x0 = _mm256_load_si256((const __m256i *)&src_ptr0[j]);
				x0 = _mm256_add_epi16(x0, INT16_MIN_EPI16);
				x1 = _mm256_load_si256((const __m256i *)&src_ptr1[j]);
				x1 = _mm256_add_epi16(x1, INT16_MIN_EPI16);
				fmadd_epi16_2(coeff01, x0, x1, accum0l, accum0h);

				x2 = _mm256_load_si256((const __m256i *)&src_ptr2[j]);
				x2 = _mm256_add_epi16(x2, INT16_MIN_EPI16);
				x3 = _mm256_load_si256((const __m256i *)&src_ptr3[j]);
				x3 = _mm256_add_epi16(x3, INT16_MIN_EPI16);
				fmadd_epi16_2(coeff23, x2, x3, accum1l, accum1h);

				x4 = _mm256_load_si256((const __m256i *)&src_ptr4[j]);
				x4 = _mm256_add_epi16(x4, INT16_MIN_EPI16);
				x5 = _mm256_load_si256((const __m256i *)&src_ptr5[j]);
				x5 = _mm256_add_epi16(x5, INT16_MIN_EPI16);
				fmadd_epi16_2(coeff45, x4, x5, accum0l, accum0h);

				x6 = _mm256_load_si256((const __m256i *)&src_ptr6[j]);
				x6 = _mm256_add_epi16(x6, INT16_MIN_EPI16);
				x7 = _mm256_load_si256((const __m256i *)&src_ptr7[j]);
				x7 = _mm256_add_epi16(x7, INT16_MIN_EPI16);
				fmadd_epi16_2(coeff67, x6, x7, accum1l, accum1h);

				accum0l = _mm256_add_epi32(accum0l, accum1l);
				accum0h = _mm256_add_epi32(accum0h, accum1h);

				if (k) {
					accum0l = _mm256_add_epi32(accum0l, _mm256_load_si256((const __m256i *)&tmp[j * 2 + 0]));
					accum0h = _mm256_add_epi32(accum0h, _mm256_load_si256((const __m256i *)&tmp[j * 2 + 16]));
				}

				if (k == filter.width() - 8) {					
					packed = pack_i30_epi32(accum0l, accum0h);
					packed = _mm256_sub_epi16(packed, INT16_MIN_EPI16);
					_mm256_store_si256((__m256i *)&dst_ptr[j], packed);
				} else {
					_mm256_store_si256((__m256i *)&tmp[j * 2 + 0], accum0l);
					_mm256_store_si256((__m256i *)&tmp[j * 2 + 16], accum0h);
				}
			}
		}
		if (filter.width() % 8) {
			ptrdiff_t m = filter.width() % 8;
			ptrdiff_t k = filter.width() - m;

			const uint16_t *src_ptr0 = src[top + k + 0];
			const uint16_t *src_ptr1 = src[top + k + 1];
			const uint16_t *src_ptr2 = src[top + k + 2];
			const uint16_t *src_ptr3 = src[top + k + 3];
			const uint16_t *src_ptr4 = src[top + k + 4];
			const uint16_t *src_ptr5 = src[top + k + 5];
			const uint16_t *src_ptr6 = src[top + k + 6];

			__m256i coeff0 = _mm256_set1_epi16(filter_row[k + 0]);
			__m256i coeff1 = _mm256_set1_epi16(filter_row[k + 1]);
			__m256i coeff2 = _mm256_set1_epi16(filter_row[k + 2]);
			__m256i coeff3 = _mm256_set1_epi16(filter_row[k + 3]);
			__m256i coeff4 = _mm256_set1_epi16(filter_row[k + 4]);
			__m256i coeff5 = _mm256_set1_epi16(filter_row[k + 5]);
			__m256i coeff6 = _mm256_set1_epi16(filter_row[k + 6]);
			__m256i coeff7 = _mm256_setzero_si256();

			__m256i coeff01 = _mm256_unpacklo_epi16(coeff0, coeff1);
			__m256i coeff23 = _mm256_unpacklo_epi16(coeff2, coeff3);
			__m256i coeff45 = _mm256_unpacklo_epi16(coeff4, coeff5);
			__m256i coeff67 = _mm256_unpacklo_epi16(coeff6, coeff7);

			for (ptrdiff_t j = 0; j < mod(src_width, 16); j += 16) {
				__m256i x0, x1, x2, x3, x4, x5, x6, x7;
				__m256i packed;

				__m256i accum0l = _mm256_setzero_si256();
				__m256i accum0h = _mm256_setzero_si256();
				__m256i accum1l = _mm256_setzero_si256();
				__m256i accum1h = _mm256_setzero_si256();

				switch (m) {
				case 7:
					x7 = INT16_MIN_EPI16;
					x6 = _mm256_load_si256((const __m256i *)&src_ptr6[j]);
					x6 = _mm256_add_epi16(x6, INT16_MIN_EPI16);
					fmadd_epi16_2(coeff67, x6, x7, accum1l, accum1h);
				case 6:
					x5 = _mm256_load_si256((const __m256i *)&src_ptr5[j]);
					x5 = _mm256_add_epi16(x5, INT16_MIN_EPI16);
				case 5:
					x4 = _mm256_load_si256((const __m256i *)&src_ptr4[j]);
					x4 = _mm256_add_epi16(x4, INT16_MIN_EPI16);
					fmadd_epi16_2(coeff45, x4, x5, accum0l, accum0h);
				case 4:
					x3 = _mm256_load_si256((const __m256i *)&src_ptr3[j]);
					x3 = _mm256_add_epi16(x3, INT16_MIN_EPI16);
				case 3:
					x2 = _mm256_load_si256((const __m256i *)&src_ptr2[j]);
					x2 = _mm256_add_epi16(x2, INT16_MIN_EPI16);
					fmadd_epi16_2(coeff23, x2, x3, accum1l, accum1h);
				case 2:
					x1 = _mm256_load_si256((const __m256i *)&src_ptr1[j]);
					x1 = _mm256_add_epi16(x1, INT16_MIN_EPI16);
				case 1:
					x0 = _mm256_load_si256((const __m256i *)&src_ptr0[j]);
					x0 = _mm256_add_epi16(x0, INT16_MIN_EPI16);
					fmadd_epi16_2(coeff01, x0, x1, accum0l, accum0h);
				}

				accum0l = _mm256_add_epi32(accum0l, accum1l);
				accum0h = _mm256_add_epi32(accum0h, accum1h);

				if (k) {
					accum0l = _mm256_add_epi32(accum0l, _mm256_load_si256((const __m256i *)&tmp[j * 2 + 0]));
					accum0h = _mm256_add_epi32(accum0h, _mm256_load_si256((const __m256i *)&tmp[j * 2 + 16]));
				}

				packed = pack_i30_epi32(accum0l, accum0h);
				packed = _mm256_sub_epi16(packed, INT16_MIN_EPI16);

				_mm256_store_si256((__m256i *)&dst_ptr[j], packed);
			}
		}
		filter_plane_v_scalar(filter, src, dst, i, i + 1, mod(src_width, 16), src_width, ScalarPolicy_U16{});
	}
}

template <class T, class Policy>
void filter_plane_fp_v(const EvaluatedFilter &filter, const ImagePlane<const T> &src, const ImagePlane<T> &dst, Policy policy)
{
	const float *filter_data = filter.data();
	const int *filter_left = filter.left();
	ptrdiff_t filter_stride = filter.stride();

	int src_width = src.width();

	for (ptrdiff_t i = 0; i < filter.height(); ++i) {
		const float *filter_row = &filter_data[i * filter_stride];
		int top = filter_left[i];
		T *dst_ptr = dst[i];

		for (ptrdiff_t k = 0; k < mod(filter.width(), 8); k += 8) {
			const T *src_ptr0 = src[top + k + 0];
			const T *src_ptr1 = src[top + k + 1];
			const T *src_ptr2 = src[top + k + 2];
			const T *src_ptr3 = src[top + k + 3];
			const T *src_ptr4 = src[top + k + 4];
			const T *src_ptr5 = src[top + k + 5];
			const T *src_ptr6 = src[top + k + 6];
			const T *src_ptr7 = src[top + k + 7];

			__m256 coeff0 = _mm256_broadcast_ss(&filter_row[k + 0]);
			__m256 coeff1 = _mm256_broadcast_ss(&filter_row[k + 1]);
			__m256 coeff2 = _mm256_broadcast_ss(&filter_row[k + 2]);
			__m256 coeff3 = _mm256_broadcast_ss(&filter_row[k + 3]);
			__m256 coeff4 = _mm256_broadcast_ss(&filter_row[k + 4]);
			__m256 coeff5 = _mm256_broadcast_ss(&filter_row[k + 5]);
			__m256 coeff6 = _mm256_broadcast_ss(&filter_row[k + 6]);
			__m256 coeff7 = _mm256_broadcast_ss(&filter_row[k + 7]);

			for (ptrdiff_t j = 0; j < mod(src_width, 8); j += 8) {
				__m256 x0, x1, x2, x3, x4, x5, x6, x7;
				__m256 accum0, accum1, accum2, accum3;

				x0 = policy.load_8(&src_ptr0[j]);
				accum0 = _mm256_mul_ps(coeff0, x0);

				x1 = policy.load_8(&src_ptr1[j]);
				accum1 = _mm256_mul_ps(coeff1, x1);

				x2 = policy.load_8(&src_ptr2[j]);
				accum2 = _mm256_mul_ps(coeff2, x2);

				x3 = policy.load_8(&src_ptr3[j]);
				accum3 = _mm256_mul_ps(coeff3, x3);

				x4 = policy.load_8(&src_ptr4[j]);
				accum0 = _mm256_fmadd_ps(coeff4, x4, accum0);

				x5 = policy.load_8(&src_ptr5[j]);
				accum1 = _mm256_fmadd_ps(coeff5, x5, accum1);

				x6 = policy.load_8(&src_ptr6[j]);
				accum2 = _mm256_fmadd_ps(coeff6, x6, accum2);

				x7 = policy.load_8(&src_ptr7[j]);
				accum3 = _mm256_fmadd_ps(coeff7, x7, accum3);

				accum0 = _mm256_add_ps(accum0, accum2);
				accum1 = _mm256_add_ps(accum1, accum3);
				accum0 = _mm256_add_ps(accum0, accum1);

				if (k)
					accum0 = _mm256_add_ps(accum0, policy.load_8(&dst_ptr[j]));

				policy.store_8(&dst_ptr[j], accum0);
			}
		}
		if (filter.width() % 8) {
			ptrdiff_t m = filter.width() % 8;
			ptrdiff_t k = filter.width() - m;

			const T *src_ptr0 = src[top + k + 0];
			const T *src_ptr1 = src[top + k + 1];
			const T *src_ptr2 = src[top + k + 2];
			const T *src_ptr3 = src[top + k + 3];
			const T *src_ptr4 = src[top + k + 4];
			const T *src_ptr5 = src[top + k + 5];
			const T *src_ptr6 = src[top + k + 6];

			__m256 coeff0 = _mm256_broadcast_ss(&filter_row[k + 0]);
			__m256 coeff1 = _mm256_broadcast_ss(&filter_row[k + 1]);
			__m256 coeff2 = _mm256_broadcast_ss(&filter_row[k + 2]);
			__m256 coeff3 = _mm256_broadcast_ss(&filter_row[k + 3]);
			__m256 coeff4 = _mm256_broadcast_ss(&filter_row[k + 4]);
			__m256 coeff5 = _mm256_broadcast_ss(&filter_row[k + 5]);
			__m256 coeff6 = _mm256_broadcast_ss(&filter_row[k + 6]);

			for (ptrdiff_t j = 0; j < mod(src_width, 8); j += 8) {
				__m256 x0, x1, x2, x3, x4, x5, x6;

				__m256 accum0 = _mm256_setzero_ps();
				__m256 accum1 = _mm256_setzero_ps();
				__m256 accum2 = _mm256_setzero_ps();
				__m256 accum3 = _mm256_setzero_ps();

				switch (m) {
				case 7:
					x6 = policy.load_8(&src_ptr6[j]);
					accum2 = _mm256_mul_ps(coeff6, x6);
				case 6:
					x5 = policy.load_8(&src_ptr5[j]);
					accum1 = _mm256_mul_ps(coeff5, x5);
				case 5:
					x4 = policy.load_8(&src_ptr4[j]);
					accum0 = _mm256_mul_ps(coeff4, x4);
				case 4:
					x3 = policy.load_8(&src_ptr3[j]);
					accum3 = _mm256_mul_ps(coeff3, x3);
				case 3:
					x2 = policy.load_8(&src_ptr2[j]);
					accum2 = _mm256_fmadd_ps(coeff2, x2, accum2);
				case 2:
					x1 = policy.load_8(&src_ptr1[j]);
					accum1 = _mm256_fmadd_ps(coeff1, x1, accum1);
				case 1:
					x0 = policy.load_8(&src_ptr0[j]);
					accum0 = _mm256_fmadd_ps(coeff0, x0, accum0);
				}

				accum0 = _mm256_add_ps(accum0, accum2);
				accum1 = _mm256_add_ps(accum1, accum3);
				accum0 = _mm256_add_ps(accum0, accum1);

				if (k)
					accum0 = _mm256_add_ps(accum0, policy.load_8(&dst_ptr[j]));

				policy.store_8(&dst_ptr[j], accum0);
			}
		}
		filter_plane_v_scalar(filter, src, dst, i, i + 1, mod(src_width, 8), src_width, policy);
	}
}

class ResizeImplAVX2 final : public ResizeImpl {
public:
	ResizeImplAVX2(const EvaluatedFilter &filter_h, const EvaluatedFilter &filter_v) : ResizeImpl(filter_h, filter_v)
	{}

	void process_u16_h(const ImagePlane<const uint16_t> &src, const ImagePlane<uint16_t> &dst, uint16_t *tmp) const override
	{
		if (m_filter_h.width() > 16)
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
		if (m_filter_h.width() > 8)
			filter_plane_fp_h<true>(m_filter_h, src, dst, VectorPolicy_F16{});
		else
			filter_plane_fp_h<false>(m_filter_h, src, dst, VectorPolicy_F16{});
	}

	void process_f16_v(const ImagePlane<const uint16_t> &src, const ImagePlane<uint16_t> &dst, uint16_t *tmp) const override
	{
		filter_plane_fp_v(m_filter_v, src, dst, VectorPolicy_F16{});
	}

	void process_f32_h(const ImagePlane<const float> &src, const ImagePlane<float> &dst, float *tmp) const override
	{
		if (m_filter_h.width() >= 8)
			filter_plane_fp_h<true>(m_filter_h, src, dst, VectorPolicy_F32{});
		else
			filter_plane_fp_h<false>(m_filter_h, src, dst, VectorPolicy_F32{});
	}

	void process_f32_v(const ImagePlane<const float> &src, const ImagePlane<float> &dst, float *tmp) const override
	{
		filter_plane_fp_v(m_filter_v, src, dst, VectorPolicy_F32{});
	}
};

} // namespace


ResizeImpl *create_resize_impl_avx2(const EvaluatedFilter &filter_h, const EvaluatedFilter &filter_v)
{
	return new ResizeImplAVX2{ filter_h, filter_v };
}

} // namespace resize;
} // namespace zimg

#endif // ZIMG_X86
