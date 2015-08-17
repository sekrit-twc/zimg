#if 0
#ifdef ZIMG_X86

#include <cstddef>
#include <cstdint>
#include <emmintrin.h>
#include "Common/align.h"
#include "Common/except.h"
#include "Common/linebuffer.h"
#include "Common/osdep.h"
#include "Common/pixel.h"
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

inline FORCE_INLINE void transpose8_epi16(__m128i &x0, __m128i &x1, __m128i &x2, __m128i &x3, __m128i &x4, __m128i &x5, __m128i &x6, __m128i &x7)
{
	__m128i t0, t1, t2, t3, t4, t5, t6, t7;
	__m128i tt0, tt1, tt2, tt3, tt4, tt5, tt6, tt7;

	t0 = _mm_unpacklo_epi16(x0, x1);
	t1 = _mm_unpacklo_epi16(x2, x3);
	t2 = _mm_unpacklo_epi16(x4, x5);
	t3 = _mm_unpacklo_epi16(x6, x7);
	t4 = _mm_unpackhi_epi16(x0, x1);
	t5 = _mm_unpackhi_epi16(x2, x3);
	t6 = _mm_unpackhi_epi16(x4, x5);
	t7 = _mm_unpackhi_epi16(x6, x7);

	tt0 = _mm_unpacklo_epi32(t0, t1);
	tt1 = _mm_unpackhi_epi32(t0, t1);
	tt2 = _mm_unpacklo_epi32(t2, t3);
	tt3 = _mm_unpackhi_epi32(t2, t3);
	tt4 = _mm_unpacklo_epi32(t4, t5);
	tt5 = _mm_unpackhi_epi32(t4, t5);
	tt6 = _mm_unpacklo_epi32(t6, t7);
	tt7 = _mm_unpackhi_epi32(t6, t7);

	x0 = _mm_unpacklo_epi64(tt0, tt2);
	x1 = _mm_unpackhi_epi64(tt0, tt2);
	x2 = _mm_unpacklo_epi64(tt1, tt3);
	x3 = _mm_unpackhi_epi64(tt1, tt3);
	x4 = _mm_unpacklo_epi64(tt4, tt6);
	x5 = _mm_unpackhi_epi64(tt4, tt6);
	x6 = _mm_unpacklo_epi64(tt5, tt7);
	x7 = _mm_unpackhi_epi64(tt5, tt7);
}

inline FORCE_INLINE void transpose_line_8x8_epi16(const uint16_t *p0, const uint16_t *p1, const uint16_t *p2, const uint16_t *p3, const uint16_t *p4, const uint16_t *p5, const uint16_t *p6, const uint16_t *p7, uint16_t *dst, unsigned n)
{
	for (unsigned j = 0; j < mod(n, 8); j += 8) {
		__m128i x0, x1, x2, x3, x4, x5, x6, x7;

		x0 = _mm_load_si128((const __m128i *)&p0[j]);
		x1 = _mm_load_si128((const __m128i *)&p1[j]);
		x2 = _mm_load_si128((const __m128i *)&p2[j]);
		x3 = _mm_load_si128((const __m128i *)&p3[j]);
		x4 = _mm_load_si128((const __m128i *)&p4[j]);
		x5 = _mm_load_si128((const __m128i *)&p5[j]);
		x6 = _mm_load_si128((const __m128i *)&p6[j]);
		x7 = _mm_load_si128((const __m128i *)&p7[j]);

		transpose8_epi16(x0, x1, x2, x3, x4, x5, x6, x7);

		_mm_store_si128((__m128i *)&dst[(j + 0) * 8], x0);
		_mm_store_si128((__m128i *)&dst[(j + 1) * 8], x1);
		_mm_store_si128((__m128i *)&dst[(j + 2) * 8], x2);
		_mm_store_si128((__m128i *)&dst[(j + 3) * 8], x3);
		_mm_store_si128((__m128i *)&dst[(j + 4) * 8], x4);
		_mm_store_si128((__m128i *)&dst[(j + 5) * 8], x5);
		_mm_store_si128((__m128i *)&dst[(j + 6) * 8], x6);
		_mm_store_si128((__m128i *)&dst[(j + 7) * 8], x7);
	}
	for (unsigned j = mod(n, 8); j < n; ++j) {
		dst[j * 8 + 0] = p0[j];
		dst[j * 8 + 1] = p1[j];
		dst[j * 8 + 2] = p2[j];
		dst[j * 8 + 3] = p3[j];
		dst[j * 8 + 4] = p4[j];
		dst[j * 8 + 5] = p5[j];
		dst[j * 8 + 6] = p6[j];
		dst[j * 8 + 7] = p7[j];
	}
}

inline FORCE_INLINE void transpose_line_4x4_ps(const float *p0, const float *p1, const float *p2, const float *p3, float *dst, unsigned n)
{
	for (unsigned j = 0; j < mod(n, 4); j += 4) {
		__m128 x0, x1, x2, x3;

		x0 = _mm_load_ps(&p0[j]);
		x1 = _mm_load_ps(&p1[j]);
		x2 = _mm_load_ps(&p2[j]);
		x3 = _mm_load_ps(&p3[j]);

		transpose4_ps(x0, x1, x2, x3);

		_mm_store_ps(&dst[(j + 0) * 4], x0);
		_mm_store_ps(&dst[(j + 1) * 4], x1);
		_mm_store_ps(&dst[(j + 2) * 4], x2);
		_mm_store_ps(&dst[(j + 3) * 4], x3);
	}
	for (unsigned j = mod(n, 4); j < n; ++j) {
		dst[j * 4 + 0] = p0[j];
		dst[j * 4 + 1] = p1[j];
		dst[j * 4 + 2] = p2[j];
		dst[j * 4 + 3] = p3[j];
	}
}

inline FORCE_INLINE void transpose_block_8x8_epi16(const uint16_t cache[8 * 16], uint16_t *p0, uint16_t *p1, uint16_t *p2, uint16_t *p3, uint16_t *p4, uint16_t *p5, uint16_t *p6, uint16_t *p7)
{
	__m128i x0, x1, x2, x3, x4, x5, x6, x7;

	x0 = _mm_load_si128((const __m128i *)&cache[0 * 8]);
	x1 = _mm_load_si128((const __m128i *)&cache[1 * 8]);
	x2 = _mm_load_si128((const __m128i *)&cache[2 * 8]);
	x3 = _mm_load_si128((const __m128i *)&cache[3 * 8]);
	x4 = _mm_load_si128((const __m128i *)&cache[4 * 8]);
	x5 = _mm_load_si128((const __m128i *)&cache[5 * 8]);
	x6 = _mm_load_si128((const __m128i *)&cache[6 * 8]);
	x7 = _mm_load_si128((const __m128i *)&cache[7 * 8]);

	transpose8_epi16(x0, x1, x2, x3, x4, x5, x6, x7);

	_mm_store_si128((__m128i *)p0, x0);
	_mm_store_si128((__m128i *)p1, x1);
	_mm_store_si128((__m128i *)p2, x2);
	_mm_store_si128((__m128i *)p3, x3);
	_mm_store_si128((__m128i *)p4, x4);
	_mm_store_si128((__m128i *)p5, x5);
	_mm_store_si128((__m128i *)p6, x6);
	_mm_store_si128((__m128i *)p7, x7);
}

inline FORCE_INLINE void transpose_block_4x4_ps(const float cache[4 * 4], float *p0, float *p1, float *p2, float *p3)
{
	__m128 x0, x1, x2, x3;

	x0 = _mm_load_ps(&cache[0 * 4]);
	x1 = _mm_load_ps(&cache[1 * 4]);
	x2 = _mm_load_ps(&cache[2 * 4]);
	x3 = _mm_load_ps(&cache[3 * 4]);

	transpose4_ps(x0, x1, x2, x3);

	_mm_store_ps(p0, x0);
	_mm_store_ps(p1, x1);
	_mm_store_ps(p2, x2);
	_mm_store_ps(p3, x3);
}

inline FORCE_INLINE void scatter_epi16(__m128i x, uint16_t *p0, uint16_t *p1, uint16_t *p2, uint16_t *p3, uint16_t *p4, uint16_t *p5, uint16_t *p6, uint16_t *p7)
{
	*p0 = _mm_extract_epi16(x, 0);
	*p1 = _mm_extract_epi16(x, 1);
	*p2 = _mm_extract_epi16(x, 2);
	*p3 = _mm_extract_epi16(x, 3);
	*p4 = _mm_extract_epi16(x, 4);
	*p5 = _mm_extract_epi16(x, 5);
	*p6 = _mm_extract_epi16(x, 6);
	*p7 = _mm_extract_epi16(x, 7);
}

inline FORCE_INLINE void scatter_ps(__m128 x, float *p0, float *p1, float *p2, float *p3)
{
	ALIGNED(16) float tmp[4];

	_mm_store_ps(tmp, x);
	*p0 = tmp[0];
	*p1 = tmp[1];
	*p2 = tmp[2];
	*p3 = tmp[3];
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

inline FORCE_INLINE __m128i pack_i30_to_epi16(__m128i lo, __m128i hi)
{
	__m128i offset = _mm_set1_epi32(1 << 13);

	lo = _mm_add_epi32(lo, offset);
	hi = _mm_add_epi32(hi, offset);

	lo = _mm_srai_epi32(lo, 14);
	hi = _mm_srai_epi32(hi, 14);

	return  _mm_packs_epi32(lo, hi);
}

template <unsigned N>
void filter_line_u16_h(const FilterContext &filter, const LineBuffer<uint16_t> &src, LineBuffer<uint16_t> &dst, unsigned n, void *tmp)
{
	__m128i INT16_MIN_EPI16 = _mm_set1_epi16(INT16_MIN);

	const int16_t *filter_data = filter.data_i16.data();
	const unsigned *filter_left = filter.left.data();
	unsigned filter_stride = filter.stride_i16;

	unsigned src_width = src.width();
	unsigned dst_width = dst.width();

	uint16_t *dst_ptr0 = dst[n + 0];
	uint16_t *dst_ptr1 = dst[n + 1];
	uint16_t *dst_ptr2 = dst[n + 2];
	uint16_t *dst_ptr3 = dst[n + 3];
	uint16_t *dst_ptr4 = dst[n + 4];
	uint16_t *dst_ptr5 = dst[n + 5];
	uint16_t *dst_ptr6 = dst[n + 6];
	uint16_t *dst_ptr7 = dst[n + 7];

	uint16_t *ttmp = (uint16_t *)tmp;
	ALIGNED(16) uint16_t cache[8 * 8];

	transpose_line_8x8_epi16(src[n + 0], src[n + 1], src[n + 2], src[n + 3], src[n + 4], src[n + 5], src[n + 6], src[n + 7], ttmp, src_width);

	for (unsigned j = 0; j < mod(dst_width, 8); ++j) {
		const int16_t *filter_row = &filter_data[j * filter_stride];
		unsigned left = filter_left[j];
		__m128i accum_lo = _mm_setzero_si128();
		__m128i accum_hi = _mm_setzero_si128();
		__m128i x0, x1, lo, hi, c;
		__m128i result;

#define ITER(xiter) \
  do { \
    x0 = _mm_load_si128((const __m128i *)&ttmp[(left + (xiter) + 0) * 8]); \
    x0 = _mm_add_epi16(x0, INT16_MIN_EPI16); \
    x1 = _mm_load_si128((const __m128i *)&ttmp[(left + (xiter) + 1) * 8]); \
    x1 = _mm_add_epi16(x1, INT16_MIN_EPI16); \
    c = _mm_loadu_si128((const __m128i *)&filter_row[(xiter)]); \
    c = _mm_shuffle_epi32(c, _MM_SHUFFLE(0, 0, 0, 0)); \
    lo = _mm_unpacklo_epi16(x0, x1); \
    lo = _mm_madd_epi16(c, lo); \
    hi = _mm_unpackhi_epi16(x0, x1); \
    hi = _mm_madd_epi16(c, hi); \
    accum_lo = _mm_add_epi32(accum_lo, lo); \
    accum_hi = _mm_add_epi32(accum_hi, hi); \
  } while (0);

		if (N) {
			switch (N) {
			case 8: case 7: ITER(6);
			case 6: case 5: ITER(4);
			case 4: case 3: ITER(2);
			case 2: case 1: ITER(0);
			}
		} else {
			for (unsigned k = 0; k < filter.filter_width; k += 2) {
				ITER(k);
			}
		}
#undef ITER

		result = pack_i30_to_epi16(accum_lo, accum_hi);
		result = _mm_sub_epi16(result, INT16_MIN_EPI16);

		_mm_store_si128((__m128i *)&cache[(j % 8) * 8], result);

		if (j % 8 == 7) {
			unsigned dst_j = j - 7;
			transpose_block_8x8_epi16(cache, &dst_ptr0[dst_j], &dst_ptr1[dst_j], &dst_ptr2[dst_j], &dst_ptr3[dst_j], &dst_ptr4[dst_j], &dst_ptr5[dst_j], &dst_ptr6[dst_j], &dst_ptr7[dst_j]);
		}
	}
	for (unsigned j = mod(dst_width, 8); j < dst_width; ++j) {
		const int16_t *filter_row = &filter_data[j * filter_stride];
		unsigned left = filter_left[j];
		__m128i accum_lo = _mm_setzero_si128();
		__m128i accum_hi = _mm_setzero_si128();
		__m128i result;

		for (unsigned k = 0; k < filter.filter_width; k += 2) {
			__m128i x0, x1, lo, hi, c;

			x0 = _mm_load_si128((const __m128i *)&ttmp[(left + k + 0) * 8]);
			x0 = _mm_add_epi16(x0, INT16_MIN_EPI16);

			x1 = _mm_load_si128((const __m128i *)&ttmp[(left + k + 1) * 8]);
			x1 = _mm_add_epi16(x1, INT16_MIN_EPI16);

			lo = _mm_unpacklo_epi16(x0, x1);
			hi = _mm_unpackhi_epi16(x0, x1);

			c = _mm_loadu_si128((const __m128i *)&filter_row[k]);
			c = _mm_shuffle_epi32(c, _MM_SHUFFLE(0, 0, 0, 0));
			lo = _mm_madd_epi16(c, lo);
			hi = _mm_madd_epi16(c, hi);

			accum_lo = _mm_add_epi32(accum_lo, lo);
			accum_hi = _mm_add_epi32(accum_hi, hi);
		}

		result = pack_i30_to_epi16(accum_lo, accum_hi);
		result = _mm_sub_epi16(result, INT16_MIN_EPI16);

		scatter_epi16(result, &dst_ptr0[j], &dst_ptr1[j], &dst_ptr2[j], &dst_ptr3[j], &dst_ptr4[j], &dst_ptr5[j], &dst_ptr6[j], &dst_ptr7[j]);
	}
}

template <unsigned N>
void filter_line_fp_h(const FilterContext &filter, const LineBuffer<float> &src, LineBuffer<float> &dst, unsigned n, void *tmp)
{
	const float *filter_data = filter.data.data();
	const unsigned *filter_left = filter.left.data();
	unsigned filter_stride = filter.stride;

	unsigned src_width = src.width();
	unsigned dst_width = dst.width();

	float *dst_ptr0 = dst[n + 0];
	float *dst_ptr1 = dst[n + 1];
	float *dst_ptr2 = dst[n + 2];
	float *dst_ptr3 = dst[n + 3];

	float *ttmp = (float *)tmp;
	ALIGNED(16) float cache[4 * 4];

	transpose_line_4x4_ps(src[n + 0], src[n + 1], src[n + 2], src[n + 3], ttmp, src_width);

	for (unsigned j = 0; j < mod(dst_width, 4); ++j) {
		const float *filter_row = &filter_data[j * filter_stride];
		unsigned left = filter_left[j];
		__m128 accum0 = _mm_setzero_ps();
		__m128 accum1 = _mm_setzero_ps();
		__m128 x, c;

#define ITER(xiter) \
  do { \
    x = _mm_load_ps(&ttmp[(left + (xiter)) * 4]); \
    c = _mm_load_ps1(&filter_row[xiter]); \
    x = _mm_mul_ps(c, x); \
    if ((xiter) % 2) accum1 = _mm_add_ps(accum1, x); else accum0 = _mm_add_ps(accum0, x); \
  } while (0);

		if (N) {
			switch (N) {
			case 8: ITER(7);
			case 7: ITER(6);
			case 6: ITER(5);
			case 5: ITER(4);
			case 4: ITER(3);
			case 3: ITER(2);
			case 2: ITER(1);
			case 1: ITER(0);
			}
		} else {
			for (unsigned k = 0; k < filter.filter_width; k += 4) {
				ITER(k + 3);
				ITER(k + 2);
				ITER(k + 1);
				ITER(k + 0);
			}
		}
#undef ITER

		if (N != 1)
			accum0 = _mm_add_ps(accum0, accum1);

		_mm_store_ps(&cache[(j % 4) * 4], accum0);

		if (j % 4 == 3) {
			unsigned dst_j = j - 3;
			transpose_block_4x4_ps(cache, &dst_ptr0[dst_j], &dst_ptr1[dst_j], &dst_ptr2[dst_j], &dst_ptr3[dst_j]);
		}
	}
	for (unsigned j = mod(dst_width, 4); j < dst_width; ++j) {
		const float *filter_row = &filter_data[j * filter_stride];
		unsigned left = filter_left[j];
		__m128 accum = _mm_setzero_ps();

		for (unsigned k = 0; k < filter.filter_width; ++k) {
			__m128 x = _mm_load_ps(&ttmp[(left + k) * 4]);
			__m128 c = _mm_load_ps1(&filter_row[k]);

			x = _mm_mul_ps(c, x);
			accum = _mm_add_ps(accum, x);
		}

		scatter_ps(accum, &dst_ptr0[j], &dst_ptr1[j], &dst_ptr2[j], &dst_ptr3[j]);
	}
}

void filter_line_u16_v(const FilterContext &filter, const LineBuffer<uint16_t> &src, LineBuffer<uint16_t> &dst, unsigned n, void *tmp)
{
	__m128i INT16_MIN_EPI16 = _mm_set1_epi16(INT16_MIN);

	const int16_t *filter_data = filter.data_i16.data();
	const unsigned *filter_left = filter.left.data();
	unsigned filter_stride = filter.stride_i16;

	unsigned width = dst.width();

	const int16_t *filter_row = &filter_data[n * filter_stride];
	unsigned top = filter_left[n];
	uint16_t *dst_ptr = dst[n];
	uint32_t *accum_tmp = (uint32_t *)tmp;

	for (unsigned k = 0; k < mod(filter.filter_width, 4); k += 4) {
		const uint16_t *src_ptr0 = src[top + k + 0];
		const uint16_t *src_ptr1 = src[top + k + 1];
		const uint16_t *src_ptr2 = src[top + k + 2];
		const uint16_t *src_ptr3 = src[top + k + 3];

		__m128i coeff0 = _mm_set1_epi16(filter_row[k + 0]);
		__m128i coeff1 = _mm_set1_epi16(filter_row[k + 1]);
		__m128i coeff2 = _mm_set1_epi16(filter_row[k + 2]);
		__m128i coeff3 = _mm_set1_epi16(filter_row[k + 3]);

		for (unsigned j = 0; j < mod(width, 8); j += 8) {
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
				accum0l = _mm_add_epi32(accum0l, _mm_load_si128((const __m128i *)&accum_tmp[j + 0]));
				accum0h = _mm_add_epi32(accum0h, _mm_load_si128((const __m128i *)&accum_tmp[j + 4]));
			}

			if (k == filter.filter_width - 4) {
				packed = pack_i30_to_epi16(accum0l, accum0h);
				packed = _mm_sub_epi16(packed, INT16_MIN_EPI16);
				_mm_store_si128((__m128i *)&dst_ptr[j], packed);
			} else {
				_mm_store_si128((__m128i *)&accum_tmp[j + 0], accum0l);
				_mm_store_si128((__m128i *)&accum_tmp[j + 4], accum0h);
			}
		}
	}
	if (filter.filter_width % 4) {
		unsigned m = filter.filter_width % 4;
		unsigned k = filter.filter_width - m;

		const uint16_t *src_ptr0 = src[top + k + 0];
		const uint16_t *src_ptr1 = src[top + k + 1];
		const uint16_t *src_ptr2 = src[top + k + 2];

		__m128i coeff0 = _mm_set1_epi16(filter_row[k + 0]);
		__m128i coeff1 = _mm_set1_epi16(filter_row[k + 1]);
		__m128i coeff2 = _mm_set1_epi16(filter_row[k + 2]);

		for (unsigned j = 0; j < mod(width, 8); j += 8) {
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
				accum0l = _mm_add_epi32(accum0l, _mm_load_si128((const __m128i *)&accum_tmp[j + 0]));
				accum0h = _mm_add_epi32(accum0h, _mm_load_si128((const __m128i *)&accum_tmp[j + 4]));
			}

			packed = pack_i30_to_epi16(accum0l, accum0h);
			packed = _mm_sub_epi16(packed, INT16_MIN_EPI16);

			_mm_store_si128((__m128i *)&dst_ptr[j], packed);
		}
	}
	filter_line_v_scalar(filter, src, dst, n, n + 1, mod(width, 8), width, ScalarPolicy_U16{});
}

void filter_line_fp_v(const FilterContext &filter, const LineBuffer<float> &src, LineBuffer<float> &dst, unsigned n)
{
	const float *filter_data = filter.data.data();
	const unsigned *filter_left = filter.left.data();
	unsigned filter_stride = filter.stride;

	unsigned width = dst.width();

	const float *filter_row = &filter_data[n * filter_stride];
	unsigned top = filter_left[n];
	float *dst_ptr = dst[n];

	for (unsigned k = 0; k < mod(filter.filter_width, 4); k += 4) {
		const float *src_ptr0 = src[top + k + 0];
		const float *src_ptr1 = src[top + k + 1];
		const float *src_ptr2 = src[top + k + 2];
		const float *src_ptr3 = src[top + k + 3];

		__m128 coeff0 = _mm_set_ps1(filter_row[k + 0]);
		__m128 coeff1 = _mm_set_ps1(filter_row[k + 1]);
		__m128 coeff2 = _mm_set_ps1(filter_row[k + 2]);
		__m128 coeff3 = _mm_set_ps1(filter_row[k + 3]);

		for (unsigned j = 0; j < mod(width, 4); j += 4) {
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
	if (filter.filter_width % 4) {
		unsigned m = filter.filter_width % 4;
		unsigned k = filter.filter_width - m;

		const float *src_ptr0 = src[top + k + 0];
		const float *src_ptr1 = src[top + k + 1];
		const float *src_ptr2 = src[top + k + 2];

		__m128 coeff0 = _mm_set_ps1(filter_row[k + 0]);
		__m128 coeff1 = _mm_set_ps1(filter_row[k + 1]);
		__m128 coeff2 = _mm_set_ps1(filter_row[k + 2]);

		for (unsigned j = 0; j < mod(width, 4); j += 4) {
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
	filter_line_v_scalar(filter, src, dst, n, n + 1, mod(width, 4), width, ScalarPolicy_F32{});
}

class ResizeImplSSE2_H : public ResizeImpl {
public:
	ResizeImplSSE2_H(const FilterContext &filter) : ResizeImpl(filter, true)
	{}

	size_t tmp_size(PixelType type, unsigned width) const override
	{
		return output_buffering(type) * align((size_t)(m_filter.input_width + 16) * pixel_size(type), ALIGNMENT);
	}

	unsigned output_buffering(PixelType type) const override
	{
		return type == PixelType::FLOAT ? 4 : 8;
	}

	void process_u16(const LineBuffer<uint16_t> &src, LineBuffer<uint16_t> &dst, unsigned n, void *tmp) const override
	{
#define CASE(x) case (x): filter_line_u16_h<x>(m_filter, src, dst, n, tmp); break;
		switch (m_filter.filter_width) {
			CASE(8) CASE(7) CASE(6) CASE(5) CASE(4) CASE(3) CASE(2) CASE(1)
		default:
			filter_line_u16_h<0>(m_filter, src, dst, n, tmp);
		}
#undef CASE
	}

	void process_f16(const LineBuffer<uint16_t> &src, LineBuffer<uint16_t> &dst, unsigned n, void *tmp) const override
	{
		throw ZimgUnsupportedError{ "f16 not supported in SSE2 impl" };
	}

	void process_f32(const LineBuffer<float> &src, LineBuffer<float> &dst, unsigned n, void *tmp) const override
	{
#define CASE(x) case (x): filter_line_fp_h<x>(m_filter, src, dst, n, tmp); break;
		switch (m_filter.filter_width) {
			CASE(8) CASE(7) CASE(6) CASE(5) CASE(4) CASE(3) CASE(2) CASE(1)
		default:
			filter_line_fp_h<0>(m_filter, src, dst, n, tmp);
		}
#undef CASE
	}
};

class ResizeImplSSE2_V : public ResizeImpl {
public:
	ResizeImplSSE2_V(const FilterContext &filter) : ResizeImpl(filter, false)
	{}

	size_t tmp_size(PixelType type, unsigned width) const override
	{
		return type == PixelType::WORD ? sizeof(uint32_t) * width : 0;
	}

	void process_u16(const LineBuffer<uint16_t> &src, LineBuffer<uint16_t> &dst, unsigned n, void *tmp) const override
	{
		filter_line_u16_v(m_filter, src, dst, n, tmp);
	}

	void process_f16(const LineBuffer<uint16_t> &src, LineBuffer<uint16_t> &dst, unsigned n, void *tmp) const override
	{
		throw ZimgUnsupportedError{ "f16 not supported in SSE2 impl" };
	}

	void process_f32(const LineBuffer<float> &src, LineBuffer<float> &dst, unsigned n, void *tmp) const override
	{
		filter_line_fp_v(m_filter, src, dst, n);
	}
};

} // namespace


ResizeImpl *create_resize_impl_sse2(const FilterContext &filter, bool horizontal)
{
	if (horizontal)
		return new ResizeImplSSE2_H{ filter };
	else
		return new ResizeImplSSE2_V{ filter };
}

} // namespace resize
} // namespace zimg

#endif // ZIMG_X86
#endif
