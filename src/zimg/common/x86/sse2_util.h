#pragma once

#ifdef ZIMG_X86

#ifndef ZIMG_X86_SSE2_UTIL_H_
#define ZIMG_X86_SSE2_UTIL_H_

#include <cstdint>
#include "common/ccdep.h"
#include "x86util.h"

namespace zimg {

// Store from [x] into [dst] the 8-bit elements with index less than [idx].
static inline FORCE_INLINE void mm_store_idxlo_epi8(__m128i *dst, __m128i x, unsigned idx)
{
	__m128i orig = _mm_load_si128(dst);
	__m128i mask = _mm_load_si128((const __m128i *)(&xmm_mask_table[idx]));

	orig = _mm_andnot_si128(mask, orig);
	x = _mm_and_si128(mask, x);
	x = _mm_or_si128(x, orig);

	_mm_store_si128(dst, x);
}

// Store from [x] into [dst] the 8-bit elements with index greater than or equal to [idx].
static inline FORCE_INLINE void mm_store_idxhi_epi8(__m128i *dst, __m128i x, unsigned idx)
{
	__m128i orig = _mm_load_si128(dst);
	__m128i mask = _mm_load_si128((const __m128i *)(&xmm_mask_table[idx]));

	orig = _mm_and_si128(mask, orig);
	x = _mm_andnot_si128(mask, x);
	x = _mm_or_si128(x, orig);

	_mm_store_si128(dst, x);
}

// Store from [x] into [dst] the 16-bit elements with index less than [idx].
static inline FORCE_INLINE void mm_store_idxlo_epi16(__m128i *dst, __m128i x, unsigned idx)
{
	mm_store_idxlo_epi8(dst, x, idx * 2);
}

// Store from [x] into [dst] the 16-bit elements with index greater than or equal to [idx].
static inline FORCE_INLINE void mm_store_idxhi_epi16(__m128i *dst, __m128i x, unsigned idx)
{
	mm_store_idxhi_epi8(dst, x, idx * 2);
}

// Store from [x] into [dst] the 32-bit elements with index less than [idx].
static inline FORCE_INLINE void mm_store_idxlo_ps(float *dst, __m128 x, unsigned idx)
{
	__m128 orig = _mm_load_ps(dst);
	__m128 mask = _mm_load_ps((const float *)(&xmm_mask_table[idx * 4]));

	orig = _mm_andnot_ps(mask, orig);
	x = _mm_and_ps(mask, x);
	x = _mm_or_ps(x, orig);

	_mm_store_ps(dst, x);
}

// Store from [x] into [dst] the 32-bit elements with index greater than or equal to [idx]
static inline FORCE_INLINE void mm_store_idxhi_ps(float *dst, __m128 x, unsigned idx)
{
	__m128 orig = _mm_load_ps(dst);
	__m128 mask = _mm_load_ps((const float *)(&xmm_mask_table[idx * 4]));

	orig = _mm_and_ps(mask, orig);
	x = _mm_andnot_ps(mask, x);
	x = _mm_or_ps(x, orig);

	_mm_store_ps(dst, x);
}

// Stores the elements of [x] into [dst0]-[dst7].
static inline FORCE_INLINE void mm_scatter_epi16(uint16_t *dst0, uint16_t *dst1, uint16_t *dst2, uint16_t *dst3,
                                                 uint16_t *dst4, uint16_t *dst5, uint16_t *dst6, uint16_t *dst7, __m128i x)
{
	*dst0 = _mm_extract_epi16(x, 0);
	*dst1 = _mm_extract_epi16(x, 1);
	*dst2 = _mm_extract_epi16(x, 2);
	*dst3 = _mm_extract_epi16(x, 3);
	*dst4 = _mm_extract_epi16(x, 4);
	*dst5 = _mm_extract_epi16(x, 5);
	*dst6 = _mm_extract_epi16(x, 6);
	*dst7 = _mm_extract_epi16(x, 7);
}

// Stores the elements of [x] into [dst0]-[dst3].
static inline FORCE_INLINE void mm_scatter_ps(float *dst0, float *dst1, float *dst2, float *dst3, __m128 x)
{
	_mm_store_ss(dst0, x);
	_mm_store_ss(dst1, _mm_shuffle_ps(x, x, _MM_SHUFFLE(3, 2, 1, 1)));
	_mm_store_ss(dst2, _mm_shuffle_ps(x, x, _MM_SHUFFLE(3, 2, 1, 2)));
	_mm_store_ss(dst3, _mm_shuffle_ps(x, x, _MM_SHUFFLE(3, 2, 1, 3)));
}

// Transpose in-place the 8x8 matrix stored in [row0]-[row7].
static inline FORCE_INLINE void mm_transpose8_epi16(__m128i &row0, __m128i &row1, __m128i &row2, __m128i &row3,
                                                    __m128i &row4, __m128i &row5, __m128i &row6, __m128i &row7)
{
	__m128i t0, t1, t2, t3, t4, t5, t6, t7;
	__m128i tt0, tt1, tt2, tt3, tt4, tt5, tt6, tt7;

	t0 = _mm_unpacklo_epi16(row0, row1);
	t1 = _mm_unpacklo_epi16(row2, row3);
	t2 = _mm_unpacklo_epi16(row4, row5);
	t3 = _mm_unpacklo_epi16(row6, row7);
	t4 = _mm_unpackhi_epi16(row0, row1);
	t5 = _mm_unpackhi_epi16(row2, row3);
	t6 = _mm_unpackhi_epi16(row4, row5);
	t7 = _mm_unpackhi_epi16(row6, row7);

	tt0 = _mm_unpacklo_epi32(t0, t1);
	tt1 = _mm_unpackhi_epi32(t0, t1);
	tt2 = _mm_unpacklo_epi32(t2, t3);
	tt3 = _mm_unpackhi_epi32(t2, t3);
	tt4 = _mm_unpacklo_epi32(t4, t5);
	tt5 = _mm_unpackhi_epi32(t4, t5);
	tt6 = _mm_unpacklo_epi32(t6, t7);
	tt7 = _mm_unpackhi_epi32(t6, t7);

	row0 = _mm_unpacklo_epi64(tt0, tt2);
	row1 = _mm_unpackhi_epi64(tt0, tt2);
	row2 = _mm_unpacklo_epi64(tt1, tt3);
	row3 = _mm_unpackhi_epi64(tt1, tt3);
	row4 = _mm_unpacklo_epi64(tt4, tt6);
	row5 = _mm_unpackhi_epi64(tt4, tt6);
	row6 = _mm_unpacklo_epi64(tt5, tt7);
	row7 = _mm_unpackhi_epi64(tt5, tt7);
}

// Saturated convert signed 32-bit to unsigned 16-bit, biased by INT16_MIN.
static inline FORCE_INLINE __m128i mm_packus_epi32_bias(__m128i a, __m128i b)
{
	const __m128i i16_min_epi32 = _mm_set1_epi32(INT16_MIN);

	a = _mm_add_epi32(a, i16_min_epi32);
	b = _mm_add_epi32(b, i16_min_epi32);

	a = _mm_packs_epi32(a, b);

	return a;
}

// Saturated convert signed 32-bit to unsigned 16-bit.
static inline FORCE_INLINE __m128i mm_packus_epi32(__m128i a, __m128i b)
{
	const __m128i i16_min_epi16 = _mm_set1_epi16(INT16_MIN);

	a = mm_packus_epi32_bias(a, b);
	a = _mm_sub_epi16(a, i16_min_epi16);

	return a;
}

} // namespace zimg

#endif // ZIMG_X86_SSE2_UTIL_H_

#endif // ZIMG_X86
