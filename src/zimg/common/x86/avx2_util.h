#pragma once

#ifdef ZIMG_X86

#ifndef ZIMG_X86_AVX2_UTIL_H_
#define ZIMG_X86_AVX2_UTIL_H_

#include "common/ccdep.h"
#include "x86util.h"

namespace zimg {

namespace _avx2 {

// Transpose two 8x8 matrices stored in the lower and upper 128-bit lanes of [row0]-[row7].
static inline FORCE_INLINE void mm256_transpose8_x2_epi16(__m256i &row0, __m256i &row1, __m256i &row2, __m256i &row3,
                                                          __m256i &row4, __m256i &row5, __m256i &row6, __m256i &row7)
{
	__m256i t0, t1, t2, t3, t4, t5, t6, t7;
	__m256i tt0, tt1, tt2, tt3, tt4, tt5, tt6, tt7;

	t0 = _mm256_unpacklo_epi16(row0, row1);
	t1 = _mm256_unpacklo_epi16(row2, row3);
	t2 = _mm256_unpacklo_epi16(row4, row5);
	t3 = _mm256_unpacklo_epi16(row6, row7);
	t4 = _mm256_unpackhi_epi16(row0, row1);
	t5 = _mm256_unpackhi_epi16(row2, row3);
	t6 = _mm256_unpackhi_epi16(row4, row5);
	t7 = _mm256_unpackhi_epi16(row6, row7);

	tt0 = _mm256_unpacklo_epi32(t0, t1);
	tt1 = _mm256_unpackhi_epi32(t0, t1);
	tt2 = _mm256_unpacklo_epi32(t2, t3);
	tt3 = _mm256_unpackhi_epi32(t2, t3);
	tt4 = _mm256_unpacklo_epi32(t4, t5);
	tt5 = _mm256_unpackhi_epi32(t4, t5);
	tt6 = _mm256_unpacklo_epi32(t6, t7);
	tt7 = _mm256_unpackhi_epi32(t6, t7);

	row0 = _mm256_unpacklo_epi64(tt0, tt2);
	row1 = _mm256_unpackhi_epi64(tt0, tt2);
	row2 = _mm256_unpacklo_epi64(tt1, tt3);
	row3 = _mm256_unpackhi_epi64(tt1, tt3);
	row4 = _mm256_unpacklo_epi64(tt4, tt6);
	row5 = _mm256_unpackhi_epi64(tt4, tt6);
	row6 = _mm256_unpacklo_epi64(tt5, tt7);
	row7 = _mm256_unpackhi_epi64(tt5, tt7);
}

// Exchange the upper 128-bit lane of [row0] with the lower 128-bit lane of [row1].
static inline FORCE_INLINE void mm256_exchange_lanes_si128(__m256i &row0, __m256i &row1)
{
	__m256i tmp0 = _mm256_permute2f128_si256(row0, row1, 0x20);
	__m256i tmp1 = _mm256_permute2f128_si256(row0, row1, 0x31);
	row0 = tmp0;
	row1 = tmp1;
}

} // namespace _avx2


// Store from [x] into [dst] the 8-bit elements with index less than [idx].
static inline FORCE_INLINE void mm256_store_idxlo_epi8(__m256i *dst, __m256i x, unsigned idx)
{
	__m256i orig = _mm256_load_si256(dst);
	__m256i mask = _mm256_load_si256((const __m256i *)(&ymm_mask_table[idx]));

	x = _mm256_blendv_epi8(orig, x, mask);
	_mm256_store_si256(dst, x);
}

// Store from [x] into [dst] the 8-bit elements with index greater than or equal to [idx].
static inline FORCE_INLINE void mm256_store_idxhi_epi8(__m256i *dst, __m256i x, unsigned idx)
{
	__m256i orig = _mm256_load_si256(dst);
	__m256i mask = _mm256_load_si256((const __m256i *)(&ymm_mask_table[idx]));

	x = _mm256_blendv_epi8(x, orig, mask);
	_mm256_store_si256(dst, x);
}

// Store from [x] into [dst] the 16-bit elements with index less than [idx].
static inline FORCE_INLINE void mm256_store_idxlo_epi16(__m256i *dst, __m256i x, unsigned idx)
{
	mm256_store_idxlo_epi8(dst, x, idx * 2);
}

// Store from [x] into [dst] the 16-bit elements with index less than [idx].
static inline FORCE_INLINE void mm256_store_idxhi_epi16(__m256i *dst, __m256i x, unsigned idx)
{
	mm256_store_idxhi_epi8(dst, x, idx * 2);
}

// Transpose in-place the 16x16 matrix stored in [row0]-[row15].
static inline FORCE_INLINE void mm256_transpose16_epi16(__m256i &row0, __m256i &row1, __m256i &row2, __m256i &row3,
                                                        __m256i &row4, __m256i &row5, __m256i &row6, __m256i &row7,
                                                        __m256i &row8, __m256i &row9, __m256i &row10, __m256i &row11,
                                                        __m256i &row12, __m256i &row13, __m256i &row14, __m256i &row15)
{
	_avx2::mm256_transpose8_x2_epi16(row0, row1, row2, row3, row4, row5, row6, row7);
	_avx2::mm256_transpose8_x2_epi16(row8, row9, row10, row11, row12, row13, row14, row15);

	_avx2::mm256_exchange_lanes_si128(row0, row8);
	_avx2::mm256_exchange_lanes_si128(row1, row9);
	_avx2::mm256_exchange_lanes_si128(row2, row10);
	_avx2::mm256_exchange_lanes_si128(row3, row11);
	_avx2::mm256_exchange_lanes_si128(row4, row12);
	_avx2::mm256_exchange_lanes_si128(row5, row13);
	_avx2::mm256_exchange_lanes_si128(row6, row14);
	_avx2::mm256_exchange_lanes_si128(row7, row15);
}

} // namespace zimg

#endif // ZIMG_X86_AVX2_UTIL_H_

#endif // ZIMG_X86
