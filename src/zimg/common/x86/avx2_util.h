#pragma once

#ifdef ZIMG_X86

#ifndef ZIMG_X86_AVX2_UTIL_H_
#define ZIMG_X86_AVX2_UTIL_H_

#include "common/ccdep.h"
#include "x86util.h"

namespace zimg {

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
	__m256i t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15;
	__m256i tt0, tt1, tt2, tt3, tt4, tt5, tt6, tt7, tt8, tt9, tt10, tt11, tt12, tt13, tt14, tt15;

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

	t0 = _mm256_unpacklo_epi64(tt0, tt2);
	t1 = _mm256_unpackhi_epi64(tt0, tt2);
	t2 = _mm256_unpacklo_epi64(tt1, tt3);
	t3 = _mm256_unpackhi_epi64(tt1, tt3);
	t4 = _mm256_unpacklo_epi64(tt4, tt6);
	t5 = _mm256_unpackhi_epi64(tt4, tt6);
	t6 = _mm256_unpacklo_epi64(tt5, tt7);
	t7 = _mm256_unpackhi_epi64(tt5, tt7);

	t8 = _mm256_unpacklo_epi16(row8, row9);
	t9 = _mm256_unpacklo_epi16(row10, row11);
	t10 = _mm256_unpacklo_epi16(row12, row13);
	t11 = _mm256_unpacklo_epi16(row14, row15);
	t12 = _mm256_unpackhi_epi16(row8, row9);
	t13 = _mm256_unpackhi_epi16(row10, row11);
	t14 = _mm256_unpackhi_epi16(row12, row13);
	t15 = _mm256_unpackhi_epi16(row14, row15);

	tt8 = _mm256_unpacklo_epi32(t8, t9);
	tt9 = _mm256_unpackhi_epi32(t8, t9);
	tt10 = _mm256_unpacklo_epi32(t10, t11);
	tt11 = _mm256_unpackhi_epi32(t10, t11);
	tt12 = _mm256_unpacklo_epi32(t12, t13);
	tt13 = _mm256_unpackhi_epi32(t12, t13);
	tt14 = _mm256_unpacklo_epi32(t14, t15);
	tt15 = _mm256_unpackhi_epi32(t14, t15);

	t8 = _mm256_unpacklo_epi64(tt8, tt10);
	t9 = _mm256_unpackhi_epi64(tt8, tt10);
	t10 = _mm256_unpacklo_epi64(tt9, tt11);
	t11 = _mm256_unpackhi_epi64(tt9, tt11);
	t12 = _mm256_unpacklo_epi64(tt12, tt14);
	t13 = _mm256_unpackhi_epi64(tt12, tt14);
	t14 = _mm256_unpacklo_epi64(tt13, tt15);
	t15 = _mm256_unpackhi_epi64(tt13, tt15);

	row0 = _mm256_permute2f128_si256(t0, t8, 0x20);
	row1 = _mm256_permute2f128_si256(t1, t9, 0x20);
	row2 = _mm256_permute2f128_si256(t2, t10, 0x20);
	row3 = _mm256_permute2f128_si256(t3, t11, 0x20);
	row4 = _mm256_permute2f128_si256(t4, t12, 0x20);
	row5 = _mm256_permute2f128_si256(t5, t13, 0x20);
	row6 = _mm256_permute2f128_si256(t6, t14, 0x20);
	row7 = _mm256_permute2f128_si256(t7, t15, 0x20);

	row8 = _mm256_permute2f128_si256(t0, t8, 0x31);
	row9 = _mm256_permute2f128_si256(t1, t9, 0x31);
	row10 = _mm256_permute2f128_si256(t2, t10, 0x31);
	row11 = _mm256_permute2f128_si256(t3, t11, 0x31);
	row12 = _mm256_permute2f128_si256(t4, t12, 0x31);
	row13 = _mm256_permute2f128_si256(t5, t13, 0x31);
	row14 = _mm256_permute2f128_si256(t6, t14, 0x31);
	row15 = _mm256_permute2f128_si256(t7, t15, 0x31);
}

} // namespace zimg

#endif // ZIMG_X86_AVX2_UTIL_H_

#endif // ZIMG_X86
