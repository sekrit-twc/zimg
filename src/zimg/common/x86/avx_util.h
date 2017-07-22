#pragma once

#ifdef ZIMG_X86

#ifndef ZIMG_X86_AVX_UTIL_H_
#define ZIMG_X86_AVX_UTIL_H_

#include "common/ccdep.h"
#include "x86util.h"

namespace zimg {

// Store from [x] into [dst] the 32-bit elements with index less than [idx].
static inline FORCE_INLINE void mm256_store_idxlo_ps(float *dst, __m256 x, unsigned idx)
{
	__m256 orig = _mm256_load_ps(dst);
	__m256 mask = _mm256_load_ps((const float *)(&ymm_mask_table[idx * 4]));

	x = _mm256_blendv_ps(orig, x, mask);
	_mm256_store_ps(dst, x);
}

// Store from [x] into [dst] the 32-bit elements with index greater than or equal to [idx].
static inline FORCE_INLINE void mm256_store_idxhi_ps(float *dst, __m256 x, unsigned idx)
{
	__m256 orig = _mm256_load_ps(dst);
	__m256 mask = _mm256_load_ps((const float *)(&ymm_mask_table[idx * 4]));

	x = _mm256_blendv_ps(x, orig, mask);
	_mm256_store_ps(dst, x);
}

// Transpose in-place the 8x8 matrix stored in [row0]-[row7]
static inline FORCE_INLINE void mm256_transpose8_ps(__m256 &row0, __m256 &row1, __m256 &row2, __m256 &row3, __m256 &row4, __m256 &row5, __m256 &row6, __m256 &row7)
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

} // namespace zimg

#endif // ZIMG_X86_AVX_UTIL_H_

#endif // ZIMG_X86

