#pragma once

#ifdef ZIMG_X86

#ifndef ZIMG_X86UTIL_H_
#define ZIMG_X86UTIL_H_

#include <cstdint>
#include "ccdep.h"

namespace zimg {;

extern const uint8_t xmm_mask_table_l alignas(16)[17][16];
extern const uint8_t xmm_mask_table_r alignas(16)[17][16];

extern const uint8_t ymm_mask_table_l alignas(32)[33][32];
extern const uint8_t ymm_mask_table_r alignas(32)[33][32];

#ifdef HAVE_CPU_SSE

// Store the right-most [count] bytes from [x] into [dst].
inline FORCE_INLINE void mm_store_left_ps(float *dst, __m128 x, unsigned count)
{
	__m128 orig = _mm_load_ps(dst);
	__m128 mask = _mm_load_ps((const float *)(&xmm_mask_table_l[count]));

	orig = _mm_andnot_ps(mask, orig);
	x = _mm_and_ps(mask, x);
	x = _mm_or_ps(x, orig);

	_mm_store_ps(dst, x);
}

// Store the left-most [count] bytes from [x] into [dst].
inline FORCE_INLINE void mm_store_right_ps(float *dst, __m128 x, unsigned count)
{
	__m128 orig = _mm_load_ps(dst);
	__m128 mask = _mm_load_ps((const float *)(&xmm_mask_table_r[count]));

	orig = _mm_andnot_ps(mask, orig);
	x = _mm_and_ps(mask, x);
	x = _mm_or_ps(x, orig);

	_mm_store_ps(dst, x);
}

#endif // HAVE_CPU_SSE

#ifdef HAVE_CPU_SSE2

// Store the right-most [count] bytes from [x] into [dst].
inline FORCE_INLINE void mm_store_left_si128(__m128i *dst, __m128i x, unsigned count)
{
	__m128i orig = _mm_load_si128(dst);
	__m128i mask = _mm_load_si128((const __m128i *)(&xmm_mask_table_l[count]));

	orig = _mm_andnot_si128(mask, orig);
	x = _mm_and_si128(mask, x);
	x = _mm_or_si128(x, orig);

	_mm_store_si128(dst, x);
}

// Store the left-most [count] bytes from [x] into [dst].
inline FORCE_INLINE void mm_store_right_si128(__m128i *dst, __m128i x, unsigned count)
{
	__m128i orig = _mm_load_si128(dst);
	__m128i mask = _mm_load_si128((const __m128i *)(&xmm_mask_table_r[count]));

	orig = _mm_andnot_si128(mask, orig);
	x = _mm_and_si128(mask, x);
	x = _mm_or_si128(x, orig);

	_mm_store_si128(dst, x);
}

#endif // HAVE_CPU_SSE2

#ifdef HAVE_CPU_AVX

// Store the right-most [count] bytes from [x] into [dst].
inline FORCE_INLINE void mm256_store_left_ps(float *dst, __m256 x, unsigned count)
{
	__m256 orig = _mm256_load_ps(dst);
	__m256 mask = _mm256_load_ps((const float *)(&ymm_mask_table_l[count]));

	orig = _mm256_andnot_ps(mask, orig);
	x = _mm256_and_ps(mask, x);
	x = _mm256_or_ps(x, orig);

	_mm256_store_ps(dst, x);
}

// Store the left-most [count] bytes from [x] into [dst].
inline FORCE_INLINE void mm256_store_right_ps(float *dst, __m256 x, unsigned count)
{
	__m256 orig = _mm256_load_ps(dst);
	__m256 mask = _mm256_load_ps((const float *)(&ymm_mask_table_r[count]));

	orig = _mm256_andnot_ps(mask, orig);
	x = _mm256_and_ps(mask, x);
	x = _mm256_or_ps(x, orig);

	_mm256_store_ps(dst, x);
}

#endif // HAVE_CPU_AVX

} // namespace zimg

#endif // ZIMG_X86UTIL_H_

#endif // ZIMG_X86
