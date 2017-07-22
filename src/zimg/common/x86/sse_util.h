#pragma once

#ifdef ZIMG_X86

#ifndef ZIMG_X86_SSE_UTIL_H_
#define ZIMG_X86_SSE_UTIL_H_

#include "common/ccdep.h"
#include "x86util.h"

namespace zimg {

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

// Stores the elements of [x] into [dst0]-[dst3].
static inline FORCE_INLINE void mm_scatter_ps(float *dst0, float *dst1, float *dst2, float *dst3, __m128 x)
{
	_mm_store_ss(dst0, x);
	_mm_store_ss(dst1, _mm_shuffle_ps(x, x, _MM_SHUFFLE(3, 2, 1, 1)));
	_mm_store_ss(dst2, _mm_shuffle_ps(x, x, _MM_SHUFFLE(3, 2, 1, 2)));
	_mm_store_ss(dst3, _mm_shuffle_ps(x, x, _MM_SHUFFLE(3, 2, 1, 3)));
}

} // namespace zimg

#endif // ZIMG_X86_SSE_UTIL_H_

#endif // ZIMG_X86
