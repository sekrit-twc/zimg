#pragma once

#ifdef ZIMG_X86

#ifndef ZIMG_DEPTH_QUANTIZE_SSE2_H_
#define ZIMG_DEPTH_QUANTIZE_SSE2_H_

#include <emmintrin.h>
#include "Common/osdep.h"
#include "Common/pixel.h"
#include "quantize.h"

namespace zimg {;
namespace depth {;

inline FORCE_INLINE __m128i blend_sse2(__m128i a, __m128i b, __m128i mask)
{
	a = _mm_and_si128(mask, a);
	b = _mm_andnot_si128(mask, b);

	return _mm_or_si128(a, b);
}

inline FORCE_INLINE __m128i min_epi32_sse2(__m128i a, __m128i b)
{
	__m128i mask = _mm_cmplt_epi32(a, b);
	return blend_sse2(a, b, mask);
}

inline FORCE_INLINE __m128i packus_epi32_sse2(__m128i a, __m128i b)
{
	a = _mm_slli_epi32(a, 16);
	a = _mm_srai_epi32(a, 16);

	b = _mm_slli_epi32(b, 16);
	b = _mm_srai_epi32(b, 16);

	return _mm_packs_epi32(a, b);
}

inline FORCE_INLINE __m128 half_to_float_sse2(__m128i x)
{
	__m128 magic = _mm_castsi128_ps(_mm_set1_epi32((uint32_t)113 << 23));
	__m128i shift_exp = _mm_set1_epi32(0x7C00UL << 13);
	__m128i sign_mask = _mm_set1_epi32(0x8000U);
	__m128i mant_mask = _mm_set1_epi32(0x7FFF);
	__m128i exp_adjust = _mm_set1_epi32((127UL - 15UL) << 23);
	__m128i exp_adjust_nan = _mm_set1_epi32((127UL - 16UL) << 23);
	__m128i exp_adjust_denorm = _mm_set1_epi32(1UL << 23);
	__m128i zero = _mm_set1_epi16(0);

	__m128i exp, ret, ret_nan, ret_denorm, sign, mask0, mask1;

	ret = _mm_and_si128(x, mant_mask);
	ret = _mm_slli_epi32(ret, 13);
	exp = _mm_and_si128(shift_exp, ret);
	ret = _mm_add_epi32(ret, exp_adjust);

	mask0 = _mm_cmpeq_epi32(exp, shift_exp);
	mask1 = _mm_cmpeq_epi32(exp, zero);

	ret_nan = _mm_add_epi32(ret, exp_adjust_nan);
	ret_denorm = _mm_add_epi32(ret, exp_adjust_denorm);
	ret_denorm = _mm_castps_si128(_mm_sub_ps(_mm_castsi128_ps(ret_denorm), magic));

	sign = _mm_and_si128(x, sign_mask);
	sign = _mm_slli_epi32(sign, 16);

	ret = blend_sse2(ret_nan, ret, mask0);
	ret = blend_sse2(ret_denorm, ret, mask1);

	ret = _mm_or_si128(ret, sign);
	return _mm_castsi128_ps(ret);
}

inline FORCE_INLINE __m128i float_to_half_sse2(__m128 x)
{
	__m128 magic = _mm_castsi128_ps(_mm_set1_epi32((uint32_t)15 << 23));
	__m128i inf = _mm_set1_epi32((uint32_t)255UL << 23);
	__m128i f16inf = _mm_set1_epi32((uint32_t)31UL << 23);
	__m128i sign_mask = _mm_set1_epi32(0x80000000UL);
	__m128i round_mask = _mm_set1_epi32(~0x0FFFU);

	__m128i ret_0x7E00 = _mm_set1_epi32(0x7E00);
	__m128i ret_0x7C00 = _mm_set1_epi32(0x7C00);

	__m128i f, sign, ge_inf, eq_inf;

	f = _mm_castps_si128(x);
	sign = _mm_and_si128(f, sign_mask);
	f = _mm_xor_si128(f, sign);

	ge_inf = _mm_cmpgt_epi32(f, inf);
	eq_inf = _mm_cmpeq_epi32(f, inf);

	f = _mm_and_si128(f, round_mask);
	f = _mm_castps_si128(_mm_mul_ps(_mm_castsi128_ps(f), magic));
	f = _mm_sub_epi32(f, round_mask);

	f = min_epi32_sse2(f, f16inf);
	f = _mm_srli_epi32(f, 13);

	f = blend_sse2(ret_0x7E00, f, ge_inf);
	f = blend_sse2(ret_0x7C00, f, eq_inf);

	sign = _mm_srli_epi32(sign, 16);
	f = _mm_or_si128(f, sign);

	return f;
}

struct UnpackByteSSE2 {
	static const int loop_step = 16;
	static const int unpacked_count = 4;

	typedef __m128i type;

	FORCE_INLINE void unpack(__m128i dst[unpacked_count], const uint8_t *ptr) const
	{
		__m128i zero = _mm_setzero_si128();
		__m128i x = _mm_load_si128((const __m128i *)ptr);

		__m128i lo_w = _mm_unpacklo_epi8(x, zero);
		__m128i hi_w = _mm_unpackhi_epi8(x, zero);

		dst[0] = _mm_unpacklo_epi16(lo_w, zero);
		dst[1] = _mm_unpackhi_epi16(lo_w, zero);
		dst[2] = _mm_unpacklo_epi16(hi_w, zero);
		dst[3] = _mm_unpackhi_epi16(hi_w, zero);
	}
};

struct UnpackWordSSE2 {
	static const int loop_step = 8;
	static const int unpacked_count = 2;

	typedef __m128i type;

	FORCE_INLINE void unpack(__m128i dst[unpacked_count], const uint16_t *ptr) const
	{
		__m128i zero = _mm_setzero_si128();
		__m128i x = _mm_load_si128((const __m128i *)ptr);

		dst[0] = _mm_unpacklo_epi16(x, zero);
		dst[1] = _mm_unpackhi_epi16(x, zero);
	}
};

struct UnpackFloatSSE2 {
	static const int loop_step = 4;
	static const int unpacked_count = 1;

	typedef __m128 type;

	FORCE_INLINE void unpack(__m128 dst[unpacked_count], const float *ptr) const
	{
		dst[0] = _mm_load_ps(ptr);
	}
};

struct PackByteSSE2 {
	static const int loop_step = 16;
	static const int unpacked_count = 4;

	typedef __m128i type;

	FORCE_INLINE void pack(uint8_t *ptr, const __m128i src[unpacked_count]) const
	{
		__m128i lo = packus_epi32_sse2(src[0], src[1]);
		__m128i hi = packus_epi32_sse2(src[2], src[3]);
		__m128i x = _mm_packus_epi16(lo, hi);

		_mm_store_si128((__m128i *)ptr, x);
	}
};

struct PackWordSSE2 {
	static const int loop_step = 8;
	static const int unpacked_count = 2;

	typedef __m128i type;

	FORCE_INLINE void pack(uint16_t *ptr, const __m128i src[unpacked_count]) const
	{
		__m128i x = packus_epi32_sse2(src[0], src[1]);
		_mm_store_si128((__m128i *)ptr, x);
	}
};

struct PackFloatSSE2 {
	static const int loop_step = 4;
	static const int unpacked_count = 1;

	typedef __m128 type;

	FORCE_INLINE void pack(float *ptr, const __m128 src[unpacked_count]) const
	{
		_mm_store_ps(ptr, src[0]);
	}
};

class IntegerToFloatSSE2 {
	float offset;
	float scale;
public:
	IntegerToFloatSSE2(int bits, bool fullrange, bool chroma)
	{
		float offset_ = (float)integer_offset(bits, fullrange, chroma);
		float scale_ = (float)integer_range(bits, fullrange, chroma);

		offset = -offset_ / scale_;
		scale = 1.0f / scale_;
	}

	FORCE_INLINE __m128 operator()(__m128i x) const
	{
		__m128 s = _mm_set_ps1(scale);
		__m128 o = _mm_set_ps1(offset);
		__m128 f = _mm_cvtepi32_ps(x);
		f = _mm_mul_ps(f, s);
		f = _mm_add_ps(f, o);
		return f;
	}
};

class FloatToIntegerSSE2 {
	float offset;
	float scale;
public:
	FloatToIntegerSSE2(int bits, bool fullrange, bool chroma)
	{
		offset = (float)integer_offset(bits, fullrange, chroma);
		scale = (float)integer_range(bits, fullrange, chroma);
	}

	FORCE_INLINE __m128i operator()(__m128 x) const
	{
		__m128 s = _mm_set_ps1(scale);
		__m128 o = _mm_set_ps1(offset);
		x = _mm_mul_ps(x, s);
		x = _mm_add_ps(x, o);
		return _mm_cvtps_epi32(x);
	}
};

inline IntegerToFloatSSE2 make_integer_to_float_sse2(const PixelFormat &fmt)
{
	return{ fmt.depth, fmt.fullrange, fmt.chroma };
}

inline FloatToIntegerSSE2 make_float_to_integer_sse2(const PixelFormat &fmt)
{
	return{ fmt.depth, fmt.fullrange, fmt.chroma };
}

} // namespace depth
} // namespace zimg

#endif // ZIMG_DEPTH_QUANTIZE_SSE2_H_

#endif // ZIMG_X86
