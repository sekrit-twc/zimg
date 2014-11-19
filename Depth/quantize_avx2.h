#pragma once

#ifdef ZIMG_X86

#ifndef ZIMG_DEPTH_QUANTIZE_AVX2_H_
#define ZIMG_DEPTH_QUANTIZE_AVX2_H_

#include <immintrin.h>
#include "Common/osdep.h"
#include "Common/pixel.h"
#include "quantize.h"

namespace zimg {;
namespace depth {;

inline FORCE_INLINE __m256 half_to_float_avx2(__m128i x)
{
	return _mm256_cvtph_ps(x);
}

inline FORCE_INLINE __m128i float_to_half_avx2(__m256 x)
{
	return _mm256_cvtps_ph(x, 0);
}

struct UnpackByteAVX2 {
	static const int loop_step = 16;
	static const int unpacked_count = 2;

	typedef __m256i type;

	FORCE_INLINE void unpack(__m256i dst[unpacked_count], const uint8_t *ptr) const
	{
		__m256i zero = _mm256_setzero_si256();
		__m256i perm = _mm256_set_epi32(7, 7, 3, 1, 7, 7, 2, 0);

		__m256i x = _mm256_castsi128_si256(_mm_load_si128((const __m128i *)ptr));
		x = _mm256_permutevar8x32_epi32(x, perm);
		x = _mm256_unpacklo_epi8(x, zero);

		dst[0] = _mm256_unpacklo_epi16(x, zero);
		dst[1] = _mm256_unpackhi_epi16(x, zero);
	}
};

struct UnpackWordAVX2 {
	static const int loop_step = 8;
	static const int unpacked_count = 1;

	typedef __m256i type;

	FORCE_INLINE void unpack(__m256i dst[unpacked_count], const uint16_t *ptr) const
	{
		__m256i zero = _mm256_setzero_si256();
		__m256i perm = _mm256_set_epi32(7, 7, 3, 2, 7, 7, 1, 0);

		__m256i x = _mm256_castsi128_si256(_mm_load_si128((const __m128i *)ptr));
		x = _mm256_permutevar8x32_epi32(x, perm);
		x = _mm256_unpacklo_epi16(x, zero);

		dst[0] = x;
	}
};

struct UnpackHalfAVX2 {
	static const int loop_step = 8;
	static const int unpacked_count = 1;

	typedef __m128i type;

	FORCE_INLINE void unpack(__m128i dst[unpacked_count], const uint16_t *ptr) const
	{
		dst[0] = _mm_load_si128((const __m128i *)ptr);
	}
};

struct UnpackFloatAVX2 {
	static const int loop_step = 8;
	static const int unpacked_count = 1;

	typedef __m256 type;

	FORCE_INLINE void unpack(__m256 dst[unpacked_count], const float *ptr) const
	{
		dst[0] = _mm256_load_ps(ptr);
	}
};

struct PackByteAVX2 {
	static const int loop_step = 16;
	static const int unpacked_count = 2;

	typedef __m256i type;

	FORCE_INLINE void pack(uint8_t *ptr, const __m256i src[unpacked_count]) const
	{
		__m256i p = _mm256_set_epi32(7, 7, 7, 7, 5, 1, 4, 0);
		__m256i x;
		
		x = _mm256_packus_epi32(src[0], src[1]);
		x = _mm256_packus_epi16(x, x);
		x = _mm256_permutevar8x32_epi32(x, p);

		_mm_store_si128((__m128i *)ptr, _mm256_castsi256_si128(x));
	}
};

struct PackWordAVX2 {
	static const int loop_step = 8;
	static const int unpacked_count = 1;

	typedef __m256i type;

	FORCE_INLINE void pack(uint16_t *ptr, const __m256i src[unpacked_count]) const
	{
		__m256i p = _mm256_set_epi32(7, 7, 7, 7, 5, 4, 1, 0);
		__m256i x;

		x = src[0];
		x = _mm256_packus_epi32(x, x);
		x = _mm256_permutevar8x32_epi32(x, p);

		_mm_store_si128((__m128i *)ptr, _mm256_castsi256_si128(x));
	}
};

struct PackHalfAVX2 {
	static const int loop_step = 8;
	static const int unpacked_count = 1;

	typedef __m128i type;

	FORCE_INLINE void pack(uint16_t *ptr, const __m128i src[unpacked_count]) const
	{
		_mm_store_si128((__m128i *)ptr, src[0]);
	}
};

struct PackFloatAVX2 {
	static const int loop_step = 8;
	static const int unpacked_count = 1;

	typedef __m256 type;

	FORCE_INLINE void pack(float *ptr, const __m256 src[unpacked_count]) const
	{
		_mm256_store_ps(ptr, src[0]);
	}
};

class IntegerToFloatAVX2 {
	__m256 offset;
	__m256 scale;
public:
	IntegerToFloatAVX2(int bits, bool fullrange, bool chroma)
	{
		float offset_ = (float)integer_offset(bits, fullrange, chroma);
		float scale_ = (float)integer_range(bits, fullrange, chroma);

		offset = _mm256_set1_ps(-offset_ / scale_);
		scale = _mm256_set1_ps(1.0f / scale_);
	}

	FORCE_INLINE __m256 operator()(__m256i x) const
	{
		return _mm256_fmadd_ps(_mm256_cvtepi32_ps(x), scale, offset);
	}
};

class FloatToIntegerAVX2 {
	__m256 offset;
	__m256 scale;
public:
	FloatToIntegerAVX2(int bits, bool fullrange, bool chroma)
	{
		offset = _mm256_set1_ps((float)integer_offset(bits, fullrange, chroma));
		scale = _mm256_set1_ps((float)integer_range(bits, fullrange, chroma));
	}

	FORCE_INLINE __m256i operator()(__m256 x) const
	{
		return _mm256_cvtps_epi32(_mm256_fmadd_ps(x, scale, offset));
	}
};

inline IntegerToFloatAVX2 make_integer_to_float_avx2(const PixelFormat &fmt)
{
	return{ fmt.depth, fmt.fullrange, fmt.chroma };
}

inline FloatToIntegerAVX2 make_float_to_integer_avx2(const PixelFormat &fmt)
{
	return{ fmt.depth, fmt.fullrange, fmt.chroma };
}

} // namespace depth
} // namespace zimg

#endif // ZIMG_DEPTH_QUANTIZE_AVX2_H_

#endif // ZIMG_X86
