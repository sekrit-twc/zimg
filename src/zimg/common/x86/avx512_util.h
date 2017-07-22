#pragma once

#ifdef ZIMG_X86_AVX512

#ifndef ZIMG_X86_AVX512_UTIL_H_
#define ZIMG_X86_AVX512_UTIL_H_

#include "common/ccdep.h"
#include "x86util.h"

namespace zimg {

namespace _avx512 {

// Transpose two 8x8 matrices stored in the lower and upper 256-bit lanes of [row0]-[row7].
static inline FORCE_INLINE void mm512_transpose8_x2_ps(__m512 &row0, __m512 &row1, __m512 &row2, __m512 &row3,
                                                       __m512 &row4, __m512 &row5, __m512 &row6, __m512 &row7)
{
	__m512 t0, t1, t2, t3, t4, t5, t6, t7;
	__m512 tt0, tt1, tt2, tt3, tt4, tt5, tt6, tt7;

	t0 = _mm512_unpacklo_ps(row0, row1);
	t1 = _mm512_unpackhi_ps(row0, row1);
	t2 = _mm512_unpacklo_ps(row2, row3);
	t3 = _mm512_unpackhi_ps(row2, row3);
	t4 = _mm512_unpacklo_ps(row4, row5);
	t5 = _mm512_unpackhi_ps(row4, row5);
	t6 = _mm512_unpacklo_ps(row6, row7);
	t7 = _mm512_unpackhi_ps(row6, row7);

	tt0 = _mm512_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
	tt1 = _mm512_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
	tt2 = _mm512_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
	tt3 = _mm512_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
	tt4 = _mm512_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
	tt5 = _mm512_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
	tt6 = _mm512_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
	tt7 = _mm512_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));

	row0 = _mm512_shuffle_f32x4(tt0, tt4, 0x88);
	row1 = _mm512_shuffle_f32x4(tt1, tt5, 0x88);
	row2 = _mm512_shuffle_f32x4(tt2, tt6, 0x88);
	row3 = _mm512_shuffle_f32x4(tt3, tt7, 0x88);
	row4 = _mm512_shuffle_f32x4(tt0, tt4, 0xdd);
	row5 = _mm512_shuffle_f32x4(tt1, tt5, 0xdd);
	row6 = _mm512_shuffle_f32x4(tt2, tt6, 0xdd);
	row7 = _mm512_shuffle_f32x4(tt3, tt7, 0xdd);
}

// Exchange the upper 256-bit lane of [row0] with the lower 256-bit lane of [row1].
static inline FORCE_INLINE void mm512_exchange_lanes_ps256(__m512 &row0, __m512 &row1)
{
	__m512 tmp0 = _mm512_shuffle_f32x4(row0, row1, 0x88);
	__m512 tmp1 = _mm512_shuffle_f32x4(row0, row1, 0xdd);
	row0 = tmp0;
	row1 = tmp1;
}

// Transpose four 8x8 matrices stored in the 128-bit lanes of [row0]-[row7].
static inline FORCE_INLINE void mm512_transpose8_x4_epi16(__m512i &row0, __m512i &row1, __m512i &row2, __m512i &row3,
                                                          __m512i &row4, __m512i &row5, __m512i &row6, __m512i &row7)
{
	__m512i t0, t1, t2, t3, t4, t5, t6, t7;
	__m512i tt0, tt1, tt2, tt3, tt4, tt5, tt6, tt7;

	t0 = _mm512_unpacklo_epi16(row0, row1);
	t1 = _mm512_unpacklo_epi16(row2, row3);
	t2 = _mm512_unpacklo_epi16(row4, row5);
	t3 = _mm512_unpacklo_epi16(row6, row7);
	t4 = _mm512_unpackhi_epi16(row0, row1);
	t5 = _mm512_unpackhi_epi16(row2, row3);
	t6 = _mm512_unpackhi_epi16(row4, row5);
	t7 = _mm512_unpackhi_epi16(row6, row7);

	tt0 = _mm512_unpacklo_epi32(t0, t1);
	tt1 = _mm512_unpackhi_epi32(t0, t1);
	tt2 = _mm512_unpacklo_epi32(t2, t3);
	tt3 = _mm512_unpackhi_epi32(t2, t3);
	tt4 = _mm512_unpacklo_epi32(t4, t5);
	tt5 = _mm512_unpackhi_epi32(t4, t5);
	tt6 = _mm512_unpacklo_epi32(t6, t7);
	tt7 = _mm512_unpackhi_epi32(t6, t7);

	row0 = _mm512_unpacklo_epi64(tt0, tt2);
	row1 = _mm512_unpackhi_epi64(tt0, tt2);
	row2 = _mm512_unpacklo_epi64(tt1, tt3);
	row3 = _mm512_unpackhi_epi64(tt1, tt3);
	row4 = _mm512_unpacklo_epi64(tt4, tt6);
	row5 = _mm512_unpackhi_epi64(tt4, tt6);
	row6 = _mm512_unpacklo_epi64(tt5, tt7);
	row7 = _mm512_unpackhi_epi64(tt5, tt7);
}

// Transpose the 4x4 matrix stored in [row0]-[row3].
static inline FORCE_INLINE void mm512_transpose4_si128(__m512i &row0, __m512i &row1, __m512i &row2, __m512i &row3)
{
	__m512i t0, t1, t2, t3;

	t0 = _mm512_shuffle_i32x4(row0, row1, 0x88);
	t1 = _mm512_shuffle_i32x4(row0, row1, 0xdd);
	t2 = _mm512_shuffle_i32x4(row2, row3, 0x88);
	t3 = _mm512_shuffle_i32x4(row2, row3, 0xdd);

	row0 = _mm512_shuffle_i32x4(t0, t2, 0x88);
	row1 = _mm512_shuffle_i32x4(t1, t3, 0x88);
	row2 = _mm512_shuffle_i32x4(t0, t2, 0xdd);
	row3 = _mm512_shuffle_i32x4(t1, t3, 0xdd);
}

} // namespace _avx512


// Return mask with lower n bits set to 1.
static inline FORCE_INLINE __mmask16 mmask16_set_lo(unsigned n)
{
	return 0xFFFFU >> (16 - n);
}

// Return mask with upper n bits set to 1.
static inline FORCE_INLINE __mmask16 mmask16_set_hi(unsigned n)
{
	return 0xFFFFU << (16 - n);
}

// Return mask with lower n bits set to 1.
static inline FORCE_INLINE __mmask32 mmask32_set_lo(unsigned n)
{
	return 0xFFFFFFFFU >> (32 - n);
}

// Return mask with upper n bits set to 1.
static inline FORCE_INLINE __mmask32 mmask32_set_hi(unsigned n)
{
	return 0xFFFFFFFFU << (32 - n);
}

// Transpose in-place the 16x16 matrix stored in [row0]-[row15].
static inline FORCE_INLINE void mm512_transpose16_ps(__m512 &row0, __m512 &row1, __m512 &row2, __m512 &row3,
                                                     __m512 &row4, __m512 &row5, __m512 &row6, __m512 &row7,
                                                     __m512 &row8, __m512 &row9, __m512 &row10, __m512 &row11,
                                                     __m512 &row12, __m512 &row13, __m512 &row14, __m512 &row15)
{
	_avx512::mm512_transpose8_x2_ps(row0, row1, row2, row3, row4, row5, row6, row7);
	_avx512::mm512_transpose8_x2_ps(row8, row9, row10, row11, row12, row13, row14, row15);

	_avx512::mm512_exchange_lanes_ps256(row0, row8);
	_avx512::mm512_exchange_lanes_ps256(row1, row9);
	_avx512::mm512_exchange_lanes_ps256(row2, row10);
	_avx512::mm512_exchange_lanes_ps256(row3, row11);
	_avx512::mm512_exchange_lanes_ps256(row4, row12);
	_avx512::mm512_exchange_lanes_ps256(row5, row13);
	_avx512::mm512_exchange_lanes_ps256(row6, row14);
	_avx512::mm512_exchange_lanes_ps256(row7, row15);
}

// Transpose in-place the 32x32 matrix stored in [row0]-[row31].
static inline FORCE_INLINE void mm512_transpose32_epi16(
	__m512i &row0, __m512i &row1, __m512i &row2, __m512i &row3, __m512i &row4, __m512i &row5, __m512i &row6, __m512i &row7,
	__m512i &row8, __m512i &row9, __m512i &row10, __m512i &row11, __m512i &row12, __m512i &row13, __m512i &row14, __m512i &row15,
	__m512i &row16, __m512i &row17, __m512i &row18, __m512i &row19, __m512i &row20, __m512i &row21, __m512i &row22, __m512i &row23,
	__m512i &row24, __m512i &row25, __m512i &row26, __m512i &row27, __m512i &row28, __m512i &row29, __m512i &row30, __m512i &row31)
{
	_avx512::mm512_transpose8_x4_epi16(row0, row1, row2, row3, row4, row5, row6, row7);
	_avx512::mm512_transpose8_x4_epi16(row8, row9, row10, row11, row12, row13, row14, row15);
	_avx512::mm512_transpose8_x4_epi16(row16, row17, row18, row19, row20, row21, row22, row23);
	_avx512::mm512_transpose8_x4_epi16(row24, row25, row26, row27, row28, row29, row30, row31);

	_avx512::mm512_transpose4_si128(row0, row8, row16, row24);
	_avx512::mm512_transpose4_si128(row1, row9, row17, row25);
	_avx512::mm512_transpose4_si128(row2, row10, row18, row26);
	_avx512::mm512_transpose4_si128(row3, row11, row19, row27);
	_avx512::mm512_transpose4_si128(row4, row12, row20, row28);
	_avx512::mm512_transpose4_si128(row5, row13, row21, row29);
	_avx512::mm512_transpose4_si128(row6, row14, row22, row30);
	_avx512::mm512_transpose4_si128(row7, row15, row23, row31);
}

} // namespace zimg

#endif // ZIMG_X86_AVX512_UTIL_H_

#endif // ZIMG_X86_AVX512
