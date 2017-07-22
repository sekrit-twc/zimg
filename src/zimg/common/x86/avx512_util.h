#pragma once

#ifdef ZIMG_X86_AVX512

#ifndef ZIMG_X86_AVX512_UTIL_H_
#define ZIMG_X86_AVX512_UTIL_H_

#include "common/ccdep.h"
#include "x86util.h"

namespace zimg {

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
	__m512 t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15;
	__m512 tt0, tt1, tt2, tt3, tt4, tt5, tt6, tt7, tt8, tt9, tt10, tt11, tt12, tt13, tt14, tt15;

	t0 = _mm512_unpacklo_ps(row0, row1);
	t1 = _mm512_unpackhi_ps(row0, row1);
	t2 = _mm512_unpacklo_ps(row2, row3);
	t3 = _mm512_unpackhi_ps(row2, row3);
	t4 = _mm512_unpacklo_ps(row4, row5);
	t5 = _mm512_unpackhi_ps(row4, row5);
	t6 = _mm512_unpacklo_ps(row6, row7);
	t7 = _mm512_unpackhi_ps(row6, row7);
	t8 = _mm512_unpacklo_ps(row8, row9);
	t9 = _mm512_unpackhi_ps(row8, row9);
	t10 = _mm512_unpacklo_ps(row10, row11);
	t11 = _mm512_unpackhi_ps(row10, row11);
	t12 = _mm512_unpacklo_ps(row12, row13);
	t13 = _mm512_unpackhi_ps(row12, row13);
	t14 = _mm512_unpacklo_ps(row14, row15);
	t15 = _mm512_unpackhi_ps(row14, row15);

	tt0 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t0), _mm512_castps_pd(t2)));
	tt1 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t0), _mm512_castps_pd(t2)));
	tt2 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t1), _mm512_castps_pd(t3)));
	tt3 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t1), _mm512_castps_pd(t3)));
	tt4 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t4), _mm512_castps_pd(t6)));
	tt5 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t4), _mm512_castps_pd(t6)));
	tt6 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t5), _mm512_castps_pd(t7)));
	tt7 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t5), _mm512_castps_pd(t7)));
	tt8 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t8), _mm512_castps_pd(t10)));
	tt9 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t8), _mm512_castps_pd(t10)));
	tt10 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t9), _mm512_castps_pd(t11)));
	tt11 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t9), _mm512_castps_pd(t11)));
	tt12 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t12), _mm512_castps_pd(t14)));
	tt13 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t12), _mm512_castps_pd(t14)));
	tt14 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t13), _mm512_castps_pd(t15)));
	tt15 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t13), _mm512_castps_pd(t15)));

	t0 = _mm512_shuffle_f32x4(tt0, tt4, 0x88);
	t1 = _mm512_shuffle_f32x4(tt1, tt5, 0x88);
	t2 = _mm512_shuffle_f32x4(tt2, tt6, 0x88);
	t3 = _mm512_shuffle_f32x4(tt3, tt7, 0x88);
	t4 = _mm512_shuffle_f32x4(tt0, tt4, 0xdd);
	t5 = _mm512_shuffle_f32x4(tt1, tt5, 0xdd);
	t6 = _mm512_shuffle_f32x4(tt2, tt6, 0xdd);
	t7 = _mm512_shuffle_f32x4(tt3, tt7, 0xdd);
	t8 = _mm512_shuffle_f32x4(tt8, tt12, 0x88);
	t9 = _mm512_shuffle_f32x4(tt9, tt13, 0x88);
	t10 = _mm512_shuffle_f32x4(tt10, tt14, 0x88);
	t11 = _mm512_shuffle_f32x4(tt11, tt15, 0x88);
	t12 = _mm512_shuffle_f32x4(tt8, tt12, 0xdd);
	t13 = _mm512_shuffle_f32x4(tt9, tt13, 0xdd);
	t14 = _mm512_shuffle_f32x4(tt10, tt14, 0xdd);
	t15 = _mm512_shuffle_f32x4(tt11, tt15, 0xdd);

	row0 = _mm512_shuffle_f32x4(t0, t8, 0x88);
	row1 = _mm512_shuffle_f32x4(t1, t9, 0x88);
	row2 = _mm512_shuffle_f32x4(t2, t10, 0x88);
	row3 = _mm512_shuffle_f32x4(t3, t11, 0x88);
	row4 = _mm512_shuffle_f32x4(t4, t12, 0x88);
	row5 = _mm512_shuffle_f32x4(t5, t13, 0x88);
	row6 = _mm512_shuffle_f32x4(t6, t14, 0x88);
	row7 = _mm512_shuffle_f32x4(t7, t15, 0x88);
	row8 = _mm512_shuffle_f32x4(t0, t8, 0xdd);
	row9 = _mm512_shuffle_f32x4(t1, t9, 0xdd);
	row10 = _mm512_shuffle_f32x4(t2, t10, 0xdd);
	row11 = _mm512_shuffle_f32x4(t3, t11, 0xdd);
	row12 = _mm512_shuffle_f32x4(t4, t12, 0xdd);
	row13 = _mm512_shuffle_f32x4(t5, t13, 0xdd);
	row14 = _mm512_shuffle_f32x4(t6, t14, 0xdd);
	row15 = _mm512_shuffle_f32x4(t7, t15, 0xdd);
}

// Transpose in-place the 32x32 matrix stored in [row0]-[row31].
static inline FORCE_INLINE void mm512_transpose32_epi16(__m512i &row0, __m512i &row1, __m512i &row2, __m512i &row3, __m512i &row4, __m512i &row5, __m512i &row6, __m512i &row7,
                                                        __m512i &row8, __m512i &row9, __m512i &row10, __m512i &row11, __m512i &row12, __m512i &row13, __m512i &row14, __m512i &row15,
                                                        __m512i &row16, __m512i &row17, __m512i &row18, __m512i &row19, __m512i &row20, __m512i &row21, __m512i &row22, __m512i &row23,
                                                        __m512i &row24, __m512i &row25, __m512i &row26, __m512i &row27, __m512i &row28, __m512i &row29, __m512i &row30, __m512i &row31)
{
	__m512i t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15;
	__m512i t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27, t28, t29, t30, t31;
	__m512i tt0, tt1, tt2, tt3, tt4, tt5, tt6, tt7, tt8, tt9, tt10, tt11, tt12, tt13, tt14, tt15;
	__m512i tt16, tt17, tt18, tt19, tt20, tt21, tt22, tt23, tt24, tt25, tt26, tt27, tt28, tt29, tt30, tt31;

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

	t0 = _mm512_unpacklo_epi64(tt0, tt2);
	t1 = _mm512_unpackhi_epi64(tt0, tt2);
	t2 = _mm512_unpacklo_epi64(tt1, tt3);
	t3 = _mm512_unpackhi_epi64(tt1, tt3);
	t4 = _mm512_unpacklo_epi64(tt4, tt6);
	t5 = _mm512_unpackhi_epi64(tt4, tt6);
	t6 = _mm512_unpacklo_epi64(tt5, tt7);
	t7 = _mm512_unpackhi_epi64(tt5, tt7);

	t8 = _mm512_unpacklo_epi16(row8, row9);
	t9 = _mm512_unpacklo_epi16(row10, row11);
	t10 = _mm512_unpacklo_epi16(row12, row13);
	t11 = _mm512_unpacklo_epi16(row14, row15);
	t12 = _mm512_unpackhi_epi16(row8, row9);
	t13 = _mm512_unpackhi_epi16(row10, row11);
	t14 = _mm512_unpackhi_epi16(row12, row13);
	t15 = _mm512_unpackhi_epi16(row14, row15);

	tt8 = _mm512_unpacklo_epi32(t8, t9);
	tt9 = _mm512_unpackhi_epi32(t8, t9);
	tt10 = _mm512_unpacklo_epi32(t10, t11);
	tt11 = _mm512_unpackhi_epi32(t10, t11);
	tt12 = _mm512_unpacklo_epi32(t12, t13);
	tt13 = _mm512_unpackhi_epi32(t12, t13);
	tt14 = _mm512_unpacklo_epi32(t14, t15);
	tt15 = _mm512_unpackhi_epi32(t14, t15);

	t8 = _mm512_unpacklo_epi64(tt8, tt10);
	t9 = _mm512_unpackhi_epi64(tt8, tt10);
	t10 = _mm512_unpacklo_epi64(tt9, tt11);
	t11 = _mm512_unpackhi_epi64(tt9, tt11);
	t12 = _mm512_unpacklo_epi64(tt12, tt14);
	t13 = _mm512_unpackhi_epi64(tt12, tt14);
	t14 = _mm512_unpacklo_epi64(tt13, tt15);
	t15 = _mm512_unpackhi_epi64(tt13, tt15);

	tt0 = _mm512_shuffle_i32x4(t0, t8, 0x88);
	tt1 = _mm512_shuffle_i32x4(t1, t9, 0x88);
	tt2 = _mm512_shuffle_i32x4(t2, t10, 0x88);
	tt3 = _mm512_shuffle_i32x4(t3, t11, 0x88);
	tt4 = _mm512_shuffle_i32x4(t4, t12, 0x88);
	tt5 = _mm512_shuffle_i32x4(t5, t13, 0x88);
	tt6 = _mm512_shuffle_i32x4(t6, t14, 0x88);
	tt7 = _mm512_shuffle_i32x4(t7, t15, 0x88);

	tt8 = _mm512_shuffle_i32x4(t0, t8, 0xdd);
	tt9 = _mm512_shuffle_i32x4(t1, t9, 0xdd);
	tt10 = _mm512_shuffle_i32x4(t2, t10, 0xdd);
	tt11 = _mm512_shuffle_i32x4(t3, t11, 0xdd);
	tt12 = _mm512_shuffle_i32x4(t4, t12, 0xdd);
	tt13 = _mm512_shuffle_i32x4(t5, t13, 0xdd);
	tt14 = _mm512_shuffle_i32x4(t6, t14, 0xdd);
	tt15 = _mm512_shuffle_i32x4(t7, t15, 0xdd);

	t16 = _mm512_unpacklo_epi16(row16, row17);
	t17 = _mm512_unpacklo_epi16(row18, row19);
	t18 = _mm512_unpacklo_epi16(row20, row21);
	t19 = _mm512_unpacklo_epi16(row22, row23);
	t20 = _mm512_unpackhi_epi16(row16, row17);
	t21 = _mm512_unpackhi_epi16(row18, row19);
	t22 = _mm512_unpackhi_epi16(row20, row21);
	t23 = _mm512_unpackhi_epi16(row22, row23);

	tt16 = _mm512_unpacklo_epi32(t16, t17);
	tt17 = _mm512_unpackhi_epi32(t16, t17);
	tt18 = _mm512_unpacklo_epi32(t18, t19);
	tt19 = _mm512_unpackhi_epi32(t18, t19);
	tt20 = _mm512_unpacklo_epi32(t20, t21);
	tt21 = _mm512_unpackhi_epi32(t20, t21);
	tt22 = _mm512_unpacklo_epi32(t22, t23);
	tt23 = _mm512_unpackhi_epi32(t22, t23);

	t16 = _mm512_unpacklo_epi64(tt16, tt18);
	t17 = _mm512_unpackhi_epi64(tt16, tt18);
	t18 = _mm512_unpacklo_epi64(tt17, tt19);
	t19 = _mm512_unpackhi_epi64(tt17, tt19);
	t20 = _mm512_unpacklo_epi64(tt20, tt22);
	t21 = _mm512_unpackhi_epi64(tt20, tt22);
	t22 = _mm512_unpacklo_epi64(tt21, tt23);
	t23 = _mm512_unpackhi_epi64(tt21, tt23);

	t24 = _mm512_unpacklo_epi16(row24, row25);
	t25 = _mm512_unpacklo_epi16(row26, row27);
	t26 = _mm512_unpacklo_epi16(row28, row29);
	t27 = _mm512_unpacklo_epi16(row30, row31);
	t28 = _mm512_unpackhi_epi16(row24, row25);
	t29 = _mm512_unpackhi_epi16(row26, row27);
	t30 = _mm512_unpackhi_epi16(row28, row29);
	t31 = _mm512_unpackhi_epi16(row30, row31);

	tt24 = _mm512_unpacklo_epi32(t24, t25);
	tt25 = _mm512_unpackhi_epi32(t24, t25);
	tt26 = _mm512_unpacklo_epi32(t26, t27);
	tt27 = _mm512_unpackhi_epi32(t26, t27);
	tt28 = _mm512_unpacklo_epi32(t28, t29);
	tt29 = _mm512_unpackhi_epi32(t28, t29);
	tt30 = _mm512_unpacklo_epi32(t30, t31);
	tt31 = _mm512_unpackhi_epi32(t30, t31);

	t24 = _mm512_unpacklo_epi64(tt24, tt26);
	t25 = _mm512_unpackhi_epi64(tt24, tt26);
	t26 = _mm512_unpacklo_epi64(tt25, tt27);
	t27 = _mm512_unpackhi_epi64(tt25, tt27);
	t28 = _mm512_unpacklo_epi64(tt28, tt30);
	t29 = _mm512_unpackhi_epi64(tt28, tt30);
	t30 = _mm512_unpacklo_epi64(tt29, tt31);
	t31 = _mm512_unpackhi_epi64(tt29, tt31);

	tt16 = _mm512_shuffle_i32x4(t16, t24, 0x88);
	tt17 = _mm512_shuffle_i32x4(t17, t25, 0x88);
	tt18 = _mm512_shuffle_i32x4(t18, t26, 0x88);
	tt19 = _mm512_shuffle_i32x4(t19, t27, 0x88);
	tt20 = _mm512_shuffle_i32x4(t20, t28, 0x88);
	tt21 = _mm512_shuffle_i32x4(t21, t29, 0x88);
	tt22 = _mm512_shuffle_i32x4(t22, t30, 0x88);
	tt23 = _mm512_shuffle_i32x4(t23, t31, 0x88);

	tt24 = _mm512_shuffle_i32x4(t16, t24, 0xdd);
	tt25 = _mm512_shuffle_i32x4(t17, t25, 0xdd);
	tt26 = _mm512_shuffle_i32x4(t18, t26, 0xdd);
	tt27 = _mm512_shuffle_i32x4(t19, t27, 0xdd);
	tt28 = _mm512_shuffle_i32x4(t20, t28, 0xdd);
	tt29 = _mm512_shuffle_i32x4(t21, t29, 0xdd);
	tt30 = _mm512_shuffle_i32x4(t22, t30, 0xdd);
	tt31 = _mm512_shuffle_i32x4(t23, t31, 0xdd);

	row0 = _mm512_shuffle_i32x4(tt0, tt16, 0x88);
	row1 = _mm512_shuffle_i32x4(tt1, tt17, 0x88);
	row2 = _mm512_shuffle_i32x4(tt2, tt18, 0x88);
	row3 = _mm512_shuffle_i32x4(tt3, tt19, 0x88);
	row4 = _mm512_shuffle_i32x4(tt4, tt20, 0x88);
	row5 = _mm512_shuffle_i32x4(tt5, tt21, 0x88);
	row6 = _mm512_shuffle_i32x4(tt6, tt22, 0x88);
	row7 = _mm512_shuffle_i32x4(tt7, tt23, 0x88);

	row8 = _mm512_shuffle_i32x4(tt8, tt24, 0x88);
	row9 = _mm512_shuffle_i32x4(tt9, tt25, 0x88);
	row10 = _mm512_shuffle_i32x4(tt10, tt26, 0x88);
	row11 = _mm512_shuffle_i32x4(tt11, tt27, 0x88);
	row12 = _mm512_shuffle_i32x4(tt12, tt28, 0x88);
	row13 = _mm512_shuffle_i32x4(tt13, tt29, 0x88);
	row14 = _mm512_shuffle_i32x4(tt14, tt30, 0x88);
	row15 = _mm512_shuffle_i32x4(tt15, tt31, 0x88);

	row16 = _mm512_shuffle_i32x4(tt0, tt16, 0xdd);
	row17 = _mm512_shuffle_i32x4(tt1, tt17, 0xdd);
	row18 = _mm512_shuffle_i32x4(tt2, tt18, 0xdd);
	row19 = _mm512_shuffle_i32x4(tt3, tt19, 0xdd);
	row20 = _mm512_shuffle_i32x4(tt4, tt20, 0xdd);
	row21 = _mm512_shuffle_i32x4(tt5, tt21, 0xdd);
	row22 = _mm512_shuffle_i32x4(tt6, tt22, 0xdd);
	row23 = _mm512_shuffle_i32x4(tt7, tt23, 0xdd);

	row24 = _mm512_shuffle_i32x4(tt8, tt24, 0xdd);
	row25 = _mm512_shuffle_i32x4(tt9, tt25, 0xdd);
	row26 = _mm512_shuffle_i32x4(tt10, tt26, 0xdd);
	row27 = _mm512_shuffle_i32x4(tt11, tt27, 0xdd);
	row28 = _mm512_shuffle_i32x4(tt12, tt28, 0xdd);
	row29 = _mm512_shuffle_i32x4(tt13, tt29, 0xdd);
	row30 = _mm512_shuffle_i32x4(tt14, tt30, 0xdd);
	row31 = _mm512_shuffle_i32x4(tt15, tt31, 0xdd);
}

} // namespace zimg

#endif // ZIMG_X86_AVX512_UTIL_H_

#endif // ZIMG_X86_AVX512
