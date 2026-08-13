#pragma once

#ifdef ZIMG_LOONGARCH

#include <cstdint>
#include "common/ccdep.h"
#include <lsxintrin.h>

namespace zimg {

extern const uint8_t lsx_mask_table alignas(16)[17][16];

static inline FORCE_INLINE void lsx_store_idxlo_u8(uint8_t *dst, __m128i x, unsigned idx)
{
	__m128i orig = __lsx_vld(dst, 0);
	__m128i mask = __lsx_vld(lsx_mask_table[idx], 0);

	orig = __lsx_vandn_v(mask, orig);
	x = __lsx_vand_v(mask, x);
	x = __lsx_vor_v(x, orig);

	__lsx_vst(x, dst, 0);
}

static inline FORCE_INLINE void lsx_store_idxhi_u8(uint8_t *dst, __m128i x, unsigned idx)
{
	__m128i orig = __lsx_vld(dst, 0);
	__m128i mask = __lsx_vld(lsx_mask_table[idx], 0);

	orig = __lsx_vand_v(mask, orig);
	x = __lsx_vandn_v(mask, x);
	x = __lsx_vor_v(x, orig);

	__lsx_vst(x, dst, 0);
}

static inline FORCE_INLINE void lsx_store_idxlo_u16(uint16_t *dst, __m128i x, unsigned idx)
{
	lsx_store_idxlo_u8(reinterpret_cast<uint8_t *>(dst), x, idx * 2);
}

static inline FORCE_INLINE void lsx_store_idxhi_u16(uint16_t *dst, __m128i x, unsigned idx)
{
	lsx_store_idxhi_u8(reinterpret_cast<uint8_t *>(dst), x, idx * 2);
}

// Store from [v] into [dst] the 32-bit elements with index less than [idx].
static inline FORCE_INLINE void lsx_store_idxlo_f32(float *dst, __m128 v, unsigned idx)
{
	switch (idx) {
	case 4:
		__lsx_vstelm_w((__m128i)v, dst, 12, 3);
	case 3:
		__lsx_vstelm_w((__m128i)v, dst, 8, 2);
	case 2:
		__lsx_vstelm_w((__m128i)v, dst, 4, 1);
	case 1:
		__lsx_vstelm_w((__m128i)v, dst, 0, 0);
	}
}

// Store from [v] into [dst] the 32-bit elements with index greater than or equal to [idx].
static inline FORCE_INLINE void lsx_store_idxhi_f32(float *dst, __m128 v, unsigned idx)
{
	switch (idx) {
	case 0:
		__lsx_vstelm_w((__m128i)v, dst, 0, 0);
	case 1:
		__lsx_vstelm_w((__m128i)v, dst, 4, 1);
	case 2:
		__lsx_vstelm_w((__m128i)v, dst, 8, 2);
	case 3:
		__lsx_vstelm_w((__m128i)v, dst, 12, 3);
	}
}

static inline FORCE_INLINE void lsx_scatter_u16(uint16_t *dst0, uint16_t *dst1, uint16_t *dst2, uint16_t *dst3,
                                                uint16_t *dst4, uint16_t *dst5, uint16_t *dst6, uint16_t *dst7, __m128i x)
{
	*dst0 = static_cast<uint16_t>(__lsx_vpickve2gr_hu(x, 0));
	*dst1 = static_cast<uint16_t>(__lsx_vpickve2gr_hu(x, 1));
	*dst2 = static_cast<uint16_t>(__lsx_vpickve2gr_hu(x, 2));
	*dst3 = static_cast<uint16_t>(__lsx_vpickve2gr_hu(x, 3));
	*dst4 = static_cast<uint16_t>(__lsx_vpickve2gr_hu(x, 4));
	*dst5 = static_cast<uint16_t>(__lsx_vpickve2gr_hu(x, 5));
	*dst6 = static_cast<uint16_t>(__lsx_vpickve2gr_hu(x, 6));
	*dst7 = static_cast<uint16_t>(__lsx_vpickve2gr_hu(x, 7));
}

static inline FORCE_INLINE void lsx_scatter_f32(float *dst0, float *dst1, float *dst2, float *dst3, __m128 x)
{
	__m128 tmp = x;
	*dst0 = tmp[0];
	*dst1 = tmp[1];
	*dst2 = tmp[2];
	*dst3 = tmp[3];
}

static inline FORCE_INLINE void lsx_transpose8_u16(__m128i &row0, __m128i &row1, __m128i &row2, __m128i &row3,
                                                   __m128i &row4, __m128i &row5, __m128i &row6, __m128i &row7)
{
	__m128i t0 = __lsx_vilvl_h(row1, row0);
	__m128i t1 = __lsx_vilvh_h(row1, row0);
	__m128i t2 = __lsx_vilvl_h(row3, row2);
	__m128i t3 = __lsx_vilvh_h(row3, row2);
	__m128i t4 = __lsx_vilvl_h(row5, row4);
	__m128i t5 = __lsx_vilvh_h(row5, row4);
	__m128i t6 = __lsx_vilvl_h(row7, row6);
	__m128i t7 = __lsx_vilvh_h(row7, row6);

	__m128i tt0 = __lsx_vilvl_w(t2, t0);
	__m128i tt1 = __lsx_vilvh_w(t2, t0);
	__m128i tt2 = __lsx_vilvl_w(t3, t1);
	__m128i tt3 = __lsx_vilvh_w(t3, t1);
	__m128i tt4 = __lsx_vilvl_w(t6, t4);
	__m128i tt5 = __lsx_vilvh_w(t6, t4);
	__m128i tt6 = __lsx_vilvl_w(t7, t5);
	__m128i tt7 = __lsx_vilvh_w(t7, t5);

	row0 = __lsx_vilvl_d(tt4, tt0);
	row1 = __lsx_vilvh_d(tt4, tt0);
	row2 = __lsx_vilvl_d(tt5, tt1);
	row3 = __lsx_vilvh_d(tt5, tt1);
	row4 = __lsx_vilvl_d(tt6, tt2);
	row5 = __lsx_vilvh_d(tt6, tt2);
	row6 = __lsx_vilvl_d(tt7, tt3);
	row7 = __lsx_vilvh_d(tt7, tt3);
}

static inline FORCE_INLINE void lsx_transpose4_f32(__m128 &row0, __m128 &row1, __m128 &row2, __m128 &row3)
{
	// Use integer interleave on the bits then reinterpret back to float
	__m128i r0 = (__m128i)row0;
	__m128i r1 = (__m128i)row1;
	__m128i r2 = (__m128i)row2;
	__m128i r3 = (__m128i)row3;

	__m128i t0 = __lsx_vilvl_w(r1, r0);
	__m128i t1 = __lsx_vilvh_w(r1, r0);
	__m128i t2 = __lsx_vilvl_w(r3, r2);
	__m128i t3 = __lsx_vilvh_w(r3, r2);

	__m128i tt0 = __lsx_vilvl_d(t2, t0);
	__m128i tt1 = __lsx_vilvh_d(t2, t0);
	__m128i tt2 = __lsx_vilvl_d(t3, t1);
	__m128i tt3 = __lsx_vilvh_d(t3, t1);

	row0 = (__m128)tt0;
	row1 = (__m128)tt1;
	row2 = (__m128)tt2;
	row3 = (__m128)tt3;
}

} // namespace zimg

#endif // ZIMG_LOONGARCH
