#pragma once

#ifdef ZIMG_ARM

#include <cstdint>
#include "common/ccdep.h"

// A32 compatibility macros
#if defined(_M_ARM) || defined(__arm__)
  #define vmovl_high_u8(x) vmovl_u8(vget_high_u8(x))
  #define vmovl_high_u16(x) vmovl_u16(vget_high_u16(x))
  #define vmovn_high_u16(r, a) vcombine_u8(r, vmovn_u16(a))
  #define vqmovn_high_u32(r, a) vcombine_u16(r, vqmovn_u32(a))
  #define vqmovn_high_s32(r, a) vcombine_s16(r, vqmovn_s32(a))
  #define vcvt_high_f16_f32(r, a) vcombine_f16(r, vcvt_f16_f32(a))
  #define vcvt_high_f32_f16(a) vcvt_f32_f16(vget_high_f16(a))
  #define vmull_high_s16(a, v) vmull_s16(vget_high_s16(a), vget_high_s16(v))
  #define vmlal_high_s16(a, b, v) vmlal_s16(a, vget_high_s16(b), vget_high_s16(v))
  #define vfmaq_f32(a, b, c) vmlaq_f32(a, b, c)
  #define vdupq_laneq_s16(vec, lane) ((lane) >= 4 ? vdupq_lane_s16(vget_high_s16(vec), (lane) % 4) : vdupq_lane_s16(vget_low_s16(vec), (lane) % 4))
  #define vdupq_laneq_f32(vec, lane) ((lane) >= 2 ? vdupq_lane_f32(vget_high_f32(vec), (lane) % 2) : vdupq_lane_f32(vget_low_f32(vec), (lane) % 2))
#endif

#if defined(_MSC_VER) && defined(_M_ARM64)
  #define __fp16 uint16_t
  #define float16x8_t uint16x8_t
  #define vld1q_f16 vld1q_u16
  #define vst1q_f16 vst1q_u16
  #define vmovn_high_u16(r, a) neon_xtn2_16(r, a)
  #define vdupq_laneq_s16(vec, lane) neon_dupqe16q(vec, lane)
  #define vdupq_laneq_f32(vec, lane) neon_dupqe32q(vec, lane)
  #define vreinterpretq_f32_f64(a) a
#endif

namespace zimg {

// The n-th mask vector has the lower n bytes set to all-ones.
extern const uint8_t neon_mask_table alignas(16)[17][16];

// Store from [x] into [dst] the 8-bit elements with index less than [idx].
static inline FORCE_INLINE void neon_store_idxlo_u8(uint8_t *dst, uint8x16_t x, unsigned idx)
{
	uint8x16_t orig = vld1q_u8(dst);
	uint8x16_t mask = vld1q_u8(neon_mask_table[idx]);

	orig = vbicq_u8(orig, mask);
	x = vandq_u8(x, mask);
	x = vorrq_u8(x, orig);

	vst1q_u8(dst, x);
}

// Store from [x] into [dst] the 8-bit elements with index greater than or equal to [idx].
static inline FORCE_INLINE void neon_store_idxhi_u8(uint8_t *dst, uint8x16_t x, unsigned idx)
{
	uint8x16_t orig = vld1q_u8(dst);
	uint8x16_t mask = vld1q_u8(neon_mask_table[idx]);

	orig = vandq_u8(orig, mask);
	x = vbicq_u8(x, mask);
	x = vorrq_u8(x, orig);

	vst1q_u8(dst, x);
}

// Store from [x] into [dst] the 16-bit elements with index less than [idx].
static inline FORCE_INLINE void neon_store_idxlo_u16(uint16_t *dst, uint16x8_t x, unsigned idx)
{
	neon_store_idxlo_u8(reinterpret_cast<uint8_t *>(dst), vreinterpretq_u8_u16(x), idx * 2);
}

// Store from [x] into [dst] the 16-bit elements with index greater than or equal to [idx].
static inline FORCE_INLINE void neon_store_idxhi_u16(uint16_t *dst, uint16x8_t x, unsigned idx)
{
	neon_store_idxhi_u8(reinterpret_cast<uint8_t *>(dst), vreinterpretq_u8_u16(x), idx * 2);
}

#if !defined(_MSC_VER) || defined(_M_ARM64)
// Store from [x] into [dst] the 16-bit elements with index less than [idx].
static inline FORCE_INLINE void neon_store_idxlo_f16(__fp16 *dst, float16x8_t x, unsigned idx)
{
	neon_store_idxlo_u8(reinterpret_cast<uint8_t *>(dst), vreinterpretq_u8_f16(x), idx * 2);
}

// Store from [x] into [dst] the 16-bit elements with index greater than or equal to [idx].
static inline FORCE_INLINE void neon_store_idxhi_f16(__fp16 *dst, float16x8_t x, unsigned idx)
{
	neon_store_idxhi_u8(reinterpret_cast<uint8_t *>(dst), vreinterpretq_u8_f16(x), idx * 2);
}
#endif // !defined(_MSC_VER) || defined(_M_ARM64)

// Store from [x] into [dst] the 32-bit elements with index less than [idx].
static inline FORCE_INLINE void neon_store_idxlo_f32(float *dst, float32x4_t x, unsigned idx)
{
	neon_store_idxlo_u8(reinterpret_cast<uint8_t *>(dst), vreinterpretq_u8_f32(x), idx * 4);
}

// Store from [x] into [dst] the 32-bit elements with index greater than or equal to [idx]
static inline FORCE_INLINE void neon_store_idxhi_f32(float *dst, float32x4_t x, unsigned idx)
{
	neon_store_idxhi_u8(reinterpret_cast<uint8_t *>(dst), vreinterpretq_u8_f32(x), idx * 4);
}

// Stores the elements of [x] into [dst0]-[dst7].
static inline FORCE_INLINE void neon_scatter_u16(uint16_t *dst0, uint16_t *dst1, uint16_t *dst2, uint16_t *dst3,
												 uint16_t *dst4, uint16_t *dst5, uint16_t *dst6, uint16_t *dst7, uint16x8_t x)
{
	*dst0 = vgetq_lane_u16(x, 0);
	*dst1 = vgetq_lane_u16(x, 1);
	*dst2 = vgetq_lane_u16(x, 2);
	*dst3 = vgetq_lane_u16(x, 3);
	*dst4 = vgetq_lane_u16(x, 4);
	*dst5 = vgetq_lane_u16(x, 5);
	*dst6 = vgetq_lane_u16(x, 6);
	*dst7 = vgetq_lane_u16(x, 7);
}

// Stores the elements of [x] into [dst0]-[dst3].
static inline FORCE_INLINE void neon_scatter_f32(float *dst0, float *dst1, float *dst2, float *dst3, float32x4_t x)
{
	*dst0 = vgetq_lane_f32(x, 0);
	*dst1 = vgetq_lane_f32(x, 1);
	*dst2 = vgetq_lane_f32(x, 2);
	*dst3 = vgetq_lane_f32(x, 3);
}

// Transpose in-place the 8x8 matrix stored in [row0]-[row7].
static inline FORCE_INLINE void neon_transpose8_u16(uint16x8_t &row0, uint16x8_t &row1, uint16x8_t &row2, uint16x8_t &row3,
													uint16x8_t &row4, uint16x8_t &row5, uint16x8_t &row6, uint16x8_t &row7)
{
    uint16x8x2_t t0_t1 = vtrnq_u16(row0, row1);
    uint16x8x2_t t2_t3 = vtrnq_u16(row2, row3);
    uint16x8x2_t t4_t5 = vtrnq_u16(row4, row5);
    uint16x8x2_t t6_t7 = vtrnq_u16(row6, row7);
    uint16x8_t t0 = t0_t1.val[0];
    uint16x8_t t1 = t0_t1.val[1];
    uint16x8_t t2 = t2_t3.val[0];
    uint16x8_t t3 = t2_t3.val[1];
    uint16x8_t t4 = t4_t5.val[0];
    uint16x8_t t5 = t4_t5.val[1];
    uint16x8_t t6 = t6_t7.val[0];
    uint16x8_t t7 = t6_t7.val[1];

    uint32x4x2_t tt0_tt1 = vtrnq_u32(vreinterpretq_u32_u16(t0), vreinterpretq_u32_u16(t2));
    uint32x4x2_t tt2_tt3 = vtrnq_u32(vreinterpretq_u32_u16(t1), vreinterpretq_u32_u16(t3));
    uint32x4x2_t tt4_tt5 = vtrnq_u32(vreinterpretq_u32_u16(t4), vreinterpretq_u32_u16(t6));
    uint32x4x2_t tt6_tt7 = vtrnq_u32(vreinterpretq_u32_u16(t5), vreinterpretq_u32_u16(t7));
    uint32x4_t tt0 = tt0_tt1.val[0];
    uint32x4_t tt1 = tt0_tt1.val[1];
    uint32x4_t tt2 = tt2_tt3.val[0];
    uint32x4_t tt3 = tt2_tt3.val[1];
    uint32x4_t tt4 = tt4_tt5.val[0];
    uint32x4_t tt5 = tt4_tt5.val[1];
    uint32x4_t tt6 = tt6_tt7.val[0];
    uint32x4_t tt7 = tt6_tt7.val[1];

#if defined(__aarch64__) || defined(_M_ARM64)
    row0 = vreinterpretq_u16_u64(vzip1q_u64(vreinterpretq_u64_u32(tt0), vreinterpretq_u64_u32(tt4)));
    row1 = vreinterpretq_u16_u64(vzip1q_u64(vreinterpretq_u64_u32(tt2), vreinterpretq_u64_u32(tt6)));
    row2 = vreinterpretq_u16_u64(vzip1q_u64(vreinterpretq_u64_u32(tt1), vreinterpretq_u64_u32(tt5)));
    row3 = vreinterpretq_u16_u64(vzip1q_u64(vreinterpretq_u64_u32(tt3), vreinterpretq_u64_u32(tt7)));
    row4 = vreinterpretq_u16_u64(vzip2q_u64(vreinterpretq_u64_u32(tt0), vreinterpretq_u64_u32(tt4)));
    row5 = vreinterpretq_u16_u64(vzip2q_u64(vreinterpretq_u64_u32(tt2), vreinterpretq_u64_u32(tt6)));
    row6 = vreinterpretq_u16_u64(vzip2q_u64(vreinterpretq_u64_u32(tt1), vreinterpretq_u64_u32(tt5)));
    row7 = vreinterpretq_u16_u64(vzip2q_u64(vreinterpretq_u64_u32(tt3), vreinterpretq_u64_u32(tt7)));
#else
    row0 = vreinterpretq_u16_u32(vcombine_u32(vget_low_u32(tt0), vget_low_u32(tt4)));
    row1 = vreinterpretq_u16_u32(vcombine_u32(vget_low_u32(tt2), vget_low_u32(tt6)));
    row2 = vreinterpretq_u16_u32(vcombine_u32(vget_low_u32(tt1), vget_low_u32(tt5)));
    row3 = vreinterpretq_u16_u32(vcombine_u32(vget_low_u32(tt3), vget_low_u32(tt7)));
    row4 = vreinterpretq_u16_u32(vcombine_u32(vget_high_u32(tt0), vget_high_u32(tt4)));
    row5 = vreinterpretq_u16_u32(vcombine_u32(vget_high_u32(tt2), vget_high_u32(tt6)));
    row6 = vreinterpretq_u16_u32(vcombine_u32(vget_high_u32(tt1), vget_high_u32(tt5)));
    row7 = vreinterpretq_u16_u32(vcombine_u32(vget_high_u32(tt3), vget_high_u32(tt7)));
#endif
}


// Transpose in-place the 4x4 matrix stored in [row0]-[row3].
static inline FORCE_INLINE void neon_transpose4_f32(float32x4_t &row0, float32x4_t &row1, float32x4_t &row2, float32x4_t &row3)
{
	float32x4x2_t t0_t1 = vtrnq_f32(row0, row1);
	float32x4x2_t t2_t3 = vtrnq_f32(row2, row3);
	float32x4_t t0 = t0_t1.val[0];
	float32x4_t t1 = t0_t1.val[1];
	float32x4_t t2 = t2_t3.val[0];
	float32x4_t t3 = t2_t3.val[1];
#if defined(__aarch64__) || defined(_M_ARM64)
	row0 = vreinterpretq_f32_f64(vzip1q_f64(vreinterpretq_f64_f32(t0), vreinterpretq_f64_f32(t2)));
	row1 = vreinterpretq_f32_f64(vzip1q_f64(vreinterpretq_f64_f32(t1), vreinterpretq_f64_f32(t3)));
	row2 = vreinterpretq_f32_f64(vzip2q_f64(vreinterpretq_f64_f32(t0), vreinterpretq_f64_f32(t2)));
	row3 = vreinterpretq_f32_f64(vzip2q_f64(vreinterpretq_f64_f32(t1), vreinterpretq_f64_f32(t3)));
#else
	row0 = vcombine_f32(vget_low_f32(t0), vget_low_f32(t2));
	row1 = vcombine_f32(vget_low_f32(t1), vget_low_f32(t3));
	row2 = vcombine_f32(vget_high_f32(t0), vget_high_f32(t2));
	row3 = vcombine_f32(vget_high_f32(t1), vget_high_f32(t3));
#endif
}

} // namespace

#endif // ZIMG_ARM
