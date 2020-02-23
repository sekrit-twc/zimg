#pragma once

#ifdef ZIMG_ARM

#include <cstdint>
#include "common/ccdep.h"

// A32 compatibility macros
#if defined(_M_ARM) || defined(__arm__)
  #define vmovl_high_u8(x) vmovl_u8(vget_high_u8(x))
  #define vmovn_high_u16(r, a) vcombine_u8(r, vmovn_u16(a))
#endif

#if defined(_MSC_VER) && defined(_M_ARM64)
  #define vmovn_high_u16(r, a) neon_xtn2_16(r, a)
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

} // namespace

#endif // ZIMG_ARM
