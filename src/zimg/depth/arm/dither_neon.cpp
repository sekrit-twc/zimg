#ifdef ZIMG_ARM

#include <cstdint>
#include <arm_neon.h>
#include "common/align.h"
#include "common/ccdep.h"
#include "dither_arm.h"

#include "common/arm/neon_util.h"

namespace zimg::depth {

namespace {

struct LoadU8 {
	typedef uint8_t type;

	static inline FORCE_INLINE void load8(const uint8_t *ptr, float32x4_t &lo, float32x4_t &hi, unsigned n = 8)
	{
		uint16x8_t x = vmovl_u8(vld1_u8(ptr));
		lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(x)));
		hi = vcvtq_f32_u32(vmovl_high_u16(x));
	}
};

struct LoadU16 {
	typedef uint16_t type;

	static inline FORCE_INLINE void load8(const uint16_t *ptr, float32x4_t &lo, float32x4_t &hi, unsigned n = 8)
	{
		uint16x8_t x = vld1q_u16(ptr);
		lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(x)));
		hi = vcvtq_f32_u32(vmovl_high_u16(x));
	}
};

struct LoadF16 {
	typedef __fp16 type;

	static inline FORCE_INLINE void load8(const __fp16 *ptr, float32x4_t &lo, float32x4_t &hi, unsigned n = 8)
	{
		float16x8_t x = vld1q_f16(ptr);
		lo = vcvt_f32_f16(vget_low_f16(x));
		hi = vcvt_high_f32_f16(x);
	}
};

struct LoadF32 {
	typedef float type;

	static inline FORCE_INLINE void load8(const float *ptr, float32x4_t &lo, float32x4_t &hi, unsigned n = 8)
	{
		lo = vld1q_f32(ptr);
		hi = n >= 4 ? vld1q_f32(ptr + 4) : vdupq_n_f32(0.0f);
	}
};

struct StoreU8 {
	typedef uint8_t type;

	static inline FORCE_INLINE void store8(uint8_t *ptr, uint16x8_t x)
	{
		vst1_u8(ptr, vmovn_u16(x));
	}

	static inline FORCE_INLINE void store8_idxlo(uint8_t *ptr, uint16x8_t x_, unsigned idx)
	{
		uint8x8_t x = vmovn_u16(x_);
		uint8x8_t orig = vld1_u8(ptr);
		uint8x8_t mask = vld1_u8(neon_mask_table[idx]);

		orig = vbic_u8(orig, mask);
		x = vand_u8(x, mask);
		x = vorr_u8(x, orig);

		vst1_u8(ptr, x);
	}

	static inline FORCE_INLINE void store8_idxhi(uint8_t *ptr, uint16x8_t x_, unsigned idx)
	{
		uint8x8_t x = vmovn_u16(x_);
		uint8x8_t orig = vld1_u8(ptr);
		uint8x8_t mask = vld1_u8(neon_mask_table[idx]);

		orig = vand_u8(orig, mask);
		x = vbic_u8(x, mask);
		x = vorr_u8(x, orig);

		vst1_u8(ptr, x);
	}
};

struct StoreU16 {
	typedef uint16_t type;

	static inline FORCE_INLINE void store8(uint16_t *ptr, uint16x8_t x)
	{
		vst1q_u16(ptr, x);
	}

	static inline FORCE_INLINE void store8_idxlo(uint16_t *ptr, uint16x8_t x, unsigned idx)
	{
		neon_store_idxlo_u16(ptr, x, idx);
	}

	static inline FORCE_INLINE void store8_idxhi(uint16_t *ptr, uint16x8_t x, unsigned idx)
	{
		neon_store_idxhi_u16(ptr, x, idx);
	}
};


inline FORCE_INLINE uint16x8_t ordered_dither_neon_xiter(float32x4_t lo, float32x4_t hi, unsigned j, const float *dither, unsigned dither_offset, unsigned dither_mask,
                                                         const float32x4_t &scale, const float32x4_t &offset, const uint16x8_t &out_max)
{
	float32x4_t dith;
	uint32x4_t lo_dw, hi_dw;
	uint16x8_t x;

	dith = vld1q_f32(dither + ((dither_offset + j + 0) & dither_mask));
	lo = vfmaq_f32(offset, lo, scale);
	lo = vaddq_f32(lo, dith);

	dith = vld1q_f32(dither + ((dither_offset + j + 4) & dither_mask));
	hi = vfmaq_f32(offset, hi, scale);
	hi = vaddq_f32(hi, dith);

	lo_dw = vcvtnq_u32_f32(lo);
	hi_dw = vcvtnq_u32_f32(hi);

	x = vqmovn_high_u32(vqmovn_u32(lo_dw), hi_dw);
	x = vminq_u16(x, out_max);

	return x;
}

template <class Load, class Store>
void ordered_dither_neon_impl(const float *dither, unsigned dither_offset, unsigned dither_mask,
                              const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right)
{
    const typename Load::type *src_p = static_cast<const typename Load::type *>(src);
    typename Store::type *dst_p = static_cast<typename Store::type *>(dst);

    unsigned vec_left = ceil_n(left, 8);
    unsigned vec_right = floor_n(right, 8);

    const float32x4_t scale_x4 = vdupq_n_f32(scale);
    const float32x4_t offset_x4 = vdupq_n_f32(offset);
    const uint16x8_t out_max = vdupq_n_u16(static_cast<uint16_t>((1 << bits) - 1));

	float32x4_t lo, hi;

#define XARGS dither, dither_offset, dither_mask, scale_x4, offset_x4, out_max
	if (left != vec_left) {
		Load::load8(src_p + vec_left - 8, lo, hi);
		uint16x8_t x = ordered_dither_neon_xiter(lo, hi, vec_left - 8, XARGS);
		Store::store8_idxhi(dst_p + vec_left - 8, x, left % 8);
	}
	for (unsigned j = vec_left; j < vec_right; j += 8) {
		Load::load8(src_p + j, lo, hi);
		uint16x8_t x = ordered_dither_neon_xiter(lo, hi, j, XARGS);
		Store::store8(dst_p + j, x);
	}
	if (right != vec_right) {
		Load::load8(src_p + vec_right, lo, hi, right % 8);
		uint16x8_t x = ordered_dither_neon_xiter(lo, hi, vec_right, XARGS);
		Store::store8_idxlo(dst_p + vec_right, x, right % 8);
	}
#undef XARGS
}

} // namespace


void ordered_dither_b2b_neon(const float *dither, unsigned dither_offset, unsigned dither_mask,
                             const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right)
{
	ordered_dither_neon_impl<LoadU8, StoreU8>(dither, dither_offset, dither_mask, src, dst, scale, offset, bits, left, right);
}

void ordered_dither_b2w_neon(const float *dither, unsigned dither_offset, unsigned dither_mask,
                             const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right)
{
	ordered_dither_neon_impl<LoadU8, StoreU16>(dither, dither_offset, dither_mask, src, dst, scale, offset, bits, left, right);
}

void ordered_dither_w2b_neon(const float *dither, unsigned dither_offset, unsigned dither_mask,
                             const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right)
{
	ordered_dither_neon_impl<LoadU16, StoreU8>(dither, dither_offset, dither_mask, src, dst, scale, offset, bits, left, right);
}

void ordered_dither_w2w_neon(const float *dither, unsigned dither_offset, unsigned dither_mask,
                             const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right)
{
	ordered_dither_neon_impl<LoadU16, StoreU16>(dither, dither_offset, dither_mask, src, dst, scale, offset, bits, left, right);
}

void ordered_dither_h2b_neon(const float *dither, unsigned dither_offset, unsigned dither_mask,
                             const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right)
{
	ordered_dither_neon_impl<LoadF16, StoreU8>(dither, dither_offset, dither_mask, src, dst, scale, offset, bits, left, right);
}

void ordered_dither_h2w_neon(const float *dither, unsigned dither_offset, unsigned dither_mask,
                             const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right)
{
	ordered_dither_neon_impl<LoadF16, StoreU16>(dither, dither_offset, dither_mask, src, dst, scale, offset, bits, left, right);
}

void ordered_dither_f2b_neon(const float *dither, unsigned dither_offset, unsigned dither_mask,
                             const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right)
{
	ordered_dither_neon_impl<LoadF32, StoreU8>(dither, dither_offset, dither_mask, src, dst, scale, offset, bits, left, right);
}

void ordered_dither_f2w_neon(const float *dither, unsigned dither_offset, unsigned dither_mask,
                             const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right)
{
	ordered_dither_neon_impl<LoadF32, StoreU16>(dither, dither_offset, dither_mask, src, dst, scale, offset, bits, left, right);
}

} // namespace zimg::depth

#endif // ZIMG_ARM
