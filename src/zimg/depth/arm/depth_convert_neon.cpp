#ifdef ZIMG_ARM

#include <cstdint>
#include <arm_neon.h>
#include "common/align.h"
#include "common/ccdep.h"
#include "depth_convert_arm.h"

#include "common/arm/neon_util.h"

namespace zimg::depth {

namespace {

// Convert unsigned 16-bit to single precision.
inline FORCE_INLINE void cvt_u16_to_f32_neon(uint16x8_t x, float32x4_t &lo, float32x4_t &hi)
{
	uint32x4_t lo_dw = vmovl_u16(vget_low_u16(x));
	uint32x4_t hi_dw = vmovl_high_u16(x);

	lo = vcvtq_f32_u32(lo_dw);
	hi = vcvtq_f32_u32(hi_dw);
}

// Convert unsigned 8-bit to single precision.
inline FORCE_INLINE void cvt_u8_to_f32_neon(uint8x16_t x, float32x4_t &lolo, float32x4_t &lohi, float32x4_t &hilo, float32x4_t &hihi)
{
	uint16x8_t lo_w = vmovl_u8(vget_low_u8(x));
	uint16x8_t hi_w = vmovl_high_u8(x);

	cvt_u16_to_f32_neon(lo_w, lolo, lohi);
	cvt_u16_to_f32_neon(hi_w, hilo, hihi);
}

inline FORCE_INLINE void depth_convert_b2f_neon_xiter(unsigned j, const uint8_t *src_p, float32x4_t scale, float32x4_t offset,
                                                      float32x4_t &lolo_out, float32x4_t &lohi_out, float32x4_t &hilo_out, float32x4_t &hihi_out)
{
	uint8x16_t x = vld1q_u8(src_p + j);
	float32x4_t lolo, lohi, hilo, hihi;

	cvt_u8_to_f32_neon(x, lolo, lohi, hilo, hihi);

	lolo_out = vfmaq_f32(offset, lolo, scale);
	lohi_out = vfmaq_f32(offset, lohi, scale);
	hilo_out = vfmaq_f32(offset, hilo, scale);
	hihi_out = vfmaq_f32(offset, hihi, scale);
}

inline FORCE_INLINE void depth_convert_w2f_neon_xiter(unsigned j, const uint16_t *src_p, float32x4_t scale, float32x4_t offset,
                                                      float32x4_t &lo_out, float32x4_t &hi_out)
{
	uint16x8_t x = vld1q_u16(src_p + j);
	float32x4_t lo, hi;

	cvt_u16_to_f32_neon(x, lo, hi);

	lo_out = vfmaq_f32(offset, lo, scale);
	hi_out = vfmaq_f32(offset, hi, scale);
}

} // namespace


void left_shift_b2b_neon(const void *src, void *dst, unsigned shift, unsigned left, unsigned right)
{
	const uint8_t *src_p = static_cast<const uint8_t *>(src);
	uint8_t *dst_p = static_cast<uint8_t *>(dst);

	unsigned vec_left = ceil_n(left, 16);
	unsigned vec_right = floor_n(right, 16);

	int8x16_t count = vdupq_n_s8(shift);

	if (left != vec_left) {
		uint8x16_t x = vld1q_u8(src_p + vec_left - 16);
		x = vshlq_u8(x, count);

		neon_store_idxhi_u8(dst_p + vec_left - 16, x, left % 16);
	}

	for (unsigned j = vec_left; j < vec_right; j += 16) {
		uint8x16_t x = vld1q_u8(src_p + j);
		x = vshlq_u8(x, count);

		vst1q_u8(dst_p + j, x);
	}

	if (right != vec_right) {
		uint8x16_t x = vld1q_u8(src_p + vec_right);
		x = vshlq_u8(x, count);

		neon_store_idxlo_u8(dst_p + vec_right, x, right % 16);
	}
}

void left_shift_b2w_neon(const void *src, void *dst, unsigned shift, unsigned left, unsigned right)
{
	const uint8_t *src_p = static_cast<const uint8_t *>(src);
	uint16_t *dst_p = static_cast<uint16_t *>(dst);

	unsigned vec_left = ceil_n(left, 16);
	unsigned vec_right = floor_n(right, 16);

	int16x8_t count = vdupq_n_s16(shift);

	if (left != vec_left) {
		uint8x16_t x = vld1q_u8(src_p + vec_left - 16);
		uint16x8_t lo = vshlq_u16(vmovl_u8(vget_low_u8(x)), count);
		uint16x8_t hi = vshlq_u16(vmovl_high_u8(x), count);

		if (vec_left - left > 8) {
			neon_store_idxhi_u16(dst_p + vec_left - 16, lo, left % 8);
			vst1q_u16(dst_p + vec_left - 8, hi);
		} else {
			neon_store_idxhi_u16(dst_p + vec_left - 8, lo, left % 8);
		}
	}

	for (unsigned j = vec_left; j < vec_right; j += 16) {
		uint8x16_t x = vld1q_u8(src_p + j);
		uint16x8_t lo = vshlq_u16(vmovl_u8(vget_low_u8(x)), count);
		uint16x8_t hi = vshlq_u16(vmovl_high_u8(x), count);

		vst1q_u16(dst_p + j + 0, lo);
		vst1q_u16(dst_p + j + 8, hi);
	}

	if (right != vec_right) {
		uint8x16_t x = vld1q_u8(src_p + vec_right);
		uint16x8_t lo = vshlq_u16(vmovl_u8(vget_low_u8(x)), count);
		uint16x8_t hi = vshlq_u16(vmovl_high_u8(x), count);

		if (right - vec_right >= 8) {
			vst1q_u16(dst_p + vec_right, lo);
			neon_store_idxlo_u16(dst_p + vec_right + 8, hi, right % 8);
		} else {
			neon_store_idxlo_u16(dst_p + vec_right, lo, right % 8);
		}
	}
}

void left_shift_w2b_neon(const void *src, void *dst, unsigned shift, unsigned left, unsigned right)
{
	const uint16_t *src_p = static_cast<const uint16_t *>(src);
	uint8_t *dst_p = static_cast<uint8_t *>(dst);

	unsigned vec_left = ceil_n(left, 16);
	unsigned vec_right = floor_n(right, 16);

	int8x16_t count = vdupq_n_s8(shift);

	if (left != vec_left) {
		uint16x8_t lo = vld1q_u16(src_p + vec_left - 16);
		uint16x8_t hi = vld1q_u16(src_p + vec_left - 8);
		uint8x16_t x = vmovn_high_u16(vmovn_u16(lo), hi);
		x = vshlq_u8(x, count);

		neon_store_idxhi_u8(dst_p + vec_left - 16, x, left % 16);
	}

	for (unsigned j = vec_left; j < vec_right; j += 16) {
		uint16x8_t lo = vld1q_u16(src_p + j + 0);
		uint16x8_t hi = vld1q_u16(src_p + j + 8);
		uint8x16_t x = vmovn_high_u16(vmovn_u16(lo), hi);
		x = vshlq_u8(x, count);

		vst1q_u8(dst_p + j, x);
	}

	if (right != vec_right) {
		uint16x8_t lo = vld1q_u16(src_p + vec_right + 0);
		uint16x8_t hi = vld1q_u16(src_p + vec_right + 8);
		uint8x16_t x = vmovn_high_u16(vmovn_u16(lo), hi);
		x = vshlq_u8(x, count);

		neon_store_idxlo_u8(dst_p + vec_right, x, right % 16);
	}
}

void left_shift_w2w_neon(const void *src, void *dst, unsigned shift, unsigned left, unsigned right)
{
	const uint16_t *src_p = static_cast<const uint16_t *>(src);
	uint16_t *dst_p = static_cast<uint16_t *>(dst);

	unsigned vec_left = ceil_n(left, 8);
	unsigned vec_right = floor_n(right, 8);

	int16x8_t count = vdupq_n_s16(shift);

	if (left != vec_left) {
		uint16x8_t x = vld1q_u16(src_p + vec_left - 8);
		x = vshlq_u16(x, count);

		neon_store_idxhi_u16(dst_p + vec_left - 8, x, left % 8);
	}

	for (unsigned j = vec_left; j < vec_right; j += 8) {
		uint16x8_t x = vld1q_u16(src_p + j);
		x = vshlq_u16(x, count);

		vst1q_u16(dst_p + j, x);
	}

	if (right != vec_right) {
		uint16x8_t x = vld1q_u16(src_p + vec_right);
		x = vshlq_u16(x, count);

		neon_store_idxlo_u16(dst_p + vec_right, x, right % 8);
	}
}

void depth_convert_b2h_neon(const void *src, void *dst, float scale, float offset, unsigned left, unsigned right)
{
	const uint8_t *src_p = static_cast<const uint8_t *>(src);
	__fp16 *dst_p = static_cast<__fp16 *>(dst);

	unsigned vec_left = ceil_n(left, 16);
	unsigned vec_right = floor_n(right, 16);

	const float32x4_t scale_x4 = vdupq_n_f32(scale);
	const float32x4_t offset_x4 = vdupq_n_f32(offset);

	float32x4_t lolo, lohi, hilo, hihi;

#define XITER depth_convert_b2f_neon_xiter
#define XARGS src_p, scale_x4, offset_x4, lolo, lohi, hilo, hihi
	if (left != vec_left) {
		XITER(vec_left - 16, XARGS);
		float16x8_t lo = vcvt_high_f16_f32(vcvt_f16_f32(lolo), lohi);
		float16x8_t hi = vcvt_high_f16_f32(vcvt_f16_f32(hilo), hihi);

		if (vec_left - left > 8) {
			neon_store_idxhi_f16(dst_p + vec_left - 16, lo, left % 8);
			vst1q_f16(dst_p + vec_left - 8, hi);
		} else {
			neon_store_idxhi_f16(dst_p + vec_left - 8, hi, left % 8);
		}
	}

	for (unsigned j = vec_left; j < vec_right; j += 16) {
		XITER(j, XARGS);
		float16x8_t lo = vcvt_high_f16_f32(vcvt_f16_f32(lolo), lohi);
		float16x8_t hi = vcvt_high_f16_f32(vcvt_f16_f32(hilo), hihi);
		vst1q_f16(dst_p + j + 0, lo);
		vst1q_f16(dst_p + j + 8, hi);
	}

	if (right != vec_right) {
		XITER(vec_right, XARGS);
		float16x8_t lo = vcvt_high_f16_f32(vcvt_f16_f32(lolo), lohi);
		float16x8_t hi = vcvt_high_f16_f32(vcvt_f16_f32(hilo), hihi);

		if (right - vec_right >= 8) {
			vst1q_f16(dst_p + vec_right + 0, lo);
			neon_store_idxlo_f16(dst_p + vec_right + 8, hi, right % 8);
		} else {
			neon_store_idxlo_f16(dst_p + vec_right, lo, right % 8);
		}
	}
#undef XITER
#undef XARGS
}

void depth_convert_b2f_neon(const void *src, void *dst, float scale, float offset, unsigned left, unsigned right)
{
	const uint8_t *src_p = static_cast<const uint8_t *>(src);
	float *dst_p = static_cast<float *>(dst);

	unsigned vec_left = ceil_n(left, 16);
	unsigned vec_right = floor_n(right, 16);

	const float32x4_t scale_x4 = vdupq_n_f32(scale);
	const float32x4_t offset_x4 = vdupq_n_f32(offset);

	float32x4_t lolo, lohi, hilo, hihi;

#define XITER depth_convert_b2f_neon_xiter
#define XARGS src_p, scale_x4, offset_x4, lolo, lohi, hilo, hihi
	if (left != vec_left) {
		XITER(vec_left - 16, XARGS);

		if (vec_left - left > 12) {
			neon_store_idxhi_f32(dst_p + vec_left - 16, lolo, left % 4);
			vst1q_f32(dst_p + vec_left - 12, lohi);
			vst1q_f32(dst_p + vec_left - 8, hilo);
			vst1q_f32(dst_p + vec_left - 4, hihi);
		} else if (vec_left - left > 8) {
			neon_store_idxhi_f32(dst_p + vec_left - 12, lohi, left % 4);
			vst1q_f32(dst_p + vec_left - 8, hilo);
			vst1q_f32(dst_p + vec_left - 4, hihi);
		} else if (vec_left - left > 4) {
			neon_store_idxhi_f32(dst_p + vec_left - 8, hilo, left % 4);
			vst1q_f32(dst_p + vec_left - 4, hihi);
		} else {
			neon_store_idxhi_f32(dst_p + vec_left - 4, hihi, left % 4);
		}
	}

	for (unsigned j = vec_left; j < vec_right; j += 16) {
		XITER(j, XARGS);

		vst1q_f32(dst_p + j + 0, lolo);
		vst1q_f32(dst_p + j + 4, lohi);
		vst1q_f32(dst_p + j + 8, hilo);
		vst1q_f32(dst_p + j + 12, hihi);
	}

	if (right != vec_right) {
		XITER(vec_right, XARGS);

		if (right - vec_right >= 12) {
			vst1q_f32(dst_p + vec_right + 0, lolo);
			vst1q_f32(dst_p + vec_right + 4, lohi);
			vst1q_f32(dst_p + vec_right + 8, hilo);
			neon_store_idxlo_f32(dst_p + vec_right + 12, hihi, right % 4);
		} else if (right - vec_right >= 8) {
			vst1q_f32(dst_p + vec_right + 0, lolo);
			vst1q_f32(dst_p + vec_right + 4, lohi);
			neon_store_idxlo_f32(dst_p + vec_right + 8, hilo, right % 4);
		} else if (right - vec_right >= 4) {
			vst1q_f32(dst_p + vec_right + 0, lolo);
			neon_store_idxlo_f32(dst_p + vec_right + 4, lohi, right % 4);
		} else {
			neon_store_idxlo_f32(dst_p + vec_right, lolo, right % 4);
		}
	}
#undef XITER
#undef XARGS
}

void depth_convert_w2h_neon(const void *src, void *dst, float scale, float offset, unsigned left, unsigned right)
{
	const uint16_t *src_p = static_cast<const uint16_t *>(src);
	__fp16 *dst_p = static_cast<__fp16 *>(dst);

	unsigned vec_left = ceil_n(left, 8);
	unsigned vec_right = floor_n(right, 8);

	const float32x4_t scale_x4 = vdupq_n_f32(scale);
	const float32x4_t offset_x4 = vdupq_n_f32(offset);

	float32x4_t lo, hi;

#define XITER depth_convert_w2f_neon_xiter
#define XARGS src_p, scale_x4, offset_x4, lo, hi
	if (left != vec_left) {
		XITER(vec_left - 8, XARGS);
		float16x8_t x = vcvt_high_f16_f32(vcvt_f16_f32(lo), hi);
		neon_store_idxhi_f16(dst_p + vec_left - 8, x, left % 8);
	}

	for (unsigned j = vec_left; j < vec_right; j += 8) {
		XITER(j, XARGS);
		float16x8_t x = vcvt_high_f16_f32(vcvt_f16_f32(lo), hi);
		vst1q_f16(dst_p + j, x);
	}

	if (right != vec_right) {
		XITER(vec_right, XARGS);
		float16x8_t x = vcvt_high_f16_f32(vcvt_f16_f32(lo), hi);
		neon_store_idxlo_f16(dst_p + vec_right, x, right % 8);
	}
#undef XITER
#undef XARGS
}

void depth_convert_w2f_neon(const void *src, void *dst, float scale, float offset, unsigned left, unsigned right)
{
	const uint16_t *src_p = static_cast<const uint16_t *>(src);
	float *dst_p = static_cast<float *>(dst);

	unsigned vec_left = ceil_n(left, 8);
	unsigned vec_right = floor_n(right, 8);

	const float32x4_t scale_x4 = vdupq_n_f32(scale);
	const float32x4_t offset_x4 = vdupq_n_f32(offset);

	float32x4_t lo, hi;

#define XITER depth_convert_w2f_neon_xiter
#define XARGS src_p, scale_x4, offset_x4, lo, hi
	if (left != vec_left) {
		XITER(vec_left - 8, XARGS);

		if (vec_left - left > 4) {
			neon_store_idxhi_f32(dst_p + vec_left - 8, lo, left % 4);
			vst1q_f32(dst_p + vec_left - 4, hi);
		} else {
			neon_store_idxhi_f32(dst_p + vec_left - 4, hi, left % 4);
		}
	}

	for (unsigned j = vec_left; j < vec_right; j += 8) {
		XITER(j, XARGS);

		vst1q_f32(dst_p + j + 0, lo);
		vst1q_f32(dst_p + j + 4, hi);
	}

	if (right != vec_right) {
		XITER(vec_right, XARGS);

		if (right - vec_right >= 4) {
			vst1q_f32(dst_p + vec_right + 0, lo);
			neon_store_idxlo_f32(dst_p + vec_right + 4, hi, right % 4);
		} else {
			neon_store_idxlo_f32(dst_p + vec_right, lo, right % 4);
		}
	}
#undef XITER
#undef XARGS
}

} // namespace zimg::depth

#endif // ZIMG_ARM
