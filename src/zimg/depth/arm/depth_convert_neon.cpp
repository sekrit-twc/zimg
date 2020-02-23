#ifdef ZIMG_ARM

#include <cstdint>
#include <arm_neon.h>
#include "common/align.h"
#include "depth_convert_arm.h"

#include "common/arm/neon_util.h"

namespace zimg {
namespace depth {

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

} // namespace depth
} // namespace zimg

#endif // ZIMG_ARM
