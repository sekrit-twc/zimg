#ifdef ZIMG_ARM

#include "common/ccdep.h"

#include <arm_neon.h>
#include "common/align.h"
#include "f16c_arm.h"

#include "common/arm/neon_util.h"

namespace zimg::depth {

void f16c_half_to_float_neon(const void *src, void *dst, unsigned left, unsigned right)
{
	const __fp16 *src_p = static_cast<const __fp16 *>(src);
	float *dst_p = static_cast<float *>(dst);

	unsigned vec_left = ceil_n(left, 4);
	unsigned vec_right = floor_n(right, 4);

	if (left != vec_left) {
		float32x4_t x = vcvt_f32_f16(vld1_f16(src_p + vec_left - 4));
		neon_store_idxhi_f32(dst_p + vec_left - 4, x, left % 4);
	}

	for (unsigned j = vec_left; j < vec_right; j += 4) {
		float32x4_t x = vcvt_f32_f16(vld1_f16(src_p + j));
		vst1q_f32(dst_p + j, x);
	}

	if (right != vec_right) {
		float32x4_t x = vcvt_f32_f16(vld1_f16(src_p + vec_right));
		neon_store_idxlo_f32(dst_p + vec_right, x, right % 4);
	}
}

void f16c_float_to_half_neon(const void *src, void *dst, unsigned left, unsigned right)
{
	const float *src_p = static_cast<const float *>(src);
	__fp16 *dst_p = static_cast<__fp16 *>(dst);

	unsigned vec_left = ceil_n(left, 4);
	unsigned vec_right = floor_n(right, 4);

	if (left != vec_left) {
		float16x4_t x = vcvt_f16_f32(vld1q_f32(src_p + vec_left - 4));
		neon_store_idxhi_f16(dst_p + vec_left - 8, vcombine_f16(vreinterpret_f16_u16(vdup_n_u16(0)), x), left % 4 + 4);
	}

	for (unsigned j = vec_left; j < vec_right; j += 4) {
		float16x4_t x = vcvt_f16_f32(vld1q_f32(src_p + j));
		vst1_f16(dst_p + j, x);
	}

	if (right != vec_right) {
		float16x4_t x = vcvt_f16_f32(vld1q_f32(src_p + vec_right));
		neon_store_idxlo_f16(dst_p + vec_right, vcombine_f16(x, vreinterpret_f16_u16(vdup_n_u16(0))), right % 4);
	}
}

} // namespace zimg::depth

#endif // ZIMG_ARM
