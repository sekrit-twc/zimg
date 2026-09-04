#ifdef ZIMG_LOONGARCH

#include <cstdint>
#include <lsxintrin.h>
#include "common/align.h"
#include "common/ccdep.h"
#include "depth/quantize.h"
#include "depth_convert_loongarch.h"

#include "common/loongarch/lsx_util.h"

namespace zimg::depth {

namespace {

inline FORCE_INLINE void float_to_half_rne_4(const float *src, uint16_t *dst)
{
	dst[0] = float_to_half(src[0]);
	dst[1] = float_to_half(src[1]);
	dst[2] = float_to_half(src[2]);
	dst[3] = float_to_half(src[3]);
}

inline FORCE_INLINE void half_to_float_4(const uint16_t *src, float *dst)
{
	dst[0] = half_to_float(src[0]);
	dst[1] = half_to_float(src[1]);
	dst[2] = half_to_float(src[2]);
	dst[3] = half_to_float(src[3]);
}

// Convert unsigned 16-bit to single precision.
inline FORCE_INLINE void cvt_u16_to_f32_lsx(__m128i x, __m128 &lo, __m128 &hi)
{
	__m128i lo_dw = __lsx_vsllwil_wu_hu(x, 0);
	__m128i hi_dw = __lsx_vexth_wu_hu(x);

	lo = (__m128)__lsx_vffint_s_wu(lo_dw);
	hi = (__m128)__lsx_vffint_s_wu(hi_dw);
}

// Convert unsigned 8-bit to single precision.
inline FORCE_INLINE void cvt_u8_to_f32_lsx(
	__m128i x,
	__m128 &lolo,
	__m128 &lohi,
	__m128 &hilo,
	__m128 &hihi)
{
	__m128i lo_w = __lsx_vsllwil_hu_bu(x, 0);
	__m128i hi_w = __lsx_vexth_hu_bu(x);

	cvt_u16_to_f32_lsx(lo_w, lolo, lohi);
	cvt_u16_to_f32_lsx(hi_w, hilo, hihi);
}

inline FORCE_INLINE void depth_convert_b2f_lsx_xiter(
	unsigned j,
	const uint8_t *src_p,
	__m128 scale,
	__m128 offset,
	__m128 &lolo_out,
	__m128 &lohi_out,
	__m128 &hilo_out,
	__m128 &hihi_out)
{
	__m128i x = __lsx_vld(src_p + j, 0);
	__m128 lolo, lohi, hilo, hihi;

	cvt_u8_to_f32_lsx(x, lolo, lohi, hilo, hihi);

	lolo_out = __lsx_vfmadd_s(lolo, scale, offset);
	lohi_out = __lsx_vfmadd_s(lohi, scale, offset);
	hilo_out = __lsx_vfmadd_s(hilo, scale, offset);
	hihi_out = __lsx_vfmadd_s(hihi, scale, offset);
}

inline FORCE_INLINE void depth_convert_w2f_lsx_xiter(
	unsigned j,
	const uint16_t *src_p,
	__m128 scale,
	__m128 offset,
	__m128 &lo_out,
	__m128 &hi_out)
{
	__m128i x = __lsx_vld(src_p + j, 0);
	__m128 lo, hi;

	cvt_u16_to_f32_lsx(x, lo, hi);

	lo_out = __lsx_vfmadd_s(lo, scale, offset);
	hi_out = __lsx_vfmadd_s(hi, scale, offset);
}

} // namespace


void left_shift_b2b_lsx(const void *src, void *dst, unsigned shift,
                        unsigned left, unsigned right)
{
	const uint8_t *src_p = static_cast<const uint8_t *>(src);
	uint8_t *dst_p = static_cast<uint8_t *>(dst);

	unsigned vec_left = ceil_n(left, 16);
	unsigned vec_right = floor_n(right, 16);

	__m128i count = __lsx_vreplgr2vr_b(static_cast<int>(shift));

	if (left != vec_left) {
		__m128i x = __lsx_vld(src_p + vec_left - 16, 0);
		x = __lsx_vsll_b(x, count);

		lsx_store_idxhi_u8(dst_p + vec_left - 16, x, left % 16);
	}

	for (unsigned j = vec_left; j < vec_right; j += 16) {
		__m128i x = __lsx_vld(src_p + j, 0);
		x = __lsx_vsll_b(x, count);

		__lsx_vst(x, dst_p + j, 0);
	}

	if (right != vec_right) {
		__m128i x = __lsx_vld(src_p + vec_right, 0);
		x = __lsx_vsll_b(x, count);

		lsx_store_idxlo_u8(dst_p + vec_right, x, right % 16);
	}
}

void left_shift_b2w_lsx(const void *src, void *dst, unsigned shift,
                        unsigned left, unsigned right)
{
	const uint8_t *src_p = static_cast<const uint8_t *>(src);
	uint16_t *dst_p = static_cast<uint16_t *>(dst);

	unsigned vec_left = ceil_n(left, 16);
	unsigned vec_right = floor_n(right, 16);

	__m128i count = __lsx_vreplgr2vr_h(static_cast<int>(shift));

	if (left != vec_left) {
		__m128i x = __lsx_vld(src_p + vec_left - 16, 0);
		__m128i lo = __lsx_vsll_h(__lsx_vsllwil_hu_bu(x, 0), count);
		__m128i hi = __lsx_vsll_h(__lsx_vexth_hu_bu(x), count);

		if (vec_left - left > 8) {
			lsx_store_idxhi_u16(dst_p + vec_left - 16,
			                          lo, left % 8);
			__lsx_vst(hi, dst_p + vec_left - 8, 0);
		} else {
			lsx_store_idxhi_u16(dst_p + vec_left - 8, hi, left % 8);
		}
	}

	for (unsigned j = vec_left; j < vec_right; j += 16) {
		__m128i x = __lsx_vld(src_p + j, 0);
		__m128i lo = __lsx_vsll_h(__lsx_vsllwil_hu_bu(x, 0), count);
		__m128i hi = __lsx_vsll_h(__lsx_vexth_hu_bu(x), count);

		__lsx_vst(lo, dst_p + j + 0, 0);
		__lsx_vst(hi, dst_p + j + 8, 0);
	}

	if (right != vec_right) {
		__m128i x = __lsx_vld(src_p + vec_right, 0);
		__m128i lo = __lsx_vsll_h(__lsx_vsllwil_hu_bu(x, 0), count);
		__m128i hi = __lsx_vsll_h(__lsx_vexth_hu_bu(x), count);

		if (right - vec_right >= 8) {
			__lsx_vst(lo, dst_p + vec_right, 0);
			lsx_store_idxlo_u16(dst_p + vec_right + 8,
			                          hi, right % 8);
		} else {
			lsx_store_idxlo_u16(dst_p + vec_right, lo, right % 8);
		}
	}
}

void left_shift_w2b_lsx(const void *src, void *dst, unsigned shift,
                        unsigned left, unsigned right)
{
	const uint16_t *src_p = static_cast<const uint16_t *>(src);
	uint8_t *dst_p = static_cast<uint8_t *>(dst);

	unsigned vec_left = ceil_n(left, 16);
	unsigned vec_right = floor_n(right, 16);

	__m128i count = __lsx_vreplgr2vr_b(static_cast<int>(shift));

	if (left != vec_left) {
		__m128i lo = __lsx_vld(src_p + vec_left - 16, 0);
		__m128i hi = __lsx_vld(src_p + vec_left - 8, 0);
		__m128i x = __lsx_vpickev_b(hi, lo);
		x = __lsx_vsll_b(x, count);

		lsx_store_idxhi_u8(dst_p + vec_left - 16, x, left % 16);
	}

	for (unsigned j = vec_left; j < vec_right; j += 16) {
		__m128i lo = __lsx_vld(src_p + j + 0, 0);
		__m128i hi = __lsx_vld(src_p + j + 8, 0);
		__m128i x = __lsx_vpickev_b(hi, lo);
		x = __lsx_vsll_b(x, count);

		__lsx_vst(x, dst_p + j, 0);
	}

	if (right != vec_right) {
		__m128i lo = __lsx_vld(src_p + vec_right + 0, 0);
		__m128i hi = right - vec_right > 8
			? __lsx_vld(src_p + vec_right + 8, 0)
			: (__m128i)__lsx_vldi(0);
		__m128i x = __lsx_vpickev_b(hi, lo);
		x = __lsx_vsll_b(x, count);

		lsx_store_idxlo_u8(dst_p + vec_right, x, right % 16);
	}
}

void left_shift_w2w_lsx(const void *src, void *dst, unsigned shift,
                        unsigned left, unsigned right)
{
	const uint16_t *src_p = static_cast<const uint16_t *>(src);
	uint16_t *dst_p = static_cast<uint16_t *>(dst);

	unsigned vec_left = ceil_n(left, 8);
	unsigned vec_right = floor_n(right, 8);

	__m128i count = __lsx_vreplgr2vr_h(static_cast<int>(shift));

	if (left != vec_left) {
		__m128i x = __lsx_vld(src_p + vec_left - 8, 0);
		x = __lsx_vsll_h(x, count);

		lsx_store_idxhi_u16(dst_p + vec_left - 8, x, left % 8);
	}

	for (unsigned j = vec_left; j < vec_right; j += 8) {
		__m128i x = __lsx_vld(src_p + j, 0);
		x = __lsx_vsll_h(x, count);

		__lsx_vst(x, dst_p + j, 0);
	}

	if (right != vec_right) {
		__m128i x = __lsx_vld(src_p + vec_right, 0);
		x = __lsx_vsll_h(x, count);

		lsx_store_idxlo_u16(dst_p + vec_right, x, right % 8);
	}
}

void depth_convert_b2h_lsx(const void *src, void *dst, float scale,
                           float offset, unsigned left, unsigned right)
{
	const uint8_t *src_p = static_cast<const uint8_t *>(src);
	uint16_t *dst_p = static_cast<uint16_t *>(dst);

	unsigned vec_left = ceil_n(left, 16);
	unsigned vec_right = floor_n(right, 16);

	const __m128 scale_x4 = (__m128)__lsx_vldrepl_w(&scale, 0);
	const __m128 offset_x4 = (__m128)__lsx_vldrepl_w(&offset, 0);

	__m128 lolo, lohi, hilo, hihi;

#define XITER depth_convert_b2f_lsx_xiter
#define XARGS src_p, scale_x4, offset_x4, lolo, lohi, hilo, hihi
	if (left != vec_left) {
		XITER(vec_left - 16, XARGS);
		uint16_t hlo[8], hhi[8];
		float_to_half_rne_4((const float *)&lolo, hlo + 0);
		float_to_half_rne_4((const float *)&lohi, hlo + 4);
		float_to_half_rne_4((const float *)&hilo, hhi + 0);
		float_to_half_rne_4((const float *)&hihi, hhi + 4);
		__m128i lo = __lsx_vld((void *)hlo, 0);
		__m128i hi = __lsx_vld((void *)hhi, 0);

		if (vec_left - left > 8) {
			lsx_store_idxhi_u16(dst_p + vec_left - 16,
			                          lo, left % 8);
			__lsx_vst(hi, dst_p + vec_left - 8, 0);
		} else {
			lsx_store_idxhi_u16(dst_p + vec_left - 8, hi, left % 8);
		}
	}

	for (unsigned j = vec_left; j < vec_right; j += 16) {
		XITER(j, XARGS);
		uint16_t hlo[8], hhi[8];
		float_to_half_rne_4((const float *)&lolo, hlo + 0);
		float_to_half_rne_4((const float *)&lohi, hlo + 4);
		float_to_half_rne_4((const float *)&hilo, hhi + 0);
		float_to_half_rne_4((const float *)&hihi, hhi + 4);
		__m128i lo = __lsx_vld((void *)hlo, 0);
		__m128i hi = __lsx_vld((void *)hhi, 0);
		__lsx_vst(lo, dst_p + j + 0, 0);
		__lsx_vst(hi, dst_p + j + 8, 0);
	}

	if (right != vec_right) {
		XITER(vec_right, XARGS);
		uint16_t hlo[8], hhi[8];
		float_to_half_rne_4((const float *)&lolo, hlo + 0);
		float_to_half_rne_4((const float *)&lohi, hlo + 4);
		float_to_half_rne_4((const float *)&hilo, hhi + 0);
		float_to_half_rne_4((const float *)&hihi, hhi + 4);
		__m128i lo = __lsx_vld((void *)hlo, 0);
		__m128i hi = __lsx_vld((void *)hhi, 0);

		if (right - vec_right >= 8) {
			__lsx_vst(lo, dst_p + vec_right + 0, 0);
			lsx_store_idxlo_u16(dst_p + vec_right + 8,
			                          hi, right % 8);
		} else {
			lsx_store_idxlo_u16(dst_p + vec_right, lo, right % 8);
		}
	}
#undef XITER
#undef XARGS
}

void depth_convert_b2f_lsx(const void *src, void *dst, float scale,
                           float offset, unsigned left, unsigned right)
{
	const uint8_t *src_p = static_cast<const uint8_t *>(src);
	float *dst_p = static_cast<float *>(dst);

	unsigned vec_left = ceil_n(left, 16);
	unsigned vec_right = floor_n(right, 16);

	const __m128 scale_x4 = (__m128)__lsx_vldrepl_w(&scale, 0);
	const __m128 offset_x4 = (__m128)__lsx_vldrepl_w(&offset, 0);

	__m128 lolo, lohi, hilo, hihi;

#define XITER depth_convert_b2f_lsx_xiter
#define XARGS src_p, scale_x4, offset_x4, lolo, lohi, hilo, hihi
	if (left != vec_left) {
		XITER(vec_left - 16, XARGS);

		if (vec_left - left > 12) {
			lsx_store_idxhi_f32(dst_p + vec_left - 16,
			                            lolo, left % 4);
			__lsx_vst((__m128i)lohi, dst_p + vec_left - 12, 0);
			__lsx_vst((__m128i)hilo, dst_p + vec_left - 8, 0);
			__lsx_vst((__m128i)hihi, dst_p + vec_left - 4, 0);
		} else if (vec_left - left > 8) {
			lsx_store_idxhi_f32(dst_p + vec_left - 12,
			                            lohi, left % 4);
			__lsx_vst((__m128i)hilo, dst_p + vec_left - 8, 0);
			__lsx_vst((__m128i)hihi, dst_p + vec_left - 4, 0);
		} else if (vec_left - left > 4) {
			lsx_store_idxhi_f32(dst_p + vec_left - 8,
			                            hilo, left % 4);
			__lsx_vst((__m128i)hihi, dst_p + vec_left - 4, 0);
		} else {
			lsx_store_idxhi_f32(dst_p + vec_left - 4,
			                            hihi, left % 4);
		}
	}

	for (unsigned j = vec_left; j < vec_right; j += 16) {
		XITER(j, XARGS);

		__lsx_vst((__m128i)lolo, dst_p + j + 0, 0);
		__lsx_vst((__m128i)lohi, dst_p + j + 4, 0);
		__lsx_vst((__m128i)hilo, dst_p + j + 8, 0);
		__lsx_vst((__m128i)hihi, dst_p + j + 12, 0);
	}

	if (right != vec_right) {
		XITER(vec_right, XARGS);

		if (right - vec_right >= 12) {
			__lsx_vst((__m128i)lolo, dst_p + vec_right + 0, 0);
			__lsx_vst((__m128i)lohi, dst_p + vec_right + 4, 0);
			__lsx_vst((__m128i)hilo, dst_p + vec_right + 8, 0);
			lsx_store_idxlo_f32(dst_p + vec_right + 12,
			                            hihi, right % 4);
		} else if (right - vec_right >= 8) {
			__lsx_vst((__m128i)lolo, dst_p + vec_right + 0, 0);
			__lsx_vst((__m128i)lohi, dst_p + vec_right + 4, 0);
			lsx_store_idxlo_f32(dst_p + vec_right + 8,
			                            hilo, right % 4);
		} else if (right - vec_right >= 4) {
			__lsx_vst((__m128i)lolo, dst_p + vec_right + 0, 0);
			lsx_store_idxlo_f32(dst_p + vec_right + 4,
			                            lohi, right % 4);
		} else {
			lsx_store_idxlo_f32(dst_p + vec_right, lolo, right % 4);
		}
	}
#undef XITER
#undef XARGS
}

void depth_convert_w2h_lsx(const void *src, void *dst, float scale,
                           float offset, unsigned left, unsigned right)
{
	const uint16_t *src_p = static_cast<const uint16_t *>(src);
	uint16_t *dst_p = static_cast<uint16_t *>(dst);

	unsigned vec_left = ceil_n(left, 8);
	unsigned vec_right = floor_n(right, 8);

	const __m128 scale_x4 = (__m128)__lsx_vldrepl_w(&scale, 0);
	const __m128 offset_x4 = (__m128)__lsx_vldrepl_w(&offset, 0);

	__m128 lo, hi;

#define XITER depth_convert_w2f_lsx_xiter
#define XARGS src_p, scale_x4, offset_x4, lo, hi
	if (left != vec_left) {
		XITER(vec_left - 8, XARGS);
		uint16_t h[8];
		float_to_half_rne_4((const float *)&lo, h + 0);
		float_to_half_rne_4((const float *)&hi, h + 4);
		__m128i x = __lsx_vld((void *)h, 0);
		lsx_store_idxhi_u16(dst_p + vec_left - 8, x, left % 8);
	}

	for (unsigned j = vec_left; j < vec_right; j += 8) {
		XITER(j, XARGS);
		uint16_t h[8];
		float_to_half_rne_4((const float *)&lo, h + 0);
		float_to_half_rne_4((const float *)&hi, h + 4);
		__m128i x = __lsx_vld((void *)h, 0);
		__lsx_vst(x, dst_p + j, 0);
	}

	if (right != vec_right) {
		XITER(vec_right, XARGS);
		uint16_t h[8];
		float_to_half_rne_4((const float *)&lo, h + 0);
		float_to_half_rne_4((const float *)&hi, h + 4);
		__m128i x = __lsx_vld((void *)h, 0);
		lsx_store_idxlo_u16(dst_p + vec_right, x, right % 8);
	}
#undef XITER
#undef XARGS
}

void depth_convert_w2f_lsx(const void *src, void *dst, float scale,
                           float offset, unsigned left, unsigned right)
{
	const uint16_t *src_p = static_cast<const uint16_t *>(src);
	float *dst_p = static_cast<float *>(dst);

	unsigned vec_left = ceil_n(left, 8);
	unsigned vec_right = floor_n(right, 8);

	const __m128 scale_x4 = (__m128)__lsx_vldrepl_w(&scale, 0);
	const __m128 offset_x4 = (__m128)__lsx_vldrepl_w(&offset, 0);

	__m128 lo, hi;

#define XITER depth_convert_w2f_lsx_xiter
#define XARGS src_p, scale_x4, offset_x4, lo, hi
	if (left != vec_left) {
		XITER(vec_left - 8, XARGS);

		if (vec_left - left > 4) {
			lsx_store_idxhi_f32(dst_p + vec_left - 8, lo, left % 4);
			__lsx_vst((__m128i)hi, dst_p + vec_left - 4, 0);
		} else {
			lsx_store_idxhi_f32(dst_p + vec_left - 4, hi, left % 4);
		}
	}

	for (unsigned j = vec_left; j < vec_right; j += 8) {
		XITER(j, XARGS);

		__lsx_vst((__m128i)lo, dst_p + j + 0, 0);
		__lsx_vst((__m128i)hi, dst_p + j + 4, 0);
	}

	if (right != vec_right) {
		XITER(vec_right, XARGS);

		if (right - vec_right >= 4) {
			__lsx_vst((__m128i)lo, dst_p + vec_right + 0, 0);
			lsx_store_idxlo_f32(dst_p + vec_right + 4,
			                            hi, right % 4);
		} else {
			lsx_store_idxlo_f32(dst_p + vec_right, lo, right % 4);
		}
	}
#undef XITER
#undef XARGS
}

void half_to_float_lsx(const void *src, void *dst, float, float,
                       unsigned left, unsigned right)
{
	const uint16_t *src_p = static_cast<const uint16_t *>(src);
	float *dst_p = static_cast<float *>(dst);

	unsigned vec_left = ceil_n(left, 4);
	unsigned vec_right = floor_n(right, 4);

	if (left != vec_left) {
		float x[4];

		half_to_float_4(src_p + vec_left - 4, x);
		__m128 v = (__m128)__lsx_vld((void *)x, 0);
		lsx_store_idxhi_f32(dst_p + vec_left - 4, v, left % 4);
	}

	for (unsigned j = vec_left; j < vec_right; j += 4) {
		float x[4];

		half_to_float_4(src_p + j, x);
		__m128 v = (__m128)__lsx_vld((void *)x, 0);
		__lsx_vst((__m128i)v, dst_p + j, 0);
	}

	if (right != vec_right) {
		float x[4];

		half_to_float_4(src_p + vec_right, x);
		__m128 v = (__m128)__lsx_vld((void *)x, 0);
		lsx_store_idxlo_f32(dst_p + vec_right, v, right % 4);
	}
}

void float_to_half_lsx(const void *src, void *dst, float, float,
                       unsigned left, unsigned right)
{
	const float *src_p = static_cast<const float *>(src);
	uint16_t *dst_p = static_cast<uint16_t *>(dst);

	unsigned vec_left = ceil_n(left, 8);
	unsigned vec_right = floor_n(right, 8);

	if (left != vec_left) {
		__m128 lo = (__m128)__lsx_vld(src_p + vec_left - 8, 0);
		__m128 hi = (__m128)__lsx_vld(src_p + vec_left - 4, 0);
		uint16_t h[8];
		float_to_half_rne_4((const float *)&lo, h + 0);
		float_to_half_rne_4((const float *)&hi, h + 4);
		__m128i x = __lsx_vld((void *)h, 0);
		lsx_store_idxhi_u16(dst_p + vec_left - 8, x, left % 8);
	}

	for (unsigned j = vec_left; j < vec_right; j += 8) {
		__m128 lo = (__m128)__lsx_vld(src_p + j, 0);
		__m128 hi = (__m128)__lsx_vld(src_p + j + 4, 0);
		uint16_t h[8];
		float_to_half_rne_4((const float *)&lo, h + 0);
		float_to_half_rne_4((const float *)&hi, h + 4);
		__m128i x = __lsx_vld((void *)h, 0);
		__lsx_vst(x, dst_p + j, 0);
	}

	if (right != vec_right) {
		float zero = 0.0f;
		__m128 lo = (__m128)__lsx_vld(src_p + vec_right, 0);
		__m128 hi = right - vec_right > 4
			? (__m128)__lsx_vld(src_p + vec_right + 4, 0)
			: (__m128)__lsx_vldrepl_w(&zero, 0);
		uint16_t h[8];
		float_to_half_rne_4((const float *)&lo, h + 0);
		float_to_half_rne_4((const float *)&hi, h + 4);
		__m128i x = __lsx_vld((void *)h, 0);
		lsx_store_idxlo_u16(dst_p + vec_right, x, right % 8);
	}
}

} // namespace zimg::depth

#endif // ZIMG_LOONGARCH
