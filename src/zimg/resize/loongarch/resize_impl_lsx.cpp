#ifdef ZIMG_LOONGARCH

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <lsxintrin.h>
#include "common/align.h"
#include "common/ccdep.h"
#include "common/checked_int.h"
#include "common/except.h"
#include "common/make_array.h"
#include "common/pixel.h"
#include "common/unroll.h"
#include "resize/resize_impl.h"
#include "resize_impl_loongarch.h"

#include "common/loongarch/lsx_util.h"

namespace zimg::resize {

namespace {

void transpose_line_8x8_u16(uint16_t * RESTRICT dst, const uint16_t * const * RESTRICT src, unsigned left, unsigned right)
{
	for (unsigned j = left; j < right; j += 8) {
		__m128i x0, x1, x2, x3, x4, x5, x6, x7;

		x0 = __lsx_vld(src[0] + j, 0);
		x1 = __lsx_vld(src[1] + j, 0);
		x2 = __lsx_vld(src[2] + j, 0);
		x3 = __lsx_vld(src[3] + j, 0);
		x4 = __lsx_vld(src[4] + j, 0);
		x5 = __lsx_vld(src[5] + j, 0);
		x6 = __lsx_vld(src[6] + j, 0);
		x7 = __lsx_vld(src[7] + j, 0);

		lsx_transpose8_u16(x0, x1, x2, x3, x4, x5, x6, x7);

		__lsx_vst(x0, dst + 0, 0);
		__lsx_vst(x1, dst + 8, 0);
		__lsx_vst(x2, dst + 16, 0);
		__lsx_vst(x3, dst + 24, 0);
		__lsx_vst(x4, dst + 32, 0);
		__lsx_vst(x5, dst + 40, 0);
		__lsx_vst(x6, dst + 48, 0);
		__lsx_vst(x7, dst + 56, 0);

		dst += 64;
	}
}

void transpose_line_4x4_f32(float * RESTRICT dst, const float *src_p0, const float *src_p1, const float *src_p2, const float *src_p3, unsigned left, unsigned right)
{
	for (unsigned j = left; j < right; j += 4) {
		__m128 x0, x1, x2, x3;

		x0 = (__m128)__lsx_vld(src_p0 + j, 0);
		x1 = (__m128)__lsx_vld(src_p1 + j, 0);
		x2 = (__m128)__lsx_vld(src_p2 + j, 0);
		x3 = (__m128)__lsx_vld(src_p3 + j, 0);

		lsx_transpose4_f32(x0, x1, x2, x3);

		__lsx_vst(x0, dst + 0, 0);
		__lsx_vst(x1, dst + 4, 0);
		__lsx_vst(x2, dst + 8, 0);
		__lsx_vst(x3, dst + 12, 0);

		dst += 16;
	}
}

inline FORCE_INLINE __m128i export_i30_u16(__m128i lo, __m128i hi)
{
	__m128i round = __lsx_vreplgr2vr_w(1 << 13);

	lo = __lsx_vadd_w(lo, round);
	hi = __lsx_vadd_w(hi, round);

	lo = __lsx_vsrai_w(lo, 14);
	hi = __lsx_vsrai_w(hi, 14);

	return __lsx_vssrarni_h_w(hi, lo, 0);
}


template <int Taps>
inline FORCE_INLINE __m128i resize_line8_h_u16_lsx_xiter(unsigned j, const unsigned *filter_left, const int16_t *filter_data, unsigned filter_stride, unsigned filter_width,
                                                         const uint16_t *src, unsigned src_base, uint16_t limit)
{
	static_assert(Taps <= 8, "only up to 8 taps can be unrolled");
	static_assert(Taps >= -7, "only up to 7 taps in epilogue");
	constexpr int Tail = Taps > 0 ? Taps : -Taps;

	const __m128i i16_min = __lsx_vreplgr2vr_h(INT16_MIN);
	const __m128i lim = __lsx_vreplgr2vr_h(static_cast<int16_t>(limit + INT16_MIN));

	const int16_t *filter_coeffs = filter_data + j * filter_stride;
	const uint16_t *src_p = src + (filter_left[j] - src_base) * 8;

	__m128i accum_lo = __lsx_vldi(0);
	__m128i accum_hi = __lsx_vldi(0);
	__m128i coeffs;

	auto f = ZIMG_UNROLL_FUNC(kk)
	{
		__m128i c, x;

		c = __lsx_vreplve_h(coeffs, static_cast<int>(kk));
		x = __lsx_vld(src_p + static_cast<unsigned>(kk) * 8, 0);
		x = __lsx_vadd_h(x, i16_min);
		accum_lo = __lsx_vmaddwev_w_h(accum_lo, c, x);
		accum_hi = __lsx_vmaddwod_w_h(accum_hi, c, x);
	};

	unsigned k_end = Taps > 0 ? 0 : floor_n(filter_width + 1, 8);

	for (unsigned k = 0; k < k_end; k += 8) {
		coeffs = __lsx_vld(filter_coeffs + k, 0);
		unroll<8>(f);
		src_p += 64;
	}

	if constexpr (Tail) {
		coeffs = __lsx_vld(filter_coeffs + k_end, 0);
		unroll<Tail>(f);
	}

	// Reshuffle even/odd grouping to low/high grouping
	// accum_lo = {r0, r2, r4, r6}, accum_hi = {r1, r3, r5, r7}
	// vilvl_w(hi, lo) = {lo[0], hi[0], lo[1], hi[1]} = {r0, r1, r2, r3}
	// vilvh_w(hi, lo) = {lo[2], hi[2], lo[3], hi[3]} = {r4, r5, r6, r7}
	{
		__m128i tmp_lo = __lsx_vilvl_w(accum_hi, accum_lo);
		accum_hi = __lsx_vilvh_w(accum_hi, accum_lo);
		accum_lo = tmp_lo;
	}

	__m128i result = export_i30_u16(accum_lo, accum_hi);
	result = __lsx_vmin_h(result, lim);
	result = __lsx_vsub_h(result, i16_min);
	return result;
}

template <int Taps>
void resize_line8_h_u16_lsx(const unsigned * RESTRICT filter_left, const int16_t * RESTRICT filter_data, unsigned filter_stride, unsigned filter_width,
                             const uint16_t * RESTRICT src, uint16_t * const * RESTRICT dst, unsigned src_base, unsigned left, unsigned right, uint16_t limit)
{
	unsigned vec_left = ceil_n(left, 8);
	unsigned vec_right = floor_n(right, 8);

#define XITER resize_line8_h_u16_lsx_xiter<Taps>
#define XARGS filter_left, filter_data, filter_stride, filter_width, src, src_base, limit
	for (unsigned j = left; j < vec_left; ++j) {
		__m128i x = XITER(j, XARGS);
		lsx_scatter_u16(dst[0] + j, dst[1] + j, dst[2] + j, dst[3] + j, dst[4] + j, dst[5] + j, dst[6] + j, dst[7] + j, x);
	}

	for (unsigned j = vec_left; j < vec_right; j += 8) {
		__m128i x0, x1, x2, x3, x4, x5, x6, x7;

		x0 = XITER(j + 0, XARGS);
		x1 = XITER(j + 1, XARGS);
		x2 = XITER(j + 2, XARGS);
		x3 = XITER(j + 3, XARGS);
		x4 = XITER(j + 4, XARGS);
		x5 = XITER(j + 5, XARGS);
		x6 = XITER(j + 6, XARGS);
		x7 = XITER(j + 7, XARGS);

		lsx_transpose8_u16(x0, x1, x2, x3, x4, x5, x6, x7);

		__lsx_vst(x0, dst[0] + j, 0);
		__lsx_vst(x1, dst[1] + j, 0);
		__lsx_vst(x2, dst[2] + j, 0);
		__lsx_vst(x3, dst[3] + j, 0);
		__lsx_vst(x4, dst[4] + j, 0);
		__lsx_vst(x5, dst[5] + j, 0);
		__lsx_vst(x6, dst[6] + j, 0);
		__lsx_vst(x7, dst[7] + j, 0);
	}

	for (unsigned j = vec_right; j < right; ++j) {
		__m128i x = XITER(j, XARGS);
		lsx_scatter_u16(dst[0] + j, dst[1] + j, dst[2] + j, dst[3] + j, dst[4] + j, dst[5] + j, dst[6] + j, dst[7] + j, x);
	}
#undef XITER
#undef XARGS
}

constexpr auto resize_line8_h_u16_lsx_jt_small = make_array(
	resize_line8_h_u16_lsx<1>,
	resize_line8_h_u16_lsx<2>,
	resize_line8_h_u16_lsx<3>,
	resize_line8_h_u16_lsx<4>,
	resize_line8_h_u16_lsx<5>,
	resize_line8_h_u16_lsx<6>,
	resize_line8_h_u16_lsx<7>,
	resize_line8_h_u16_lsx<8>);

constexpr auto resize_line8_h_u16_lsx_jt_large = make_array(
	resize_line8_h_u16_lsx<0>,
	resize_line8_h_u16_lsx<-1>,
	resize_line8_h_u16_lsx<-2>,
	resize_line8_h_u16_lsx<-3>,
	resize_line8_h_u16_lsx<-4>,
	resize_line8_h_u16_lsx<-5>,
	resize_line8_h_u16_lsx<-6>,
	resize_line8_h_u16_lsx<-7>);


template <int Taps>
inline FORCE_INLINE __m128 resize_line4_h_f32_lsx_xiter(unsigned j, const unsigned *filter_left, const float *filter_data, unsigned filter_stride, unsigned filter_width,
                                                        const float *src, unsigned src_base)
{
	static_assert(Taps <= 8, "only up to 8 taps can be unrolled");
	static_assert(Taps >= -3, "only up to 3 taps in epilogue");
	constexpr int Tail = Taps >= 4 ? Taps - 4 : Taps > 0 ? Taps : -Taps;

	const float *filter_coeffs = filter_data + j * filter_stride;
	const float *src_p = src + (filter_left[j] - src_base) * 4;

	__m128 accum0 = (__m128)__lsx_vldi(0);
	__m128 accum1 = (__m128)__lsx_vldi(0);
	__m128 coeffs;

	auto f = ZIMG_UNROLL_FUNC(kk)
	{
		__m128 &acc = kk % 2 ? accum1 : accum0;

		__m128 c = (__m128)__lsx_vreplve_w(*(__m128i *)&coeffs, static_cast<int>(kk));
		__m128 x = (__m128)__lsx_vld(src_p + 4 * static_cast<int>(kk), 0);
		acc = __lsx_vfmadd_s(c, x, acc);
	};

	unsigned k_end = Taps >= 4 ? 4 : Taps > 0 ? 0 : floor_n(filter_width, 4);

	for (unsigned k = 0; k < k_end; k += 4) {
		coeffs = (__m128)__lsx_vld(filter_coeffs + k, 0);
		unroll<4>(f);
		src_p += 16;
	}

	if constexpr (Tail) {
		coeffs = (__m128)__lsx_vld(filter_coeffs + k_end, 0);
		unroll<Tail>(f);
	}

	if constexpr (Taps <= 0 || Taps >= 2)
		accum0 = __lsx_vfadd_s(accum0, accum1);

	return accum0;
}

template <int Taps>
void resize_line4_h_f32_lsx(const unsigned * RESTRICT filter_left, const float * RESTRICT filter_data, unsigned filter_stride, unsigned filter_width,
                            const float * RESTRICT src, float * const * RESTRICT dst, unsigned src_base, unsigned left, unsigned right)
{
	unsigned vec_left = ceil_n(left, 4);
	unsigned vec_right = floor_n(right, 4);

#define XITER resize_line4_h_f32_lsx_xiter<Taps>
#define XARGS filter_left, filter_data, filter_stride, filter_width, src, src_base
	for (unsigned j = left; j < vec_left; ++j) {
		__m128 x = XITER(j, XARGS);
		lsx_scatter_f32(dst[0] + j, dst[1] + j, dst[2] + j, dst[3] + j, x);
	}

	for (unsigned j = vec_left; j < vec_right; j += 4) {
		__m128 x0, x1, x2, x3;

		x0 = XITER(j + 0, XARGS);
		x1 = XITER(j + 1, XARGS);
		x2 = XITER(j + 2, XARGS);
		x3 = XITER(j + 3, XARGS);

		lsx_transpose4_f32(x0, x1, x2, x3);

		__lsx_vst(x0, dst[0] + j, 0);
		__lsx_vst(x1, dst[1] + j, 0);
		__lsx_vst(x2, dst[2] + j, 0);
		__lsx_vst(x3, dst[3] + j, 0);
	}

	for (unsigned j = vec_right; j < right; ++j) {
		__m128 x = XITER(j, XARGS);
		lsx_scatter_f32(dst[0] + j, dst[1] + j, dst[2] + j, dst[3] + j, x);
	}
#undef XITER
#undef XARGS
}

constexpr auto resize_line4_h_f32_lsx_jt_small = make_array(
	resize_line4_h_f32_lsx<1>,
	resize_line4_h_f32_lsx<2>,
	resize_line4_h_f32_lsx<3>,
	resize_line4_h_f32_lsx<4>,
	resize_line4_h_f32_lsx<5>,
	resize_line4_h_f32_lsx<6>,
	resize_line4_h_f32_lsx<7>,
	resize_line4_h_f32_lsx<8>);

constexpr auto resize_line4_h_f32_lsx_jt_large = make_array(
	resize_line4_h_f32_lsx<0>,
	resize_line4_h_f32_lsx<-1>,
	resize_line4_h_f32_lsx<-2>,
	resize_line4_h_f32_lsx<-3>);


constexpr unsigned V_ACCUM_NONE = 0;
constexpr unsigned V_ACCUM_INITIAL = 1;
constexpr unsigned V_ACCUM_UPDATE = 2;
constexpr unsigned V_ACCUM_FINAL = 3;

template <unsigned Taps, unsigned AccumMode>
inline FORCE_INLINE __m128i resize_line_v_u16_lsx_xiter(unsigned j, unsigned accum_base, const uint16_t * const srcp[8],
                                                        int32_t * RESTRICT accum_p, const __m128i c[8], uint16_t limit)
{
	static_assert(Taps >= 1 && Taps <= 8, "must have between 1-8 taps");

	const __m128i i16_min = __lsx_vreplgr2vr_h(INT16_MIN);
	__attribute__((unused)) const __m128i lim = __lsx_vreplgr2vr_h(static_cast<int16_t>(limit + INT16_MIN));

	__m128i accum_lo = __lsx_vldi(0);
	__m128i accum_hi = __lsx_vldi(0);

	unroll<Taps>(ZIMG_UNROLL_FUNC(k)
	{
		__m128i x;

		x = __lsx_vld(srcp[k] + j, 0);
		x = __lsx_vadd_h(x, i16_min);

		if constexpr (k == 0 && (AccumMode == V_ACCUM_UPDATE || AccumMode == V_ACCUM_FINAL)) {
			accum_lo = __lsx_vmaddwev_w_h(__lsx_vld(accum_p + j - accum_base + 0, 0), c[k], x);
			accum_hi = __lsx_vmaddwod_w_h(__lsx_vld(accum_p + j - accum_base + 4, 0), c[k], x);
		} else if constexpr (k == 0) {
			accum_lo = __lsx_vmulwev_w_h(c[k], x);
			accum_hi = __lsx_vmulwod_w_h(c[k], x);
		} else {
			accum_lo = __lsx_vmaddwev_w_h(accum_lo, c[k], x);
			accum_hi = __lsx_vmaddwod_w_h(accum_hi, c[k], x);
		}
	});

	if constexpr (AccumMode == V_ACCUM_INITIAL || AccumMode == V_ACCUM_UPDATE) {
		__lsx_vst(accum_lo, accum_p + j - accum_base + 0, 0);
		__lsx_vst(accum_hi, accum_p + j - accum_base + 4, 0);
		return __lsx_vldi(0);
	} else {
		__m128i tmp_lo = __lsx_vilvl_w(accum_hi, accum_lo);
		accum_hi = __lsx_vilvh_w(accum_hi, accum_lo);
		accum_lo = tmp_lo;
		__m128i result = export_i30_u16(accum_lo, accum_hi);
		result = __lsx_vmin_h(result, lim);
		result = __lsx_vsub_h(result, i16_min);
		return result;
	}
}

template <unsigned Taps, unsigned AccumMode>
void resize_line_v_u16_lsx(const int16_t * RESTRICT filter_data, const uint16_t * const * RESTRICT src, uint16_t * RESTRICT dst, int32_t * RESTRICT accum, unsigned left, unsigned right, uint16_t limit)
{
	const uint16_t *srcp[8] = {src[0], src[1], src[2], src[3], src[4], src[5], src[6], src[7]};
	unsigned vec_left = ceil_n(left, 8);
	unsigned vec_right = floor_n(right, 8);
	unsigned accum_base = floor_n(left, 8);

	const __m128i c[8] = {
		__lsx_vreplgr2vr_h(filter_data[0]),
		__lsx_vreplgr2vr_h(filter_data[1]),
		__lsx_vreplgr2vr_h(filter_data[2]),
		__lsx_vreplgr2vr_h(filter_data[3]),
		__lsx_vreplgr2vr_h(filter_data[4]),
		__lsx_vreplgr2vr_h(filter_data[5]),
		__lsx_vreplgr2vr_h(filter_data[6]),
		__lsx_vreplgr2vr_h(filter_data[7]),
	};

#define XITER resize_line_v_u16_lsx_xiter<Taps, AccumMode>
#define XARGS accum_base, srcp, accum, c, limit
	if (left != vec_left) {
		__m128i out = XITER(vec_left - 8, XARGS);

		if (AccumMode == V_ACCUM_NONE || AccumMode == V_ACCUM_FINAL)
			lsx_store_idxhi_u16(dst + vec_left - 8, out, left % 8);
	}

	for (unsigned j = vec_left; j < vec_right; j += 8) {
		__m128i out = XITER(j, XARGS);

		if (AccumMode == V_ACCUM_NONE || AccumMode == V_ACCUM_FINAL)
			__lsx_vst(out, dst + j, 0);
	}

	if (right != vec_right) {
		__m128i out = XITER(vec_right, XARGS);

		if (AccumMode == V_ACCUM_NONE || AccumMode == V_ACCUM_FINAL)
			lsx_store_idxlo_u16(dst + vec_right, out, right % 8);
	}
#undef XITER
#undef XARGS
}

constexpr auto resize_line_v_u16_lsx_jt_small = make_array(
	resize_line_v_u16_lsx<1, V_ACCUM_NONE>,
	resize_line_v_u16_lsx<2, V_ACCUM_NONE>,
	resize_line_v_u16_lsx<3, V_ACCUM_NONE>,
	resize_line_v_u16_lsx<4, V_ACCUM_NONE>,
	resize_line_v_u16_lsx<5, V_ACCUM_NONE>,
	resize_line_v_u16_lsx<6, V_ACCUM_NONE>,
	resize_line_v_u16_lsx<7, V_ACCUM_NONE>,
	resize_line_v_u16_lsx<8, V_ACCUM_NONE>);

constexpr auto resize_line_v_u16_lsx_initial = resize_line_v_u16_lsx<8, V_ACCUM_INITIAL>;
constexpr auto resize_line_v_u16_lsx_update = resize_line_v_u16_lsx<8, V_ACCUM_UPDATE>;

constexpr auto resize_line_v_u16_lsx_jt_final = make_array(
	resize_line_v_u16_lsx<1, V_ACCUM_FINAL>,
	resize_line_v_u16_lsx<2, V_ACCUM_FINAL>,
	resize_line_v_u16_lsx<3, V_ACCUM_FINAL>,
	resize_line_v_u16_lsx<4, V_ACCUM_FINAL>,
	resize_line_v_u16_lsx<5, V_ACCUM_FINAL>,
	resize_line_v_u16_lsx<6, V_ACCUM_FINAL>,
	resize_line_v_u16_lsx<7, V_ACCUM_FINAL>,
	resize_line_v_u16_lsx<8, V_ACCUM_FINAL>);


template <unsigned Taps, bool Continue>
inline FORCE_INLINE __m128 resize_line_v_f32_lsx_xiter(unsigned j, const float * const srcp[8], const float *accum_p, const __m128 c[8])
{
	static_assert(Taps >= 1 && Taps <= 8, "must have between 1-8 taps");

	__m128 accum0 = (__m128)__lsx_vldi(0);
	__m128 accum1 = (__m128)__lsx_vldi(0);

	unroll<Taps>(ZIMG_UNROLL_FUNC(k)
	{
		__m128 &acc = k % 2 ? accum1 : accum0;
		__m128 x;

		x = (__m128)__lsx_vld(srcp[k] + j, 0);

		if constexpr (k == 0 && Continue)
			acc = __lsx_vfmadd_s(c[k], x, (__m128)__lsx_vld(accum_p + j, 0));
		else if constexpr (k == 0 || k == 1)
			acc = __lsx_vfmul_s(c[k], x);
		else
			acc = __lsx_vfmadd_s(c[k], x, acc);
	});

	if constexpr (Taps >= 2) accum0 = __lsx_vfadd_s(accum0, accum1);
	return accum0;
}

template <unsigned Taps, bool Continue>
void resize_line_v_f32_lsx(const float * RESTRICT filter_data, const float * const * RESTRICT src, float * RESTRICT dst, unsigned left, unsigned right)
{
	const float *srcp[8] = { src[0], src[1], src[2], src[3], src[4], src[5], src[6], src[7] };
	unsigned vec_left = ceil_n(left, 4);
	unsigned vec_right = floor_n(right, 4);

	const float repl_vals[8] = {
		filter_data[0], filter_data[1], filter_data[2], filter_data[3],
		filter_data[4], filter_data[5], filter_data[6], filter_data[7]
	};
	int32_t repl_vals_i32[8];
	memcpy(repl_vals_i32, repl_vals, sizeof(repl_vals));
	const __m128 c[8] = {
		(__m128)__lsx_vreplgr2vr_w(repl_vals_i32[0]),
		(__m128)__lsx_vreplgr2vr_w(repl_vals_i32[1]),
		(__m128)__lsx_vreplgr2vr_w(repl_vals_i32[2]),
		(__m128)__lsx_vreplgr2vr_w(repl_vals_i32[3]),
		(__m128)__lsx_vreplgr2vr_w(repl_vals_i32[4]),
		(__m128)__lsx_vreplgr2vr_w(repl_vals_i32[5]),
		(__m128)__lsx_vreplgr2vr_w(repl_vals_i32[6]),
		(__m128)__lsx_vreplgr2vr_w(repl_vals_i32[7]),
	};

#define XITER resize_line_v_f32_lsx_xiter<Taps, Continue>
#define XARGS srcp, dst, c
	if (left != vec_left) {
		__m128 accum = XITER(vec_left - 4, XARGS);
		lsx_store_idxhi_f32(dst + vec_left - 4, accum, left % 4);
	}

	for (unsigned j = vec_left; j < vec_right; j += 4) {
		__m128 accum = XITER(j, XARGS);
		__lsx_vst(accum, dst + j, 0);
	}

	if (right != vec_right) {
		__m128 accum = XITER(vec_right, XARGS);
		lsx_store_idxlo_f32(dst + vec_right, accum, right % 4);
	}
#undef XITER
#undef XARGS
}

constexpr auto resize_line_v_f32_lsx_jt_init = make_array(
	resize_line_v_f32_lsx<1, false>,
	resize_line_v_f32_lsx<2, false>,
	resize_line_v_f32_lsx<3, false>,
	resize_line_v_f32_lsx<4, false>,
	resize_line_v_f32_lsx<5, false>,
	resize_line_v_f32_lsx<6, false>,
	resize_line_v_f32_lsx<7, false>,
	resize_line_v_f32_lsx<8, false>);

constexpr auto resize_line_v_f32_lsx_jt_cont = make_array(
	resize_line_v_f32_lsx<1, true>,
	resize_line_v_f32_lsx<2, true>,
	resize_line_v_f32_lsx<3, true>,
	resize_line_v_f32_lsx<4, true>,
	resize_line_v_f32_lsx<5, true>,
	resize_line_v_f32_lsx<6, true>,
	resize_line_v_f32_lsx<7, true>,
	resize_line_v_f32_lsx<8, true>);


class ResizeImplH_U16_Lsx final : public ResizeImplH {
	decltype(resize_line8_h_u16_lsx_jt_small)::value_type m_func;
	uint16_t m_pixel_max;
public:
	ResizeImplH_U16_Lsx(const FilterContext &filter, unsigned height, unsigned depth) try :
		ResizeImplH(filter, height, PixelType::WORD),
		m_func{},
		m_pixel_max{ static_cast<uint16_t>((1UL << depth) - 1) }
	{
		m_desc.step = 8;
		m_desc.scratchpad_size = (ceil_n(checked_size_t{ filter.input_width }, 8) * sizeof(uint16_t) * 8).get();

		if (filter.filter_width <= 8)
			m_func = resize_line8_h_u16_lsx_jt_small[filter.filter_width - 1];
		else
			m_func = resize_line8_h_u16_lsx_jt_large[filter.filter_width % 8];
	} catch (const std::overflow_error &) {
		error::throw_<error::OutOfMemory>();
	}

	void process(const graphengine::BufferDescriptor *in, const graphengine::BufferDescriptor *out,
	             unsigned i, unsigned left, unsigned right, void *, void *tmp) const noexcept override
	{
		auto range = get_col_deps(left, right);

		const uint16_t *src_ptr[8] = { 0 };
		uint16_t *dst_ptr[8] = { 0 };
		uint16_t *transpose_buf = static_cast<uint16_t *>(tmp);
		unsigned height = m_desc.format.height;

		for (unsigned n = 0; n < 8; ++n) {
			src_ptr[n] = in->get_line<uint16_t>(std::min(i + n, height - 1));
		}

		transpose_line_8x8_u16(transpose_buf, src_ptr, floor_n(range.first, 8), ceil_n(range.second, 8));

		for (unsigned n = 0; n < 8; ++n) {
			dst_ptr[n] = out->get_line<uint16_t>(std::min(i + n, height - 1));
		}

		m_func(m_filter.left.data(), m_filter.data_i16.data(), m_filter.stride_i16, m_filter.filter_width,
		       transpose_buf, dst_ptr, floor_n(range.first, 8), left, right, m_pixel_max);
	}
};


class ResizeImplH_F32_Lsx final : public ResizeImplH {
	decltype(resize_line4_h_f32_lsx_jt_small)::value_type m_func;
public:
	ResizeImplH_F32_Lsx(const FilterContext &filter, unsigned height) try :
		ResizeImplH(filter, height, PixelType::FLOAT),
		m_func{}
	{
		m_desc.step = 4;
		m_desc.scratchpad_size = (ceil_n(checked_size_t{ filter.input_width }, 4) * sizeof(float) * 4).get();

		if (filter.filter_width <= 8)
			m_func = resize_line4_h_f32_lsx_jt_small[filter.filter_width - 1];
		else
			m_func = resize_line4_h_f32_lsx_jt_large[filter.filter_width % 4];
	} catch (const std::overflow_error &) {
		error::throw_<error::OutOfMemory>();
	}

	void process(const graphengine::BufferDescriptor *in, const graphengine::BufferDescriptor *out,
	             unsigned i, unsigned left, unsigned right, void *, void *tmp) const noexcept override
	{
		auto range = get_col_deps(left, right);

		const float *src_ptr[4] = { 0 };
		float *dst_ptr[4] = { 0 };
		float *transpose_buf = static_cast<float *>(tmp);
		unsigned height = m_desc.format.height;

		src_ptr[0] = in->get_line<float>(std::min(i + 0, height - 1));
		src_ptr[1] = in->get_line<float>(std::min(i + 1, height - 1));
		src_ptr[2] = in->get_line<float>(std::min(i + 2, height - 1));
		src_ptr[3] = in->get_line<float>(std::min(i + 3, height - 1));

		transpose_line_4x4_f32(transpose_buf, src_ptr[0], src_ptr[1], src_ptr[2], src_ptr[3], floor_n(range.first, 4), ceil_n(range.second, 4));

		dst_ptr[0] = out->get_line<float>(std::min(i + 0, height - 1));
		dst_ptr[1] = out->get_line<float>(std::min(i + 1, height - 1));
		dst_ptr[2] = out->get_line<float>(std::min(i + 2, height - 1));
		dst_ptr[3] = out->get_line<float>(std::min(i + 3, height - 1));

		m_func(m_filter.left.data(), m_filter.data.data(), m_filter.stride, m_filter.filter_width,
		       transpose_buf, dst_ptr, floor_n(range.first, 4), left, right);
	}
};


class ResizeImplV_U16_Lsx : public ResizeImplV {
	uint16_t m_pixel_max;
public:
	ResizeImplV_U16_Lsx(const FilterContext &filter, unsigned width, unsigned depth) try :
		ResizeImplV(filter, width, PixelType::WORD),
		m_pixel_max{ static_cast<uint16_t>((1UL << depth) - 1) }
	{
		if (m_filter.filter_width > 8)
			m_desc.scratchpad_size = (ceil_n(checked_size_t{ width }, 8) * sizeof(uint32_t)).get();
	} catch (const std::overflow_error &) {
		error::throw_<error::OutOfMemory>();
	}

	void process(const graphengine::BufferDescriptor *in, const graphengine::BufferDescriptor *out,
	             unsigned i, unsigned left, unsigned right, void *, void *tmp) const noexcept override
	{
		const int16_t *filter_data = m_filter.data_i16.data() + i * m_filter.stride_i16;
		unsigned filter_width = m_filter.filter_width;
		unsigned src_height = m_filter.input_width;

		const uint16_t *src_lines[8] = { 0 };
		uint16_t *dst_line = out->get_line<uint16_t>(i);
		int32_t *accum_buf = static_cast<int32_t *>(tmp);

		unsigned top = m_filter.left[i];

		auto gather_8_lines = [&](unsigned i)
		{
			for (unsigned n = 0; n < 8; ++n) {
				src_lines[n] = in->get_line<uint16_t>(std::min(i + n, src_height - 1));
			}
		};

#define XARGS src_lines, dst_line, accum_buf, left, right, m_pixel_max
		if (filter_width <= 8) {
			gather_8_lines(top);
			resize_line_v_u16_lsx_jt_small[filter_width - 1](filter_data, XARGS);
		} else {
			unsigned k_end = ceil_n(filter_width, 8) - 8;

			gather_8_lines(top);
			resize_line_v_u16_lsx_initial(filter_data + 0, XARGS);

			for (unsigned k = 8; k < k_end; k += 8) {
				gather_8_lines(top + k);
				resize_line_v_u16_lsx_update(filter_data + k, XARGS);
			}

			gather_8_lines(top + k_end);
			resize_line_v_u16_lsx_jt_final[filter_width - k_end - 1](filter_data + k_end, XARGS);
		}
#undef XARGS
	}
};


class ResizeImplV_F32_Lsx : public ResizeImplV {
public:
	ResizeImplV_F32_Lsx(const FilterContext &filter, unsigned width) :
		ResizeImplV(filter, width, PixelType::FLOAT)
	{}

	void process(const graphengine::BufferDescriptor *in, const graphengine::BufferDescriptor *out,
	             unsigned i, unsigned left, unsigned right, void *, void *) const noexcept override
	{
		const float *filter_data = m_filter.data.data() + i * m_filter.stride;
		unsigned filter_width = m_filter.filter_width;
		unsigned src_height = m_filter.input_width;

		const float *src_lines[8] = { 0 };
		float *dst_line = out->get_line<float>(i);

		{
			unsigned taps_remain = std::min(filter_width - 0, 8U);
			unsigned top = m_filter.left[i] + 0;

			src_lines[0] = in->get_line<float>(std::min(top + 0, src_height - 1));
			src_lines[1] = in->get_line<float>(std::min(top + 1, src_height - 1));
			src_lines[2] = in->get_line<float>(std::min(top + 2, src_height - 1));
			src_lines[3] = in->get_line<float>(std::min(top + 3, src_height - 1));
			src_lines[4] = in->get_line<float>(std::min(top + 4, src_height - 1));
			src_lines[5] = in->get_line<float>(std::min(top + 5, src_height - 1));
			src_lines[6] = in->get_line<float>(std::min(top + 6, src_height - 1));
			src_lines[7] = in->get_line<float>(std::min(top + 7, src_height - 1));

			resize_line_v_f32_lsx_jt_init[taps_remain - 1](filter_data + 0, src_lines, dst_line, left, right);
		}

		for (unsigned k = 8; k < filter_width; k += 8) {
			unsigned taps_remain = std::min(filter_width - k, 8U);
			unsigned top = m_filter.left[i] + k;

			src_lines[0] = in->get_line<float>(std::min(top + 0, src_height - 1));
			src_lines[1] = in->get_line<float>(std::min(top + 1, src_height - 1));
			src_lines[2] = in->get_line<float>(std::min(top + 2, src_height - 1));
			src_lines[3] = in->get_line<float>(std::min(top + 3, src_height - 1));
			src_lines[4] = in->get_line<float>(std::min(top + 4, src_height - 1));
			src_lines[5] = in->get_line<float>(std::min(top + 5, src_height - 1));
			src_lines[6] = in->get_line<float>(std::min(top + 6, src_height - 1));
			src_lines[7] = in->get_line<float>(std::min(top + 7, src_height - 1));

			resize_line_v_f32_lsx_jt_cont[taps_remain - 1](filter_data + k, src_lines, dst_line, left, right);
		}
	}
};

} // namespace


std::unique_ptr<graphengine::Filter> create_resize_impl_h_lsx(const FilterContext &context, unsigned height, PixelType type, unsigned depth)
{
	std::unique_ptr<graphengine::Filter> ret;

	if (type == PixelType::FLOAT)
		ret = std::make_unique<ResizeImplH_F32_Lsx>(context, height);
	else if (type == PixelType::WORD)
		ret = std::make_unique<ResizeImplH_U16_Lsx>(context, height, depth);

	return ret;
}

std::unique_ptr<graphengine::Filter> create_resize_impl_v_lsx(const FilterContext &context, unsigned width, PixelType type, unsigned depth)
{
	std::unique_ptr<graphengine::Filter> ret;

	if (type == PixelType::FLOAT)
		ret = std::make_unique<ResizeImplV_F32_Lsx>(context, width);
	else if (type == PixelType::WORD)
		ret = std::make_unique<ResizeImplV_U16_Lsx>(context, width, depth);

	return ret;
}

} // namespace zimg::resize

#endif // ZIMG_LOONGARCH
