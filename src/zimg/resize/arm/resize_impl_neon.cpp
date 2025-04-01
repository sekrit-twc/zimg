#ifdef ZIMG_ARM

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <arm_neon.h>
#include "common/align.h"
#include "common/ccdep.h"
#include "common/checked_int.h"
#include "common/except.h"
#include "common/make_array.h"
#include "common/pixel.h"
#include "common/unroll.h"
#include "resize/resize_impl.h"
#include "resize_impl_arm.h"

#include "common/arm/neon_util.h"

namespace zimg::resize {

namespace {

void transpose_line_8x8_u16(uint16_t * RESTRICT dst, const uint16_t * const * RESTRICT src, unsigned left, unsigned right)
{
	for (unsigned j = left; j < right; j += 8) {
		uint16x8_t x0, x1, x2, x3, x4, x5, x6, x7;

		x0 = vld1q_u16(src[0] + j);
		x1 = vld1q_u16(src[1] + j);
		x2 = vld1q_u16(src[2] + j);
		x3 = vld1q_u16(src[3] + j);
		x4 = vld1q_u16(src[4] + j);
		x5 = vld1q_u16(src[5] + j);
		x6 = vld1q_u16(src[6] + j);
		x7 = vld1q_u16(src[7] + j);

		neon_transpose8_u16(x0, x1, x2, x3, x4, x5, x6, x7);

		vst1q_u16(dst + 0, x0);
		vst1q_u16(dst + 8, x1);
		vst1q_u16(dst + 16, x2);
		vst1q_u16(dst + 24, x3);
		vst1q_u16(dst + 32, x4);
		vst1q_u16(dst + 40, x5);
		vst1q_u16(dst + 48, x6);
		vst1q_u16(dst + 56, x7);

		dst += 64;
	}
}

void transpose_line_4x4_f32(float * RESTRICT dst, const float *src_p0, const float *src_p1, const float *src_p2, const float *src_p3, unsigned left, unsigned right)
{
	for (unsigned j = left; j < right; j += 4) {
		float32x4_t x0, x1, x2, x3;

		x0 = vld1q_f32(src_p0 + j);
		x1 = vld1q_f32(src_p1 + j);
		x2 = vld1q_f32(src_p2 + j);
		x3 = vld1q_f32(src_p3 + j);

		neon_transpose4_f32(x0, x1, x2, x3);

		vst1q_f32(dst + 0, x0);
		vst1q_f32(dst + 4, x1);
		vst1q_f32(dst + 8, x2);
		vst1q_f32(dst + 12, x3);

		dst += 16;
	}
}

inline FORCE_INLINE int16x8_t export_i30_u16(int32x4_t lo, int32x4_t hi)
{
	const int32x4_t round = vdupq_n_s32(1 << 13);

	lo = vaddq_s32(lo, round);
	hi = vaddq_s32(hi, round);

	lo = vshrq_n_s32(lo, 14);
	hi = vshrq_n_s32(hi, 14);

	return vqmovn_high_s32(vqmovn_s32(lo), hi);
}


template <int Taps>
inline FORCE_INLINE uint16x8_t resize_line8_h_u16_neon_xiter(unsigned j, const unsigned *filter_left, const int16_t *filter_data, unsigned filter_stride, unsigned filter_width,
                                                             const uint16_t *src, unsigned src_base, uint16_t limit)
{
	static_assert(Taps <= 8, "only up to 8 taps can be unrolled");
	static_assert(Taps >= -7, "only up to 7 taps in epilogue");
	constexpr int Tail = Taps > 0 ? Taps : -Taps;

	const int16x8_t i16_min = vdupq_n_s16(INT16_MIN);
	const int16x8_t lim = vdupq_n_s16(limit + INT16_MIN);

	const int16_t *filter_coeffs = filter_data + j * filter_stride;
	const uint16_t *src_p = src + (filter_left[j] - src_base) * 8;

	int32x4_t accum_lo = vdupq_n_s32(0);
	int32x4_t accum_hi = vdupq_n_s32(0);
	int16x8_t coeffs;

	auto f = ZIMG_UNROLL_FUNC(kk)
	{
		int16x8_t c, x;

		c = vdupq_laneq_s16(coeffs, static_cast<unsigned>(kk));
		x = vreinterpretq_s16_u16(vld1q_u16(src_p + kk * 8));
		x = vaddq_s16(x, i16_min);
		accum_lo = vmlal_s16(accum_lo, vget_low_s16(x), vget_low_s16(c));
		accum_hi = vmlal_high_s16(accum_hi, x, c);
	};

	unsigned k_end = Taps > 0 ? 0 : floor_n(filter_width + 1, 8);

	for (unsigned k = 0; k < k_end; k += 8) {
		coeffs = vld1q_s16(filter_coeffs + k);
		unroll<8>(f);
		src_p += 64;
	}

	if constexpr (Tail) {
		coeffs = vld1q_s16(filter_coeffs + k_end);
		unroll<Tail>(f);
	}

	int16x8_t result = export_i30_u16(accum_lo, accum_hi);
	result = vminq_s16(result, lim);
	result = vsubq_s16(result, i16_min);
	return vreinterpretq_u16_s16(result);
}

template <int Taps>
void resize_line8_h_u16_neon(const unsigned * RESTRICT filter_left, const int16_t * RESTRICT filter_data, unsigned filter_stride, unsigned filter_width,
                             const uint16_t * RESTRICT src, uint16_t * const * RESTRICT dst, unsigned src_base, unsigned left, unsigned right, uint16_t limit)
{
	unsigned vec_left = ceil_n(left, 8);
	unsigned vec_right = floor_n(right, 8);

#define XITER resize_line8_h_u16_neon_xiter<Taps>
#define XARGS filter_left, filter_data, filter_stride, filter_width, src, src_base, limit
	for (unsigned j = left; j < vec_left; ++j) {
		uint16x8_t x = XITER(j, XARGS);
		neon_scatter_u16(dst[0] + j, dst[1] + j, dst[2] + j, dst[3] + j, dst[4] + j, dst[5] + j, dst[6] + j, dst[7] + j, x);
	}

	for (unsigned j = vec_left; j < vec_right; j += 8) {
		uint16x8_t x0, x1, x2, x3, x4, x5, x6, x7;

		x0 = XITER(j + 0, XARGS);
		x1 = XITER(j + 1, XARGS);
		x2 = XITER(j + 2, XARGS);
		x3 = XITER(j + 3, XARGS);
		x4 = XITER(j + 4, XARGS);
		x5 = XITER(j + 5, XARGS);
		x6 = XITER(j + 6, XARGS);
		x7 = XITER(j + 7, XARGS);

		neon_transpose8_u16(x0, x1, x2, x3, x4, x5, x6, x7);

		vst1q_u16(dst[0] + j, x0);
		vst1q_u16(dst[1] + j, x1);
		vst1q_u16(dst[2] + j, x2);
		vst1q_u16(dst[3] + j, x3);
		vst1q_u16(dst[4] + j, x4);
		vst1q_u16(dst[5] + j, x5);
		vst1q_u16(dst[6] + j, x6);
		vst1q_u16(dst[7] + j, x7);
	}

	for (unsigned j = vec_right; j < right; ++j) {
		uint16x8_t x = XITER(j, XARGS);
		neon_scatter_u16(dst[0] + j, dst[1] + j, dst[2] + j, dst[3] + j, dst[4] + j, dst[5] + j, dst[6] + j, dst[7] + j, x);
	}
#undef XITER
#undef XARGS
}

constexpr auto resize_line8_h_u16_neon_jt_small = make_array(
	resize_line8_h_u16_neon<1>,
	resize_line8_h_u16_neon<2>,
	resize_line8_h_u16_neon<3>,
	resize_line8_h_u16_neon<4>,
	resize_line8_h_u16_neon<5>,
	resize_line8_h_u16_neon<6>,
	resize_line8_h_u16_neon<7>,
	resize_line8_h_u16_neon<8>);

constexpr auto resize_line8_h_u16_neon_jt_large = make_array(
	resize_line8_h_u16_neon<0>,
	resize_line8_h_u16_neon<-1>,
	resize_line8_h_u16_neon<-2>,
	resize_line8_h_u16_neon<-3>,
	resize_line8_h_u16_neon<-4>,
	resize_line8_h_u16_neon<-5>,
	resize_line8_h_u16_neon<-6>,
	resize_line8_h_u16_neon<-7>);


template <int Taps>
inline FORCE_INLINE float32x4_t resize_line4_h_f32_neon_xiter(unsigned j, const unsigned *filter_left, const float *filter_data, unsigned filter_stride, unsigned filter_width,
                                                              const float *src, unsigned src_base)
{
	static_assert(Taps <= 8, "only up to 8 taps can be unrolled");
	static_assert(Taps >= -3, "only up to 3 taps in epilogue");
	constexpr int Tail = Taps >= 4 ? Taps - 4 : Taps > 0 ? Taps : -Taps;

	const float *filter_coeffs = filter_data + j * filter_stride;
	const float *src_p = src + (filter_left[j] - src_base) * 4;

	float32x4_t accum0 = vdupq_n_f32(0.0f);
	float32x4_t accum1 = vdupq_n_f32(0.0f);
	float32x4_t coeffs;

	auto f = ZIMG_UNROLL_FUNC(kk)
	{
		float32x4_t &acc = kk % 2 ? accum1 : accum0;

		float32x4_t c = vdupq_laneq_f32(coeffs, static_cast<int>(kk));
		float32x4_t x = vld1q_f32(src_p + 4 * static_cast<int>(kk));
		acc = vfmaq_f32(acc, c, x);
	};

	unsigned k_end = Taps >= 4 ? 4 : Taps > 0 ? 0 : floor_n(filter_width, 4);

	for (unsigned k = 0; k < k_end; k += 4) {
		coeffs = vld1q_f32(filter_coeffs + k);
		unroll<4>(f);
		src_p += 16;
	}

	if constexpr (Tail) {
		coeffs = vld1q_f32(filter_coeffs + k_end);
		unroll<Tail>(f);
	}

	if constexpr (Taps <= 0 || Taps >= 2)
		accum0 = vaddq_f32(accum0, accum1);

	return accum0;
}

template <int Taps>
void resize_line4_h_f32_neon(const unsigned * RESTRICT filter_left, const float * RESTRICT filter_data, unsigned filter_stride, unsigned filter_width,
                            const float * RESTRICT src, float * const * RESTRICT dst, unsigned src_base, unsigned left, unsigned right)
{
	unsigned vec_left = ceil_n(left, 4);
	unsigned vec_right = floor_n(right, 4);

#define XITER resize_line4_h_f32_neon_xiter<Taps>
#define XARGS filter_left, filter_data, filter_stride, filter_width, src, src_base
	for (unsigned j = left; j < vec_left; ++j) {
		float32x4_t x = XITER(j, XARGS);
		neon_scatter_f32(dst[0] + j, dst[1] + j, dst[2] + j, dst[3] + j, x);
	}

	for (unsigned j = vec_left; j < vec_right; j += 4) {
		float32x4_t x0, x1, x2, x3;

		x0 = XITER(j + 0, XARGS);
		x1 = XITER(j + 1, XARGS);
		x2 = XITER(j + 2, XARGS);
		x3 = XITER(j + 3, XARGS);

		neon_transpose4_f32(x0, x1, x2, x3);

		vst1q_f32(dst[0] + j, x0);
		vst1q_f32(dst[1] + j, x1);
		vst1q_f32(dst[2] + j, x2);
		vst1q_f32(dst[3] + j, x3);
	}

	for (unsigned j = vec_right; j < right; ++j) {
		float32x4_t x = XITER(j, XARGS);
		neon_scatter_f32(dst[0] + j, dst[1] + j, dst[2] + j, dst[3] + j, x);
	}
#undef XITER
#undef XARGS
}

constexpr auto resize_line4_h_f32_neon_jt_small = make_array(
	resize_line4_h_f32_neon<1>,
	resize_line4_h_f32_neon<2>,
	resize_line4_h_f32_neon<3>,
	resize_line4_h_f32_neon<4>,
	resize_line4_h_f32_neon<5>,
	resize_line4_h_f32_neon<6>,
	resize_line4_h_f32_neon<7>,
	resize_line4_h_f32_neon<8>);

constexpr auto resize_line4_h_f32_neon_jt_large = make_array(
	resize_line4_h_f32_neon<0>,
	resize_line4_h_f32_neon<-1>,
	resize_line4_h_f32_neon<-2>,
	resize_line4_h_f32_neon<-3>);


constexpr unsigned V_ACCUM_NONE = 0;
constexpr unsigned V_ACCUM_INITIAL = 1;
constexpr unsigned V_ACCUM_UPDATE = 2;
constexpr unsigned V_ACCUM_FINAL = 3;

template <unsigned Taps, unsigned AccumMode>
inline FORCE_INLINE uint16x8_t resize_line_v_u16_neon_xiter(unsigned j, unsigned accum_base, const uint16_t * const srcp[8],
                                                            int32_t * RESTRICT accum_p, const int16x8_t c[8], uint16_t limit)
{
	static_assert(Taps >= 1 && Taps <= 8, "must have between 1-8 taps");

	const int16x8_t i16_min = vdupq_n_s16(INT16_MIN);
	const int16x8_t lim = vdupq_n_s16(limit + INT16_MIN);

	int32x4_t accum_lo = vdupq_n_s32(0);
	int32x4_t accum_hi = vdupq_n_s32(0);

	unroll<Taps>(ZIMG_UNROLL_FUNC(k)
	{
		int16x8_t x;

		x = vreinterpretq_s16_u16(vld1q_u16(srcp[k] + j));
		x = vaddq_s16(x, i16_min);

		if constexpr (k == 0 && (AccumMode == V_ACCUM_UPDATE || AccumMode == V_ACCUM_FINAL)) {
			accum_lo = vmlal_s16(vld1q_s32(accum_p + j - accum_base + 0), vget_low_s16(c[k]), vget_low_s16(x));
			accum_hi = vmlal_high_s16(vld1q_s32(accum_p + j - accum_base + 4), c[k], x);
		} else if constexpr (k == 0) {
			accum_lo = vmull_s16(vget_low_s16(c[k]), vget_low_s16(x));
			accum_hi = vmull_high_s16(c[k], x);
		} else {
			accum_lo = vmlal_s16(accum_lo, vget_low_s16(c[k]), vget_low_s16(x));
			accum_hi = vmlal_high_s16(accum_hi, c[k], x);
		}
	});

	if constexpr (AccumMode == V_ACCUM_INITIAL || AccumMode == V_ACCUM_UPDATE) {
		vst1q_s32(accum_p + j - accum_base + 0, accum_lo);
		vst1q_s32(accum_p + j - accum_base + 4, accum_hi);
		return vdupq_n_u16(0);
	} else {
		int16x8_t result = export_i30_u16(accum_lo, accum_hi);
		result = vminq_s16(result, lim);
		result = vsubq_s16(result, i16_min);
		return vreinterpretq_u16_s16(result);
	}
}

template <unsigned Taps, unsigned AccumMode>
void resize_line_v_u16_neon(const int16_t * RESTRICT filter_data, const uint16_t * const * RESTRICT src, uint16_t * RESTRICT dst, int32_t * RESTRICT accum, unsigned left, unsigned right, uint16_t limit)
{
	const uint16_t *srcp[8] = {src[0], src[1], src[2], src[3], src[4], src[5], src[6], src[7]};
	unsigned vec_left = ceil_n(left, 8);
	unsigned vec_right = floor_n(right, 8);
	unsigned accum_base = floor_n(left, 8);

	const int16x8_t c[8] = {
		vdupq_n_s16(filter_data[0]),
		vdupq_n_s16(filter_data[1]),
		vdupq_n_s16(filter_data[2]),
		vdupq_n_s16(filter_data[3]),
		vdupq_n_s16(filter_data[4]),
		vdupq_n_s16(filter_data[5]),
		vdupq_n_s16(filter_data[6]),
		vdupq_n_s16(filter_data[7]),
	};

#define XITER resize_line_v_u16_neon_xiter<Taps, AccumMode>
#define XARGS accum_base, srcp, accum, c, limit
	if (left != vec_left) {
		uint16x8_t out = XITER(vec_left - 8, XARGS);

		if (AccumMode == V_ACCUM_NONE || AccumMode == V_ACCUM_FINAL)
			neon_store_idxhi_u16(dst + vec_left - 8, out, left % 8);
	}

	for (unsigned j = vec_left; j < vec_right; j += 8) {
		uint16x8_t out = XITER(j, XARGS);

		if (AccumMode == V_ACCUM_NONE || AccumMode == V_ACCUM_FINAL)
			vst1q_u16(dst + j, out);
	}

	if (right != vec_right) {
		uint16x8_t out = XITER(vec_right, XARGS);

		if (AccumMode == V_ACCUM_NONE || AccumMode == V_ACCUM_FINAL)
			neon_store_idxlo_u16(dst + vec_right, out, right % 8);
	}
#undef XITER
#undef XARGS
}

constexpr auto resize_line_v_u16_neon_jt_small = make_array(
	resize_line_v_u16_neon<1, V_ACCUM_NONE>,
	resize_line_v_u16_neon<2, V_ACCUM_NONE>,
	resize_line_v_u16_neon<3, V_ACCUM_NONE>,
	resize_line_v_u16_neon<4, V_ACCUM_NONE>,
	resize_line_v_u16_neon<5, V_ACCUM_NONE>,
	resize_line_v_u16_neon<6, V_ACCUM_NONE>,
	resize_line_v_u16_neon<7, V_ACCUM_NONE>,
	resize_line_v_u16_neon<8, V_ACCUM_NONE>);

constexpr auto resize_line_v_u16_neon_initial = resize_line_v_u16_neon<8, V_ACCUM_INITIAL>;
constexpr auto resize_line_v_u16_neon_update = resize_line_v_u16_neon<8, V_ACCUM_UPDATE>;

constexpr auto resize_line_v_u16_neon_jt_final = make_array(
	resize_line_v_u16_neon<1, V_ACCUM_FINAL>,
	resize_line_v_u16_neon<2, V_ACCUM_FINAL>,
	resize_line_v_u16_neon<3, V_ACCUM_FINAL>,
	resize_line_v_u16_neon<4, V_ACCUM_FINAL>,
	resize_line_v_u16_neon<5, V_ACCUM_FINAL>,
	resize_line_v_u16_neon<6, V_ACCUM_FINAL>,
	resize_line_v_u16_neon<7, V_ACCUM_FINAL>,
	resize_line_v_u16_neon<8, V_ACCUM_FINAL>);


template <unsigned Taps, bool Continue>
inline FORCE_INLINE float32x4_t resize_line_v_f32_neon_xiter(unsigned j, const float * const srcp[8], const float *accum_p, const float32x4_t c[8])
{
	static_assert(Taps >= 1 && Taps <= 8, "must have between 1-8 taps");

	float32x4_t accum0 = vdupq_n_f32(0.0f);
	float32x4_t accum1 = vdupq_n_f32(0.0f);

	unroll<Taps>(ZIMG_UNROLL_FUNC(k)
	{
		float32x4_t &acc = k % 2 ? accum1 : accum0;
		float32x4_t x;

		x = vld1q_f32(srcp[k] + j);

		if constexpr (k == 0 && Continue)
			acc = vfmaq_f32(vld1q_f32(accum_p + j), c[k], x);
		else if constexpr (k == 0 || k == 1)
			acc = vmulq_f32(c[k], x);
		else
			acc = vfmaq_f32(acc, c[k], x);
	});

	if constexpr (Taps >= 2) accum0 = vaddq_f32(accum0, accum1);
	return accum0;
}

template <unsigned Taps, bool Continue>
void resize_line_v_f32_neon(const float * RESTRICT filter_data, const float * const * RESTRICT src, float * RESTRICT dst, unsigned left, unsigned right)
{
	const float *srcp[8] = { src[0], src[1], src[2], src[3], src[4], src[5], src[6], src[7] };
	unsigned vec_left = ceil_n(left, 4);
	unsigned vec_right = floor_n(right, 4);

	const float32x4_t c[8] = {
		vdupq_n_f32(filter_data[0]),
		vdupq_n_f32(filter_data[1]),
		vdupq_n_f32(filter_data[2]),
		vdupq_n_f32(filter_data[3]),
		vdupq_n_f32(filter_data[4]),
		vdupq_n_f32(filter_data[5]),
		vdupq_n_f32(filter_data[6]),
		vdupq_n_f32(filter_data[7]),
	};

#define XITER resize_line_v_f32_neon_xiter<Taps, Continue>
#define XARGS srcp, dst, c
	if (left != vec_left) {
		float32x4_t accum = XITER(vec_left - 4, XARGS);
		neon_store_idxhi_f32(dst + vec_left - 4, accum, left % 4);
	}

	for (unsigned j = vec_left; j < vec_right; j += 4) {
		float32x4_t accum = XITER(j, XARGS);
		vst1q_f32(dst + j, accum);
	}

	if (right != vec_right) {
		float32x4_t accum = XITER(vec_right, XARGS);
		neon_store_idxlo_f32(dst + vec_right, accum, right % 4);
	}
#undef XITER
#undef XARGS
}

constexpr auto resize_line_v_f32_neon_jt_init = make_array(
	resize_line_v_f32_neon<1, false>,
	resize_line_v_f32_neon<2, false>,
	resize_line_v_f32_neon<3, false>,
	resize_line_v_f32_neon<4, false>,
	resize_line_v_f32_neon<5, false>,
	resize_line_v_f32_neon<6, false>,
	resize_line_v_f32_neon<7, false>,
	resize_line_v_f32_neon<8, false>);

constexpr auto resize_line_v_f32_neon_jt_cont = make_array(
	resize_line_v_f32_neon<1, true>,
	resize_line_v_f32_neon<2, true>,
	resize_line_v_f32_neon<3, true>,
	resize_line_v_f32_neon<4, true>,
	resize_line_v_f32_neon<5, true>,
	resize_line_v_f32_neon<6, true>,
	resize_line_v_f32_neon<7, true>,
	resize_line_v_f32_neon<8, true>);


class ResizeImplH_U16_Neon final : public ResizeImplH {
	decltype(resize_line8_h_u16_neon_jt_small)::value_type m_func;
	uint16_t m_pixel_max;
public:
	ResizeImplH_U16_Neon(const FilterContext &filter, unsigned height, unsigned depth) try :
		ResizeImplH(filter, height, PixelType::WORD),
		m_func{},
		m_pixel_max{ static_cast<uint16_t>((1UL << depth) - 1) }
	{
		m_desc.step = 8;
		m_desc.scratchpad_size = (ceil_n(checked_size_t{ filter.input_width }, 8) * sizeof(uint16_t) * 8).get();

		if (filter.filter_width <= 8)
			m_func = resize_line8_h_u16_neon_jt_small[filter.filter_width - 1];
		else
			m_func = resize_line8_h_u16_neon_jt_large[filter.filter_width % 8];
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


class ResizeImplH_F32_Neon final : public ResizeImplH {
	decltype(resize_line4_h_f32_neon_jt_small)::value_type m_func;
public:
	ResizeImplH_F32_Neon(const FilterContext &filter, unsigned height) try :
		ResizeImplH(filter, height, PixelType::FLOAT),
		m_func{}
	{
		m_desc.step = 4;
		m_desc.scratchpad_size = (ceil_n(checked_size_t{ filter.input_width }, 4) * sizeof(float) * 4).get();

		if (filter.filter_width <= 8)
			m_func = resize_line4_h_f32_neon_jt_small[filter.filter_width - 1];
		else
			m_func = resize_line4_h_f32_neon_jt_large[filter.filter_width % 4];
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


class ResizeImplV_U16_Neon : public ResizeImplV {
	uint16_t m_pixel_max;
public:
	ResizeImplV_U16_Neon(const FilterContext &filter, unsigned width, unsigned depth) try :
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
			resize_line_v_u16_neon_jt_small[filter_width - 1](filter_data, XARGS);
		} else {
			unsigned k_end = ceil_n(filter_width, 8) - 8;

			gather_8_lines(top);
			resize_line_v_u16_neon_initial(filter_data + 0, XARGS);

			for (unsigned k = 8; k < k_end; k += 8) {
				gather_8_lines(top + k);
				resize_line_v_u16_neon_update(filter_data + k, XARGS);
			}

			gather_8_lines(top + k_end);
			resize_line_v_u16_neon_jt_final[filter_width - k_end - 1](filter_data + k_end, XARGS);
		}
#undef XARGS
	}
};


class ResizeImplV_F32_Neon : public ResizeImplV {
public:
	ResizeImplV_F32_Neon(const FilterContext &filter, unsigned width) :
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

			resize_line_v_f32_neon_jt_init[taps_remain - 1](filter_data + 0, src_lines, dst_line, left, right);
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

			resize_line_v_f32_neon_jt_cont[taps_remain - 1](filter_data + k, src_lines, dst_line, left, right);
		}
	}
};

} // namespace


std::unique_ptr<graphengine::Filter> create_resize_impl_h_neon(const FilterContext &context, unsigned height, PixelType type, unsigned depth)
{
	std::unique_ptr<graphengine::Filter> ret;

	if (type == PixelType::FLOAT)
		ret = std::make_unique<ResizeImplH_F32_Neon>(context, height);
	else if (type == PixelType::WORD)
		ret = std::make_unique<ResizeImplH_U16_Neon>(context, height, depth);

	return ret;
}

std::unique_ptr<graphengine::Filter> create_resize_impl_v_neon(const FilterContext &context, unsigned width, PixelType type, unsigned depth)
{
	std::unique_ptr<graphengine::Filter> ret;

	if (type == PixelType::FLOAT)
		ret = std::make_unique<ResizeImplV_F32_Neon>(context, width);
	else if (type == PixelType::WORD)
		ret = std::make_unique<ResizeImplV_U16_Neon>(context, width, depth);

	return ret;
}

} // namespace zimg::resize

#endif // ZIMG_ARM
