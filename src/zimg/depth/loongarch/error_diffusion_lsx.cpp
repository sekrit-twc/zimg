#ifdef ZIMG_LOONGARCH

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <lsxintrin.h>
#include "common/align.h"
#include "common/ccdep.h"
#include "common/checked_int.h"
#include "common/except.h"
#include "common/pixel.h"
#include "common/zassert.h"
#include "depth/quantize.h"
#include "graph/filter_base.h"
#include "dither_loongarch.h"

#include "common/loongarch/lsx_util.h"

namespace zimg::depth {

namespace {

template <class T>
struct Buffer {
	const graphengine::BufferDescriptor &buffer;

	Buffer(const graphengine::BufferDescriptor &buffer)
		: buffer{ buffer }
	{}

	T *operator[](unsigned i) const { return buffer.get_line<T>(i); }
};


struct error_state {
	float err_left[8];
	float err_top_right[8];
	float err_top[8];
	float err_top_left[8];
};


template <PixelType Type>
struct error_diffusion_traits;

template <>
struct error_diffusion_traits<PixelType::BYTE> {
	typedef uint8_t type;

	static float load1(const uint8_t *ptr) { return *ptr; }
	static void store1(uint8_t *ptr, uint32_t x)
	{
		*ptr = static_cast<uint8_t>(x);
	}

	static void load8(const uint8_t *ptr, __m128 &lo, __m128 &hi)
	{
		__m128i x = __lsx_vld((void *)ptr, 0);
		__m128i lo_w = __lsx_vsllwil_hu_bu(x, 0);
		lo = (__m128)__lsx_vffint_s_wu(__lsx_vsllwil_wu_hu(lo_w, 0));
		hi = (__m128)__lsx_vffint_s_wu(__lsx_vexth_wu_hu(lo_w));
	}

	static void store8(uint8_t *ptr, __m128i x)
	{
		__m128i b = __lsx_vpickev_b(x, x);
		__lsx_vstelm_d(b, ptr, 0, 0);
	}
};

template <>
struct error_diffusion_traits<PixelType::WORD> {
	typedef uint16_t type;

	static float load1(const uint16_t *ptr) { return *ptr; }
	static void store1(uint16_t *ptr, uint32_t x)
	{
		*ptr = static_cast<uint16_t>(x);
	}

	static void load8(const uint16_t *ptr, __m128 &lo, __m128 &hi)
	{
		__m128i x = __lsx_vld((void *)ptr, 0);
		lo = (__m128)__lsx_vffint_s_wu(__lsx_vsllwil_wu_hu(x, 0));
		hi = (__m128)__lsx_vffint_s_wu(__lsx_vexth_wu_hu(x));
	}

	static void store8(uint16_t *ptr, __m128i x)
	{
		__lsx_vst(x, ptr, 0);
	}
};

template <>
struct error_diffusion_traits<PixelType::HALF> {
	typedef uint16_t type;

	static float load1(const uint16_t *ptr)
	{
		return half_to_float(*ptr);
	}

	static void load8(const uint16_t *ptr, __m128 &lo, __m128 &hi)
	{
		// No native fp16->fp32 SIMD carries the exact bit semantics of the
		// reference (NaN payloads); so convert each half scalar and pack.
		float v[8];
		for (unsigned i = 0; i < 8; ++i)
			v[i] = half_to_float(ptr[i]);

		lo = (__m128)__lsx_vld((void *)(v + 0), 0);
		hi = (__m128)__lsx_vld((void *)(v + 4), 0);
	}
};

template <>
struct error_diffusion_traits<PixelType::FLOAT> {
	typedef float type;

	static float load1(const float *ptr) { return *ptr; }

	static void load8(const float *ptr, __m128 &lo, __m128 &hi)
	{
		lo = (__m128)__lsx_vld((void *)(ptr + 0), 0);
		hi = (__m128)__lsx_vld((void *)(ptr + 4), 0);
	}
};


template <PixelType SrcType, PixelType DstType>
void error_diffusion_scalar(const void *src, void *dst,
                             const float * RESTRICT error_top,
                             float * RESTRICT error_cur, float scale,
                             float offset, unsigned bits, unsigned width)
{
	typedef error_diffusion_traits<SrcType> src_traits;
	typedef error_diffusion_traits<DstType> dst_traits;

	const typename src_traits::type *src_p =
		static_cast<const typename src_traits::type *>(src);
	typename dst_traits::type *dst_p =
		static_cast<typename dst_traits::type *>(dst);

	const float max_val = static_cast<float>((1UL << bits) - 1);

	float err_left = error_cur[0];
	float err_top_right;
	float err_top = error_top[0 + 1];
	float err_top_left = error_top[0];

	for (unsigned j = 0; j < width; ++j) {
		// Error array is padded by one left, two on the right.
		unsigned j_err = j + 1;
		err_top_right = error_top[j_err + 1];

		float x = std::fma(src_traits::load1(src_p + j), scale, offset);
		float err0, err1, err;

		err0 = err_left * (7.0f / 16.0f);
		err0 = std::fma(err_top_right, 3.0f / 16.0f, err0);
		err1 = err_top * (5.0f / 16.0f);
		err1 = std::fma(err_top_left, 1.0f / 16.0f, err1);
		err = err0 + err1;

		x += err;
		x = std::clamp(x, 0.0f, max_val);

		uint32_t q = static_cast<uint32_t>(std::lrint(x));
		err = x - static_cast<float>(q);

		dst_traits::store1(dst_p + j, q);
		error_cur[j_err] = err;

		err_left = err;
		err_top_left = err_top;
		err_top = err_top_right;
	}
}

auto select_error_diffusion_scalar_func(PixelType pixel_in, PixelType pixel_out)
{
	if (pixel_in == PixelType::BYTE && pixel_out == PixelType::BYTE)
		return error_diffusion_scalar<PixelType::BYTE, PixelType::BYTE>;
	else if (pixel_in == PixelType::BYTE && pixel_out == PixelType::WORD)
		return error_diffusion_scalar<PixelType::BYTE, PixelType::WORD>;
	else if (pixel_in == PixelType::WORD && pixel_out == PixelType::BYTE)
		return error_diffusion_scalar<PixelType::WORD, PixelType::BYTE>;
	else if (pixel_in == PixelType::WORD && pixel_out == PixelType::WORD)
		return error_diffusion_scalar<PixelType::WORD, PixelType::WORD>;
	else if (pixel_in == PixelType::HALF && pixel_out == PixelType::BYTE)
		return error_diffusion_scalar<PixelType::HALF, PixelType::BYTE>;
	else if (pixel_in == PixelType::HALF && pixel_out == PixelType::WORD)
		return error_diffusion_scalar<PixelType::HALF, PixelType::WORD>;
	else if (pixel_in == PixelType::FLOAT && pixel_out == PixelType::BYTE)
		return error_diffusion_scalar<PixelType::FLOAT,
		                                PixelType::BYTE>;
	else if (pixel_in == PixelType::FLOAT && pixel_out == PixelType::WORD)
		return error_diffusion_scalar<PixelType::FLOAT,
		                                PixelType::WORD>;
	else
		error::throw_<error::InternalError>(
		"no conversion between pixel types");
}


struct wf_error_state {
	__m128 err_left_lo;
	__m128 err_left_hi;
	__m128 err_top_right_lo;
	__m128 err_top_right_hi;
	__m128 err_top_lo;
	__m128 err_top_hi;
	__m128 err_top_left_lo;
	__m128 err_top_left_hi;
};

// err_rot_lo = [lo[3], hi[0], hi[1], hi[2]]
// err_rot_hi = [hi[3], lo[0], lo[1], lo[2]]
inline FORCE_INLINE void lsx_rotate_err(const __m128 &lo, const __m128 &hi,
                                        __m128 &rot_lo, __m128 &rot_hi)
{
	// rot_lo = [lo[3], hi[0], hi[1], hi[2]]: shift hi right one lane,
	// then insert lo[3] into lane 0 (vextrins.w, imm=0x03).
	rot_lo = (__m128)__lsx_vextrins_w(
		__lsx_vshuf4i_w((__m128i)hi, 0x90), (__m128i)lo, 0x03);
	// rot_hi = [hi[3], lo[0], lo[1], lo[2]]: shift lo right one lane,
	// then insert hi[3] into lane 0.
	rot_hi = (__m128)__lsx_vextrins_w(
		__lsx_vshuf4i_w((__m128i)lo, 0x90), (__m128i)hi, 0x03);
}

inline FORCE_INLINE __m128i error_diffusion_wf_lsx_xiter(
	__m128 v_lo,
	__m128 v_hi,
	unsigned j,
	const float *error_top,
	float *error_cur,
	const __m128 &max_val,
	const __m128 &err_left_w,
	const __m128 &err_top_right_w,
	const __m128 &err_top_w,
	const __m128 &err_top_left_w,
	wf_error_state &state)
{
	unsigned j_err = j + 1;

	__m128 err0_lo, err0_hi, err1_lo, err1_hi;

	err0_lo = __lsx_vfmul_s(state.err_left_lo, err_left_w);
	err0_lo = __lsx_vfmadd_s(
		state.err_top_right_lo, err_top_right_w, err0_lo);
	err1_lo = __lsx_vfmul_s(state.err_top_lo, err_top_w);
	err1_lo = __lsx_vfmadd_s(
		state.err_top_left_lo, err_top_left_w, err1_lo);
	err0_lo = __lsx_vfadd_s(err0_lo, err1_lo);

	err0_hi = __lsx_vfmul_s(state.err_left_hi, err_left_w);
	err0_hi = __lsx_vfmadd_s(
		state.err_top_right_hi, err_top_right_w, err0_hi);
	err1_hi = __lsx_vfmul_s(state.err_top_hi, err_top_w);
	err1_hi = __lsx_vfmadd_s(
		state.err_top_left_hi, err_top_left_w, err1_hi);
	err0_hi = __lsx_vfadd_s(err0_hi, err1_hi);

	__m128 x_lo = __lsx_vfadd_s(v_lo, err0_lo);
	__m128 x_hi = __lsx_vfadd_s(v_hi, err0_hi);

	float zero_f = 0.0f;
	x_lo = __lsx_vfmax_s(x_lo, (__m128)__lsx_vldrepl_w(&zero_f, 0));
	x_hi = __lsx_vfmax_s(x_hi, (__m128)__lsx_vldrepl_w(&zero_f, 0));

	x_lo = __lsx_vfmin_s(x_lo, max_val);
	x_hi = __lsx_vfmin_s(x_hi, max_val);

	__m128i q_lo = __lsx_vftintrne_w_s(x_lo);
	__m128i q_hi = __lsx_vftintrne_w_s(x_hi);

	__m128 y_lo = (__m128)__lsx_vffint_s_wu(q_lo);
	__m128 y_hi = (__m128)__lsx_vffint_s_wu(q_hi);

	err0_lo = __lsx_vfsub_s(x_lo, y_lo);
	err0_hi = __lsx_vfsub_s(x_hi, y_hi);

	// Left-rotate err0 by 32 bits across the 8-lane [err0_hi, err0_lo].
	__m128 err_rot_lo, err_rot_hi;
	lsx_rotate_err(err0_hi, err0_lo, err_rot_lo, err_rot_hi);

	// Extract the previous high error.
	__lsx_vstelm_w((__m128i)err_rot_lo, error_cur + j_err + 0, 0, 0);

	// Insert the next error into the low position.
	float next_err = error_top[j_err + 14 + 2];
	uint32_t next_bits;
	memcpy(&next_bits, &next_err, sizeof(next_bits));
	err_rot_lo = (__m128)__lsx_vinsgr2vr_w(
		(__m128i)err_rot_lo, next_bits, 0);

	// Narrow q (s32) to u16 x8.
	__m128i q = __lsx_vpickev_h(q_hi, q_lo);

	state.err_left_lo = err0_lo;
	state.err_left_hi = err0_hi;
	state.err_top_left_lo = state.err_top_lo;
	state.err_top_left_hi = state.err_top_hi;
	state.err_top_lo = state.err_top_right_lo;
	state.err_top_hi = state.err_top_right_hi;
	state.err_top_right_lo = err_rot_lo;
	state.err_top_right_hi = err_rot_hi;

	return q;
}

template <PixelType SrcType, PixelType DstType, class T, class U>
void error_diffusion_wf_lsx(const Buffer<const T> &src,
                            const Buffer<U> &dst,
                            unsigned i, const float *error_top,
                            float *error_cur, error_state *state,
                            float scale, float offset, unsigned bits,
                            unsigned width)
{
	typedef error_diffusion_traits<SrcType> src_traits;
	typedef error_diffusion_traits<DstType> dst_traits;

	typedef typename src_traits::type src_type;
	typedef typename dst_traits::type dst_type;

	static_assert(std::is_same_v<T, src_type>);
	static_assert(std::is_same_v<U, dst_type>);

	float v7f = 7.0f / 16.0f;
	float v3f = 3.0f / 16.0f;
	float v5f = 5.0f / 16.0f;
	float v1f = 1.0f / 16.0f;
	float sc = scale, of = offset;
	float maxv = static_cast<float>((1UL << bits) - 1);

	const __m128 err_left_w = (__m128)__lsx_vldrepl_w(&v7f, 0);
	const __m128 err_top_right_w = (__m128)__lsx_vldrepl_w(&v3f, 0);
	const __m128 err_top_w = (__m128)__lsx_vldrepl_w(&v5f, 0);
	const __m128 err_top_left_w = (__m128)__lsx_vldrepl_w(&v1f, 0);

	const __m128 scale_ps = (__m128)__lsx_vldrepl_w(&sc, 0);
	const __m128 offset_ps = (__m128)__lsx_vldrepl_w(&of, 0);
	const __m128 max_val = (__m128)__lsx_vldrepl_w(&maxv, 0);

	wf_error_state st{};
	st.err_left_lo = (__m128)__lsx_vld((void *)(state->err_left + 0), 0);
	st.err_left_hi = (__m128)__lsx_vld((void *)(state->err_left + 4), 0);
	st.err_top_right_lo = (__m128)__lsx_vld(
		(void *)(state->err_top_right + 0), 0);
	st.err_top_right_hi = (__m128)__lsx_vld(
		(void *)(state->err_top_right + 4), 0);
	st.err_top_lo = (__m128)__lsx_vld((void *)(state->err_top + 0), 0);
	st.err_top_hi = (__m128)__lsx_vld((void *)(state->err_top + 4), 0);
	st.err_top_left_lo = (__m128)__lsx_vld(
		(void *)(state->err_top_left + 0), 0);
	st.err_top_left_hi = (__m128)__lsx_vld(
		(void *)(state->err_top_left + 4), 0);

	for (unsigned j = 0; j < width; j += 8) {
		__m128 row_lo[8];
		__m128 row_hi[8];

		src_traits::load8(src[i + 0] + j + 14, row_lo[0], row_hi[0]);
		src_traits::load8(src[i + 1] + j + 12, row_lo[1], row_hi[1]);
		src_traits::load8(src[i + 2] + j + 10, row_lo[2], row_hi[2]);
		src_traits::load8(src[i + 3] + j + 8, row_lo[3], row_hi[3]);
		src_traits::load8(src[i + 4] + j + 6, row_lo[4], row_hi[4]);
		src_traits::load8(src[i + 5] + j + 4, row_lo[5], row_hi[5]);
		src_traits::load8(src[i + 6] + j + 2, row_lo[6], row_hi[6]);
		src_traits::load8(src[i + 7] + j + 0, row_lo[7], row_hi[7]);

		for (unsigned k = 0; k < 8; ++k) {
			row_lo[k] = __lsx_vfmadd_s(
				row_lo[k], scale_ps, offset_ps);
			row_hi[k] = __lsx_vfmadd_s(
				row_hi[k], scale_ps, offset_ps);
		}

		lsx_transpose4_f32(row_lo[0], row_lo[1], row_lo[2], row_lo[3]);
		lsx_transpose4_f32(row_lo[4], row_lo[5], row_lo[6], row_lo[7]);
		lsx_transpose4_f32(row_hi[0], row_hi[1], row_hi[2], row_hi[3]);
		lsx_transpose4_f32(row_hi[4], row_hi[5], row_hi[6], row_hi[7]);

		const __m128 diag_lo[8] = {
			row_lo[0], row_lo[1], row_lo[2], row_lo[3],
			row_hi[0], row_hi[1], row_hi[2], row_hi[3] };
		const __m128 diag_hi[8] = {
			row_lo[4], row_lo[5], row_lo[6], row_lo[7],
			row_hi[4], row_hi[5], row_hi[6], row_hi[7] };

#define XITER error_diffusion_wf_lsx_xiter
#define XARGS error_top, error_cur, max_val, err_left_w, \
                err_top_right_w, err_top_w, err_top_left_w, st
		__m128i out0 = XITER(diag_lo[0], diag_hi[0], j + 0, XARGS);
		__m128i out1 = XITER(diag_lo[1], diag_hi[1], j + 1, XARGS);
		__m128i out2 = XITER(diag_lo[2], diag_hi[2], j + 2, XARGS);
		__m128i out3 = XITER(diag_lo[3], diag_hi[3], j + 3, XARGS);
		__m128i out4 = XITER(diag_lo[4], diag_hi[4], j + 4, XARGS);
		__m128i out5 = XITER(diag_lo[5], diag_hi[5], j + 5, XARGS);
		__m128i out6 = XITER(diag_lo[6], diag_hi[6], j + 6, XARGS);
		__m128i out7 = XITER(diag_lo[7], diag_hi[7], j + 7, XARGS);
#undef XITER
#undef XARGS

		lsx_transpose8_u16(
			out0, out1, out2, out3, out4, out5, out6, out7);

		dst_traits::store8(dst[i + 0] + j + 14, out0);
		dst_traits::store8(dst[i + 1] + j + 12, out1);
		dst_traits::store8(dst[i + 2] + j + 10, out2);
		dst_traits::store8(dst[i + 3] + j + 8, out3);
		dst_traits::store8(dst[i + 4] + j + 6, out4);
		dst_traits::store8(dst[i + 5] + j + 4, out5);
		dst_traits::store8(dst[i + 6] + j + 2, out6);
		dst_traits::store8(dst[i + 7] + j + 0, out7);
	}

	__lsx_vst((__m128i)st.err_left_lo, state->err_left + 0, 0);
	__lsx_vst((__m128i)st.err_left_hi, state->err_left + 4, 0);
	__lsx_vst((__m128i)st.err_top_right_lo, state->err_top_right + 0, 0);
	__lsx_vst((__m128i)st.err_top_right_hi, state->err_top_right + 4, 0);
	__lsx_vst((__m128i)st.err_top_lo, state->err_top + 0, 0);
	__lsx_vst((__m128i)st.err_top_hi, state->err_top + 4, 0);
	__lsx_vst((__m128i)st.err_top_left_lo, state->err_top_left + 0, 0);
	__lsx_vst((__m128i)st.err_top_left_hi, state->err_top_left + 4, 0);
}

template <PixelType SrcType, PixelType DstType>
void error_diffusion_lsx(const Buffer<const void> &src_,
                         const Buffer<void> &dst_, unsigned i,
                         const float *error_top, float *error_cur,
                         float scale, float offset, unsigned bits,
                         unsigned width)
{
	typedef error_diffusion_traits<SrcType> src_traits;
	typedef error_diffusion_traits<DstType> dst_traits;

	typedef typename src_traits::type src_type;
	typedef typename dst_traits::type dst_type;

	Buffer<const src_type> src{ src_.buffer };
	Buffer<dst_type> dst{ dst_.buffer };

	error_state state alignas(16) = {};
	float error_tmp[7][24] = {};

	// Prologue.
		error_diffusion_scalar<SrcType, DstType>(
			src[i + 0], dst[i + 0], error_top,
			error_tmp[0], scale, offset,
			bits, 14);
		error_diffusion_scalar<SrcType, DstType>(
			src[i + 1], dst[i + 1], error_tmp[0],
			error_tmp[1], scale, offset,
			bits, 12);
		error_diffusion_scalar<SrcType, DstType>(
			src[i + 2], dst[i + 2], error_tmp[1],
			error_tmp[2], scale, offset,
			bits, 10);
		error_diffusion_scalar<SrcType, DstType>(
			src[i + 3], dst[i + 3], error_tmp[2],
			error_tmp[3], scale, offset,
			bits, 8);
		error_diffusion_scalar<SrcType, DstType>(
			src[i + 4], dst[i + 4], error_tmp[3],
			error_tmp[4], scale, offset,
			bits, 6);
		error_diffusion_scalar<SrcType, DstType>(
			src[i + 5], dst[i + 5], error_tmp[4],
			error_tmp[5], scale, offset,
			bits, 4);
		error_diffusion_scalar<SrcType, DstType>(
			src[i + 6], dst[i + 6], error_tmp[5],
			error_tmp[6], scale, offset,
			bits, 2);
	// Wavefront.
	state.err_left[0] = error_tmp[0][13 + 1];
	state.err_left[1] = error_tmp[1][11 + 1];
	state.err_left[2] = error_tmp[2][9 + 1];
	state.err_left[3] = error_tmp[3][7 + 1];
	state.err_left[4] = error_tmp[4][5 + 1];
	state.err_left[5] = error_tmp[5][3 + 1];
	state.err_left[6] = error_tmp[6][1 + 1];
	state.err_left[7] = 0.0f;

	state.err_top_right[0] = error_top[15 + 1];
	state.err_top_right[1] = error_tmp[0][13 + 1];
	state.err_top_right[2] = error_tmp[1][11 + 1];
	state.err_top_right[3] = error_tmp[2][9 + 1];
	state.err_top_right[4] = error_tmp[3][7 + 1];
	state.err_top_right[5] = error_tmp[4][5 + 1];
	state.err_top_right[6] = error_tmp[5][3 + 1];
	state.err_top_right[7] = error_tmp[6][1 + 1];

	state.err_top[0] = error_top[14 + 1];
	state.err_top[1] = error_tmp[0][12 + 1];
	state.err_top[2] = error_tmp[1][10 + 1];
	state.err_top[3] = error_tmp[2][8 + 1];
	state.err_top[4] = error_tmp[3][6 + 1];
	state.err_top[5] = error_tmp[4][4 + 1];
	state.err_top[6] = error_tmp[5][2 + 1];
	state.err_top[7] = error_tmp[6][0 + 1];

	state.err_top_left[0] = error_top[13 + 1];
	state.err_top_left[1] = error_tmp[0][11 + 1];
	state.err_top_left[2] = error_tmp[1][9 + 1];
	state.err_top_left[3] = error_tmp[2][7 + 1];
	state.err_top_left[4] = error_tmp[3][5 + 1];
	state.err_top_left[5] = error_tmp[4][3 + 1];
	state.err_top_left[6] = error_tmp[5][1 + 1];
	state.err_top_left[7] = 0.0f;

	unsigned vec_count = floor_n(width - 14, 8);
	error_diffusion_wf_lsx<SrcType, DstType>(
		src, dst, i, error_top, error_cur, &state,
		scale, offset, bits, vec_count);

	error_tmp[0][13 + 1] = state.err_top_right[1];
	error_tmp[0][12 + 1] = state.err_top[1];
	error_tmp[0][11 + 1] = state.err_top_left[1];

	error_tmp[1][11 + 1] = state.err_top_right[2];
	error_tmp[1][10 + 1] = state.err_top[2];
	error_tmp[1][9 + 1] = state.err_top_left[2];

	error_tmp[2][9 + 1] = state.err_top_right[3];
	error_tmp[2][8 + 1] = state.err_top[3];
	error_tmp[2][7 + 1] = state.err_top_left[3];

	error_tmp[3][7 + 1] = state.err_top_right[4];
	error_tmp[3][6 + 1] = state.err_top[4];
	error_tmp[3][5 + 1] = state.err_top_left[4];

	error_tmp[4][5 + 1] = state.err_top_right[5];
	error_tmp[4][4 + 1] = state.err_top[5];
	error_tmp[4][3 + 1] = state.err_top_left[5];

	error_tmp[5][3 + 1] = state.err_top_right[6];
	error_tmp[5][2 + 1] = state.err_top[6];
	error_tmp[5][1 + 1] = state.err_top_left[6];

	error_tmp[6][1 + 1] = state.err_top_right[7];
	error_tmp[6][0 + 1] = state.err_top[7];
	error_tmp[6][0] = state.err_top_left[7];

	// Epilogue.
	error_diffusion_scalar<SrcType, DstType>(
		src[i + 0] + vec_count + 14,
		dst[i + 0] + vec_count + 14,
		error_top + vec_count + 14, error_tmp[0] + 14,
		scale, offset, bits, width - vec_count - 14);
	error_diffusion_scalar<SrcType, DstType>(
		src[i + 1] + vec_count + 12, dst[i + 1] + vec_count + 12,
		    error_tmp[0] + 12, error_tmp[1] + 12,
		scale, offset, bits, width - vec_count - 12);
	error_diffusion_scalar<SrcType, DstType>(
		src[i + 2] + vec_count + 10, dst[i + 2] + vec_count + 10,
		    error_tmp[1] + 10, error_tmp[2] + 10,
		scale, offset, bits, width - vec_count - 10);
	error_diffusion_scalar<SrcType, DstType>(
		src[i + 3] + vec_count + 8, dst[i + 3] + vec_count + 8,
		    error_tmp[2] + 8, error_tmp[3] + 8,
		scale, offset, bits, width - vec_count - 8);
	error_diffusion_scalar<SrcType, DstType>(
		src[i + 4] + vec_count + 6, dst[i + 4] + vec_count + 6,
		    error_tmp[3] + 6, error_tmp[4] + 6,
		scale, offset, bits, width - vec_count - 6);
	error_diffusion_scalar<SrcType, DstType>(
		src[i + 5] + vec_count + 4, dst[i + 5] + vec_count + 4,
		    error_tmp[4] + 4, error_tmp[5] + 4,
		scale, offset, bits, width - vec_count - 4);
	error_diffusion_scalar<SrcType, DstType>(
		src[i + 6] + vec_count + 2, dst[i + 6] + vec_count + 2,
		    error_tmp[5] + 2, error_tmp[6] + 2,
		scale, offset, bits, width - vec_count - 2);
	error_diffusion_scalar<SrcType, DstType>(
		src[i + 7] + vec_count + 0,
		dst[i + 7] + vec_count + 0,
		error_tmp[6] + 0, error_cur + vec_count + 0,
		scale, offset, bits, width - vec_count - 0);
}

auto select_error_diffusion_lsx_func(PixelType pixel_in, PixelType pixel_out)
{
	if (pixel_in == PixelType::BYTE && pixel_out == PixelType::BYTE)
		return error_diffusion_lsx<PixelType::BYTE, PixelType::BYTE>;
	else if (pixel_in == PixelType::BYTE && pixel_out == PixelType::WORD)
		return error_diffusion_lsx<PixelType::BYTE, PixelType::WORD>;
	else if (pixel_in == PixelType::WORD && pixel_out == PixelType::BYTE)
		return error_diffusion_lsx<PixelType::WORD, PixelType::BYTE>;
	else if (pixel_in == PixelType::WORD && pixel_out == PixelType::WORD)
		return error_diffusion_lsx<PixelType::WORD, PixelType::WORD>;
	else if (pixel_in == PixelType::HALF && pixel_out == PixelType::BYTE)
		return error_diffusion_lsx<PixelType::HALF, PixelType::BYTE>;
	else if (pixel_in == PixelType::HALF && pixel_out == PixelType::WORD)
		return error_diffusion_lsx<PixelType::HALF, PixelType::WORD>;
	else if (pixel_in == PixelType::FLOAT && pixel_out == PixelType::BYTE)
		return error_diffusion_lsx<PixelType::FLOAT, PixelType::BYTE>;
	else if (pixel_in == PixelType::FLOAT && pixel_out == PixelType::WORD)
		return error_diffusion_lsx<PixelType::FLOAT, PixelType::WORD>;
	else
		error::throw_<error::InternalError>(
		"no conversion between pixel types");
}


class ErrorDiffusionLSX final : public graph::FilterBase {
	decltype(select_error_diffusion_scalar_func({}, {})) m_scalar_func;
	decltype(select_error_diffusion_lsx_func({}, {})) m_vec_func;

	float m_scale;
	float m_offset;
	unsigned m_depth;

	void process_scalar(void *ctx, const void *src, void *dst,
	                    bool parity) const
	{
		float *ctx_a = reinterpret_cast<float *>(ctx);
		float *ctx_b = reinterpret_cast<float *>(
			static_cast<unsigned char *>(ctx) +
			m_desc.context_size / 2);

		float *error_top = parity ? ctx_a : ctx_b;
		float *error_cur = parity ? ctx_b : ctx_a;

		m_scalar_func(
			src, dst, error_top, error_cur, m_scale, m_offset,
			m_depth, m_desc.format.width);
	}

	void process_vector(void *ctx,
	                     const graphengine::BufferDescriptor *in,
	                     const graphengine::BufferDescriptor *out,
	                     unsigned i) const
	{
		float *ctx_a = reinterpret_cast<float *>(ctx);
		float *ctx_b = reinterpret_cast<float *>(
			static_cast<unsigned char *>(ctx) +
			m_desc.context_size / 2);

		float *error_top = (i / 8) % 2 ? ctx_a : ctx_b;
		float *error_cur = (i / 8) % 2 ? ctx_b : ctx_a;

		m_vec_func(
			*in, *out, i, error_top, error_cur, m_scale, m_offset,
			m_depth, m_desc.format.width);
	}

public:
	ErrorDiffusionLSX(unsigned width, unsigned height,
	                  const PixelFormat &pixel_in,
	                  const PixelFormat &pixel_out) try :
		m_scalar_func{ select_error_diffusion_scalar_func(
			pixel_in.type, pixel_out.type) },
		m_vec_func{ select_error_diffusion_lsx_func(
			pixel_in.type, pixel_out.type) },
		m_scale{}, m_offset{}, m_depth{ pixel_out.depth }
	{
		zassert_d(width <= pixel_max_width(pixel_in.type), "overflow");
		zassert_d(width <= pixel_max_width(pixel_out.type), "overflow");

		if (!pixel_is_integer(pixel_out.type))
			error::throw_<error::InternalError>(
				"cannot dither to non-integer format");

		m_desc.format = { width, height, pixel_size(pixel_out.type) };
		m_desc.num_deps = 1;
		m_desc.num_planes = 1;
		m_desc.step = 8;
		m_desc.context_size = ((static_cast<checked_size_t>(width) + 3)
		                       * sizeof(float) * 2).get();

		m_desc.flags.stateful = 1;
		m_desc.flags.in_place =
			pixel_size(pixel_in.type) == pixel_size(pixel_out.type);
		m_desc.flags.entire_row = 1;

		std::tie(m_scale, m_offset) =
			get_scale_offset(pixel_in, pixel_out);
	} catch (const std::overflow_error &) {
		error::throw_<error::OutOfMemory>();
	}

	pair_unsigned get_row_deps(unsigned i) const noexcept override
	{
		unsigned last = std::min(i, UINT_MAX - 8) + 8;
		return{ i, std::min(last, m_desc.format.height) };
	}

	pair_unsigned get_col_deps(unsigned, unsigned) const noexcept override
	{
		return{ 0, m_desc.format.width };
	}

	void init_context(void *ctx) const noexcept override
	{
		std::fill_n(
			static_cast<unsigned char *>(ctx),
			m_desc.context_size, 0);
	}

	void process(const graphengine::BufferDescriptor *in,
	             const graphengine::BufferDescriptor *out, unsigned i,
	             unsigned left, unsigned right, void *context,
	             void *) const noexcept override
	{
		(void)left;
		(void)right;

		if (m_desc.format.height - i < 8) {
			bool parity = !!((i / 8) % 2);

			for (unsigned ii = i; ii < m_desc.format.height; ++ii) {
				process_scalar(
					context, in->get_line(ii),
			out->get_line(ii), parity);
				parity = !parity;
			}
		} else {
			process_vector(context, in, out, i);
		}
	}
};

} // namespace

std::unique_ptr<graphengine::Filter> create_error_diffusion_lsx(
	unsigned width, unsigned height, const PixelFormat &pixel_in,
	const PixelFormat &pixel_out)
{
	if (width < 14)
		return nullptr;

	return std::make_unique<ErrorDiffusionLSX>(
		width, height, pixel_in, pixel_out);
}

} // namespace zimg::depth

#endif // ZIMG_LOONGARCH
