#ifdef ZIMG_X86

#include <algorithm>
#include <climits>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <immintrin.h>
#include "common/align.h"
#include "common/ccdep.h"
#include "common/checked_int.h"
#include "common/except.h"
#include "common/pixel.h"
#include "common/zassert.h"
#include "depth/quantize.h"
#include "graphengine/filter.h"
#include "dither_x86.h"

#include "common/x86/avx_util.h"

namespace zimg {
namespace depth {

namespace {

template <class T>
struct Buffer {
	const graphengine::BufferDescriptor &buffer;

	Buffer(const graphengine::BufferDescriptor &buffer) : buffer{ buffer } {}

	T *operator[](unsigned i) const { return buffer.get_line<T>(i); }
};


struct error_state {
	float err_left[8];
	float err_top_right[8];
	float err_top[8];
	float err_top_left[8];
};


template <PixelType SrcType>
struct error_diffusion_traits;

template <>
struct error_diffusion_traits<PixelType::BYTE> {
	typedef uint8_t type;

	static float load1(const uint8_t *ptr) { return *ptr; }
	static void store1(uint8_t *ptr, uint32_t x) { *ptr = static_cast<uint8_t>(x); }

	static __m256 load8(const uint8_t *ptr)
	{
		return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)ptr)));
	}

	static void store8(uint8_t *ptr, __m256i x)
	{
		x = _mm256_packs_epi32(x, x);
		x = _mm256_permute4x64_epi64(x, _MM_SHUFFLE(0, 0, 2, 0));
		x = _mm256_packus_epi16(x, x);
		_mm_storel_epi64((__m128i *)ptr, _mm256_castsi256_si128(x));
	}
};

template <>
struct error_diffusion_traits<PixelType::WORD> {
	typedef uint16_t type;

	static float load1(const uint16_t *ptr) { return *ptr; }
	static void store1(uint16_t *ptr, uint32_t x) { *ptr = static_cast<uint32_t>(x); }

	static __m256 load8(const uint16_t *ptr)
	{
		return _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)ptr)));
	}

	static void store8(uint16_t *ptr, __m256i x)
	{
		x = _mm256_packus_epi32(x, x);
		x = _mm256_permute4x64_epi64(x, _MM_SHUFFLE(0, 0, 2, 0));
		_mm_storeu_si128((__m128i *)ptr, _mm256_castsi256_si128(x));
	}
};

template <>
struct error_diffusion_traits<PixelType::HALF> {
	typedef uint16_t type;

	static float load1(const uint16_t *ptr)
	{
		return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(*ptr)));
	}

	static __m256 load8(const uint16_t *ptr) {
		return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)ptr));
	}
};

template <>
struct error_diffusion_traits<PixelType::FLOAT> {
	typedef float type;

	static float load1(const float *ptr) { return *ptr; }
	static __m256 load8(const float *ptr) { return _mm256_loadu_ps(ptr); }
};


inline FORCE_INLINE float fma(float a, float b, float c)
{
	return _mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(a), _mm_set_ss(b), _mm_set_ss(c)));
}

inline FORCE_INLINE float max(float x, float y)
{
	return _mm_cvtss_f32(_mm_max_ss(_mm_set_ss(x), _mm_set_ss(y)));
}

inline FORCE_INLINE float min(float x, float y)
{
	return _mm_cvtss_f32(_mm_min_ss(_mm_set_ss(x), _mm_set_ss(y)));
}


template <PixelType SrcType, PixelType DstType>
void error_diffusion_scalar(const void *src, void *dst, const float * RESTRICT error_top, float * RESTRICT error_cur,
                            float scale, float offset, unsigned bits, unsigned width)
{
	typedef error_diffusion_traits<SrcType> src_traits;
	typedef error_diffusion_traits<DstType> dst_traits;

	const typename src_traits::type *src_p = static_cast<const typename src_traits::type *>(src);
	typename dst_traits::type *dst_p = static_cast<typename dst_traits::type *>(dst);

	float err_left = error_cur[0];
	float err_top_right;
	float err_top = error_top[0 + 1];
	float err_top_left = error_top[0];

	for (unsigned j = 0; j < width; ++j) {
		// Error array is padded by one on each side.
		unsigned j_err = j + 1;
		err_top_right = error_top[j_err + 1];

		float x = fma(src_traits::load1(src_p + j), scale, offset);
		float err, err0, err1;

		err0 = err_left * (7.0f / 16.0f);
		err0 = fma(err_top_right, 3.0f / 16.0f, err0);
		err1 = err_top * (5.0f / 16.0f);
		err1 = fma(err_top_left, 1.0f / 16.0f, err1);
		err = err0 + err1;

		x += err;
		x = min(max(x, 0.0f), static_cast<float>(1L << bits) - 1);

		uint32_t q = _mm_cvt_ss2si(_mm_set_ss(x));
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
		return error_diffusion_scalar<PixelType::FLOAT, PixelType::BYTE>;
	else if (pixel_in == PixelType::FLOAT && pixel_out == PixelType::WORD)
		return error_diffusion_scalar<PixelType::FLOAT, PixelType::WORD>;
	else
		error::throw_<error::InternalError>("no conversion between pixel types");
}


inline FORCE_INLINE void error_diffusion_wf_avx2_xiter(__m256 &v, unsigned j, const float *error_top, float *error_cur, const __m256 &max_val,
                                                       const __m256 &err_left_w, const __m256 &err_top_right_w, const __m256 &err_top_w, const __m256 &err_top_left_w,
                                                       __m256 &err_left, __m256 &err_top_right, __m256 &err_top, __m256 &err_top_left)
{
	const __m256i rot_mask = _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 7);

	unsigned j_err = j + 1;

	__m256 x, y, err0, err1, err_rot;
	__m256i q;

	err0 = _mm256_mul_ps(err_left_w, err_left);
	err0 = _mm256_fmadd_ps(err_top_right_w, err_top_right, err0);
	err1 = _mm256_mul_ps(err_top_w, err_top);
	err1 = _mm256_fmadd_ps(err_top_left_w, err_top_left, err1);
	err0 = _mm256_add_ps(err0, err1);

	x = _mm256_add_ps(v, err0);
	x = _mm256_max_ps(x, _mm256_setzero_ps());
	x = _mm256_min_ps(x, max_val);
	q = _mm256_cvtps_epi32(x);
	v = _mm256_castsi256_ps(q);

	y = _mm256_cvtepi32_ps(q);
	err0 = _mm256_sub_ps(x, y);

	// Left-rotate err0 by 32 bits.
	err_rot = _mm256_permutevar8x32_ps(err0, rot_mask);

	// Extract the previous high error.
	error_cur[j_err + 0] = _mm_cvtss_f32(_mm256_castps256_ps128(err_rot));

	// Insert the next error into the low position.
	err_rot = _mm256_blend_ps(err_rot, _mm256_castps128_ps256(_mm_set_ss(error_top[j_err + 14 + 2])), 1);

	err_left = err0;
	err_top_left = err_top;
	err_top = err_top_right;
	err_top_right = err_rot;
}

template <PixelType SrcType, PixelType DstType, class T, class U>
void error_diffusion_wf_avx2(const Buffer<const T> &src, const Buffer<U> &dst, unsigned i,
                             const float *error_top, float *error_cur, error_state *state, float scale, float offset, unsigned bits, unsigned width)
{
	typedef error_diffusion_traits<SrcType> src_traits;
	typedef error_diffusion_traits<DstType> dst_traits;

	typedef typename src_traits::type src_type;
	typedef typename dst_traits::type dst_type;

	static_assert(std::is_same<T, src_type>::value, "wrong type");
	static_assert(std::is_same<U, dst_type>::value, "wrong type");

	const __m256 err_left_w = _mm256_set1_ps(7.0f / 16.0f);
	const __m256 err_top_right_w = _mm256_set1_ps(3.0f / 16.0f);
	const __m256 err_top_w = _mm256_set1_ps(5.0f / 16.0f);
	const __m256 err_top_left_w = _mm256_set1_ps(1.0f / 16.0f);

	const __m256 scale_ps = _mm256_set1_ps(scale);
	const __m256 offset_ps = _mm256_set1_ps(offset);

	const __m256 max_val = _mm256_set1_ps(static_cast<float>((1UL << bits) - 1));

	__m256 err_left = _mm256_load_ps(state->err_left);
	__m256 err_top_right = _mm256_load_ps(state->err_top_right);
	__m256 err_top = _mm256_load_ps(state->err_top);
	__m256 err_top_left = _mm256_load_ps(state->err_top_left);

#define XITER error_diffusion_wf_avx2_xiter
#define XARGS error_top, error_cur, max_val, err_left_w, err_top_right_w, err_top_w, err_top_left_w, err_left, err_top_right, err_top, err_top_left
	for (unsigned j = 0; j < width; j += 8) {
		__m256 v0 = src_traits::load8(src[i + 0] + j + 14);
		__m256 v1 = src_traits::load8(src[i + 1] + j + 12);
		__m256 v2 = src_traits::load8(src[i + 2] + j + 10);
		__m256 v3 = src_traits::load8(src[i + 3] + j + 8);
		__m256 v4 = src_traits::load8(src[i + 4] + j + 6);
		__m256 v5 = src_traits::load8(src[i + 5] + j + 4);
		__m256 v6 = src_traits::load8(src[i + 6] + j + 2);
		__m256 v7 = src_traits::load8(src[i + 7] + j + 0);

		v0 = _mm256_fmadd_ps(v0, scale_ps, offset_ps);
		v1 = _mm256_fmadd_ps(v1, scale_ps, offset_ps);
		v2 = _mm256_fmadd_ps(v2, scale_ps, offset_ps);
		v3 = _mm256_fmadd_ps(v3, scale_ps, offset_ps);
		v4 = _mm256_fmadd_ps(v4, scale_ps, offset_ps);
		v5 = _mm256_fmadd_ps(v5, scale_ps, offset_ps);
		v6 = _mm256_fmadd_ps(v6, scale_ps, offset_ps);
		v7 = _mm256_fmadd_ps(v7, scale_ps, offset_ps);

		mm256_transpose8_ps(v0, v1, v2, v3, v4, v5, v6, v7);

		XITER(v0, j + 0, XARGS);
		XITER(v1, j + 1, XARGS);
		XITER(v2, j + 2, XARGS);
		XITER(v3, j + 3, XARGS);
		XITER(v4, j + 4, XARGS);
		XITER(v5, j + 5, XARGS);
		XITER(v6, j + 6, XARGS);
		XITER(v7, j + 7, XARGS);

		mm256_transpose8_ps(v0, v1, v2, v3, v4, v5, v6, v7);

		dst_traits::store8(dst[i + 0] + j + 14, _mm256_castps_si256(v0));
		dst_traits::store8(dst[i + 1] + j + 12, _mm256_castps_si256(v1));
		dst_traits::store8(dst[i + 2] + j + 10, _mm256_castps_si256(v2));
		dst_traits::store8(dst[i + 3] + j + 8, _mm256_castps_si256(v3));
		dst_traits::store8(dst[i + 4] + j + 6, _mm256_castps_si256(v4));
		dst_traits::store8(dst[i + 5] + j + 4, _mm256_castps_si256(v5));
		dst_traits::store8(dst[i + 6] + j + 2, _mm256_castps_si256(v6));
		dst_traits::store8(dst[i + 7] + j + 0, _mm256_castps_si256(v7));
	}
#undef XITER
#undef XARGS

	_mm256_store_ps(state->err_left, err_left);
	_mm256_store_ps(state->err_top_right, err_top_right);
	_mm256_store_ps(state->err_top, err_top);
	_mm256_store_ps(state->err_top_left, err_top_left);
}

template <PixelType SrcType, PixelType DstType>
void error_diffusion_avx2(const Buffer<const void> &src_, const Buffer<void> &dst_, unsigned i,
                          const float *error_top, float *error_cur, float scale, float offset, unsigned bits, unsigned width)
{
	typedef error_diffusion_traits<SrcType> src_traits;
	typedef error_diffusion_traits<DstType> dst_traits;

	typedef typename src_traits::type src_type;
	typedef typename dst_traits::type dst_type;

	Buffer<const src_type> src = src_.buffer;
	Buffer<dst_type> dst = dst_.buffer;

	error_state state alignas(32) = {};
	float error_tmp[7][24] = {};

	// Prologue.
	error_diffusion_scalar<SrcType, DstType>(src[i + 0], dst[i + 0], error_top, error_tmp[0], scale, offset, bits, 14);
	error_diffusion_scalar<SrcType, DstType>(src[i + 1], dst[i + 1], error_tmp[0], error_tmp[1], scale, offset, bits, 12);
	error_diffusion_scalar<SrcType, DstType>(src[i + 2], dst[i + 2], error_tmp[1], error_tmp[2], scale, offset, bits, 10);
	error_diffusion_scalar<SrcType, DstType>(src[i + 3], dst[i + 3], error_tmp[2], error_tmp[3], scale, offset, bits, 8);
	error_diffusion_scalar<SrcType, DstType>(src[i + 4], dst[i + 4], error_tmp[3], error_tmp[4], scale, offset, bits, 6);
	error_diffusion_scalar<SrcType, DstType>(src[i + 5], dst[i + 5], error_tmp[4], error_tmp[5], scale, offset, bits, 4);
	error_diffusion_scalar<SrcType, DstType>(src[i + 6], dst[i + 6], error_tmp[5], error_tmp[6], scale, offset, bits, 2);

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
	error_diffusion_wf_avx2<SrcType, DstType>(src, dst, i, error_top, error_cur, &state, scale, offset, bits, vec_count);

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
	error_diffusion_scalar<SrcType, DstType>(src[i + 0] + vec_count + 14, dst[i + 0] + vec_count + 14, error_top + vec_count + 14, error_tmp[0] + 14,
	                                         scale, offset, bits, width - vec_count - 14);
	error_diffusion_scalar<SrcType, DstType>(src[i + 1] + vec_count + 12, dst[i + 1] + vec_count + 12, error_tmp[0] + 12, error_tmp[1] + 12,
	                                         scale, offset, bits, width - vec_count - 12);
	error_diffusion_scalar<SrcType, DstType>(src[i + 2] + vec_count + 10, dst[i + 2] + vec_count + 10, error_tmp[1] + 10, error_tmp[2] + 10,
	                                         scale, offset, bits, width - vec_count - 10);
	error_diffusion_scalar<SrcType, DstType>(src[i + 3] + vec_count + 8, dst[i + 3] + vec_count + 8, error_tmp[2] + 8, error_tmp[3] + 8,
	                                         scale, offset, bits, width - vec_count - 8);
	error_diffusion_scalar<SrcType, DstType>(src[i + 4] + vec_count + 6, dst[i + 4] + vec_count + 6, error_tmp[3] + 6, error_tmp[4] + 6,
	                                         scale, offset, bits, width - vec_count - 6);
	error_diffusion_scalar<SrcType, DstType>(src[i + 5] + vec_count + 4, dst[i + 5] + vec_count + 4, error_tmp[4] + 4, error_tmp[5] + 4,
	                                         scale, offset, bits, width - vec_count - 4);
	error_diffusion_scalar<SrcType, DstType>(src[i + 6] + vec_count + 2, dst[i + 6] + vec_count + 2, error_tmp[5] + 2, error_tmp[6] + 2,
	                                         scale, offset, bits, width - vec_count - 2);
	error_diffusion_scalar<SrcType, DstType>(src[i + 7] + vec_count + 0, dst[i + 7] + vec_count + 0, error_tmp[6] + 0, error_cur + vec_count + 0,
	                                         scale, offset, bits, width - vec_count - 0);
}

auto select_error_diffusion_avx2_func(PixelType pixel_in, PixelType pixel_out)
{
	if (pixel_in == PixelType::BYTE && pixel_out == PixelType::BYTE)
		return error_diffusion_avx2<PixelType::BYTE, PixelType::BYTE>;
	else if (pixel_in == PixelType::BYTE && pixel_out == PixelType::WORD)
		return error_diffusion_avx2<PixelType::BYTE, PixelType::WORD>;
	else if (pixel_in == PixelType::WORD && pixel_out == PixelType::BYTE)
		return error_diffusion_avx2<PixelType::WORD, PixelType::BYTE>;
	else if (pixel_in == PixelType::WORD && pixel_out == PixelType::WORD)
		return error_diffusion_avx2<PixelType::WORD, PixelType::WORD>;
	else if (pixel_in == PixelType::HALF && pixel_out == PixelType::BYTE)
		return error_diffusion_avx2<PixelType::HALF, PixelType::BYTE>;
	else if (pixel_in == PixelType::HALF && pixel_out == PixelType::WORD)
		return error_diffusion_avx2<PixelType::HALF, PixelType::WORD>;
	else if (pixel_in == PixelType::FLOAT && pixel_out == PixelType::BYTE)
		return error_diffusion_avx2<PixelType::FLOAT, PixelType::BYTE>;
	else if (pixel_in == PixelType::FLOAT && pixel_out == PixelType::WORD)
		return error_diffusion_avx2<PixelType::FLOAT, PixelType::WORD>;
	else
		error::throw_<error::InternalError>("no conversion between pixel types");
}


class ErrorDiffusionAVX2 final : public graphengine::Filter {
	graphengine::FilterDescriptor m_desc;

	decltype(select_error_diffusion_scalar_func({}, {})) m_scalar_func;
	decltype(select_error_diffusion_avx2_func({}, {})) m_avx2_func;

	float m_scale;
	float m_offset;
	unsigned m_depth;

	void process_scalar(void *ctx, const void *src, void *dst, bool parity) const
	{
		float *ctx_a = reinterpret_cast<float *>(ctx);
		float *ctx_b = reinterpret_cast<float *>(static_cast<unsigned char *>(ctx) + m_desc.context_size / 2);

		float *error_top = parity ? ctx_a : ctx_b;
		float *error_cur = parity ? ctx_b : ctx_a;

		m_scalar_func(src, dst, error_top, error_cur, m_scale, m_offset, m_depth, m_desc.format.width);
	}

	void process_vector(void *ctx, const graphengine::BufferDescriptor *in, const graphengine::BufferDescriptor *out, unsigned i) const
	{
		float *ctx_a = reinterpret_cast<float *>(ctx);
		float *ctx_b = reinterpret_cast<float *>(static_cast<unsigned char *>(ctx) + m_desc.context_size / 2);

		float *error_top = (i / 8) % 2 ? ctx_a : ctx_b;
		float *error_cur = (i / 8) % 2 ? ctx_b : ctx_a;

		m_avx2_func(*in, *out, i, error_top, error_cur, m_scale, m_offset, m_depth, m_desc.format.width);
	}
public:
	ErrorDiffusionAVX2(unsigned width, unsigned height, const PixelFormat &pixel_in, const PixelFormat &pixel_out) try :
		m_desc{},
		m_scalar_func{ select_error_diffusion_scalar_func(pixel_in.type, pixel_out.type) },
		m_avx2_func{ select_error_diffusion_avx2_func(pixel_in.type, pixel_out.type) },
		m_scale{},
		m_offset{},
		m_depth{ pixel_out.depth }
	{
		zassert_d(width <= pixel_max_width(pixel_in.type), "overflow");
		zassert_d(width <= pixel_max_width(pixel_out.type), "overflow");

		if (!pixel_is_integer(pixel_out.type))
			error::throw_<error::InternalError>("cannot dither to non-integer format");

		m_desc.format = { width, height, pixel_size(pixel_out.type) };
		m_desc.num_deps = 1;
		m_desc.num_planes = 1;
		m_desc.step = 8;
		m_desc.context_size = ((static_cast<checked_size_t>(width) + 2) * sizeof(float) * 2).get();

		m_desc.flags.stateful = 1;
		m_desc.flags.in_place = pixel_size(pixel_in.type) == pixel_size(pixel_out.type);
		m_desc.flags.entire_row = 1;

		std::tie(m_scale, m_offset) = get_scale_offset(pixel_in, pixel_out);
	} catch (const std::overflow_error &) {
		error::throw_<error::OutOfMemory>();
	}

	const graphengine::FilterDescriptor &descriptor() const noexcept override { return m_desc; }

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
		std::fill_n(static_cast<unsigned char *>(ctx), m_desc.context_size, 0);
	}

	void process(const graphengine::BufferDescriptor *in, const graphengine::BufferDescriptor *out,
	             unsigned i, unsigned left, unsigned right, void *context, void *) const noexcept override
	{
		if (m_desc.format.height - i < 8) {
			bool parity = !!((i / 8) % 2);

			for (unsigned ii = i; ii < m_desc.format.height; ++ii) {
				process_scalar(context, in->get_line(ii), out->get_line(ii), parity);
				parity = !parity;
			}
		} else {
			process_vector(context, in, out, i);
		}
	}
};

} // namespace


std::unique_ptr<graphengine::Filter> create_error_diffusion_avx2(unsigned width, unsigned height, const PixelFormat &pixel_in, const PixelFormat &pixel_out)
{
	if (width < 14)
		return nullptr;

	return std::make_unique<ErrorDiffusionAVX2>(width, height, pixel_in, pixel_out);
}

} // namespace depth
} // namespace zimg

#endif // ZIMG_X86
