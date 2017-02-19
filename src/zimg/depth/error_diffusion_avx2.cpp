#ifdef ZIMG_X86

#include <algorithm>
#include <tuple>
#include <type_traits>
#include <immintrin.h>
#include "common/align.h"
#include "common/ccdep.h"
#include "common/except.h"
#include "common/make_unique.h"
#include "common/pixel.h"

#define HAVE_CPU_AVX
#include "common/x86util.h"
#undef HAVE_CPU_AVX

#include "common/zassert.h"
#include "graph/image_buffer.h"
#include "graph/image_filter.h"
#include "dither_x86.h"
#include "quantize.h"

namespace zimg {
namespace depth {

namespace {

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
		__m256i x = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i *)ptr));
		x = _mm256_unpacklo_epi8(x, _mm256_setzero_si256());
		x = _mm256_permute4x64_epi64(x, _MM_SHUFFLE(0, 1, 0, 0));
		x = _mm256_unpacklo_epi16(x, _mm256_setzero_si256());
		return _mm256_cvtepi32_ps(x);
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
		__m256i x = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)ptr));
		x = _mm256_permute4x64_epi64(x, _MM_SHUFFLE(0, 1, 0, 0));
		x = _mm256_unpacklo_epi16(x, _mm256_setzero_si256());
		return _mm256_cvtepi32_ps(x);
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

inline FORCE_INLINE float extract_hi_ps(__m256 x)
{
	__m128 y = _mm256_extractf128_ps(x, 1);
	return _mm_cvtss_f32(_mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(y), 12)));
}

inline FORCE_INLINE __m256 rotate_insert_lo(__m256 x, float y)
{
	__m256i mask = _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 7);

	x = _mm256_permutevar8x32_ps(x, mask);
	x = _mm256_blend_ps(x, _mm256_castps128_ps256(_mm_set_ss(y)), 1);

	return x;
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
	float err_top_right = error_top[1 + 1];
	float err_top = error_top[0 + 1];
	float err_top_left = error_top[0];

	for (unsigned j = 0; j < width; ++j) {
		// Error array is padded by one on each side.
		unsigned j_err = j + 1;

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
		err_top_right = error_top[j_err + 2];
	}
}

decltype(&error_diffusion_scalar<PixelType::BYTE, PixelType::BYTE>) select_error_diffusion_scalar_func(PixelType pixel_in, PixelType pixel_out)
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
		throw error::InternalError{ "no conversion between pixel types" };
}


inline FORCE_INLINE void error_diffusion_wf_avx2_xiter(__m256 &v, unsigned j, const float *error_top, float *error_cur, const __m256 &max_val,
                                                       const __m256 &err_left_w, const __m256 &err_top_right_w, const __m256 &err_top_w, const __m256 &err_top_left_w,
													   __m256 &err_left, __m256 &err_top_right, __m256 &err_top, __m256 &err_top_left)
{
	unsigned j_err = j + 1;

	__m256 x, y, err0, err1;
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

	error_cur[j_err + 0] = extract_hi_ps(err0);

	err_left = err0;
	err_top_left = err_top;
	err_top = err_top_right;
	err_top_right = rotate_insert_lo(err0, error_top[j_err + 14 + 2]);
}

template <PixelType SrcType, PixelType DstType, class T, class U>
void error_diffusion_wf_avx2(const graph::ImageBuffer<const T> &src, const graph::ImageBuffer<U> &dst, unsigned i,
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
void error_diffusion_avx2(const graph::ImageBuffer<const void> &src, const graph::ImageBuffer<void> &dst, unsigned i,
                          const float *error_top, float *error_cur, float scale, float offset, unsigned bits, unsigned width)
{
	typedef error_diffusion_traits<SrcType> src_traits;
	typedef error_diffusion_traits<DstType> dst_traits;

	typedef typename src_traits::type src_type;
	typedef typename dst_traits::type dst_type;

	const graph::ImageBuffer<const src_type> &src_buf = graph::static_buffer_cast<const src_type>(src);
	const graph::ImageBuffer<dst_type> &dst_buf = graph::static_buffer_cast<dst_type>(dst);

	error_state state alignas(32) = {};
	float error_tmp[7][24] = {};

	// Prologue.
	error_diffusion_scalar<SrcType, DstType>(src_buf[i + 0], dst_buf[i + 0], error_top, error_tmp[0], scale, offset, bits, 14);
	error_diffusion_scalar<SrcType, DstType>(src_buf[i + 1], dst_buf[i + 1], error_tmp[0], error_tmp[1], scale, offset, bits, 12);
	error_diffusion_scalar<SrcType, DstType>(src_buf[i + 2], dst_buf[i + 2], error_tmp[1], error_tmp[2], scale, offset, bits, 10);
	error_diffusion_scalar<SrcType, DstType>(src_buf[i + 3], dst_buf[i + 3], error_tmp[2], error_tmp[3], scale, offset, bits, 8);
	error_diffusion_scalar<SrcType, DstType>(src_buf[i + 4], dst_buf[i + 4], error_tmp[3], error_tmp[4], scale, offset, bits, 6);
	error_diffusion_scalar<SrcType, DstType>(src_buf[i + 5], dst_buf[i + 5], error_tmp[4], error_tmp[5], scale, offset, bits, 4);
	error_diffusion_scalar<SrcType, DstType>(src_buf[i + 6], dst_buf[i + 6], error_tmp[5], error_tmp[6], scale, offset, bits, 2);

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
	error_diffusion_wf_avx2<SrcType, DstType>(src_buf, dst_buf, i, error_top, error_cur, &state, scale, offset, bits, vec_count);

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
	error_diffusion_scalar<SrcType, DstType>(src_buf[i + 0] + vec_count + 14, dst_buf[i + 0] + vec_count + 14, error_top + vec_count + 14, error_tmp[0] + 14,
	                                         scale, offset, bits, width - vec_count - 14);
	error_diffusion_scalar<SrcType, DstType>(src_buf[i + 1] + vec_count + 12, dst_buf[i + 1] + vec_count + 12, error_tmp[0] + 12, error_tmp[1] + 12,
	                                         scale, offset, bits, width - vec_count - 12);
	error_diffusion_scalar<SrcType, DstType>(src_buf[i + 2] + vec_count + 10, dst_buf[i + 2] + vec_count + 10, error_tmp[1] + 10, error_tmp[2] + 10,
	                                         scale, offset, bits, width - vec_count - 10);
	error_diffusion_scalar<SrcType, DstType>(src_buf[i + 3] + vec_count + 8, dst_buf[i + 3] + vec_count + 8, error_tmp[2] + 8, error_tmp[3] + 8,
	                                         scale, offset, bits, width - vec_count - 8);
	error_diffusion_scalar<SrcType, DstType>(src_buf[i + 4] + vec_count + 6, dst_buf[i + 4] + vec_count + 6, error_tmp[3] + 6, error_tmp[4] + 6,
	                                         scale, offset, bits, width - vec_count - 6);
	error_diffusion_scalar<SrcType, DstType>(src_buf[i + 5] + vec_count + 4, dst_buf[i + 5] + vec_count + 4, error_tmp[4] + 4, error_tmp[5] + 4,
	                                         scale, offset, bits, width - vec_count - 4);
	error_diffusion_scalar<SrcType, DstType>(src_buf[i + 6] + vec_count + 2, dst_buf[i + 6] + vec_count + 2, error_tmp[5] + 2, error_tmp[6] + 2,
	                                         scale, offset, bits, width - vec_count - 2);
	error_diffusion_scalar<SrcType, DstType>(src_buf[i + 7] + vec_count + 0, dst_buf[i + 7] + vec_count + 0, error_tmp[6] + 0, error_cur + vec_count + 0,
	                                         scale, offset, bits, width - vec_count - 0);
}

decltype(&error_diffusion_avx2<PixelType::BYTE, PixelType::BYTE>) select_error_diffusion_avx2_func(PixelType pixel_in, PixelType pixel_out)
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
		throw error::InternalError{ "no conversion between pixel types" };
}


class ErrorDiffusionAVX2 final : public graph::ImageFilter {
	decltype(&error_diffusion_scalar<PixelType::BYTE, PixelType::BYTE>) m_scalar_func;
	decltype(&error_diffusion_avx2<PixelType::BYTE, PixelType::BYTE>) m_avx2_func;

	PixelType m_pixel_in;
	PixelType m_pixel_out;

	float m_scale;
	float m_offset;
	unsigned m_depth;

	unsigned m_width;
	unsigned m_height;

	void process_scalar(void *ctx, const void *src, void *dst, bool parity) const
	{
		float *ctx_a = reinterpret_cast<float *>(ctx);
		float *ctx_b = reinterpret_cast<float *>(static_cast<unsigned char *>(ctx) + get_context_size() / 2);

		float *error_top = parity ? ctx_a : ctx_b;
		float *error_cur = parity ? ctx_b : ctx_a;

		m_scalar_func(src, dst, error_top, error_cur, m_scale, m_offset, m_depth, m_width);
	}

	void process_vector(void *ctx, const graph::ImageBuffer<const void> &src, const graph::ImageBuffer<void> &dst, unsigned i) const
	{
		float *ctx_a = reinterpret_cast<float *>(ctx);
		float *ctx_b = reinterpret_cast<float *>(static_cast<unsigned char *>(ctx) + get_context_size() / 2);

		float *error_top = (i / 8) % 2 ? ctx_a : ctx_b;
		float *error_cur = (i / 8) % 2 ? ctx_b : ctx_a;

		m_avx2_func(src, dst, i, error_top, error_cur, m_scale, m_offset, m_depth, m_width);
	}
public:
	ErrorDiffusionAVX2(unsigned width, unsigned height, const PixelFormat &format_in, const PixelFormat &format_out) :
		m_scalar_func{ select_error_diffusion_scalar_func(format_in.type, format_out.type) },
		m_avx2_func{ select_error_diffusion_avx2_func(format_in.type, format_out.type) },
		m_pixel_in{ format_in.type },
		m_pixel_out{ format_out.type },
		m_scale{},
		m_offset{},
		m_depth{ format_out.depth },
		m_width{ width },
		m_height{ height }
	{
		zassert_d(width <= pixel_max_width(format_in.type), "overflow");
		zassert_d(width <= pixel_max_width(format_out.type), "overflow");

		if (!pixel_is_integer(format_out.type))
			throw error::InternalError{ "cannot dither to non-integer format" };

		std::tie(m_scale, m_offset) = get_scale_offset(format_in, format_out);
	}

	filter_flags get_flags() const override
	{
		filter_flags flags{};

		flags.has_state = true;
		flags.same_row = true;
		flags.in_place = pixel_size(m_pixel_in) == pixel_size(m_pixel_out);
		flags.entire_row = true;

		return flags;
	}

	image_attributes get_image_attributes() const override
	{
		return{ m_width, m_height, m_pixel_out };
	}

	pair_unsigned get_required_row_range(unsigned i) const override
	{
		return{ i, std::min(i + 8, m_height) };
	}

	pair_unsigned get_required_col_range(unsigned, unsigned) const override
	{
		return{ 0, get_image_attributes().width };
	}

	unsigned get_simultaneous_lines() const override { return 8; }

	unsigned get_max_buffering() const override { return 8; }

	size_t get_context_size() const override
	{
		return (m_width + 2) * sizeof(float) * 2;
	}

	size_t get_tmp_size(unsigned, unsigned) const override { return 0; }

	void init_context(void *ctx) const override
	{
		std::fill_n(static_cast<unsigned char *>(ctx), get_context_size(), 0);
	}

	void process(void *ctx, const graph::ImageBuffer<const void> *src, const graph::ImageBuffer<void> *dst, void *, unsigned i, unsigned, unsigned) const override
	{
		if (m_height - i < 8) {
			bool parity = !!((i / 8) % 2);

			for (unsigned ii = i; ii < m_height; ++ii) {
				process_scalar(ctx, (*src)[ii], (*dst)[ii], parity);
				parity = !parity;
			}
		} else {
			process_vector(ctx, *src, *dst, i);
		}
	}
};

} // namespace


std::unique_ptr<graph::ImageFilter> create_error_diffusion_avx2(unsigned width, unsigned height, const PixelFormat &pixel_in, const PixelFormat &pixel_out)
{
	if (width < 14)
		return nullptr;

	return ztd::make_unique<ErrorDiffusionAVX2>(width, height, pixel_in, pixel_out);
}

} // namespace depth
} // namespace zimg

#endif // ZIMG_X86
