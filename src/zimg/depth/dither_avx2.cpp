#ifdef ZIMG_X86

#include "common/ccdep.h"

#include <cstdint>
#include <immintrin.h>
#include "common/align.h"

#define HAVE_CPU_SSE2
#define HAVE_CPU_AVX2
  #include "common/x86util.h"
#undef HAVE_CPU_SSE2
#undef HAVE_CPU_AVX2

#include "dither_x86.h"

namespace zimg {
namespace depth {

namespace {

struct LoadF16 {
	typedef uint16_t type;

	static __m256 load(const uint16_t *ptr) { return _mm256_cvtph_ps(_mm_load_si128((const __m128i *)ptr)); }
};

struct LoadF32 {
	typedef float type;

	static __m256 load(const float *ptr) { return _mm256_load_ps(ptr); }
};

inline FORCE_INLINE void mm_store_left_epi8(uint8_t *dst, __m128i x, unsigned count)
{
	mm_store_left_si128((__m128i *)dst, x, count);
}

inline FORCE_INLINE void mm_store_right_epi8(uint8_t *dst, __m128i x, unsigned count)
{
	mm_store_right_si128((__m128i *)dst, x, count);
}

inline FORCE_INLINE void mm256_store_left_epi16(uint16_t *dst, __m256i x, unsigned count)
{
	mm256_store_left_si256((__m256i *)dst, x, count * 2);
}

inline FORCE_INLINE void mm256_store_right_epi16(uint16_t *dst, __m256i x, unsigned count)
{
	mm256_store_right_si256((__m256i *)dst, x, count * 2);
}

inline FORCE_INLINE __m256i mm256_zeroextend_epi8(__m128i y)
{
	__m256i x;

	x = _mm256_permute4x64_epi64(_mm256_castsi128_si256(y), _MM_SHUFFLE(1, 1, 0, 0));
	x = _mm256_unpacklo_epi8(x, _mm256_setzero_si256());

	return x;
}

inline FORCE_INLINE __m128i mm256_packus_epi16_si256_si128(__m256i x)
{
	x = _mm256_packus_epi16(x, x);
	x = _mm256_permute4x64_epi64(x, _MM_SHUFFLE(3, 1, 2, 0));

	return _mm256_castsi256_si128(x);
}

inline FORCE_INLINE void mm256_cvtepu16_ps(__m256i x, __m256 &lo, __m256 &hi)
{
	__m256i lo_dw, hi_dw;

	x = _mm256_permute4x64_epi64(x, _MM_SHUFFLE(3, 1, 2, 0));

	lo_dw = _mm256_unpacklo_epi16(x, _mm256_setzero_si256());
	hi_dw = _mm256_unpackhi_epi16(x, _mm256_setzero_si256());

	lo = _mm256_cvtepi32_ps(lo_dw);
	hi = _mm256_cvtepi32_ps(hi_dw);
}

inline FORCE_INLINE __m256i mm256_cvtps_epu16(__m256 lo, __m256 hi)
{
	__m256i lo_dw = _mm256_cvtps_epi32(lo);
	__m256i hi_dw = _mm256_cvtps_epi32(hi);
	__m256i x;

	x = _mm256_packus_epi32(lo_dw, hi_dw); // 0 1 2 3 8 9 a b | 4 5 6 7 c d e f
	x = _mm256_permute4x64_epi64(x, _MM_SHUFFLE(3, 1, 2, 0));

	return x;
}


inline FORCE_INLINE __m128i ordered_dither_b2b_avx2_xiter(unsigned j, const float *dither, unsigned dither_offset, unsigned dither_mask,
                                                          const uint8_t *src_p, __m256 scale, __m256 offset, __m128i out_max)
{
	__m128i y = _mm_load_si128((const __m128i *)(src_p + j));
	__m256i x = mm256_zeroextend_epi8(y);
	__m256 lo, hi;
	__m256 dith;

	mm256_cvtepu16_ps(x, lo, hi);

	dith = _mm256_load_ps(dither + ((dither_offset + j + 0) & dither_mask));
	lo = _mm256_fmadd_ps(scale, lo, offset);
	lo = _mm256_add_ps(lo, dith);

	dith = _mm256_load_ps(dither + ((dither_offset + j + 8) & dither_mask));
	hi = _mm256_fmadd_ps(scale, hi, offset);
	hi = _mm256_add_ps(hi, dith);

	x = mm256_cvtps_epu16(lo, hi);
	y = mm256_packus_epi16_si256_si128(x);
	y = _mm_min_epu8(y, out_max);

	return y;
}

inline FORCE_INLINE __m256i ordered_dither_b2w_avx2_xiter(unsigned j, const float *dither, unsigned dither_offset, unsigned dither_mask,
                                                          const uint8_t *src_p, __m256 scale, __m256 offset, __m256i out_max)
{
	__m128i y = _mm_load_si128((const __m128i *)(src_p + j));
	__m256i x = mm256_zeroextend_epi8(y);
	__m256 lo, hi;
	__m256 dith;

	mm256_cvtepu16_ps(x, lo, hi);

	dith = _mm256_load_ps(dither + ((dither_offset + j + 0) & dither_mask));
	lo = _mm256_fmadd_ps(scale, lo, offset);
	lo = _mm256_add_ps(lo, dith);

	dith = _mm256_load_ps(dither + ((dither_offset + j + 8) & dither_mask));
	hi = _mm256_fmadd_ps(scale, hi, offset);
	hi = _mm256_add_ps(hi, dith);

	x = mm256_cvtps_epu16(lo, hi);
	x = _mm256_min_epu16(x, out_max);

	return x;
}

inline FORCE_INLINE __m128i ordered_dither_w2b_avx2_xiter(unsigned j, const float *dither, unsigned dither_offset, unsigned dither_mask,
                                                          const uint16_t *src_p, __m256 scale, __m256 offset, __m128i out_max)
{
	__m256i x = _mm256_load_si256((const __m256i *)(src_p + j));
	__m128i y;
	__m256 lo, hi;
	__m256 dith;

	mm256_cvtepu16_ps(x, lo, hi);

	dith = _mm256_load_ps(dither + ((dither_offset + j + 0) & dither_mask));
	lo = _mm256_fmadd_ps(scale, lo, offset);
	lo = _mm256_add_ps(lo, dith);

	dith = _mm256_load_ps(dither + ((dither_offset + j + 8) & dither_mask));
	hi = _mm256_fmadd_ps(scale, hi, offset);
	hi = _mm256_add_ps(hi, dith);

	x = mm256_cvtps_epu16(lo, hi);
	y = mm256_packus_epi16_si256_si128(x);
	y = _mm_min_epu8(y, out_max);

	return y;
}

inline FORCE_INLINE __m256i ordered_dither_w2w_avx2_xiter(unsigned j, const float *dither, unsigned dither_offset, unsigned dither_mask,
                                                          const uint16_t *src_p, __m256 scale, __m256 offset, __m256i out_max)
{
	__m256i x = _mm256_load_si256((const __m256i *)(src_p + j));
	__m256 lo, hi;
	__m256 dith;

	mm256_cvtepu16_ps(x, lo, hi);

	dith = _mm256_load_ps(dither + ((dither_offset + j + 0) & dither_mask));
	lo = _mm256_fmadd_ps(scale, lo, offset);
	lo = _mm256_add_ps(lo, dith);

	dith = _mm256_load_ps(dither + ((dither_offset + j + 8) & dither_mask));
	hi = _mm256_fmadd_ps(scale, hi, offset);
	hi = _mm256_add_ps(hi, dith);

	x = mm256_cvtps_epu16(lo, hi);
	x = _mm256_min_epu16(x, out_max);

	return x;
}

template <class Load>
inline FORCE_INLINE __m128i ordered_dither_f2b_avx2_xiter(unsigned j, const float *dither, unsigned dither_offset, unsigned dither_mask,
                                                          const typename Load::type *src_p, __m256 scale, __m256 offset, __m128i out_max)
{
	__m256 lo = Load::load(src_p + j + 0);
	__m256 hi = Load::load(src_p + j + 8);
	__m256 dith;
	__m256i x;
	__m128i y;

	dith = _mm256_load_ps(dither + ((dither_offset + j + 0) & dither_mask));
	lo = _mm256_fmadd_ps(scale, lo, offset);
	lo = _mm256_add_ps(lo, dith);

	dith = _mm256_load_ps(dither + ((dither_offset + j + 8) & dither_mask));
	hi = _mm256_fmadd_ps(scale, hi, offset);
	hi = _mm256_add_ps(hi, dith);

	x = mm256_cvtps_epu16(lo, hi);
	y = mm256_packus_epi16_si256_si128(x);
	y = _mm_min_epu8(y, out_max);

	return y;
}

template <class Load>
inline FORCE_INLINE __m256i ordered_dither_f2w_avx2_xiter(unsigned j, const float *dither, unsigned dither_offset, unsigned dither_mask,
                                                          const typename Load::type *src_p, __m256 scale, __m256 offset, __m256i out_max)
{
	__m256 lo = Load::load(src_p + j + 0);
	__m256 hi = Load::load(src_p + j + 8);
	__m256 dith;
	__m256i x;

	dith = _mm256_load_ps(dither + ((dither_offset + j + 0) & dither_mask));
	lo = _mm256_fmadd_ps(scale, lo, offset);
	lo = _mm256_add_ps(lo, dith);

	dith = _mm256_load_ps(dither + ((dither_offset + j + 8) & dither_mask));
	hi = _mm256_fmadd_ps(scale, hi, offset);
	hi = _mm256_add_ps(hi, dith);

	x = mm256_cvtps_epu16(lo, hi);
	x = _mm256_min_epu16(x, out_max);

	return x;
}

} // namespace


void ordered_dither_b2b_avx2(const float *dither, unsigned dither_offset, unsigned dither_mask,
                             const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right)
{
	const uint8_t *src_p = static_cast<const uint8_t *>(src);
	uint8_t *dst_p = static_cast<uint8_t *>(dst);

	unsigned vec_left = ceil_n(left, 16);
	unsigned vec_right = floor_n(right, 16);

	const __m256 scale_ps = _mm256_set1_ps(scale);
	const __m256 offset_ps = _mm256_set1_ps(offset);
	const __m128i out_max = _mm_set1_epi8((uint8_t)((1 << bits) - 1));

#define XITER ordered_dither_b2b_avx2_xiter
#define XARGS dither, dither_offset, dither_mask, src_p, scale_ps, offset_ps, out_max
	if (left != vec_left) {
		__m128i x = XITER(vec_left - 16, XARGS);
		mm_store_left_epi8(dst_p + vec_left - 16, x, vec_left - left);
	}

	for (unsigned j = vec_left; j < vec_right; j += 16) {
		__m128i x = XITER(j, XARGS);
		_mm_store_si128((__m128i *)(dst_p + j), x);
	}

	if (right != vec_right) {
		__m128i x = XITER(vec_right, XARGS);
		mm_store_right_epi8(dst_p + vec_right, x, right - vec_right);
	}
#undef XITER
#undef XARGS
}

void ordered_dither_b2w_avx2(const float *dither, unsigned dither_offset, unsigned dither_mask,
                             const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right)
{
	const uint8_t *src_p = static_cast<const uint8_t *>(src);
	uint16_t *dst_p = static_cast<uint16_t *>(dst);

	unsigned vec_left = ceil_n(left, 16);
	unsigned vec_right = floor_n(right, 16);

	const __m256 scale_ps = _mm256_set1_ps(scale);
	const __m256 offset_ps = _mm256_set1_ps(offset);
	const __m256i out_max = _mm256_set1_epi16((uint16_t)((1 << bits) - 1));

#define XITER ordered_dither_b2w_avx2_xiter
#define XARGS dither, dither_offset, dither_mask, src_p, scale_ps, offset_ps, out_max
	if (left != vec_left) {
		__m256i x = XITER(vec_left - 16, XARGS);
		mm256_store_left_epi16(dst_p + vec_left - 16, x, vec_left - left);
	}

	for (unsigned j = vec_left; j < vec_right; j += 16) {
		__m256i x = XITER(j, XARGS);
		_mm256_store_si256((__m256i *)(dst_p + j), x);
	}

	if (right != vec_right) {
		__m256i x = XITER(vec_right, XARGS);
		mm256_store_right_epi16(dst_p + vec_right, x, right - vec_right);
	}
#undef XITER
#undef XARGS
}

void ordered_dither_w2b_avx2(const float *dither, unsigned dither_offset, unsigned dither_mask,
                             const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right)
{
	const uint16_t *src_p = static_cast<const uint16_t *>(src);
	uint8_t *dst_p = static_cast<uint8_t *>(dst);

	unsigned vec_left = ceil_n(left, 16);
	unsigned vec_right = floor_n(right, 16);

	const __m256 scale_ps = _mm256_set1_ps(scale);
	const __m256 offset_ps = _mm256_set1_ps(offset);
	const __m128i out_max = _mm_set1_epi8((uint8_t)((1 << bits) - 1));

#define XITER ordered_dither_w2b_avx2_xiter
#define XARGS dither, dither_offset, dither_mask, src_p, scale_ps, offset_ps, out_max
	if (left != vec_left) {
		__m128i x = XITER(vec_left - 16, XARGS);
		mm_store_left_epi8(dst_p + vec_left - 16, x, vec_left - left);
	}

	for (unsigned j = vec_left; j < vec_right; j += 16) {
		__m128i x = XITER(j, XARGS);
		_mm_store_si128((__m128i *)(dst_p + j), x);
	}

	if (right != vec_right) {
		__m128i x = XITER(vec_right, XARGS);
		mm_store_right_epi8(dst_p + vec_right, x, right - vec_right);
	}
#undef XITER
#undef XARGS
}

void ordered_dither_w2w_avx2(const float *dither, unsigned dither_offset, unsigned dither_mask,
                             const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right)
{
	const uint16_t *src_p = static_cast<const uint16_t *>(src);
	uint16_t *dst_p = static_cast<uint16_t *>(dst);

	unsigned vec_left = ceil_n(left, 16);
	unsigned vec_right = floor_n(right, 16);

	const __m256 scale_ps = _mm256_set1_ps(scale);
	const __m256 offset_ps = _mm256_set1_ps(offset);
	const __m256i out_max = _mm256_set1_epi16((uint16_t)((1 << bits) - 1));

#define XITER ordered_dither_w2w_avx2_xiter
#define XARGS dither, dither_offset, dither_mask, src_p, scale_ps, offset_ps, out_max
	if (left != vec_left) {
		__m256i x = XITER(vec_left - 16, XARGS);
		mm256_store_left_epi16(dst_p + vec_left - 16, x, vec_left - left);
	}

	for (unsigned j = vec_left; j < vec_right; j += 16) {
		__m256i x = XITER(j, XARGS);
		_mm256_store_si256((__m256i *)(dst_p + j), x);
	}

	if (right != vec_right) {
		__m256i x = XITER(vec_right, XARGS);
		mm256_store_right_epi16(dst_p + vec_right, x, right - vec_right);
	}
#undef XITER
#undef XARGS
}

void ordered_dither_h2b_avx2(const float *dither, unsigned dither_offset, unsigned dither_mask,
                             const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right)
{
	const uint16_t *src_p = static_cast<const uint16_t *>(src);
	uint8_t *dst_p = static_cast<uint8_t *>(dst);

	unsigned vec_left = ceil_n(left, 16);
	unsigned vec_right = floor_n(right, 16);

	const __m256 scale_ps = _mm256_set1_ps(scale);
	const __m256 offset_ps = _mm256_set1_ps(offset);
	const __m128i out_max = _mm_set1_epi8((uint8_t)((1 << bits) - 1));

#define XITER ordered_dither_f2b_avx2_xiter<LoadF16>
#define XARGS dither, dither_offset, dither_mask, src_p, scale_ps, offset_ps, out_max
	if (left != vec_left) {
		__m128i x = XITER(vec_left - 16, XARGS);
		mm_store_left_epi8(dst_p + vec_left - 16, x, vec_left - left);
	}

	for (unsigned j = vec_left; j < vec_right; j += 16) {
		__m128i x = XITER(j, XARGS);
		_mm_store_si128((__m128i *)(dst_p + j), x);
	}

	if (right != vec_right) {
		__m128i x = XITER(vec_right, XARGS);
		mm_store_right_epi8(dst_p + vec_right, x, right - vec_right);
	}
#undef XITER
#undef XARGS
}

void ordered_dither_h2w_avx2(const float *dither, unsigned dither_offset, unsigned dither_mask,
                             const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right)
{
	const uint16_t *src_p = static_cast<const uint16_t *>(src);
	uint16_t *dst_p = static_cast<uint16_t *>(dst);

	unsigned vec_left = ceil_n(left, 16);
	unsigned vec_right = floor_n(right, 16);

	const __m256 scale_ps = _mm256_set1_ps(scale);
	const __m256 offset_ps = _mm256_set1_ps(offset);
	const __m256i out_max = _mm256_set1_epi16((uint16_t)((1 << bits) - 1));

#define XITER ordered_dither_f2w_avx2_xiter<LoadF16>
#define XARGS dither, dither_offset, dither_mask, src_p, scale_ps, offset_ps, out_max
	if (left != vec_left) {
		__m256i x = XITER(vec_left - 16, XARGS);
		mm256_store_left_epi16(dst_p + vec_left - 16, x, vec_left - left);
	}

	for (unsigned j = vec_left; j < vec_right; j += 16) {
		__m256i x = XITER(j, XARGS);
		_mm256_store_si256((__m256i *)(dst_p + j), x);
	}

	if (right != vec_right) {
		__m256i x = XITER(vec_right, XARGS);
		mm256_store_right_epi16(dst_p + vec_right, x, right - vec_right);
	}
#undef XITER
#undef XARGS
}

void ordered_dither_f2b_avx2(const float *dither, unsigned dither_offset, unsigned dither_mask,
                             const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right)
{
	const float *src_p = static_cast<const float *>(src);
	uint8_t *dst_p = static_cast<uint8_t *>(dst);

	unsigned vec_left = ceil_n(left, 16);
	unsigned vec_right = floor_n(right, 16);

	const __m256 scale_ps = _mm256_set1_ps(scale);
	const __m256 offset_ps = _mm256_set1_ps(offset);
	const __m128i out_max = _mm_set1_epi8((uint8_t)((1 << bits) - 1));

#define XITER ordered_dither_f2b_avx2_xiter<LoadF32>
#define XARGS dither, dither_offset, dither_mask, src_p, scale_ps, offset_ps, out_max
	if (left != vec_left) {
		__m128i x = XITER(vec_left - 16, XARGS);
		mm_store_left_epi8(dst_p + vec_left - 16, x, vec_left - left);
	}

	for (unsigned j = vec_left; j < vec_right; j += 16) {
		__m128i x = XITER(j, XARGS);
		_mm_store_si128((__m128i *)(dst_p + j), x);
	}

	if (right != vec_right) {
		__m128i x = XITER(vec_right, XARGS);
		mm_store_right_epi8(dst_p + vec_right, x, right - vec_right);
	}
#undef XITER
#undef XARGS
}

void ordered_dither_f2w_avx2(const float *dither, unsigned dither_offset, unsigned dither_mask,
                             const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right)
{
	const float *src_p = static_cast<const float *>(src);
	uint16_t *dst_p = static_cast<uint16_t *>(dst);

	unsigned vec_left = ceil_n(left, 16);
	unsigned vec_right = floor_n(right, 16);

	const __m256 scale_ps = _mm256_set1_ps(scale);
	const __m256 offset_ps = _mm256_set1_ps(offset);
	const __m256i out_max = _mm256_set1_epi16((uint16_t)((1 << bits) - 1));

#define XITER ordered_dither_f2w_avx2_xiter<LoadF32>
#define XARGS dither, dither_offset, dither_mask, src_p, scale_ps, offset_ps, out_max
	if (left != vec_left) {
		__m256i x = XITER(vec_left - 16, XARGS);
		mm256_store_left_epi16(dst_p + vec_left - 16, x, vec_left - left);
	}

	for (unsigned j = vec_left; j < vec_right; j += 16) {
		__m256i x = XITER(j, XARGS);
		_mm256_store_si256((__m256i *)(dst_p + j), x);
	}

	if (right != vec_right) {
		__m256i x = XITER(vec_right, XARGS);
		mm256_store_right_epi16(dst_p + vec_right, x, right - vec_right);
	}
#undef XITER
#undef XARGS
}

} // namespace depth
} // namespace zimg

#endif // ZIMG_X86
