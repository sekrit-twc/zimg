#ifdef ZIMG_X86

#include <cstdint>
#include <immintrin.h>
#include "common/align.h"
#include "dither_x86.h"

#include "common/x86/sse2_util.h"
#include "common/x86/avx2_util.h"

namespace zimg::depth {

namespace {

// Convert the packed unsigned 16-bit integers in [x] to unsigned 8-bit integers using saturation.
static inline FORCE_INLINE __m128i mm256_cvtusepi16_epi8(__m256i x)
{
	x = _mm256_packus_epi16(x, x);
	x = _mm256_permute4x64_epi64(x, _MM_SHUFFLE(3, 1, 2, 0));

	return _mm256_castsi256_si128(x);
}


struct LoadU8 {
	typedef uint8_t type;

	static inline FORCE_INLINE __m256 load8(const uint8_t *ptr)
	{
		return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)ptr)));
	}
};

struct LoadU16 {
	typedef uint16_t type;

	static inline FORCE_INLINE __m256 load8(const uint16_t *ptr)
	{
		return _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm_load_si128((const __m128i *)ptr)));
	}
};

struct LoadF16 {
	typedef uint16_t type;

	static inline FORCE_INLINE __m256 load8(const uint16_t *ptr)
	{
		return _mm256_cvtph_ps(_mm_load_si128((const __m128i *)ptr));
	}
};

struct LoadF32 {
	typedef float type;

	static inline FORCE_INLINE __m256 load8(const float *ptr)
	{
		return _mm256_load_ps(ptr);
	}
};

struct StoreU8 {
	typedef uint8_t type;

	static inline FORCE_INLINE void store16(uint8_t *ptr, __m256i x)
	{
		_mm_store_si128((__m128i *)ptr, mm256_cvtusepi16_epi8(x));
	}

	static inline FORCE_INLINE void store16_idxlo(uint8_t *ptr, __m256i x, unsigned idx)
	{
		mm_store_idxlo_epi8((__m128i *)ptr, mm256_cvtusepi16_epi8(x), idx);
	}

	static inline FORCE_INLINE void store16_idxhi(uint8_t *ptr, __m256i x, unsigned idx)
	{
		mm_store_idxhi_epi8((__m128i *)ptr, mm256_cvtusepi16_epi8(x), idx);
	}
};

struct StoreU16 {
	typedef uint16_t type;

	static inline FORCE_INLINE void store16(uint16_t *ptr, __m256i x) { _mm256_store_si256((__m256i *)ptr, x); }

	static inline FORCE_INLINE void store16_idxlo(uint16_t *ptr, __m256i x, unsigned idx) { mm256_store_idxlo_epi16((__m256i *)ptr, x, idx); }

	static inline FORCE_INLINE void store16_idxhi(uint16_t *ptr, __m256i x, unsigned idx) { mm256_store_idxhi_epi16((__m256i *)ptr, x, idx); }
};


// Convert the 32-bit floating point values in [lo | hi] to unsigned 16-bit integers using saturation.
inline FORCE_INLINE __m256i mm256_cvt2ps_epu16(__m256 lo, __m256 hi)
{
	__m256i lo_dw = _mm256_cvtps_epi32(lo);
	__m256i hi_dw = _mm256_cvtps_epi32(hi);
	__m256i x;

	x = _mm256_packus_epi32(lo_dw, hi_dw); // 0 1 2 3 8 9 a b | 4 5 6 7 c d e f
	x = _mm256_permute4x64_epi64(x, _MM_SHUFFLE(3, 1, 2, 0));

	return x;
}

inline FORCE_INLINE __m256i ordered_dither_avx2_xiter(__m256 lo, __m256 hi, unsigned j, const float *dither, unsigned dither_offset, unsigned dither_mask,
                                                      const __m256 &scale, const __m256 &offset, const __m256i &out_max)
{
	__m256 dith;
	__m256i x;

	dith = _mm256_load_ps(dither + ((dither_offset + j + 0) & dither_mask));
	lo = _mm256_fmadd_ps(scale, lo, offset);
	lo = _mm256_add_ps(lo, dith);

	dith = _mm256_load_ps(dither + ((dither_offset + j + 8) & dither_mask));
	hi = _mm256_fmadd_ps(scale, hi, offset);
	hi = _mm256_add_ps(hi, dith);

	x = mm256_cvt2ps_epu16(lo, hi);
	x = _mm256_min_epu16(x, out_max);

	return x;
}

template <class Load, class Store>
inline FORCE_INLINE void ordered_dither_avx2_impl(const float *dither, unsigned dither_offset, unsigned dither_mask,
                                                  const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right)
{
	const typename Load::type *src_p = static_cast<const typename Load::type *>(src);
	typename Store::type *dst_p = static_cast<typename Store::type *>(dst);

	unsigned vec_left = ceil_n(left, 16);
	unsigned vec_right = floor_n(right, 16);

	const __m256 scale_ps = _mm256_set1_ps(scale);
	const __m256 offset_ps = _mm256_set1_ps(offset);
	const __m256i out_max = _mm256_set1_epi16(static_cast<uint16_t>((1 << bits) - 1));

#define XARGS dither, dither_offset, dither_mask, scale_ps, offset_ps, out_max
	if (left != vec_left) {
		__m256 lo = Load::load8(src_p + vec_left - 16 + 0);
		__m256 hi = Load::load8(src_p + vec_left - 16 + 8);
		__m256i x = ordered_dither_avx2_xiter(lo, hi, vec_left - 16, XARGS);
		Store::store16_idxhi(dst_p + vec_left - 16, x, left % 16);
	}
	for (unsigned j = vec_left; j < vec_right; j += 16) {
		__m256 lo = Load::load8(src_p + j + 0);
		__m256 hi = Load::load8(src_p + j + 8);
		__m256i x = ordered_dither_avx2_xiter(lo, hi, j, XARGS);
		Store::store16(dst_p + j, x);
	}
	if (right != vec_right) {
		__m256 lo = Load::load8(src_p + vec_right + 0);
		__m256 hi = Load::load8(src_p + vec_right + 8);
		__m256i x = ordered_dither_avx2_xiter(lo, hi, vec_right, XARGS);
		Store::store16_idxlo(dst_p + vec_right, x, right % 16);
	}
#undef XARGS
}

} // namespace


void ordered_dither_b2b_avx2(const float *dither, unsigned dither_offset, unsigned dither_mask,
                             const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right)
{
	ordered_dither_avx2_impl<LoadU8, StoreU8>(dither, dither_offset, dither_mask, src, dst, scale, offset, bits, left, right);
}

void ordered_dither_b2w_avx2(const float *dither, unsigned dither_offset, unsigned dither_mask,
                             const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right)
{
	ordered_dither_avx2_impl<LoadU8, StoreU16>(dither, dither_offset, dither_mask, src, dst, scale, offset, bits, left, right);
}

void ordered_dither_w2b_avx2(const float *dither, unsigned dither_offset, unsigned dither_mask,
                             const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right)
{
	ordered_dither_avx2_impl<LoadU16, StoreU8>(dither, dither_offset, dither_mask, src, dst, scale, offset, bits, left, right);
}

void ordered_dither_w2w_avx2(const float *dither, unsigned dither_offset, unsigned dither_mask,
                             const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right)
{
	ordered_dither_avx2_impl<LoadU16, StoreU16>(dither, dither_offset, dither_mask, src, dst, scale, offset, bits, left, right);
}

void ordered_dither_h2b_avx2(const float *dither, unsigned dither_offset, unsigned dither_mask,
                             const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right)
{
	ordered_dither_avx2_impl<LoadF16, StoreU8>(dither, dither_offset, dither_mask, src, dst, scale, offset, bits, left, right);
}

void ordered_dither_h2w_avx2(const float *dither, unsigned dither_offset, unsigned dither_mask,
                             const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right)
{
	ordered_dither_avx2_impl<LoadF16, StoreU16>(dither, dither_offset, dither_mask, src, dst, scale, offset, bits, left, right);
}

void ordered_dither_f2b_avx2(const float *dither, unsigned dither_offset, unsigned dither_mask,
                             const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right)
{
	ordered_dither_avx2_impl<LoadF32, StoreU8>(dither, dither_offset, dither_mask, src, dst, scale, offset, bits, left, right);
}

void ordered_dither_f2w_avx2(const float *dither, unsigned dither_offset, unsigned dither_mask,
                             const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right)
{
	ordered_dither_avx2_impl<LoadF32, StoreU16>(dither, dither_offset, dither_mask, src, dst, scale, offset, bits, left, right);
}

} // namespace zimg::depth

#endif // ZIMG_X86
