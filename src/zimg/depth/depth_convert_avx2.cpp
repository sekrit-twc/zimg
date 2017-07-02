#ifdef ZIMG_X86

#include <algorithm>
#include <cstdint>
#include <immintrin.h>
#include "common/align.h"
#include "common/ccdep.h"

#define HAVE_CPU_SSE2
#define HAVE_CPU_AVX
#define HAVE_CPU_AVX2
  #include "common/x86util.h"
#undef HAVE_CPU_SSE2
#undef HAVE_CPU_AVX
#undef HAVE_CPU_AVX2

#include "depth_convert_x86.h"

namespace zimg {
namespace depth {

namespace {

struct PackF16 {
	typedef __m128i type;

	static __m128i pack(__m256 x) { return _mm256_cvtps_ph(x, 0); }
};

struct PackF32 {
	typedef __m256 type;

	static __m256 pack(__m256 x) { return x; }
};


inline FORCE_INLINE __m256i mm256_zeroextend_epi8(__m128i y)
{
	__m256i x;

	x = _mm256_permute4x64_epi64(_mm256_castsi128_si256(y), _MM_SHUFFLE(1, 1, 0, 0));
	x = _mm256_unpacklo_epi8(x, _mm256_setzero_si256());

	return x;
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

template <class Pack>
inline FORCE_INLINE void depth_convert_b2f_avx2_xiter(unsigned j, const uint8_t *src_p, __m256 scale, __m256 offset,
                                                      typename Pack::type &lo_out, typename Pack::type &hi_out)
{
	__m128i y = _mm_load_si128((const __m128i *)(src_p + j));
	__m256i x = mm256_zeroextend_epi8(y);
	__m256 lo, hi;

	mm256_cvtepu16_ps(x, lo, hi);

	lo = _mm256_fmadd_ps(scale, lo, offset);
	hi = _mm256_fmadd_ps(scale, hi, offset);

	lo_out = Pack::pack(lo);
	hi_out = Pack::pack(hi);
}

template <class Pack>
inline FORCE_INLINE void depth_convert_w2f_avx2_xiter(unsigned j, const uint16_t *src_p, __m256 scale, __m256 offset,
                                                      typename Pack::type &lo_out, typename Pack::type &hi_out)
{
	__m256i x = _mm256_load_si256((const __m256i *)(src_p + j));
	__m256 lo, hi;

	mm256_cvtepu16_ps(x, lo, hi);

	lo = _mm256_fmadd_ps(scale, lo, offset);
	hi = _mm256_fmadd_ps(scale, hi, offset);

	lo_out = Pack::pack(lo);
	hi_out = Pack::pack(hi);
}

} // namespace


void depth_convert_b2h_avx2(const void *src, void *dst, float scale, float offset, unsigned left, unsigned right)
{
	const uint8_t *src_p = static_cast<const uint8_t *>(src);
	uint16_t *dst_p = static_cast<uint16_t *>(dst);

	unsigned vec_left = ceil_n(left, 16);
	unsigned vec_right = floor_n(right, 16);

	const __m256 scale_ps = _mm256_set1_ps(scale);
	const __m256 offset_ps = _mm256_set1_ps(offset);

	__m128i lo, hi;

#define XITER depth_convert_b2f_avx2_xiter<PackF16>
#define XARGS src_p, scale_ps, offset_ps, lo, hi
	if (left != vec_left) {
		XITER(vec_left - 16, XARGS);

		if (vec_left - left > 8) {
			mm_store_idxhi_epi16((__m128i *)(dst_p + vec_left - 16), lo, vec_left - left - 8);
			_mm_store_si128((__m128i *)(dst_p + vec_left - 8), hi);
		} else {
			mm_store_idxhi_epi16((__m128i *)(dst_p + vec_left - 8), hi, vec_left - left);
		}
	}

	for (unsigned j = vec_left; j < vec_right; j += 16) {
		XITER(j, XARGS);

		_mm_store_si128((__m128i *)(dst_p + j + 0), lo);
		_mm_store_si128((__m128i *)(dst_p + j + 8), hi);
	}

	if (right != vec_right) {
		XITER(vec_right, XARGS);

		if (right - vec_right > 8) {
			_mm_store_si128((__m128i *)(dst_p + vec_right + 0), lo);
			mm_store_idxlo_epi16((__m128i *)(dst_p + vec_right + 8), hi, right - vec_right - 8);
		} else {
			mm_store_idxlo_epi16((__m128i *)(dst_p + vec_right), lo, right - vec_right);
		}
	}
#undef XITER
#undef XARGS
}

void depth_convert_b2f_avx2(const void *src, void *dst, float scale, float offset, unsigned left, unsigned right)
{
	const uint8_t *src_p = static_cast<const uint8_t *>(src);
	float *dst_p = static_cast<float *>(dst);

	unsigned vec_left = ceil_n(left, 16);
	unsigned vec_right = floor_n(right, 16);

	const __m256 scale_ps = _mm256_set1_ps(scale);
	const __m256 offset_ps = _mm256_set1_ps(offset);

	__m256 lo, hi;

#define XITER depth_convert_b2f_avx2_xiter<PackF32>
#define XARGS src_p, scale_ps, offset_ps, lo, hi
	if (left != vec_left) {
		XITER(vec_left - 16, XARGS);

		if (vec_left - left > 8) {
			mm256_store_idxhi_ps(dst_p + vec_left - 16, lo, vec_left - left - 8);
			_mm256_store_ps(dst_p + vec_left - 8, hi);
		} else {
			mm256_store_idxhi_ps(dst_p + vec_left - 8, hi, vec_left - left);
		}
	}

	for (unsigned j = vec_left; j < vec_right; j += 16) {
		XITER(j, XARGS);

		_mm256_store_ps(dst_p + j + 0, lo);
		_mm256_store_ps(dst_p + j + 8, hi);
	}

	if (right != vec_right) {
		XITER(vec_right, XARGS);

		if (right - vec_right > 8) {
			_mm256_store_ps(dst_p + vec_right + 0, lo);
			mm256_store_idxlo_ps(dst_p + vec_right + 8, hi, right - vec_right - 8);
		} else {
			mm256_store_idxlo_ps(dst_p + vec_right, lo, right - vec_right);
		}
	}
#undef XITER
#undef XARGS
}

void depth_convert_w2h_avx2(const void *src, void *dst, float scale, float offset, unsigned left, unsigned right)
{
	const uint16_t *src_p = static_cast<const uint16_t *>(src);
	uint16_t *dst_p = static_cast<uint16_t *>(dst);

	unsigned vec_left = ceil_n(left, 16);
	unsigned vec_right = floor_n(right, 16);

	const __m256 scale_ps = _mm256_set1_ps(scale);
	const __m256 offset_ps = _mm256_set1_ps(offset);

	__m128i lo, hi;

#define XITER depth_convert_w2f_avx2_xiter<PackF16>
#define XARGS src_p, scale_ps, offset_ps, lo, hi
	if (left != vec_left) {
		XITER(vec_left - 16, XARGS);

		if (vec_left - left > 8) {
			mm_store_idxhi_epi16((__m128i *)(dst_p + vec_left - 16), lo, vec_left - left - 8);
			_mm_store_si128((__m128i *)(dst_p + vec_left - 8), hi);
		} else {
			mm_store_idxhi_epi16((__m128i *)(dst_p + vec_left - 8), hi, vec_left - left);
		}
	}

	for (unsigned j = vec_left; j < vec_right; j += 16) {
		XITER(j, XARGS);

		_mm_store_si128((__m128i *)(dst_p + j + 0), lo);
		_mm_store_si128((__m128i *)(dst_p + j + 8), hi);
	}

	if (right != vec_right) {
		XITER(vec_right, XARGS);

		if (right - vec_right > 8) {
			_mm_store_si128((__m128i *)(dst_p + vec_right + 0), lo);
			mm_store_idxlo_epi16((__m128i *)(dst_p + vec_right + 8), hi, right - vec_right - 8);
		} else {
			mm_store_idxlo_epi16((__m128i *)(dst_p + vec_right), lo, right - vec_right);
		}
	}
#undef XITER
#undef XARGS
}

void depth_convert_w2f_avx2(const void *src, void *dst, float scale, float offset, unsigned left, unsigned right)
{
	const uint16_t *src_p = static_cast<const uint16_t *>(src);
	float *dst_p = static_cast<float *>(dst);

	unsigned vec_left = ceil_n(left, 16);
	unsigned vec_right = floor_n(right, 16);

	const __m256 scale_ps = _mm256_set1_ps(scale);
	const __m256 offset_ps = _mm256_set1_ps(offset);

	__m256 lo, hi;

#define XITER depth_convert_w2f_avx2_xiter<PackF32>
#define XARGS src_p, scale_ps, offset_ps, lo, hi
	if (left != vec_left) {
		XITER(vec_left - 16, XARGS);

		if (vec_left - left > 8) {
			mm256_store_idxhi_ps(dst_p + vec_left - 16, lo, vec_left - left - 8);
			_mm256_store_ps(dst_p + vec_left - 8, hi);
		} else {
			mm256_store_idxhi_ps(dst_p + vec_left - 8, hi, vec_left - left);
		}
	}

	for (unsigned j = vec_left; j < vec_right; j += 16) {
		XITER(j, XARGS);

		_mm256_store_ps(dst_p + j + 0, lo);
		_mm256_store_ps(dst_p + j + 8, hi);
	}

	if (right != vec_right) {
		XITER(vec_right, XARGS);

		if (right - vec_right > 8) {
			_mm256_store_ps(dst_p + vec_right + 0, lo);
			mm256_store_idxlo_ps(dst_p + vec_right + 8, hi, right - vec_right - 8);
		} else {
			mm256_store_idxlo_ps(dst_p + vec_right, lo, right - vec_right);
		}
	}
#undef XITER
#undef XARGS
}

} // namespace depth
} // namespace zimg

#endif // ZIMG_X86
