#ifdef ZIMG_X86

#include <immintrin.h>
#ifdef __clang__
  #include <x86intrin.h>
#endif
#include "common/align.h"

#define HAVE_CPU_SSE2
#define HAVE_CPU_AVX
  #include "common/x86util.h"
#undef HAVE_CPU_SSE2
#undef HAVE_CPU_AVX

#include "f16c_x86.h"

namespace zimg {;
namespace depth {;

namespace {;

inline FORCE_INLINE void mm_store_left_epi16(uint16_t *dst, __m128i x, unsigned count)
{
	mm_store_left_si128((__m128i *)dst, x, count * 2);
}

inline FORCE_INLINE void mm_store_right_epi16(uint16_t *dst, __m128i x, unsigned count)
{
	mm_store_right_si128((__m128i *)dst, x, count * 2);
}

inline FORCE_INLINE void mm256_store_left(float *dst, __m256 x, unsigned count)
{
	mm256_store_left_ps(dst, x, count * 4);
}

inline FORCE_INLINE void mm256_store_right(float *dst, __m256 x, unsigned count)
{
	mm256_store_right_ps(dst, x, count * 4);
}

} // namespace

void f16c_half_to_float_ivb(const void *src, void *dst, unsigned left, unsigned right)
{
	const uint16_t *src_p = static_cast<const uint16_t *>(src);
	float *dst_p = static_cast<float *>(dst);

	unsigned vec_left = ceil_n(left, 8);
	unsigned vec_right = floor_n(right, 8);

	if (left != vec_left) {
		__m256 x = _mm256_cvtph_ps(_mm_load_si128((const __m128i *)(src_p + vec_left - 8)));
		mm256_store_left(dst_p + vec_left - 8, x, vec_left - left);
	}

	for (unsigned j = vec_left; j < vec_right; j += 8) {
		__m256 x = _mm256_cvtph_ps(_mm_load_si128((const __m128i *)(src_p + j)));
		_mm256_store_ps(dst_p + j, x);
	}

	if (right != vec_right) {
		__m256 x = _mm256_cvtph_ps(_mm_load_si128((const __m128i *)(src_p + vec_right)));
		mm256_store_right(dst_p + vec_right, x, right - vec_right);
	}
}

void f16c_float_to_half_ivb(const void *src, void *dst, unsigned left, unsigned right)
{
	const float *src_p = static_cast<const float *>(src);
	uint16_t *dst_p = static_cast<uint16_t *>(dst);

	unsigned vec_left = ceil_n(left, 8);
	unsigned vec_right = floor_n(right, 8);

	if (left != vec_left) {
		__m128i x = _mm256_cvtps_ph(_mm256_load_ps(src_p + vec_left - 8), 0);
		mm_store_left_epi16(dst_p + vec_left - 8, x, vec_left - left);
	}

	for (unsigned j = vec_left; j < vec_right; j += 8) {
		__m128i x = _mm256_cvtps_ph(_mm256_load_ps(src_p + j), 0);
		_mm_store_si128((__m128i *)(dst_p + j), x);
	}

	if (right != vec_right) {
		__m128i x = _mm256_cvtps_ph(_mm256_load_ps(src_p + vec_right), 0);
		mm_store_right_epi16(dst_p + vec_right, x, right - vec_right);
	}
}

} // namespace depth
} // namespace zimg

#endif // ZIMG_X86
