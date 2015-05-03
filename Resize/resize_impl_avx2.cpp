#ifdef ZIMG_X86

#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include "Common/align.h"
#include "Common/linebuffer.h"
#include "Common/osdep.h"
#include "Common/pixel.h"
#include "filter.h"
#include "resize_impl.h"
#include "resize_impl_x86.h"

namespace zimg {;
namespace resize {;

namespace {;

struct ScalarPolicy_F16 {
	typedef float num_type;

	FORCE_INLINE float coeff(const FilterContext &filter, unsigned row, unsigned k)
	{
		return filter.data[row * filter.stride + k];
	}

	FORCE_INLINE float load(const uint16_t *src) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16(*src))); }

	FORCE_INLINE void store(uint16_t *dst, float x) { *dst = _mm_extract_epi16(_mm_cvtps_ph(_mm_set_ps1(x), 0), 0); }
};

struct VectorPolicy_F16 : public ScalarPolicy_F16 {
	FORCE_INLINE __m256 load_8(const uint16_t *p) { return _mm256_cvtph_ps(_mm_load_si128((const __m128i *)p)); }
	FORCE_INLINE __m256 loadu_8(const uint16_t *p) { return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)p)); }

	FORCE_INLINE void store_8(uint16_t *p, __m256 x) { _mm_store_si128((__m128i *)p, _mm256_cvtps_ph(x, 0)); }
};

struct VectorPolicy_F32 : public ScalarPolicy_F32 {
	FORCE_INLINE __m256 load_8(const float *p) { return _mm256_load_ps(p); }
	FORCE_INLINE __m256 loadu_8(const float *p) { return _mm256_loadu_ps(p); }

	FORCE_INLINE void store_8(float *p, __m256 x) { _mm256_store_ps(p, x); }
};

inline FORCE_INLINE void transpose8_ps(__m256 &row0, __m256 &row1, __m256 &row2, __m256 &row3, __m256 &row4, __m256 &row5, __m256 &row6, __m256 &row7)
{
	__m256 t0, t1, t2, t3, t4, t5, t6, t7;
	__m256 tt0, tt1, tt2, tt3, tt4, tt5, tt6, tt7;

	t0 = _mm256_unpacklo_ps(row0, row1);
	t1 = _mm256_unpackhi_ps(row0, row1);
	t2 = _mm256_unpacklo_ps(row2, row3);
	t3 = _mm256_unpackhi_ps(row2, row3);
	t4 = _mm256_unpacklo_ps(row4, row5);
	t5 = _mm256_unpackhi_ps(row4, row5);
	t6 = _mm256_unpacklo_ps(row6, row7);
	t7 = _mm256_unpackhi_ps(row6, row7);

	tt0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
	tt1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
	tt2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
	tt3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
	tt4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
	tt5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
	tt6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
	tt7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));

	row0 = _mm256_permute2f128_ps(tt0, tt4, 0x20);
	row1 = _mm256_permute2f128_ps(tt1, tt5, 0x20);
	row2 = _mm256_permute2f128_ps(tt2, tt6, 0x20);
	row3 = _mm256_permute2f128_ps(tt3, tt7, 0x20);
	row4 = _mm256_permute2f128_ps(tt0, tt4, 0x31);
	row5 = _mm256_permute2f128_ps(tt1, tt5, 0x31);
	row6 = _mm256_permute2f128_ps(tt2, tt6, 0x31);
	row7 = _mm256_permute2f128_ps(tt3, tt7, 0x31);
}

inline FORCE_INLINE void transpose16x8_epi16(__m256i &row0, __m256i &row1, __m256i &row2, __m256i &row3, __m256i &row4, __m256i &row5, __m256i &row6, __m256i &row7)
{
	__m256i t0, t1, t2, t3, t4, t5, t6, t7;
	__m256i tt0, tt1, tt2, tt3, tt4, tt5, tt6, tt7;

	t0 = _mm256_unpacklo_epi16(row0, row1);
	t1 = _mm256_unpacklo_epi16(row2, row3);
	t2 = _mm256_unpacklo_epi16(row4, row5);
	t3 = _mm256_unpacklo_epi16(row6, row7);
	t4 = _mm256_unpackhi_epi16(row0, row1);
	t5 = _mm256_unpackhi_epi16(row2, row3);
	t6 = _mm256_unpackhi_epi16(row4, row5);
	t7 = _mm256_unpackhi_epi16(row6, row7);

	tt0 = _mm256_unpacklo_epi32(t0, t1);
	tt1 = _mm256_unpackhi_epi32(t0, t1);
	tt2 = _mm256_unpacklo_epi32(t2, t3);
	tt3 = _mm256_unpackhi_epi32(t2, t3);
	tt4 = _mm256_unpacklo_epi32(t4, t5);
	tt5 = _mm256_unpackhi_epi32(t4, t5);
	tt6 = _mm256_unpacklo_epi32(t6, t7);
	tt7 = _mm256_unpackhi_epi32(t6, t7);

	t0 = _mm256_unpacklo_epi64(tt0, tt2);
	t1 = _mm256_unpackhi_epi64(tt0, tt2);
	t2 = _mm256_unpacklo_epi64(tt1, tt3);
	t3 = _mm256_unpackhi_epi64(tt1, tt3);
	t4 = _mm256_unpacklo_epi64(tt4, tt6);
	t5 = _mm256_unpackhi_epi64(tt4, tt6);
	t6 = _mm256_unpacklo_epi64(tt5, tt7);
	t7 = _mm256_unpackhi_epi64(tt5, tt7);

	row0 = _mm256_permute2x128_si256(t0, t1, 0x20);
	row1 = _mm256_permute2x128_si256(t2, t3, 0x20);
	row2 = _mm256_permute2x128_si256(t4, t5, 0x20);
	row3 = _mm256_permute2x128_si256(t6, t7, 0x20);
	row4 = _mm256_permute2x128_si256(t0, t1, 0x31);
	row5 = _mm256_permute2x128_si256(t2, t3, 0x31);
	row6 = _mm256_permute2x128_si256(t4, t5, 0x31);
	row7 = _mm256_permute2x128_si256(t6, t7, 0x31);
}

inline FORCE_INLINE void transpose8x16_epi16(__m256i &row01, __m256i &row23, __m256i &row45, __m256i &row67, __m256i &row89, __m256i &rowab, __m256i &rowcd, __m256i &rowef)
{
	__m256i t0, t1, t2, t3, t4, t5, t6, t7;
	__m256i tt0, tt1, tt2, tt3, tt4, tt5, tt6, tt7;

	tt0 = _mm256_permute2x128_si256(row01, row89, 0x20);
	tt1 = _mm256_permute2x128_si256(row01, row89, 0x31);
	tt2 = _mm256_permute2x128_si256(row23, rowab, 0x20);
	tt3 = _mm256_permute2x128_si256(row23, rowab, 0x31);
	tt4 = _mm256_permute2x128_si256(row45, rowcd, 0x20);
	tt5 = _mm256_permute2x128_si256(row45, rowcd, 0x31);
	tt6 = _mm256_permute2x128_si256(row67, rowef, 0x20);
	tt7 = _mm256_permute2x128_si256(row67, rowef, 0x31);

	t0 = _mm256_unpacklo_epi16(tt0, tt1);
	t1 = _mm256_unpacklo_epi16(tt2, tt3);
	t2 = _mm256_unpacklo_epi16(tt4, tt5);
	t3 = _mm256_unpacklo_epi16(tt6, tt7);
	t4 = _mm256_unpackhi_epi16(tt0, tt1);
	t5 = _mm256_unpackhi_epi16(tt2, tt3);
	t6 = _mm256_unpackhi_epi16(tt4, tt5);
	t7 = _mm256_unpackhi_epi16(tt6, tt7);

	tt0 = _mm256_unpacklo_epi32(t0, t1);
	tt1 = _mm256_unpackhi_epi32(t0, t1);
	tt2 = _mm256_unpacklo_epi32(t2, t3);
	tt3 = _mm256_unpackhi_epi32(t2, t3);
	tt4 = _mm256_unpacklo_epi32(t4, t5);
	tt5 = _mm256_unpackhi_epi32(t4, t5);
	tt6 = _mm256_unpacklo_epi32(t6, t7);
	tt7 = _mm256_unpackhi_epi32(t6, t7);

	row01 = _mm256_unpacklo_epi64(tt0, tt2);
	row23 = _mm256_unpackhi_epi64(tt0, tt2);
	row45 = _mm256_unpacklo_epi64(tt1, tt3);
	row67 = _mm256_unpackhi_epi64(tt1, tt3);
	row89 = _mm256_unpacklo_epi64(tt4, tt6);
	rowab = _mm256_unpackhi_epi64(tt4, tt6);
	rowcd = _mm256_unpacklo_epi64(tt5, tt7);
	rowef = _mm256_unpackhi_epi64(tt5, tt7);
}

inline FORCE_INLINE void transpose_line_8x8_ps(const float *p0, const float *p1, const float *p2, const float *p3, const float *p4, const float *p5, const float *p6, const float *p7, float *dst, unsigned n)
{
	for (unsigned j = 0; j < mod(n, 8); j += 8) {
		__m256 x0, x1, x2, x3, x4, x5, x6, x7;

		x0 = _mm256_load_ps(&p0[j]);
		x1 = _mm256_load_ps(&p1[j]);
		x2 = _mm256_load_ps(&p2[j]);
		x3 = _mm256_load_ps(&p3[j]);
		x4 = _mm256_load_ps(&p4[j]);
		x5 = _mm256_load_ps(&p5[j]);
		x6 = _mm256_load_ps(&p6[j]);
		x7 = _mm256_load_ps(&p7[j]);

		transpose8_ps(x0, x1, x2, x3, x4, x5, x6, x7);

		_mm256_store_ps(&dst[(j + 0) * 8], x0);
		_mm256_store_ps(&dst[(j + 1) * 8], x1);
		_mm256_store_ps(&dst[(j + 2) * 8], x2);
		_mm256_store_ps(&dst[(j + 3) * 8], x3);
		_mm256_store_ps(&dst[(j + 4) * 8], x4);
		_mm256_store_ps(&dst[(j + 5) * 8], x5);
		_mm256_store_ps(&dst[(j + 6) * 8], x6);
		_mm256_store_ps(&dst[(j + 7) * 8], x7);
	}
	for (unsigned j = mod(n, 8); j < n; ++j) {
		dst[j * 8 + 0] = p0[j];
		dst[j * 8 + 1] = p1[j];
		dst[j * 8 + 2] = p2[j];
		dst[j * 8 + 3] = p3[j];
		dst[j * 8 + 4] = p4[j];
		dst[j * 8 + 5] = p5[j];
		dst[j * 8 + 6] = p6[j];
		dst[j * 8 + 7] = p7[j];
	}
}

inline FORCE_INLINE void transpose_line_16x8_epi16(const uint16_t *p0, const uint16_t *p1, const uint16_t *p2, const uint16_t *p3, const uint16_t *p4, const uint16_t *p5, const uint16_t *p6, const uint16_t *p7, uint16_t *dst, unsigned n)
{
	for (unsigned j = 0; j < mod(n, 16); j += 16) {
		__m256i x0, x1, x2, x3, x4, x5, x6, x7;

		x0 = _mm256_load_si256((const __m256i *)&p0[j]);
		x1 = _mm256_load_si256((const __m256i *)&p1[j]);
		x2 = _mm256_load_si256((const __m256i *)&p2[j]);
		x3 = _mm256_load_si256((const __m256i *)&p3[j]);
		x4 = _mm256_load_si256((const __m256i *)&p4[j]);
		x5 = _mm256_load_si256((const __m256i *)&p5[j]);
		x6 = _mm256_load_si256((const __m256i *)&p6[j]);
		x7 = _mm256_load_si256((const __m256i *)&p7[j]);

		transpose16x8_epi16(x0, x1, x2, x3, x4, x5, x6, x7);

		_mm256_store_si256((__m256i *)&dst[(j + 0) * 8], x0);
		_mm256_store_si256((__m256i *)&dst[(j + 2) * 8], x1);
		_mm256_store_si256((__m256i *)&dst[(j + 4) * 8], x2);
		_mm256_store_si256((__m256i *)&dst[(j + 6) * 8], x3);
		_mm256_store_si256((__m256i *)&dst[(j + 8) * 8], x4);
		_mm256_store_si256((__m256i *)&dst[(j + 10) * 8], x5);
		_mm256_store_si256((__m256i *)&dst[(j + 12) * 8], x6);
		_mm256_store_si256((__m256i *)&dst[(j + 14) * 8], x7);
	}
	for (unsigned j = mod(n, 16); j < n; ++j) {
		dst[j * 8 + 0] = p0[j];
		dst[j * 8 + 1] = p1[j];
		dst[j * 8 + 2] = p2[j];
		dst[j * 8 + 3] = p3[j];
		dst[j * 8 + 4] = p4[j];
		dst[j * 8 + 5] = p5[j];
		dst[j * 8 + 6] = p6[j];
		dst[j * 8 + 7] = p7[j];
	}
}

inline FORCE_INLINE void transpose_block_8x8_ps(const float cache[8 * 8], float *p0, float *p1, float *p2, float *p3, float *p4, float *p5, float *p6, float *p7)
{
	__m256 x0, x1, x2, x3, x4, x5, x6, x7;

	x0 = _mm256_load_ps(&cache[0 * 8]);
	x1 = _mm256_load_ps(&cache[1 * 8]);
	x2 = _mm256_load_ps(&cache[2 * 8]);
	x3 = _mm256_load_ps(&cache[3 * 8]);
	x4 = _mm256_load_ps(&cache[4 * 8]);
	x5 = _mm256_load_ps(&cache[5 * 8]);
	x6 = _mm256_load_ps(&cache[6 * 8]);
	x7 = _mm256_load_ps(&cache[7 * 8]);

	transpose8_ps(x0, x1, x2, x3, x4, x5, x6, x7);

	_mm256_store_ps(p0, x0);
	_mm256_store_ps(p1, x1);
	_mm256_store_ps(p2, x2);
	_mm256_store_ps(p3, x3);
	_mm256_store_ps(p4, x4);
	_mm256_store_ps(p5, x5);
	_mm256_store_ps(p6, x6);
	_mm256_store_ps(p7, x7);
}

inline FORCE_INLINE void transpose_block_8x16_epi16(const uint16_t cache[8 * 16], uint16_t *p0, uint16_t *p1, uint16_t *p2, uint16_t *p3, uint16_t *p4, uint16_t *p5, uint16_t *p6, uint16_t *p7)
{
	__m256i x0, x1, x2, x3, x4, x5, x6, x7;

	x0 = _mm256_load_si256((const __m256i *)&cache[0 * 16]);
	x1 = _mm256_load_si256((const __m256i *)&cache[1 * 16]);
	x2 = _mm256_load_si256((const __m256i *)&cache[2 * 16]);
	x3 = _mm256_load_si256((const __m256i *)&cache[3 * 16]);
	x4 = _mm256_load_si256((const __m256i *)&cache[4 * 16]);
	x5 = _mm256_load_si256((const __m256i *)&cache[5 * 16]);
	x6 = _mm256_load_si256((const __m256i *)&cache[6 * 16]);
	x7 = _mm256_load_si256((const __m256i *)&cache[7 * 16]);

	transpose8x16_epi16(x0, x1, x2, x3, x4, x5, x6, x7);

	_mm256_store_si256((__m256i *)p0, x0);
	_mm256_store_si256((__m256i *)p1, x1);
	_mm256_store_si256((__m256i *)p2, x2);
	_mm256_store_si256((__m256i *)p3, x3);
	_mm256_store_si256((__m256i *)p4, x4);
	_mm256_store_si256((__m256i *)p5, x5);
	_mm256_store_si256((__m256i *)p6, x6);
	_mm256_store_si256((__m256i *)p7, x7);
}

inline FORCE_INLINE void scatter_ps(__m256 x, float *p0, float *p1, float *p2, float *p3, float *p4, float *p5, float *p6, float *p7)
{
	ALIGNED float tmp[8];

	_mm256_store_ps(tmp, x);
	*p0 = tmp[0];
	*p1 = tmp[1];
	*p2 = tmp[2];
	*p3 = tmp[3];
	*p4 = tmp[4];
	*p5 = tmp[5];
	*p6 = tmp[6];
	*p7 = tmp[7];
}

inline FORCE_INLINE void scatter_epi16(__m128i x, uint16_t *p0, uint16_t *p1, uint16_t *p2, uint16_t *p3, uint16_t *p4, uint16_t *p5, uint16_t *p6, uint16_t *p7)
{
	*p0 = _mm_extract_epi16(x, 0);
	*p1 = _mm_extract_epi16(x, 1);
	*p2 = _mm_extract_epi16(x, 2);
	*p3 = _mm_extract_epi16(x, 3);
	*p4 = _mm_extract_epi16(x, 4);
	*p5 = _mm_extract_epi16(x, 5);
	*p6 = _mm_extract_epi16(x, 6);
	*p7 = _mm_extract_epi16(x, 7);
}

// Performs accum[0,1] += x * unpack[lo,hi](a,b)
inline FORCE_INLINE void fmadd_epi16_2(__m256i x, __m256i a, __m256i b, __m256i &accum0, __m256i &accum1)
{
	__m256i tmplo, tmphi;

	tmplo = _mm256_unpacklo_epi16(a, b);
	tmphi = _mm256_unpackhi_epi16(a, b);
	tmplo = _mm256_madd_epi16(x, tmplo);
	tmphi = _mm256_madd_epi16(x, tmphi);
	accum0 = _mm256_add_epi32(accum0, tmplo);
	accum1 = _mm256_add_epi32(accum1, tmphi);
}

inline FORCE_INLINE __m256i pack_i30_to_epi16(__m256i lo, __m256i hi)
{
	__m256i offset = _mm256_set1_epi32(1 << 13);

	lo = _mm256_add_epi32(lo, offset);
	hi = _mm256_add_epi32(hi, offset);

	lo = _mm256_srai_epi32(lo, 14);
	hi = _mm256_srai_epi32(hi, 14);

	return _mm256_packs_epi32(lo, hi);
}

inline FORCE_INLINE __m256i interleave_lanes_epi16(__m256i x)
{
	__m256i MASK = _mm256_set_epi8(15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0,
	                               15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0);

	x = _mm256_permute4x64_epi64(x, _MM_SHUFFLE(3, 1, 2, 0));
	x = _mm256_shuffle_epi8(x, MASK);

	return x;
}

inline FORCE_INLINE __m128i pack1_i30_to_epi16(__m256i x)
{
	__m256i offset = _mm256_set1_epi32(1 << 13);

	x = _mm256_add_epi32(x, offset);
	x = _mm256_srai_epi32(x, 14);
	x = _mm256_packs_epi32(x, x);

	x = _mm256_permute4x64_epi64(x, _MM_SHUFFLE(3, 1, 2, 0));

	return _mm256_castsi256_si128(x);
}

template <unsigned N>
void filter_line_u16_h(const FilterContext &filter, const LineBuffer<uint16_t> &src, LineBuffer<uint16_t> &dst, unsigned n, void *tmp)
{
	__m256i INT16_MIN_EPI16 = _mm256_set1_epi16(INT16_MIN);

	const int16_t *filter_data = filter.data_i16.data();
	const unsigned *filter_left = filter.left.data();
	unsigned filter_stride = filter.stride_i16;

	unsigned src_width = src.width();
	unsigned dst_width = dst.width();

	uint16_t *dst_ptr0 = dst[n + 0];
	uint16_t *dst_ptr1 = dst[n + 1];
	uint16_t *dst_ptr2 = dst[n + 2];
	uint16_t *dst_ptr3 = dst[n + 3];
	uint16_t *dst_ptr4 = dst[n + 4];
	uint16_t *dst_ptr5 = dst[n + 5];
	uint16_t *dst_ptr6 = dst[n + 6];
	uint16_t *dst_ptr7 = dst[n + 7];

	uint16_t *ttmp = (uint16_t *)tmp;
	ALIGNED uint16_t cache[8 * 16];

	transpose_line_16x8_epi16(src[n + 0], src[n + 1], src[n + 2], src[n + 3], src[n + 4], src[n + 5], src[n + 6], src[n + 7], ttmp, src_width);

	for (unsigned j = 0; j < mod(dst_width, 16); ++j) {
		const int16_t *filter_row = &filter_data[j * filter_stride];
		unsigned left = filter_left[j];
		__m256i accum = _mm256_setzero_si256();
		__m256i x, c;
		__m128i result;

#define ITER(xiter) \
  do { \
    x = _mm256_loadu_si256((const __m256i *)&ttmp[(left + (xiter)) * 8]); \
    c = _mm256_broadcastd_epi32(_mm_loadu_si128((const __m128i *)&filter_row[xiter])); \
    x = _mm256_add_epi16(x, INT16_MIN_EPI16); \
    x = interleave_lanes_epi16(x); \
    x = _mm256_madd_epi16(c, x); \
    accum = _mm256_add_epi32(accum, x); \
  } while (0);

		if (N) {
			switch (N) {
			case 8: case 7: ITER(6);
			case 6: case 5: ITER(4);
			case 4: case 3: ITER(2);
			case 2: case 1: ITER(0);
			}
		} else {
			for (unsigned k = 0; k < filter.filter_width; k += 2) {
				ITER(k);
			}
		}
#undef ITER

		result = pack1_i30_to_epi16(accum);
		result = _mm_sub_epi16(result, _mm256_castsi256_si128(INT16_MIN_EPI16));

		_mm_store_si128((__m128i *)&cache[(j % 16) * 8], result);

		if (j % 16 == 15) {
			unsigned dst_j = j - 15;
			transpose_block_8x16_epi16(cache, &dst_ptr0[dst_j], &dst_ptr1[dst_j], &dst_ptr2[dst_j], &dst_ptr3[dst_j], &dst_ptr4[dst_j], &dst_ptr5[dst_j], &dst_ptr6[dst_j], &dst_ptr7[dst_j]);
		}
	}
	for (unsigned j = mod(dst_width, 16); j < dst_width; ++j) {
		const int16_t *filter_row = &filter_data[j * filter_stride];
		unsigned left = filter_left[j];
		__m256i accum = _mm256_setzero_si256();
		__m128i result;

		for (unsigned k = 0; k < filter.filter_width; k += 2) {
			__m256i x = _mm256_loadu_si256((const __m256i *)&ttmp[(left + k) * 8]);
			__m256i c = _mm256_broadcastd_epi32(_mm_set1_epi32(*(const int32_t *)&filter_row[k]));

			x = _mm256_add_epi16(x, INT16_MIN_EPI16);
			x = interleave_lanes_epi16(x);
			x = _mm256_madd_epi16(c, x);

			accum = _mm256_add_epi32(accum, x);
		}

		result = pack1_i30_to_epi16(accum);
		result = _mm_sub_epi16(result, _mm256_castsi256_si128(INT16_MIN_EPI16));

		scatter_epi16(result, &dst_ptr0[j], &dst_ptr1[j], &dst_ptr2[j], &dst_ptr3[j], &dst_ptr4[j], &dst_ptr5[j], &dst_ptr6[j], &dst_ptr7[j]);
	}
}

template <unsigned N>
void filter_line_f16_h(const FilterContext &filter, const LineBuffer<uint16_t> &src, LineBuffer<uint16_t> &dst, unsigned n, void *tmp)
{
	const float *filter_data = filter.data.data();
	const unsigned *filter_left = filter.left.data();
	unsigned filter_stride = filter.stride;

	unsigned src_width = src.width();
	unsigned dst_width = dst.width();

	uint16_t *dst_ptr0 = dst[n + 0];
	uint16_t *dst_ptr1 = dst[n + 1];
	uint16_t *dst_ptr2 = dst[n + 2];
	uint16_t *dst_ptr3 = dst[n + 3];
	uint16_t *dst_ptr4 = dst[n + 4];
	uint16_t *dst_ptr5 = dst[n + 5];
	uint16_t *dst_ptr6 = dst[n + 6];
	uint16_t *dst_ptr7 = dst[n + 7];

	uint16_t *ttmp = (uint16_t *)tmp;
	ALIGNED uint16_t cache[8 * 16];

	transpose_line_16x8_epi16(src[n + 0], src[n + 1], src[n + 2], src[n + 3], src[n + 4], src[n + 5], src[n + 6], src[n + 7], ttmp, src_width);

	for (unsigned j = 0; j < mod(dst_width, 16); ++j) {
		const float *filter_row = &filter_data[j * filter_stride];
		unsigned left = filter_left[j];
		__m256 accum0 = _mm256_setzero_ps();
		__m256 accum1 = _mm256_setzero_ps();
		__m256 x, c;

#define ITER(xiter) \
  do { \
    x = _mm256_cvtph_ps(_mm_load_si128((const __m128i *)&ttmp[(left + (xiter)) * 8])); \
    c = _mm256_broadcast_ss(&filter_row[xiter]); \
    if ((xiter) % 2) accum1 = _mm256_fmadd_ps(c, x, accum1); else accum0 = _mm256_fmadd_ps(c, x, accum0); \
  } while (0);

		if (N) {
			switch (N) {
			case 8: ITER(7);
			case 7: ITER(6);
			case 6: ITER(5);
			case 5: ITER(4);
			case 4: ITER(3);
			case 3: ITER(2);
			case 2: ITER(1);
			case 1: ITER(0);
			}
		} else {
			for (unsigned k = 0; k < filter.filter_width; k += 4) {
				ITER(k + 3);
				ITER(k + 2);
				ITER(k + 1);
				ITER(k + 0);
			}
		}
#undef ITER

		if (N != 1)
			accum0 = _mm256_add_ps(accum0, accum1);

		_mm_store_si128((__m128i *)&cache[(j % 16) * 8], _mm256_cvtps_ph(accum0, 0));

		if (j % 16 == 15) {
			unsigned dst_j = j - 15;
			transpose_block_8x16_epi16(cache, &dst_ptr0[dst_j], &dst_ptr1[dst_j], &dst_ptr2[dst_j], &dst_ptr3[dst_j], &dst_ptr4[dst_j], &dst_ptr5[dst_j], &dst_ptr6[dst_j], &dst_ptr7[dst_j]);
		}
	}
	for (unsigned j = mod(dst_width, 16); j < dst_width; ++j) {
		const float *filter_row = &filter_data[j * filter_stride];
		unsigned left = filter_left[j];
		__m256 accum = _mm256_setzero_ps();

		for (unsigned k = 0; k < filter.filter_width; ++k) {
			__m256 x = _mm256_cvtph_ps(_mm_load_si128((const __m128i *)&ttmp[(left + k) * 8]));
			__m256 c = _mm256_broadcast_ss(&filter_row[k]);

			accum = _mm256_fmadd_ps(c, x, accum);
		}

		scatter_epi16(_mm256_cvtps_ph(accum, 0), &dst_ptr0[j], &dst_ptr1[j], &dst_ptr2[j], &dst_ptr3[j], &dst_ptr4[j], &dst_ptr5[j], &dst_ptr6[j], &dst_ptr7[j]);
	}
}

template <unsigned N>
void filter_line_f32_h(const FilterContext &filter, const LineBuffer<float> &src, LineBuffer<float> &dst, unsigned n, void *tmp)
{
	const float *filter_data = filter.data.data();
	const unsigned *filter_left = filter.left.data();
	unsigned filter_stride = filter.stride;

	unsigned src_width = src.width();
	unsigned dst_width = dst.width();

	float *dst_ptr0 = dst[n + 0];
	float *dst_ptr1 = dst[n + 1];
	float *dst_ptr2 = dst[n + 2];
	float *dst_ptr3 = dst[n + 3];
	float *dst_ptr4 = dst[n + 4];
	float *dst_ptr5 = dst[n + 5];
	float *dst_ptr6 = dst[n + 6];
	float *dst_ptr7 = dst[n + 7];

	float *ttmp = (float *)tmp;
	ALIGNED float cache[8 * 8];

	transpose_line_8x8_ps(src[n + 0], src[n + 1], src[n + 2], src[n + 3], src[n + 4], src[n + 5], src[n + 6], src[n + 7], ttmp, src_width);

	for (unsigned j = 0; j < mod(dst_width, 8); ++j) {
		const float *filter_row = &filter_data[j * filter_stride];
		unsigned left = filter_left[j];
		__m256 accum0 = _mm256_setzero_ps();
		__m256 accum1 = _mm256_setzero_ps();
		__m256 x, c;

#define ITER(xiter) \
  do { \
    x = _mm256_load_ps(&ttmp[(left + (xiter)) * 8]); \
    c = _mm256_broadcast_ss(&filter_row[xiter]); \
    if ((xiter) % 2) accum1 = _mm256_fmadd_ps(c, x, accum1); else accum0 = _mm256_fmadd_ps(c, x, accum0); \
  } while (0);

		if (N) {
			switch (N) {
			case 8: ITER(7);
			case 7: ITER(6);
			case 6: ITER(5);
			case 5: ITER(4);
			case 4: ITER(3);
			case 3: ITER(2);
			case 2: ITER(1);
			case 1: ITER(0);
			}
		} else {
			for (unsigned k = 0; k < filter.filter_width; k += 4) {
				ITER(k + 3);
				ITER(k + 2);
				ITER(k + 1);
				ITER(k + 0);
			}
		}
#undef ITER

		if (N != 1)
			accum0 = _mm256_add_ps(accum0, accum1);

		_mm256_store_ps(&cache[(j % 8) * 8], accum0);

		if (j % 8 == 7) {
			unsigned dst_j = j - 7;
			transpose_block_8x8_ps(cache, &dst_ptr0[dst_j], &dst_ptr1[dst_j], &dst_ptr2[dst_j], &dst_ptr3[dst_j], &dst_ptr4[dst_j], &dst_ptr5[dst_j], &dst_ptr6[dst_j], &dst_ptr7[dst_j]);
		}
	}
	for (unsigned j = mod(dst_width, 8); j < dst_width; ++j) {
		const float *filter_row = &filter_data[j * filter_stride];
		unsigned left = filter_left[j];
		__m256 accum = _mm256_setzero_ps();

		for (unsigned k = 0; k < filter.filter_width; ++k) {
			__m256 x = _mm256_load_ps(&ttmp[(left + k) * 8]);
			__m256 c = _mm256_broadcast_ss(&filter_row[k]);

			accum = _mm256_fmadd_ps(c, x, accum);
		}

		scatter_ps(accum, &dst_ptr0[j], &dst_ptr1[j], &dst_ptr2[j], &dst_ptr3[j], &dst_ptr4[j], &dst_ptr5[j], &dst_ptr6[j], &dst_ptr7[j]);
	}
}

void filter_line_u16_v(const FilterContext &filter, const LineBuffer<uint16_t> &src, LineBuffer<uint16_t> &dst, unsigned n, void *tmp)
{
	__m256i INT16_MIN_EPI16 = _mm256_set1_epi16(INT16_MIN);

	const int16_t *filter_data = filter.data_i16.data();
	const unsigned *filter_left = filter.left.data();
	unsigned filter_stride = filter.stride_i16;

	unsigned width = dst.width();

	const int16_t *filter_row = &filter_data[n * filter_stride];
	unsigned top = filter_left[n];
	uint16_t *dst_ptr = dst[n];
	uint32_t *accum_tmp = (uint32_t *)tmp;

	for (unsigned k = 0; k < mod(filter.filter_width, 8); k += 8) {
		const uint16_t *src_ptr0 = src[top + k + 0];
		const uint16_t *src_ptr1 = src[top + k + 1];
		const uint16_t *src_ptr2 = src[top + k + 2];
		const uint16_t *src_ptr3 = src[top + k + 3];
		const uint16_t *src_ptr4 = src[top + k + 4];
		const uint16_t *src_ptr5 = src[top + k + 5];
		const uint16_t *src_ptr6 = src[top + k + 6];
		const uint16_t *src_ptr7 = src[top + k + 7];

		__m256i coeff0 = _mm256_set1_epi16(filter_row[k + 0]);
		__m256i coeff1 = _mm256_set1_epi16(filter_row[k + 1]);
		__m256i coeff2 = _mm256_set1_epi16(filter_row[k + 2]);
		__m256i coeff3 = _mm256_set1_epi16(filter_row[k + 3]);
		__m256i coeff4 = _mm256_set1_epi16(filter_row[k + 4]);
		__m256i coeff5 = _mm256_set1_epi16(filter_row[k + 5]);
		__m256i coeff6 = _mm256_set1_epi16(filter_row[k + 6]);
		__m256i coeff7 = _mm256_set1_epi16(filter_row[k + 7]);

		__m256i coeff01 = _mm256_unpacklo_epi16(coeff0, coeff1);
		__m256i coeff23 = _mm256_unpacklo_epi16(coeff2, coeff3);
		__m256i coeff45 = _mm256_unpacklo_epi16(coeff4, coeff5);
		__m256i coeff67 = _mm256_unpacklo_epi16(coeff6, coeff7);

		for (unsigned j = 0; j < mod(width, 16); j += 16) {
			__m256i x0, x1, x2, x3, x4, x5, x6, x7;
			__m256i packed;

			__m256i accum0l = _mm256_setzero_si256();
			__m256i accum0h = _mm256_setzero_si256();
			__m256i accum1l = _mm256_setzero_si256();
			__m256i accum1h = _mm256_setzero_si256();

			x0 = _mm256_load_si256((const __m256i *)&src_ptr0[j]);
			x0 = _mm256_add_epi16(x0, INT16_MIN_EPI16);
			x1 = _mm256_load_si256((const __m256i *)&src_ptr1[j]);
			x1 = _mm256_add_epi16(x1, INT16_MIN_EPI16);
			fmadd_epi16_2(coeff01, x0, x1, accum0l, accum0h);

			x2 = _mm256_load_si256((const __m256i *)&src_ptr2[j]);
			x2 = _mm256_add_epi16(x2, INT16_MIN_EPI16);
			x3 = _mm256_load_si256((const __m256i *)&src_ptr3[j]);
			x3 = _mm256_add_epi16(x3, INT16_MIN_EPI16);
			fmadd_epi16_2(coeff23, x2, x3, accum1l, accum1h);

			x4 = _mm256_load_si256((const __m256i *)&src_ptr4[j]);
			x4 = _mm256_add_epi16(x4, INT16_MIN_EPI16);
			x5 = _mm256_load_si256((const __m256i *)&src_ptr5[j]);
			x5 = _mm256_add_epi16(x5, INT16_MIN_EPI16);
			fmadd_epi16_2(coeff45, x4, x5, accum0l, accum0h);

			x6 = _mm256_load_si256((const __m256i *)&src_ptr6[j]);
			x6 = _mm256_add_epi16(x6, INT16_MIN_EPI16);
			x7 = _mm256_load_si256((const __m256i *)&src_ptr7[j]);
			x7 = _mm256_add_epi16(x7, INT16_MIN_EPI16);
			fmadd_epi16_2(coeff67, x6, x7, accum1l, accum1h);

			accum0l = _mm256_add_epi32(accum0l, accum1l);
			accum0h = _mm256_add_epi32(accum0h, accum1h);

			if (k) {
				accum0l = _mm256_add_epi32(accum0l, _mm256_load_si256((const __m256i *)&accum_tmp[j + 0]));
				accum0h = _mm256_add_epi32(accum0h, _mm256_load_si256((const __m256i *)&accum_tmp[j + 8]));
			}

			if (k == filter.filter_width - 8) {					
				packed = pack_i30_to_epi16(accum0l, accum0h);
				packed = _mm256_sub_epi16(packed, INT16_MIN_EPI16);
				_mm256_store_si256((__m256i *)&dst_ptr[j], packed);
			} else {
				_mm256_store_si256((__m256i *)&accum_tmp[j + 0], accum0l);
				_mm256_store_si256((__m256i *)&accum_tmp[j + 8], accum0h);
			}
		}
	}
	if (filter.filter_width % 8) {
		unsigned m = filter.filter_width % 8;
		unsigned k = filter.filter_width - m;

		const uint16_t *src_ptr0 = src[top + k + 0];
		const uint16_t *src_ptr1 = src[top + k + 1];
		const uint16_t *src_ptr2 = src[top + k + 2];
		const uint16_t *src_ptr3 = src[top + k + 3];
		const uint16_t *src_ptr4 = src[top + k + 4];
		const uint16_t *src_ptr5 = src[top + k + 5];
		const uint16_t *src_ptr6 = src[top + k + 6];

		__m256i coeff0 = _mm256_set1_epi16(filter_row[k + 0]);
		__m256i coeff1 = _mm256_set1_epi16(filter_row[k + 1]);
		__m256i coeff2 = _mm256_set1_epi16(filter_row[k + 2]);
		__m256i coeff3 = _mm256_set1_epi16(filter_row[k + 3]);
		__m256i coeff4 = _mm256_set1_epi16(filter_row[k + 4]);
		__m256i coeff5 = _mm256_set1_epi16(filter_row[k + 5]);
		__m256i coeff6 = _mm256_set1_epi16(filter_row[k + 6]);
		__m256i coeff7 = _mm256_setzero_si256();

		__m256i coeff01 = _mm256_unpacklo_epi16(coeff0, coeff1);
		__m256i coeff23 = _mm256_unpacklo_epi16(coeff2, coeff3);
		__m256i coeff45 = _mm256_unpacklo_epi16(coeff4, coeff5);
		__m256i coeff67 = _mm256_unpacklo_epi16(coeff6, coeff7);

		for (unsigned j = 0; j < mod(width, 16); j += 16) {
			__m256i x0, x1, x2, x3, x4, x5, x6, x7;
			__m256i packed;

			__m256i accum0l = _mm256_setzero_si256();
			__m256i accum0h = _mm256_setzero_si256();
			__m256i accum1l = _mm256_setzero_si256();
			__m256i accum1h = _mm256_setzero_si256();

			switch (m) {
			case 7:
				x7 = INT16_MIN_EPI16;
				x6 = _mm256_load_si256((const __m256i *)&src_ptr6[j]);
				x6 = _mm256_add_epi16(x6, INT16_MIN_EPI16);
				fmadd_epi16_2(coeff67, x6, x7, accum1l, accum1h);
			case 6:
				x5 = _mm256_load_si256((const __m256i *)&src_ptr5[j]);
				x5 = _mm256_add_epi16(x5, INT16_MIN_EPI16);
			case 5:
				x4 = _mm256_load_si256((const __m256i *)&src_ptr4[j]);
				x4 = _mm256_add_epi16(x4, INT16_MIN_EPI16);
				fmadd_epi16_2(coeff45, x4, x5, accum0l, accum0h);
			case 4:
				x3 = _mm256_load_si256((const __m256i *)&src_ptr3[j]);
				x3 = _mm256_add_epi16(x3, INT16_MIN_EPI16);
			case 3:
				x2 = _mm256_load_si256((const __m256i *)&src_ptr2[j]);
				x2 = _mm256_add_epi16(x2, INT16_MIN_EPI16);
				fmadd_epi16_2(coeff23, x2, x3, accum1l, accum1h);
			case 2:
				x1 = _mm256_load_si256((const __m256i *)&src_ptr1[j]);
				x1 = _mm256_add_epi16(x1, INT16_MIN_EPI16);
			case 1:
				x0 = _mm256_load_si256((const __m256i *)&src_ptr0[j]);
				x0 = _mm256_add_epi16(x0, INT16_MIN_EPI16);
				fmadd_epi16_2(coeff01, x0, x1, accum0l, accum0h);
			}

			accum0l = _mm256_add_epi32(accum0l, accum1l);
			accum0h = _mm256_add_epi32(accum0h, accum1h);

			if (k) {
				accum0l = _mm256_add_epi32(accum0l, _mm256_load_si256((const __m256i *)&accum_tmp[j + 0]));
				accum0h = _mm256_add_epi32(accum0h, _mm256_load_si256((const __m256i *)&accum_tmp[j + 8]));
			}

			packed = pack_i30_to_epi16(accum0l, accum0h);
			packed = _mm256_sub_epi16(packed, INT16_MIN_EPI16);

			_mm256_store_si256((__m256i *)&dst_ptr[j], packed);
		}
	}
	filter_line_v_scalar(filter, src, dst, n, n + 1, mod(width, 16), width, ScalarPolicy_U16{});
}

template <class T, class Policy>
void filter_line_fp_v(const FilterContext &filter, const LineBuffer<T> &src, LineBuffer<T> &dst, unsigned n, Policy policy)
{
	const float *filter_data = filter.data.data();
	const unsigned *filter_left = filter.left.data();
	unsigned filter_stride = filter.stride;

	unsigned width = dst.width();

	const float *filter_row = &filter_data[n * filter_stride];
	unsigned top = filter_left[n];
	T *dst_ptr = dst[n];

	for (unsigned k = 0; k < mod(filter.filter_width, 8); k += 8) {
		const T *src_ptr0 = src[top + k + 0];
		const T *src_ptr1 = src[top + k + 1];
		const T *src_ptr2 = src[top + k + 2];
		const T *src_ptr3 = src[top + k + 3];
		const T *src_ptr4 = src[top + k + 4];
		const T *src_ptr5 = src[top + k + 5];
		const T *src_ptr6 = src[top + k + 6];
		const T *src_ptr7 = src[top + k + 7];

		__m256 coeff0 = _mm256_broadcast_ss(&filter_row[k + 0]);
		__m256 coeff1 = _mm256_broadcast_ss(&filter_row[k + 1]);
		__m256 coeff2 = _mm256_broadcast_ss(&filter_row[k + 2]);
		__m256 coeff3 = _mm256_broadcast_ss(&filter_row[k + 3]);
		__m256 coeff4 = _mm256_broadcast_ss(&filter_row[k + 4]);
		__m256 coeff5 = _mm256_broadcast_ss(&filter_row[k + 5]);
		__m256 coeff6 = _mm256_broadcast_ss(&filter_row[k + 6]);
		__m256 coeff7 = _mm256_broadcast_ss(&filter_row[k + 7]);

		for (unsigned j = 0; j < mod(width, 8); j += 8) {
			__m256 x0, x1, x2, x3, x4, x5, x6, x7;
			__m256 accum0, accum1;

			x0 = policy.load_8(&src_ptr0[j]);
			accum0 = _mm256_mul_ps(coeff0, x0);

			x1 = policy.load_8(&src_ptr1[j]);
			accum1 = _mm256_mul_ps(coeff1, x1);

			x2 = policy.load_8(&src_ptr2[j]);
			accum0 = _mm256_fmadd_ps(coeff2, x2, accum0);

			x3 = policy.load_8(&src_ptr3[j]);
			accum1 = _mm256_fmadd_ps(coeff3, x3, accum1);

			x4 = policy.load_8(&src_ptr4[j]);
			accum0 = _mm256_fmadd_ps(coeff4, x4, accum0);

			x5 = policy.load_8(&src_ptr5[j]);
			accum1 = _mm256_fmadd_ps(coeff5, x5, accum1);

			x6 = policy.load_8(&src_ptr6[j]);
			accum0 = _mm256_fmadd_ps(coeff6, x6, accum0);

			x7 = policy.load_8(&src_ptr7[j]);
			accum1 = _mm256_fmadd_ps(coeff7, x7, accum1);

			accum0 = _mm256_add_ps(accum0, accum1);
			if (k)
				accum0 = _mm256_add_ps(accum0, policy.load_8(&dst_ptr[j]));

			policy.store_8(&dst_ptr[j], accum0);
		}
	}
	if (filter.filter_width % 8) {
		unsigned m = filter.filter_width % 8;
		unsigned k = filter.filter_width - m;

		const T *src_ptr0 = src[top + k + 0];
		const T *src_ptr1 = src[top + k + 1];
		const T *src_ptr2 = src[top + k + 2];
		const T *src_ptr3 = src[top + k + 3];
		const T *src_ptr4 = src[top + k + 4];
		const T *src_ptr5 = src[top + k + 5];
		const T *src_ptr6 = src[top + k + 6];

		__m256 coeff0 = _mm256_broadcast_ss(&filter_row[k + 0]);
		__m256 coeff1 = _mm256_broadcast_ss(&filter_row[k + 1]);
		__m256 coeff2 = _mm256_broadcast_ss(&filter_row[k + 2]);
		__m256 coeff3 = _mm256_broadcast_ss(&filter_row[k + 3]);
		__m256 coeff4 = _mm256_broadcast_ss(&filter_row[k + 4]);
		__m256 coeff5 = _mm256_broadcast_ss(&filter_row[k + 5]);
		__m256 coeff6 = _mm256_broadcast_ss(&filter_row[k + 6]);

		for (unsigned j = 0; j < mod(width, 8); j += 8) {
			__m256 x0, x1, x2, x3, x4, x5, x6;

			__m256 accum0 = _mm256_setzero_ps();
			__m256 accum1 = _mm256_setzero_ps();

			switch (m) {
			case 7:
				x6 = policy.load_8(&src_ptr6[j]);
				accum0 = _mm256_mul_ps(coeff6, x6);
			case 6:
				x5 = policy.load_8(&src_ptr5[j]);
				accum1 = _mm256_mul_ps(coeff5, x5);
			case 5:
				x4 = policy.load_8(&src_ptr4[j]);
				accum0 = _mm256_fmadd_ps(coeff4, x4, accum0);
			case 4:
				x3 = policy.load_8(&src_ptr3[j]);
				accum1 = _mm256_fmadd_ps(coeff3, x3, accum1);
			case 3:
				x2 = policy.load_8(&src_ptr2[j]);
				accum0 = _mm256_fmadd_ps(coeff2, x2, accum0);
			case 2:
				x1 = policy.load_8(&src_ptr1[j]);
				accum1 = _mm256_fmadd_ps(coeff1, x1, accum1);
			case 1:
				x0 = policy.load_8(&src_ptr0[j]);
				accum0 = _mm256_fmadd_ps(coeff0, x0, accum0);
			}

			accum0 = _mm256_add_ps(accum0, accum1);
			if (k)
				accum0 = _mm256_add_ps(accum0, policy.load_8(&dst_ptr[j]));

			policy.store_8(&dst_ptr[j], accum0);
		}
	}
	filter_line_v_scalar(filter, src, dst, n, n + 1, mod(width, 8), width, policy);
}

class ResizeImplAVX2_H final : public ResizeImpl {
public:
	ResizeImplAVX2_H(const FilterContext &filter) : ResizeImpl(filter, true)
	{}

	size_t tmp_size(PixelType type, unsigned width) const override
	{
		return 8 * align((size_t)(m_filter.input_width + 32) * pixel_size(type), ALIGNMENT);
	}

	unsigned output_buffering(PixelType type) const override
	{
		return 8;
	}

	void process_u16(const LineBuffer<uint16_t> &src, LineBuffer<uint16_t> &dst, unsigned n, void *tmp) const override
	{
#define CASE(x) case (x): filter_line_u16_h<x>(m_filter, src, dst, n, tmp); break;
		switch (m_filter.filter_width) {
			CASE(8) CASE(7) CASE(6) CASE(5) CASE(4) CASE(3) CASE(2) CASE(1)
		default:
			filter_line_u16_h<0>(m_filter, src, dst, n, tmp);
		}
#undef CASE
	}

	void process_f16(const LineBuffer<uint16_t> &src, LineBuffer<uint16_t> &dst, unsigned n, void *tmp) const override
	{
#define CASE(x) case (x): filter_line_f16_h<x>(m_filter, src, dst, n, tmp); break;
		switch (m_filter.filter_width) {
		CASE(8) CASE(7) CASE(6) CASE(5) CASE(4) CASE(3) CASE(2) CASE(1)
		default:
			filter_line_f16_h<0>(m_filter, src, dst, n, tmp);
		}
#undef CASE
	}

	void process_f32(const LineBuffer<float> &src, LineBuffer<float> &dst, unsigned n, void *tmp) const override
	{
#define CASE(x) case (x): filter_line_f32_h<x>(m_filter, src, dst, n, tmp); break;
		switch (m_filter.filter_width) {
			CASE(8) CASE(7) CASE(6) CASE(5) CASE(4) CASE(3) CASE(2) CASE(1)
		default:
			filter_line_f32_h<0>(m_filter, src, dst, n, tmp);
		}
#undef CASE
	}
};

class ResizeImplAVX2_V final : public ResizeImpl {
public:
	ResizeImplAVX2_V(const FilterContext &filter) : ResizeImpl(filter, false)
	{}

	size_t tmp_size(PixelType type, unsigned width) const override
	{
		return type == PixelType::WORD ? sizeof(uint32_t) * width : 0;
	}

	void process_u16(const LineBuffer<uint16_t> &src, LineBuffer<uint16_t> &dst, unsigned n, void *tmp) const override
	{
		filter_line_u16_v(m_filter, src, dst, n, tmp);
	}

	void process_f16(const LineBuffer<uint16_t> &src, LineBuffer<uint16_t> &dst, unsigned n, void *tmp) const override
	{
		filter_line_fp_v(m_filter, src, dst, n, VectorPolicy_F16{});
	}

	void process_f32(const LineBuffer<float> &src, LineBuffer<float> &dst, unsigned n, void *tmp) const override
	{
		filter_line_fp_v(m_filter, src, dst, n, VectorPolicy_F32{});
	}
};

} // namespace


ResizeImpl *create_resize_impl_avx2(const FilterContext &filter, bool horizontal)
{
	if (horizontal)
		return new ResizeImplAVX2_H{ filter };
	else
		return new ResizeImplAVX2_V{ filter };
}

} // namespace resize;
} // namespace zimg

#endif // ZIMG_X86
