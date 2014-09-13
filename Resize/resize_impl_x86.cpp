#ifdef RESIZE_X86

#include <algorithm>
#include <intrin.h>
#include "osdep.h"
#include "resize_impl.h"

namespace resize {;

namespace {;

struct LoadStoreManagerF32 {
	FORCE_INLINE float load_1(const float *p) { return *p; }
	FORCE_INLINE void store_1(float *p, float x) { *p = x; }

	FORCE_INLINE __m256 load_8(const float *p) { return _mm256_load_ps(p); }
	FORCE_INLINE __m256 loadu_8(const float *p) { return _mm256_loadu_ps(p); }

	FORCE_INLINE void store_8(float *p, __m256 x) { _mm256_store_ps(p, x); }
	FORCE_INLINE void storeu_8(float *p, __m256 x) { _mm256_storeu_ps(p, x); }
};

struct LoadStoreManagerF16 {
	FORCE_INLINE float load_1(const uint16_t *p)
	{
		__m128i sh = _mm_set1_epi16(*p);
		__m128 ss = _mm_cvtph_ps(sh);
		return _mm_cvtss_f32(ss);
	}

	FORCE_INLINE void store_1(uint16_t *p, float x)
	{
		__m128 ss = _mm_set_ps1(x);
		__m128i sh = _mm_cvtps_ph(ss, 0);
		*p = _mm_extract_epi16(sh, 0);
	}

	FORCE_INLINE __m256 load_8(const uint16_t *p) { return _mm256_cvtph_ps(_mm_load_si128((const __m128i *)p)); }
	FORCE_INLINE __m256 loadu_8(const uint16_t *p) { return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)p)); }

	FORCE_INLINE void store_8(uint16_t *p, __m256 x) { _mm_store_si128((__m128i *)p, _mm256_cvtps_ph(x, 0)); }
	FORCE_INLINE void storeu_8(uint16_t *p, __m256 x) { _mm_storeu_si128((__m128i *)p, _mm256_cvtps_ph(x, 0)); }
};

FORCE_INLINE void transpose8_ps_avx(__m256 &row0, __m256 &row1, __m256 &row2, __m256 &row3, __m256 &row4, __m256 &row5, __m256 &row6, __m256 &row7)
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

FORCE_INLINE void transpose8_epi32_avx(__m256i &row0, __m256i &row1, __m256i &row2, __m256i &row3, __m256i &row4, __m256i &row5, __m256i &row6, __m256i &row7)
{
	__m256 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;

	tmp0 = _mm256_castsi256_ps(row0);
	tmp1 = _mm256_castsi256_ps(row1);
	tmp2 = _mm256_castsi256_ps(row2);
	tmp3 = _mm256_castsi256_ps(row3);
	tmp4 = _mm256_castsi256_ps(row4);
	tmp5 = _mm256_castsi256_ps(row5);
	tmp6 = _mm256_castsi256_ps(row6);
	tmp7 = _mm256_castsi256_ps(row7);

	transpose8_ps_avx(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7);

	row0 = _mm256_castps_si256(tmp0);
	row1 = _mm256_castps_si256(tmp1);
	row2 = _mm256_castps_si256(tmp2);
	row3 = _mm256_castps_si256(tmp3);
	row4 = _mm256_castps_si256(tmp4);
	row5 = _mm256_castps_si256(tmp5);
	row6 = _mm256_castps_si256(tmp6);
	row7 = _mm256_castps_si256(tmp7);
}

FORCE_INLINE __m256i mhadd_epi16_epi32_avx(__m256i a, __m256i b)
{
	__m256i lo, hi, uplo, uphi;

	lo = _mm256_mullo_epi16(a, b);
	hi = _mm256_mulhi_epi16(a, b);

	uplo = _mm256_unpacklo_epi16(lo, hi);
	uphi = _mm256_unpackhi_epi16(lo, hi);

	return _mm256_add_epi32(uplo, uphi);
}

FORCE_INLINE void fmadd_epi16_epi32_avx(__m256i a, __m256i b, __m256i &accum0, __m256i &accum1)
{
	__m256i lo, hi, uplo, uphi;

	lo = _mm256_mullo_epi16(a, b);
	hi = _mm256_mulhi_epi16(a, b);

	uplo = _mm256_unpacklo_epi16(lo, hi);
	uphi = _mm256_unpackhi_epi16(lo, hi);

	accum0 = _mm256_add_epi32(accum0, uplo);
	accum1 = _mm256_add_epi32(accum1, uphi);
}

FORCE_INLINE __m256i pack_i30_epi32(__m256i lo, __m256i hi)
{
	__m256i offset = _mm256_set1_epi32(1 << 13);

	lo = _mm256_add_epi32(lo, offset);
	hi = _mm256_add_epi32(hi, offset);

	lo = _mm256_srai_epi32(lo, 14);
	hi = _mm256_srai_epi32(hi, 14);

	return  _mm256_packs_epi32(lo, hi);
}

template <bool DoLoop>
void filter_plane_u16_h_avx(const EvaluatedFilter &filter, const uint16_t * RESTRICT src, uint16_t * RESTRICT dst,
                            int src_width, int src_height, int src_stride, int dst_stride)
{
	__m256i INT16_MIN_EPI16 = _mm256_set1_epi16(INT16_MIN);

	for (int i = 0; i < mod(src_height, 8); i += 8) {
		int j;

		for (j = 0; j < mod(filter.height(), 16); ++j) {
			__m256i x0, x1, x2, x3, x4, x5, x6, x7;
			__m256i accum = _mm256_setzero_si256();
			__m256i cached[16];

			const int16_t *filter_row = filter.data_i16() + j * filter.stride_i16();
			int left = filter.left()[j];

			if (left + filter.stride_i16() > src_width)
				break;

			for (int k = 0; k < (DoLoop ? filter.width() : 16); k += 16) {
				__m256i coeff = _mm256_load_si256((const __m256i *)(filter_row + k));

				x0 = _mm256_loadu_si256((const __m256i *)(src + (i + 0) * src_stride + left + k));
				x0 = _mm256_add_epi32(x0, INT16_MIN_EPI16);
				x0 = mhadd_epi16_epi32_avx(coeff, x0);

				x1 = _mm256_loadu_si256((const __m256i *)(src + (i + 1) * src_stride + left + k));
				x1 = _mm256_add_epi32(x1, INT16_MIN_EPI16);
				x1 = mhadd_epi16_epi32_avx(coeff, x1);

				x2 = _mm256_loadu_si256((const __m256i *)(src + (i + 2) * src_stride + left + k));
				x2 = _mm256_add_epi32(x2, INT16_MIN_EPI16);
				x2 = mhadd_epi16_epi32_avx(coeff, x2);

				x3 = _mm256_loadu_si256((const __m256i *)(src + (i + 3) * src_stride + left + k));
				x3 = _mm256_add_epi32(x3, INT16_MIN_EPI16);
				x3 = mhadd_epi16_epi32_avx(coeff, x3);

				x4 = _mm256_loadu_si256((const __m256i *)(src + (i + 4) * src_stride + left + k));
				x4 = _mm256_add_epi32(x4, INT16_MIN_EPI16);
				x4 = mhadd_epi16_epi32_avx(coeff, x4);

				x5 = _mm256_loadu_si256((const __m256i *)(src + (i + 5) * src_stride + left + k));
				x5 = _mm256_add_epi32(x5, INT16_MIN_EPI16);
				x5 = mhadd_epi16_epi32_avx(coeff, x5);

				x6 = _mm256_loadu_si256((const __m256i *)(src + (i + 6) * src_stride + left + k));
				x6 = _mm256_add_epi32(x6, INT16_MIN_EPI16);
				x6 = mhadd_epi16_epi32_avx(coeff, x6);

				x7 = _mm256_loadu_si256((const __m256i *)(src + (i + 7) * src_stride + left + k));
				x7 = _mm256_add_epi32(x7, INT16_MIN_EPI16);
				x7 = mhadd_epi16_epi32_avx(coeff, x7);

				transpose8_epi32_avx(x0, x1, x2, x3, x4, x5, x6, x7);

				x0 = _mm256_add_epi32(x0, x4);
				x1 = _mm256_add_epi32(x1, x5);
				x2 = _mm256_add_epi32(x2, x6);
				x3 = _mm256_add_epi32(x3, x7);

				x0 = _mm256_add_epi32(x0, x2);
				x1 = _mm256_add_epi32(x1, x3);

				accum = _mm256_add_epi32(accum, x0);
				accum = _mm256_add_epi32(accum, x1);
			}
			cached[j % 16] = accum;

			if (j % 16 == 15) {
				__m256i packed;
				int dst_j = mod(j, 16);

				transpose8_epi32_avx(cached[0], cached[1], cached[2], cached[3], cached[8], cached[9], cached[10], cached[11]);
				transpose8_epi32_avx(cached[4], cached[5], cached[6], cached[7], cached[12], cached[13], cached[14], cached[15]);

				packed = pack_i30_epi32(cached[0], cached[4]);
				packed = _mm256_sub_epi16(packed, INT16_MIN_EPI16);
				_mm256_store_si256((__m256i *)(dst + (i + 0) * dst_stride + dst_j), packed);

				packed = pack_i30_epi32(cached[1], cached[5]);
				packed = _mm256_sub_epi16(packed, INT16_MIN_EPI16);
				_mm256_store_si256((__m256i *)(dst + (i + 1) * dst_stride + dst_j), packed);

				packed = pack_i30_epi32(cached[2], cached[6]);
				packed = _mm256_sub_epi16(packed, INT16_MIN_EPI16);
				_mm256_store_si256((__m256i *)(dst + (i + 2) * dst_stride + dst_j), packed);

				packed = pack_i30_epi32(cached[3], cached[7]);
				packed = _mm256_sub_epi16(packed, INT16_MIN_EPI16);
				_mm256_store_si256((__m256i *)(dst + (i + 3) * dst_stride + dst_j), packed);

				packed = pack_i30_epi32(cached[8], cached[12]);
				packed = _mm256_sub_epi16(packed, INT16_MIN_EPI16);
				_mm256_store_si256((__m256i *)(dst + (i + 4) * dst_stride + dst_j), packed);

				packed = pack_i30_epi32(cached[9], cached[13]);
				packed = _mm256_sub_epi16(packed, INT16_MIN_EPI16);
				_mm256_store_si256((__m256i *)(dst + (i + 5) * dst_stride + dst_j), packed);

				packed = pack_i30_epi32(cached[10], cached[14]);
				packed = _mm256_sub_epi16(packed, INT16_MIN_EPI16);
				_mm256_store_si256((__m256i *)(dst + (i + 6) * dst_stride + dst_j), packed);

				packed = pack_i30_epi32(cached[11], cached[15]);
				packed = _mm256_sub_epi16(packed, INT16_MIN_EPI16);
				_mm256_store_si256((__m256i *)(dst + (i + 7) * dst_stride + dst_j), packed);
			}
		}

		for (int ii = i; ii < i + 8; ++ii) {
			for (j = mod(j, 16); j < filter.height(); ++j) {
				int left = filter.left()[j];
				int32_t accum = 0;

				for (int k = 0; k < filter.width(); ++k) {
					int32_t coeff = filter.data_i16()[j * filter.stride_i16() + k];
					int32_t x = unpack_u16(src[ii * src_stride + left + k]);
					accum += coeff * x;
				}
				dst[ii * dst_stride + j] = pack_i30(accum);
			}
		}
	}
	for (int i = mod(src_height, 8); i < src_height; ++i) {
		for (int j = 0; j < filter.height(); ++j) {
			int left = filter.left()[j];
			int32_t accum = 0;

			for (int k = 0; k < filter.width(); ++k) {
				int32_t coeff = filter.data_i16()[j * filter.stride_i16() + k];
				int32_t x = unpack_u16(src[i * src_stride + left + k]);
				accum += coeff * x;
			}
			dst[i * dst_stride + j] = pack_i30(accum);
		}
	}
}

template <bool DoLoop, class T, class MemoryManager>
void filter_plane_fp_h_avx(const EvaluatedFilter &filter, const T * RESTRICT src, T * RESTRICT dst,
                            int src_width, int src_height, int src_stride, int dst_stride, MemoryManager mem)
{
	for (int i = 0; i < mod(src_height, 8); i += 8) {
		int j;

		for (j = 0; j < mod(filter.height(), 8); ++j) {
			__m256 x0, x1, x2, x3, x4, x5, x6, x7;
			__m256 accum = _mm256_setzero_ps();
			__m256 cached[8];

			const float *filter_row = filter.data() + j * filter.stride();
			int left = filter.left()[j];

			if (left + filter.stride() > src_width)
				break;

			for (int k = 0; k < (DoLoop ? filter.width() : 8); k += 8) {
				__m256 coeff = _mm256_load_ps(filter_row + k);

				x0 = mem.loadu_8(src + (i + 0) * src_stride + left + k);
				x0 = _mm256_mul_ps(coeff, x0);
				
				x1 = mem.loadu_8(src + (i + 1) * src_stride + left + k);
				x1 = _mm256_mul_ps(coeff, x1);

				x2 = mem.loadu_8(src + (i + 2) * src_stride + left + k);
				x2 = _mm256_mul_ps(coeff, x2);

				x3 = mem.loadu_8(src + (i + 3) * src_stride + left + k);
				x3 = _mm256_mul_ps(coeff, x3);

				x4 = mem.loadu_8(src + (i + 4) * src_stride + left + k);
				x4 = _mm256_mul_ps(coeff, x4);

				x5 = mem.loadu_8(src + (i + 5) * src_stride + left + k);
				x5 = _mm256_mul_ps(coeff, x5);

				x6 = mem.loadu_8(src + (i + 6) * src_stride + left + k);
				x6 = _mm256_mul_ps(coeff, x6);

				x7 = mem.loadu_8(src + (i + 7) * src_stride + left + k);
				x7 = _mm256_mul_ps(coeff, x7);

				transpose8_ps_avx(x0, x1, x2, x3, x4, x5, x6, x7);

				x0 = _mm256_add_ps(x0, x4);
				x1 = _mm256_add_ps(x1, x5);
				x2 = _mm256_add_ps(x2, x6);
				x3 = _mm256_add_ps(x3, x7);

				x0 = _mm256_add_ps(x0, x2);
				x1 = _mm256_add_ps(x1, x3);

				accum = _mm256_add_ps(accum, x0);
				accum = _mm256_add_ps(accum, x1);
			}
			cached[j % 8] = accum;

			if (j % 8 == 7) {
				int dst_j = mod(j, 8);

				transpose8_ps_avx(cached[0], cached[1], cached[2], cached[3], cached[4], cached[5], cached[6], cached[7]);

				mem.store_8(dst + (i + 0) * dst_stride + dst_j, cached[0]);
				mem.store_8(dst + (i + 1) * dst_stride + dst_j, cached[1]);
				mem.store_8(dst + (i + 2) * dst_stride + dst_j, cached[2]);
				mem.store_8(dst + (i + 3) * dst_stride + dst_j, cached[3]);
				mem.store_8(dst + (i + 4) * dst_stride + dst_j, cached[4]);
				mem.store_8(dst + (i + 5) * dst_stride + dst_j, cached[5]);
				mem.store_8(dst + (i + 6) * dst_stride + dst_j, cached[6]);
				mem.store_8(dst + (i + 7) * dst_stride + dst_j, cached[7]);
			}
		}

		for (int ii = i; ii < i + 8; ++ii) {
			for (int jj = mod(j, 8); jj < filter.height(); ++jj) {
				int left = filter.left()[jj];
				float accum = 0.f;

				for (int k = 0; k < filter.width(); ++k) {
					float coeff = filter.data()[jj * filter.stride() + k];
					float x = mem.load_1(src + ii * src_stride + left + k);
					accum += coeff * x;
				}
				mem.store_1(dst + ii * dst_stride + jj, accum);
			}
		}
	}
	for (int i = mod(src_height, 8); i < src_height; ++i) {
		for (int j = 0; j < filter.height(); ++j) {
			int left = filter.left()[j];
			float accum = 0.f;

			for (int k = 0; k < filter.width(); ++k) {
				float coeff = filter.data()[j * filter.stride() + k];
				float x = mem.load_1(src + i * src_stride + left + k);
				accum += coeff * x;
			}
			mem.store_1(dst + i * dst_stride + j, accum);
		}
	}
}

void filter_plane_u16_v_avx(const EvaluatedFilter &filter, const uint16_t * RESTRICT src, uint16_t * RESTRICT dst, uint16_t * RESTRICT tmp,
                            int src_width, int src_height, int src_stride, int dst_stride)
{
	__m256i INT16_MIN_EPI16 = _mm256_set1_epi16(INT16_MIN);

	for (int i = 0; i < filter.height(); ++i) {
		__m256i coeff0, coeff1, coeff2, coeff3, coeff4, coeff5, coeff6, coeff7;
		__m256i x0, x1, x2, x3, x4, x5, x6, x7;
		__m256i accum0l, accum0h, accum1l, accum1h;
		__m256i packed;

		const uint16_t *src_ptr0, *src_ptr1, *src_ptr2, *src_ptr3, *src_ptr4, *src_ptr5, *src_ptr6, *src_ptr7;
		uint16_t *dst_ptr = dst + i * dst_stride;

		for (int k = 0; k < mod(filter.width(), 8); k += 8) {
			src_ptr0 = src + (filter.left()[i] + k + 0) * src_stride;
			src_ptr1 = src + (filter.left()[i] + k + 1) * src_stride;
			src_ptr2 = src + (filter.left()[i] + k + 2) * src_stride;
			src_ptr3 = src + (filter.left()[i] + k + 3) * src_stride;
			src_ptr4 = src + (filter.left()[i] + k + 4) * src_stride;
			src_ptr5 = src + (filter.left()[i] + k + 5) * src_stride;
			src_ptr6 = src + (filter.left()[i] + k + 6) * src_stride;
			src_ptr7 = src + (filter.left()[i] + k + 7) * src_stride;

			coeff0 = _mm256_set1_epi16(filter.data_i16()[i * filter.stride_i16() + k + 0]);
			coeff1 = _mm256_set1_epi16(filter.data_i16()[i * filter.stride_i16() + k + 1]);
			coeff2 = _mm256_set1_epi16(filter.data_i16()[i * filter.stride_i16() + k + 2]);
			coeff3 = _mm256_set1_epi16(filter.data_i16()[i * filter.stride_i16() + k + 3]);
			coeff4 = _mm256_set1_epi16(filter.data_i16()[i * filter.stride_i16() + k + 4]);
			coeff5 = _mm256_set1_epi16(filter.data_i16()[i * filter.stride_i16() + k + 5]);
			coeff6 = _mm256_set1_epi16(filter.data_i16()[i * filter.stride_i16() + k + 6]);
			coeff7 = _mm256_set1_epi16(filter.data_i16()[i * filter.stride_i16() + k + 7]);

			for (int j = 0; j < mod(src_width, 16); j += 16) {
				accum0l = _mm256_setzero_si256();
				accum0h = _mm256_setzero_si256();
				accum1l = _mm256_setzero_si256();
				accum1h = _mm256_setzero_si256();

				x0 = _mm256_load_si256((const __m256i *)(src_ptr0 + j));
				x0 = _mm256_add_epi16(x0, INT16_MIN_EPI16);
				fmadd_epi16_epi32_avx(coeff0, x0, accum0l, accum0h);

				x1 = _mm256_load_si256((const __m256i *)(src_ptr1 + j));
				x1 = _mm256_add_epi16(x1, INT16_MIN_EPI16);
				fmadd_epi16_epi32_avx(coeff1, x1, accum1l, accum1h);

				x2 = _mm256_load_si256((const __m256i *)(src_ptr2 + j));
				x2 = _mm256_add_epi16(x2, INT16_MIN_EPI16);
				fmadd_epi16_epi32_avx(coeff2, x2, accum0l, accum0h);

				x3 = _mm256_load_si256((const __m256i *)(src_ptr3 + j));
				x3 = _mm256_add_epi16(x3, INT16_MIN_EPI16);
				fmadd_epi16_epi32_avx(coeff3, x3, accum1l, accum1h);

				x4 = _mm256_load_si256((const __m256i *)(src_ptr4 + j));
				x4 = _mm256_add_epi16(x4, INT16_MIN_EPI16);
				fmadd_epi16_epi32_avx(coeff4, x4, accum0l, accum0h);

				x5 = _mm256_load_si256((const __m256i *)(src_ptr5 + j));
				x5 = _mm256_add_epi16(x5, INT16_MIN_EPI16);
				fmadd_epi16_epi32_avx(coeff5, x5, accum1l, accum1h);

				x6 = _mm256_load_si256((const __m256i *)(src_ptr6 + j));
				x6 = _mm256_add_epi16(x6, INT16_MIN_EPI16);
				fmadd_epi16_epi32_avx(coeff6, x6, accum0l, accum0h);

				x7 = _mm256_load_si256((const __m256i *)(src_ptr7 + j));
				x7 = _mm256_add_epi16(x7, INT16_MIN_EPI16);
				fmadd_epi16_epi32_avx(coeff7, x7, accum1l, accum1h);

				accum0l = _mm256_add_epi32(accum0l, accum1l);
				accum0h = _mm256_add_epi32(accum0h, accum1h);

				if (k) {
					__m256i cachel = _mm256_load_si256((const __m256i *)(tmp + j * 2 + 0));
					__m256i cacheh = _mm256_load_si256((const __m256i *)(tmp + j * 2 + 16));

					accum0l = _mm256_add_epi32(accum0l, cachel);
					accum0h = _mm256_add_epi32(accum0h, cacheh);
				}

				if (k == filter.width() - 8) {					
					packed = pack_i30_epi32(accum0l, accum0h);
					packed = _mm256_sub_epi16(packed, INT16_MIN_EPI16);

					_mm256_store_si256((__m256i *)(dst_ptr + j), packed);
				} else {
					_mm256_store_si256((__m256i *)(tmp + j * 2 + 0), accum0l);
					_mm256_store_si256((__m256i *)(tmp + j * 2 + 16), accum0h);
				}
			}
		}
		if (filter.width() % 8) {
			int m = filter.width() % 8;
			int k = filter.width() - m;

			coeff6 = _mm256_set1_epi16(filter.data_i16()[i * filter.stride_i16() + k + 6]);
			coeff5 = _mm256_set1_epi16(filter.data_i16()[i * filter.stride_i16() + k + 5]);
			coeff4 = _mm256_set1_epi16(filter.data_i16()[i * filter.stride_i16() + k + 4]);
			coeff3 = _mm256_set1_epi16(filter.data_i16()[i * filter.stride_i16() + k + 3]);
			coeff2 = _mm256_set1_epi16(filter.data_i16()[i * filter.stride_i16() + k + 2]);
			coeff1 = _mm256_set1_epi16(filter.data_i16()[i * filter.stride_i16() + k + 1]);
			coeff0 = _mm256_set1_epi16(filter.data_i16()[i * filter.stride_i16() + k + 0]);

			src_ptr6 = src + (filter.left()[i] + k + 6) * src_stride;
			src_ptr5 = src + (filter.left()[i] + k + 5) * src_stride;
			src_ptr4 = src + (filter.left()[i] + k + 4) * src_stride;
			src_ptr3 = src + (filter.left()[i] + k + 3) * src_stride;
			src_ptr2 = src + (filter.left()[i] + k + 2) * src_stride;
			src_ptr1 = src + (filter.left()[i] + k + 1) * src_stride;
			src_ptr0 = src + (filter.left()[i] + k + 0) * src_stride;

			for (int j = 0; j < mod(src_width, 16); j += 16) {
				accum0l = _mm256_setzero_si256();
				accum0h = _mm256_setzero_si256();
				accum1l = _mm256_setzero_si256();
				accum1h = _mm256_setzero_si256();

				switch (m) {
				case 7:
					x6 = _mm256_load_si256((const __m256i *)(src_ptr6 + j));
					x6 = _mm256_add_epi16(x6, INT16_MIN_EPI16);
					fmadd_epi16_epi32_avx(coeff6, x6, accum0l, accum0h);
				case 6:
					x5 = _mm256_load_si256((const __m256i *)(src_ptr5 + j));
					x5 = _mm256_add_epi16(x5, INT16_MIN_EPI16);
					fmadd_epi16_epi32_avx(coeff5, x5, accum1l, accum1h);
				case 5:
					x4 = _mm256_load_si256((const __m256i *)(src_ptr4 + j));
					x4 = _mm256_add_epi16(x4, INT16_MIN_EPI16);
					fmadd_epi16_epi32_avx(coeff4, x4, accum0l, accum0h);
				case 4:
					x3 = _mm256_load_si256((const __m256i *)(src_ptr3 + j));
					x3 = _mm256_add_epi16(x3, INT16_MIN_EPI16);
					fmadd_epi16_epi32_avx(coeff3, x3, accum1l, accum1h);
				case 3:
					x2 = _mm256_load_si256((const __m256i *)(src_ptr2 + j));
					x2 = _mm256_add_epi16(x2, INT16_MIN_EPI16);
					fmadd_epi16_epi32_avx(coeff2, x2, accum0l, accum0h);
				case 2:
					x1 = _mm256_load_si256((const __m256i *)(src_ptr1 + j));
					x1 = _mm256_add_epi16(x1, INT16_MIN_EPI16);
					fmadd_epi16_epi32_avx(coeff1, x1, accum1l, accum1h);
				case 1:
					x0 = _mm256_load_si256((const __m256i *)(src_ptr0 + j));
					x0 = _mm256_add_epi16(x0, INT16_MIN_EPI16);
					fmadd_epi16_epi32_avx(coeff0, x0, accum0l, accum0h);
				}

				accum0l = _mm256_add_epi32(accum0l, accum1l);
				accum0h = _mm256_add_epi32(accum0h, accum1h);

				if (k) {
					accum0l = _mm256_add_epi32(accum0l, _mm256_load_si256((const __m256i *)(tmp + j * 2 + 0)));
					accum0h = _mm256_add_epi32(accum0h, _mm256_load_si256((const __m256i *)(tmp + j * 2 + 16)));
				}

				packed = pack_i30_epi32(accum0l, accum0h);
				packed = _mm256_sub_epi16(packed, INT16_MIN_EPI16);

				_mm256_store_si256((__m256i *)(dst_ptr + j), packed);
			}
		}

		for (int j = mod(src_width, 16); j < src_width; ++j) {
			int top = filter.left()[i];
			int32_t accum = 0;

			for (int k = 0; k < filter.width(); ++k) {
				int32_t coeff = filter.data_i16()[i * filter.stride_i16() + k];
				int32_t x = unpack_u16(src[(top + k) * src_stride + j]);

				accum += coeff * x;
			}

			dst[i * dst_stride + j] = pack_i30(accum);
		}
	}

}

template <class T, class MemoryManager>
void filter_plane_fp_v_avx(const EvaluatedFilter &filter, const T * RESTRICT src, T * RESTRICT dst,
                            int src_width, int src_height, int src_stride, int dst_stride, MemoryManager mem)
{
	for (int i = 0; i < filter.height(); ++i) {
		__m256 coeff0, coeff1, coeff2, coeff3, coeff4, coeff5, coeff6, coeff7;
		__m256 x0, x1, x2, x3, x4, x5, x6, x7;
		__m256 accum0, accum1, accum2, accum3;

		const T *src_ptr0, *src_ptr1, *src_ptr2, *src_ptr3, *src_ptr4, *src_ptr5, *src_ptr6, *src_ptr7;
		T *dst_ptr = dst + i * dst_stride;

		for (int k = 0; k < mod(filter.width(), 8); k += 8) {
			src_ptr0 = src + (filter.left()[i] + k + 0) * src_stride;
			src_ptr1 = src + (filter.left()[i] + k + 1) * src_stride;
			src_ptr2 = src + (filter.left()[i] + k + 2) * src_stride;
			src_ptr3 = src + (filter.left()[i] + k + 3) * src_stride;
			src_ptr4 = src + (filter.left()[i] + k + 4) * src_stride;
			src_ptr5 = src + (filter.left()[i] + k + 5) * src_stride;
			src_ptr6 = src + (filter.left()[i] + k + 6) * src_stride;
			src_ptr7 = src + (filter.left()[i] + k + 7) * src_stride;

			coeff0 = _mm256_broadcast_ss(filter.data() + i * filter.stride() + k + 0);
			coeff1 = _mm256_broadcast_ss(filter.data() + i * filter.stride() + k + 1);
			coeff2 = _mm256_broadcast_ss(filter.data() + i * filter.stride() + k + 2);
			coeff3 = _mm256_broadcast_ss(filter.data() + i * filter.stride() + k + 3);
			coeff4 = _mm256_broadcast_ss(filter.data() + i * filter.stride() + k + 4);
			coeff5 = _mm256_broadcast_ss(filter.data() + i * filter.stride() + k + 5);
			coeff6 = _mm256_broadcast_ss(filter.data() + i * filter.stride() + k + 6);
			coeff7 = _mm256_broadcast_ss(filter.data() + i * filter.stride() + k + 7);

			for (int j = 0; j < mod(src_width, 8); j += 8) {
				x0 = mem.load_8(src_ptr0 + j);
				accum0 = _mm256_mul_ps(coeff0, x0);

				x1 = mem.load_8(src_ptr1 + j);
				accum1 = _mm256_mul_ps(coeff1, x1);

				x2 = mem.load_8(src_ptr2 + j);
				accum2 = _mm256_mul_ps(coeff2, x2);

				x3 = mem.load_8(src_ptr3 + j);
				accum3 = _mm256_mul_ps(coeff3, x3);

				x4 = mem.load_8(src_ptr4 + j);
				accum0 = _mm256_fmadd_ps(coeff4, x4, accum0);

				x5 = mem.load_8(src_ptr5 + j);
				accum1 = _mm256_fmadd_ps(coeff5, x5, accum1);

				x6 = mem.load_8(src_ptr6 + j);
				accum2 = _mm256_fmadd_ps(coeff6, x6, accum2);

				x7 = mem.load_8(src_ptr7 + j);
				accum3 = _mm256_fmadd_ps(coeff7, x7, accum3);

				accum0 = _mm256_add_ps(accum0, accum2);
				accum1 = _mm256_add_ps(accum1, accum3);
				accum0 = _mm256_add_ps(accum0, accum1);

				if (k)
					accum0 = _mm256_add_ps(accum0, mem.load_8(dst_ptr + j));

				mem.store_8(dst_ptr + j, accum0);
			}
		}
		if (filter.width() % 8) {
			int m = filter.width() % 8;
			int k = filter.width() - m;

			coeff6 = _mm256_broadcast_ss(filter.data() + i * filter.stride() + k + 6);
			coeff5 = _mm256_broadcast_ss(filter.data() + i * filter.stride() + k + 5);
			coeff4 = _mm256_broadcast_ss(filter.data() + i * filter.stride() + k + 4);
			coeff3 = _mm256_broadcast_ss(filter.data() + i * filter.stride() + k + 3);
			coeff2 = _mm256_broadcast_ss(filter.data() + i * filter.stride() + k + 2);
			coeff1 = _mm256_broadcast_ss(filter.data() + i * filter.stride() + k + 1);
			coeff0 = _mm256_broadcast_ss(filter.data() + i * filter.stride() + k + 0);

			src_ptr6 = src + (filter.left()[i] + k + 6) * src_stride;
			src_ptr5 = src + (filter.left()[i] + k + 5) * src_stride;
			src_ptr4 = src + (filter.left()[i] + k + 4) * src_stride;
			src_ptr3 = src + (filter.left()[i] + k + 3) * src_stride;
			src_ptr2 = src + (filter.left()[i] + k + 2) * src_stride;
			src_ptr1 = src + (filter.left()[i] + k + 1) * src_stride;
			src_ptr0 = src + (filter.left()[i] + k + 0) * src_stride;

			for (int j = 0; j < mod(src_width, 8); j += 8) {
				accum0 = _mm256_setzero_ps();
				accum1 = _mm256_setzero_ps();
				accum2 = _mm256_setzero_ps();
				accum3 = _mm256_setzero_ps();

				switch (m) {
				case 7:
					x6 = mem.load_8(src_ptr6 + j);
					accum2 = _mm256_mul_ps(coeff6, x6);
				case 6:
					x5 = mem.load_8(src_ptr5 + j);
					accum1 = _mm256_mul_ps(coeff5, x5);
				case 5:
					x4 = mem.load_8(src_ptr4 + j);
					accum0 = _mm256_mul_ps(coeff4, x4);
				case 4:
					x3 = mem.load_8(src_ptr3 + j);
					accum3 = _mm256_mul_ps(coeff3, x3);
				case 3:
					x2 = mem.load_8(src_ptr2 + j);
					accum2 = _mm256_fmadd_ps(coeff2, x2, accum2);
				case 2:
					x1 = mem.load_8(src_ptr1 + j);
					accum1 = _mm256_fmadd_ps(coeff1, x1, accum1);
				case 1:
					x0 = mem.load_8(src_ptr0 + j);
					accum0 = _mm256_fmadd_ps(coeff0, x0, accum0);
				}

				accum0 = _mm256_add_ps(accum0, accum2);
				accum1 = _mm256_add_ps(accum1, accum3);
				accum0 = _mm256_add_ps(accum0, accum1);

				if (k)
					accum0 = _mm256_add_ps(accum0, mem.load_8(dst_ptr + j));

				mem.store_8(dst_ptr + j, accum0);
			}
		}

		for (int j = mod(src_width, 8); j < src_width; ++j) {
			int top = filter.left()[i];
			float accum = 0.f;

			for (int k = 0; k < filter.width(); ++k) {
				float coeff = filter.data()[i * filter.stride() + k];
				float x = mem.load_1(src + (top + k) * src_stride + j);
				accum += coeff * x;
			}
			mem.store_1(dst + i * dst_stride + j, accum);
		}
	}
}

class ResizeImplX86 final : public ResizeImpl {
	template <class T, class MemoryManager>
	void process_fp_h(const T * RESTRICT src, T * RESTRICT dst, T * RESTRICT tmp,
	                  int src_width, int src_height, int src_stride, int dst_stride, MemoryManager mem) const
	{
		if (m_filter_h.width() >= 8)
			filter_plane_fp_h_avx<true>(m_filter_h, src, dst, src_width, src_height, src_stride, dst_stride, mem);
		else
			filter_plane_fp_h_avx<false>(m_filter_h, src, dst, src_width, src_height, src_stride, dst_stride, mem);
	}

	template <class T, class MemoryManager>
	void process_fp_v(const T * RESTRICT src, T * RESTRICT dst, T * RESTRICT tmp,
	                  int src_width, int src_height, int src_stride, int dst_stride, MemoryManager mem) const
	{
		filter_plane_fp_v_avx(m_filter_v, src, dst, src_width, src_height, src_stride, dst_stride, mem);
	}
public:
	ResizeImplX86(const EvaluatedFilter &filter_h, const EvaluatedFilter &filter_v) : ResizeImpl(filter_h, filter_v)
	{}

	void process_u16_h(const uint16_t * RESTRICT src, uint16_t * RESTRICT dst, uint16_t * RESTRICT tmp,
	                   int src_width, int src_height, int src_stride, int dst_stride) const override
	{
		if (m_filter_h.width() >= 16)
			filter_plane_u16_h_avx<true>(m_filter_h, src, dst, src_width, src_height, src_stride, dst_stride);
		else
			filter_plane_u16_h_avx<false>(m_filter_h, src, dst, src_width, src_height, src_stride, dst_stride);
	}

	void process_u16_v(const uint16_t * RESTRICT src, uint16_t * RESTRICT dst, uint16_t * RESTRICT tmp,
	                   int src_width, int src_height, int src_stride, int dst_stride) const override
	{
		filter_plane_u16_v_avx(m_filter_v, src, dst, tmp, src_width, src_height, src_stride, dst_stride);
	}

	void process_f16_h(const uint16_t * RESTRICT src, uint16_t * RESTRICT dst, uint16_t * RESTRICT tmp,
	                   int src_width, int src_height, int src_stride, int dst_stride) const override
	{
		process_fp_h(src, dst, tmp, src_width, src_height, src_stride, dst_stride, LoadStoreManagerF16{});
	}

	void process_f16_v(const uint16_t * RESTRICT src, uint16_t * RESTRICT dst, uint16_t * RESTRICT tmp,
	                   int src_width, int src_height, int src_stride, int dst_stride) const override
	{
		process_fp_v(src, dst, tmp, src_width, src_height, src_stride, dst_stride, LoadStoreManagerF16{});
	}

	void process_f32_h(const float * RESTRICT src, float * RESTRICT dst, float * RESTRICT tmp,
	                   int src_width, int src_height, int src_stride, int dst_stride) const override
	{
		process_fp_h(src, dst, tmp, src_width, src_height, src_stride, dst_stride, LoadStoreManagerF32{});
	}

	void process_f32_v(const float * RESTRICT src, float * RESTRICT dst, float * RESTRICT tmp,
	                   int src_width, int src_height, int src_stride, int dst_stride) const override
	{
		process_fp_v(src, dst, tmp, src_width, src_height, src_stride, dst_stride, LoadStoreManagerF32{});
	}
};

} // namespace


ResizeImpl *create_resize_impl_x86(const EvaluatedFilter &filter_h, const EvaluatedFilter &filter_v)
{
	return new ResizeImplX86{ filter_h, filter_v };
}

} // namespace resize;

#endif // RESIZE_X86
