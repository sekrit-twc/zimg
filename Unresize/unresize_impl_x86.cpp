#ifdef ZIMG_X86
#include <emmintrin.h>
#include "Common/osdep.h"
#include "unresize_impl.h"

namespace zimg {;
namespace unresize {;

namespace {;

#define DISTRIBUTE_PS(x, i) (_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(x), _MM_SHUFFLE(i, i, i, i))))

void transpose_4x4_ps(__m128 &x0, __m128 &x1, __m128 &x2, __m128 &x3)
{
	__m128d t0 = _mm_castps_pd(_mm_unpacklo_ps(x0, x1));
	__m128d t1 = _mm_castps_pd(_mm_unpacklo_ps(x2, x3));
	__m128d t2 = _mm_castps_pd(_mm_unpackhi_ps(x0, x1));
	__m128d t3 = _mm_castps_pd(_mm_unpackhi_ps(x2, x3));

	__m128d o0 = _mm_unpacklo_pd(t0, t1);
	__m128d o1 = _mm_unpackhi_pd(t0, t1);
	__m128d o2 = _mm_unpacklo_pd(t2, t3);
	__m128d o3 = _mm_unpackhi_pd(t2, t3);

	x0 = _mm_castpd_ps(o0);
	x1 = _mm_castpd_ps(o1);
	x2 = _mm_castpd_ps(o2);
	x3 = _mm_castpd_ps(o3);
}

void loadu_4x4_ps(const float *p, int stride, __m128 &x0, __m128 &x1, __m128 &x2, __m128 &x3)
{
	x0 = _mm_loadu_ps(&p[stride * 0]);
	x1 = _mm_loadu_ps(&p[stride * 1]);
	x2 = _mm_loadu_ps(&p[stride * 2]);
	x3 = _mm_loadu_ps(&p[stride * 3]);
}

void storeu_4x4_ps(float *p, int stride, __m128 &x0, __m128 &x1, __m128 &x2, __m128 &x3)
{
	_mm_storeu_ps(&p[stride * 0], x0);
	_mm_storeu_ps(&p[stride * 1], x1);
	_mm_storeu_ps(&p[stride * 2], x2);
	_mm_storeu_ps(&p[stride * 3], x3);
}

} // namespace


template <int HWIDTH, int VWIDTH>
UnresizeImplX86<HWIDTH, VWIDTH>::UnresizeImplX86(const BilinearContext &hcontext, const BilinearContext &vcontext) :
UnresizeImpl(hcontext, vcontext)
{}

template <int HWIDTH, int VWIDTH>
template <int WIDTH>
FORCE_INLINE void UnresizeImplX86<HWIDTH, VWIDTH>::unresize_scanline4(const BilinearContext &ctx, const float * RESTRICT src, float * RESTRICT dst, float * RESTRICT tmp, int src_stride, int dst_stride) const
{
	// Compile time unrollowing for matrix-vector loop.
	int matrix_row_size = WIDTH > 0 ? WIDTH : ctx.matrix_row_size;
	int dst_width = ctx.dst_width;

	// LU data.
	const float *pc = ctx.lu_c.data();
	const float *pl = ctx.lu_l.data();
	const float *pu = ctx.lu_u.data();

	__m128 w = _mm_setzero_ps();
	__m128 z = _mm_setzero_ps();

	// Input, matrix-vector product, and forward substitution loop.
	for (int i = 0; i < dst_width; ++i) {
		const float *matrix_row = &ctx.matrix_coefficients[i * matrix_row_size];
		int left = ctx.matrix_row_offsets[i];

		// Matrix-vector product.
		__m128 accum = _mm_setzero_ps();
		for (int j = 0; j < matrix_row_size; j += 4) {
			__m128 matrix_coeffs = _mm_loadu_ps(&matrix_row[j]);
			__m128 v0, v1, v2, v3;

			loadu_4x4_ps(&src[left + j], src_stride, v0, v1, v2, v3);
			v0 = _mm_mul_ps(matrix_coeffs, v0);
			v1 = _mm_mul_ps(matrix_coeffs, v1);
			v2 = _mm_mul_ps(matrix_coeffs, v2);
			v3 = _mm_mul_ps(matrix_coeffs, v3);
			transpose_4x4_ps(v0, v1, v2, v3);

			__m128 s1 = _mm_add_ps(v0, v1);
			__m128 s2 = _mm_add_ps(v2, v3);
			accum = _mm_add_ps(accum, s1);
			accum = _mm_add_ps(accum, s2);
		}

		// Forward substitution.
		__m128 f, c, l;
		f = accum;
		c = _mm_load_ps1(&pc[i]);
		l = _mm_load_ps1(&pl[i]);

		z = _mm_mul_ps(c, z);
		z = _mm_sub_ps(f, z);
		z = _mm_mul_ps(z, l);
		_mm_storeu_ps(&tmp[i * 4], z);
	}

	// Backward substitution and output loop.
	for (int i = dst_width - 1; i >= mod(dst_width, 4); --i) {
		__m128 u = _mm_load_ps1(&pu[i]);

		z = _mm_loadu_ps(&tmp[i * 4]);
		w = _mm_mul_ps(u, w);
		w = _mm_sub_ps(z, w);

		_mm_store_ss(&dst[dst_stride * 0 + i], DISTRIBUTE_PS(w, 0));
		_mm_store_ss(&dst[dst_stride * 1 + i], DISTRIBUTE_PS(w, 1));
		_mm_store_ss(&dst[dst_stride * 2 + i], DISTRIBUTE_PS(w, 2));
		_mm_store_ss(&dst[dst_stride * 3 + i], DISTRIBUTE_PS(w, 3));
	}
	for (int i = mod(dst_width, 4) - 1; i >= 3; i -= 4) {
		__m128 u = _mm_load_ps(&pu[i - 3]);
		__m128 z0, z1, z2, z3;
		__m128 w0, w1, w2, w3;

		loadu_4x4_ps(&tmp[(i - 3) * 4], 4, z0, z1, z2, z3);

		w = _mm_mul_ps(DISTRIBUTE_PS(u, 3), w);
		w = _mm_sub_ps(z3, w);
		w3 = w;

		w = _mm_mul_ps(DISTRIBUTE_PS(u, 2), w);
		w = _mm_sub_ps(z2, w);
		w2 = w;

		w = _mm_mul_ps(DISTRIBUTE_PS(u, 1), w);
		w = _mm_sub_ps(z1, w);
		w1 = w;

		w = _mm_mul_ps(DISTRIBUTE_PS(u, 0), w);
		w = _mm_sub_ps(z0, w);
		w0 = w;

		transpose_4x4_ps(w0, w1, w2, w3);
		storeu_4x4_ps(&dst[i - 3], dst_stride, w0, w1, w2, w3);
	}
}

template <int HWIDTH, int VWIDTH>
void UnresizeImplX86<HWIDTH, VWIDTH>::unresize_scanline4_h(const float *src, float *dst, float *tmp, int src_stride, int dst_stride) const
{
	unresize_scanline4<HWIDTH>(m_hcontext, src, dst, tmp, src_stride, dst_stride);
}

template <int HWIDTH, int VWIDTH>
void UnresizeImplX86<HWIDTH, VWIDTH>::unresize_scanline4_v(const float *src, float *dst, float *tmp, int src_stride, int dst_stride) const
{
	unresize_scanline4<VWIDTH>(m_vcontext, src, dst, tmp, src_stride, dst_stride);
}

template <int HWIDTH, int VWIDTH>
void UnresizeImplX86<HWIDTH, VWIDTH>::transpose_plane(const float *src, float *dst, int src_width, int src_height, int src_stride, int dst_stride) const
{
	for (int i = 0; i < mod(src_height, 16); i += 16) {
		for (int j = 0; j < mod(src_width, 16); j += 16) {
			for (int k = 0; k < 16; k += 4) {
				for (int kk = 0; kk < 16; kk += 4) {
					__m128 v0, v1, v2, v3;

					loadu_4x4_ps(&src[(i + k) * src_stride + (j + kk)], src_stride, v0, v1, v2, v3);
					transpose_4x4_ps(v0, v1, v2, v3);
					storeu_4x4_ps(&dst[(j + kk) * dst_stride + (i + k)], dst_stride, v0, v1, v2, v3);
				}
			}
		}
	}
	for (int i = mod(src_height, 16); i < src_height; ++i) {
		for (int j = 0; j < src_width; ++j) {
			dst[j * dst_stride + i] = src[i * src_stride + j];
		}
	}
	for (int j = mod(src_width, 16); j < src_width; ++j) {
		for (int i = 0; i < src_height; ++i) {
			dst[j * dst_stride + i] = src[i * src_stride + j];
		}
	}
}

template <int HWIDTH, int VWIDTH>
void UnresizeImplX86<HWIDTH, VWIDTH>::load_scanline_u8(const uint8_t * RESTRICT src, float * RESTRICT dst, int width) const
{
	const __m128 UINT8_MAX_FLOAT = _mm_set1_ps(1.0f / (float)UINT8_MAX);

	for (int i = 0; i < mod(width, 16); i += 16) {
		__m128i zero = _mm_set1_epi32(0);

		__m128i u8 = _mm_loadu_si128((__m128i *)&src[i]);
		__m128i u0, u1, u2, u3;

		u2 = _mm_unpacklo_epi8(u8, zero);
		u3 = _mm_unpackhi_epi8(u8, zero);

		u0 = _mm_unpacklo_epi16(u2, zero);
		u1 = _mm_unpackhi_epi16(u2, zero);
		u2 = _mm_unpacklo_epi16(u3, zero);
		u3 = _mm_unpackhi_epi16(u3, zero);

		__m128 f0, f1, f2, f3;
		f0 = _mm_cvtepi32_ps(u0);
		f1 = _mm_cvtepi32_ps(u1);
		f2 = _mm_cvtepi32_ps(u2);
		f3 = _mm_cvtepi32_ps(u3);

		f0 = _mm_mul_ps(f0, UINT8_MAX_FLOAT);
		f1 = _mm_mul_ps(f1, UINT8_MAX_FLOAT);
		f2 = _mm_mul_ps(f2, UINT8_MAX_FLOAT);
		f3 = _mm_mul_ps(f3, UINT8_MAX_FLOAT);

		storeu_4x4_ps(&dst[i], 4, f0, f1, f2, f3);
	}
	for (int i = mod(width, 16); i < width; ++i) {
		dst[i] = (float)src[i] / UINT8_MAX;
	}
}

template <int HWIDTH, int VWIDTH>
void UnresizeImplX86<HWIDTH, VWIDTH>::load_scanline_u16(const uint16_t * RESTRICT src, float * RESTRICT dst, int width) const
{
	const __m128 UINT16_MAX_FLOAT = _mm_set1_ps(1.0f / (float)UINT16_MAX);

	for (int i = 0; i < mod(width, 8); i += 8) {
		__m128i zero = _mm_set1_epi32(0);

		__m128i u16 = _mm_loadu_si128((__m128i *)&src[i]);

		__m128i u0, u1;
		u0 = _mm_unpacklo_epi16(u16, zero);
		u1 = _mm_unpackhi_epi16(u16, zero);

		__m128 f0, f1;
		f0 = _mm_cvtepi32_ps(u0);
		f1 = _mm_cvtepi32_ps(u1);

		f0 = _mm_mul_ps(f0, UINT16_MAX_FLOAT);
		f1 = _mm_mul_ps(f1, UINT16_MAX_FLOAT);

		_mm_storeu_ps(&dst[i + 0], f0);
		_mm_storeu_ps(&dst[i + 4], f1);
	}
	for (int i = mod(width, 8); i < width; ++i) {
		dst[i] = (float)src[i] / UINT16_MAX;
	}
}

template <int HWIDTH, int VWIDTH>
void UnresizeImplX86<HWIDTH, VWIDTH>::store_scanline_u8(const float * RESTRICT src, uint8_t *dst, int width) const
{
	const __m128 UINT8_MAX_FLOAT = _mm_set1_ps((float)UINT8_MAX);

	for (int i = 0; i < mod(width, 16); i += 16) {
		__m128 f0, f1, f2, f3;
		__m128i u0, u1, u2, u3;

		loadu_4x4_ps(&src[i], 4, f0, f1, f2, f3);

		f0 = _mm_mul_ps(f0, UINT8_MAX_FLOAT);
		f1 = _mm_mul_ps(f1, UINT8_MAX_FLOAT);
		f2 = _mm_mul_ps(f2, UINT8_MAX_FLOAT);
		f3 = _mm_mul_ps(f3, UINT8_MAX_FLOAT);

		u0 = _mm_cvtps_epi32(f0);
		u1 = _mm_cvtps_epi32(f1);
		u2 = _mm_cvtps_epi32(f2);
		u3 = _mm_cvtps_epi32(f3);

		u0 = _mm_packs_epi32(u0, u1);
		u1 = _mm_packs_epi32(u2, u3);

		u0 = _mm_packus_epi16(u0, u1);

		_mm_storeu_si128((__m128i *)&dst[i], u0);
	}
	for (int i = mod(width, 16); i < width; ++i) {
		dst[i] = clamp_float<uint8_t>(src[i]);
	}
}

template <int HWIDTH, int VWIDTH>
void UnresizeImplX86<HWIDTH, VWIDTH>::store_scanline_u16(const float * RESTRICT src, uint16_t *dst, int width) const
{
	const __m128 UINT16_MAX_FLOAT = _mm_set1_ps((float)UINT16_MAX);

	for (int i = 0; i < mod(width, 8); i += 8) {
		__m128 f0, f1;
		__m128i u0, u1;

		f0 = _mm_loadu_ps(&src[i + 0]);
		f1 = _mm_loadu_ps(&src[i + 4]);

		f0 = _mm_mul_ps(f0, UINT16_MAX_FLOAT);
		f1 = _mm_mul_ps(f1, UINT16_MAX_FLOAT);

		u0 = _mm_cvtps_epi32(f0);
		u1 = _mm_cvtps_epi32(f1);

		// Workaround for _mm_packs_epi32 saturation.
		u0 = _mm_slli_epi32(u0, 16);
		u0 = _mm_srai_epi32(u0, 16);
		u1 = _mm_slli_epi32(u1, 16);
		u1 = _mm_srai_epi32(u1, 16);

		u0 = _mm_packs_epi32(u0, u1);

		_mm_storeu_si128((__m128i *)&dst[i], u0);
	}
	for (int i = mod(width, 8); i < width; ++i) {
		dst[i] = clamp_float<uint16_t>(src[i]);
	}
}

// Explicit instantiations.
template class UnresizeImplX86<0, 0>; 
template class UnresizeImplX86<4, 0>;
template class UnresizeImplX86<4, 4>;
template class UnresizeImplX86<4, 8>;
template class UnresizeImplX86<4, 12>;
template class UnresizeImplX86<8, 0>;
template class UnresizeImplX86<8, 4>;
template class UnresizeImplX86<8, 8>;
template class UnresizeImplX86<8, 12>;
template class UnresizeImplX86<12, 0>;
template class UnresizeImplX86<12, 4>;
template class UnresizeImplX86<12, 8>;
template class UnresizeImplX86<12, 12>;

} // namespace unresize
} // namespace zimg

#endif // ZIMG_X86
