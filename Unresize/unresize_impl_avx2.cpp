#ifdef ZIMG_X86
#include <immintrin.h>
#include "Common/osdep.h"
#include "bilinear.h"
#include "unresize_impl.h"
#include "unresize_impl_x86.h"

namespace zimg {;
namespace unresize {;

namespace {;

FORCE_INLINE void transpose8_ps(__m256 &row0, __m256 &row1, __m256 &row2, __m256 &row3, __m256 &row4, __m256 &row5, __m256 &row6, __m256 &row7)
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

void filter_plane_h_avx2(const BilinearContext &ctx, const ImagePlane<const float> &src, const ImagePlane<float> &dst, float *tmp)
{
	const float * RESTRICT src_p = src.data();
	float * RESTRICT dst_p = dst.data();
	int src_width = src.width();
	int src_height = src.height();
	int src_stride = src.stride();
	int dst_stride = dst.stride();

	const float *pc = ctx.lu_c.data();
	const float *pl = ctx.lu_l.data();
	const float *pu = ctx.lu_u.data();

	for (ptrdiff_t i = 0; i < mod(src_height, 8); i += 8) {
		__m256 w, z, f, c, l;
		ptrdiff_t j;

		z = _mm256_setzero_ps();

		// Input, matrix-vector product, and forward substitution loop.
		for (j = 0; j < ctx.dst_width; ++j) {
			const float *matrix_row = &ctx.matrix_coefficients[j * ctx.matrix_row_stride];
			ptrdiff_t left = ctx.matrix_row_offsets[j];

			if (left + ctx.matrix_row_stride > src_width)
				break;

			// Matrix-vector product.
			__m256 accum0 = _mm256_setzero_ps();
			__m256 accum1 = _mm256_setzero_ps();
			__m256 accum2 = _mm256_setzero_ps();
			__m256 accum3 = _mm256_setzero_ps();

			for (ptrdiff_t k = 0; k < ctx.matrix_row_size; k += 8) {
				__m256 coeffs = _mm256_loadu_ps(&matrix_row[k]);
				__m256 v0, v1, v2, v3, v4, v5, v6, v7;

				v0 = _mm256_loadu_ps(&src_p[(i + 0) * src_stride + left + k]);
				v0 = _mm256_mul_ps(coeffs, v0);

				v1 = _mm256_loadu_ps(&src_p[(i + 1) * src_stride + left + k]);
				v1 = _mm256_mul_ps(coeffs, v1);

				v2 = _mm256_loadu_ps(&src_p[(i + 2) * src_stride + left + k]);
				v2 = _mm256_mul_ps(coeffs, v2);

				v3 = _mm256_loadu_ps(&src_p[(i + 3) * src_stride + left + k]);
				v3 = _mm256_mul_ps(coeffs, v3);

				v4 = _mm256_loadu_ps(&src_p[(i + 4) * src_stride + left + k]);
				v4 = _mm256_mul_ps(coeffs, v4);

				v5 = _mm256_loadu_ps(&src_p[(i + 5) * src_stride + left + k]);
				v5 = _mm256_mul_ps(coeffs, v5);

				v6 = _mm256_loadu_ps(&src_p[(i + 6) * src_stride + left + k]);
				v6 = _mm256_mul_ps(coeffs, v6);

				v7 = _mm256_loadu_ps(&src_p[(i + 7) * src_stride + left + k]);
				v7 = _mm256_mul_ps(coeffs, v7);

				transpose8_ps(v0, v1, v2, v3, v4, v5, v6, v7);

				accum0 = _mm256_add_ps(accum0, v0);
				accum1 = _mm256_add_ps(accum1, v1);
				accum2 = _mm256_add_ps(accum2, v2);
				accum3 = _mm256_add_ps(accum3, v3);
				accum0 = _mm256_add_ps(accum0, v4);
				accum1 = _mm256_add_ps(accum1, v5);
				accum2 = _mm256_add_ps(accum2, v6);
				accum3 = _mm256_add_ps(accum3, v7);
			}

			// Forward substitution.
			accum0 = _mm256_add_ps(accum0, accum2);
			accum1 = _mm256_add_ps(accum1, accum3);

			f = _mm256_add_ps(accum0, accum1);
			c = _mm256_broadcast_ss(&pc[j]);
			l = _mm256_broadcast_ss(&pl[j]);

			z = _mm256_fnmadd_ps(c, z, f);
			z = _mm256_mul_ps(z, l);

			_mm256_store_ps(&tmp[j * 8], z);
		}
		// Handle remainder of line.
		for (; j < ctx.dst_width; ++j) {
			const float *matrix_row = &ctx.matrix_coefficients[j * ctx.matrix_row_stride];
			ptrdiff_t left = ctx.matrix_row_offsets[j];

			for (ptrdiff_t ii = 0; ii < 8; ++ii) {
				float accum = 0;
				for (ptrdiff_t k = 0; k < ctx.matrix_row_size; ++k) {
					accum += matrix_row[k] * src_p[(i + ii) * src_stride + left + k];
				}

				tmp[j * 8 + ii] = (accum - pc[j] * tmp[(j - 1) * 8 + ii]) * pl[j];
			}
		}

		w = _mm256_setzero_ps();

		// Backward substitution and output loop.
		for (ptrdiff_t j = ctx.dst_width; j > mod(ctx.dst_width, 8); --j) {
			float w_buf[8];

			_mm256_storeu_ps(w_buf, w);
			for (ptrdiff_t ii = 0; ii < 8; ++ii) {
				w_buf[ii] = tmp[(j - 1) * 8 + ii] - pu[j - 1] * w_buf[ii];
				dst_p[(i + ii) * dst_stride + j - 1] = w_buf[ii];
			}
			w = _mm256_loadu_ps(w_buf);
		}
		for (ptrdiff_t j = mod(ctx.dst_width, 8); j > 0; j -= 8) {
			__m256 u0, u1, u2, u3, u4, u5, u6, u7;
			__m256 z0, z1, z2, z3, z4, z5, z6, z7;
			__m256 w0, w1, w2, w3, w4, w5, w6, w7;

			z0 = _mm256_load_ps(&tmp[(j - 8) * 8]);
			z1 = _mm256_load_ps(&tmp[(j - 7) * 8]);
			z2 = _mm256_load_ps(&tmp[(j - 6) * 8]);
			z3 = _mm256_load_ps(&tmp[(j - 5) * 8]);
			z4 = _mm256_load_ps(&tmp[(j - 4) * 8]);
			z5 = _mm256_load_ps(&tmp[(j - 3) * 8]);
			z6 = _mm256_load_ps(&tmp[(j - 2) * 8]);
			z7 = _mm256_load_ps(&tmp[(j - 1) * 8]);

			u7 = _mm256_broadcast_ss(pu + j - 8);
			w = _mm256_fnmadd_ps(u7, w, z7);
			w7 = w;

			u6 = _mm256_broadcast_ss(pu + j - 7);
			w = _mm256_fnmadd_ps(u6, w, z6);
			w6 = w;

			u5 = _mm256_broadcast_ss(pu + j - 6);
			w = _mm256_fnmadd_ps(u5, w, z5);
			w5 = w;

			u4 = _mm256_broadcast_ss(pu + j - 5);
			w = _mm256_fnmadd_ps(u4, w, z4);
			w4 = w;

			u3 = _mm256_broadcast_ss(pu + j - 4);
			w = _mm256_fnmadd_ps(u3, w, z3);
			w3 = w;

			u2 = _mm256_broadcast_ss(pu + j - 3);
			w = _mm256_fnmadd_ps(u2, w, z2);
			w2 = w;

			u1 = _mm256_broadcast_ss(pu + j - 2);
			w = _mm256_fnmadd_ps(u1, w, z1);
			w1 = w;

			u0 = _mm256_broadcast_ss(pu + j - 1);
			w = _mm256_fnmadd_ps(u0, w, z0);
			w0 = w;

			transpose8_ps(w0, w1, w2, w3, w4, w5, w6, w7);

			_mm256_store_ps(&dst_p[(i + 0) * dst_stride + j - 8], w0);
			_mm256_store_ps(&dst_p[(i + 1) * dst_stride + j - 8], w1);
			_mm256_store_ps(&dst_p[(i + 2) * dst_stride + j - 8], w2);
			_mm256_store_ps(&dst_p[(i + 3) * dst_stride + j - 8], w3);
			_mm256_store_ps(&dst_p[(i + 4) * dst_stride + j - 8], w4);
			_mm256_store_ps(&dst_p[(i + 5) * dst_stride + j - 8], w5);
			_mm256_store_ps(&dst_p[(i + 6) * dst_stride + j - 8], w6);
			_mm256_store_ps(&dst_p[(i + 7) * dst_stride + j - 8], w7);
		}
	}
	for (ptrdiff_t i = mod(src_height, 8); i < src_height; ++i) {
		filter_scanline_h_forward(ctx, src, tmp, i, 0, ctx.dst_width);
		filter_scanline_h_back(ctx, tmp, dst, i, ctx.dst_width, 0);
	}
}

void filter_plane_v_avx2(const BilinearContext &ctx, const ImagePlane<const float> &src, const ImagePlane<float> &dst)
{
	const float * RESTRICT src_p = src.data();
	float * RESTRICT dst_p = dst.data();
	int src_width = src.width();
	int src_height = src.height();
	int src_stride = src.stride();
	int dst_stride = dst.stride();

	const float *pc = ctx.lu_c.data();
	const float *pl = ctx.lu_l.data();
	const float *pu = ctx.lu_u.data();

	__m256 w, z, f, c, l, u;

	for (ptrdiff_t i = 0; i < ctx.dst_width; ++i) {
		__m256 coeff0, coeff1, coeff2, coeff3, coeff4, coeff5, coeff6, coeff7;
		__m256 x0, x1, x2, x3, x4, x5, x6, x7;
		__m256 accum0, accum1, accum2, accum3;

		const float *row = &ctx.matrix_coefficients.data()[i * ctx.matrix_row_stride];
		ptrdiff_t top = ctx.matrix_row_offsets[i];

		const float *src_ptr0, *src_ptr1, *src_ptr2, *src_ptr3, *src_ptr4, *src_ptr5, *src_ptr6, *src_ptr7;
		float *dst_ptr = &dst_p[i * dst_stride];

		// Matrix-vector product.
		for (ptrdiff_t k = 0; k < mod(ctx.matrix_row_size, 8); k += 8) {
			src_ptr0 = &src_p[(top + k + 0) * src_stride];
			src_ptr1 = &src_p[(top + k + 1) * src_stride];
			src_ptr2 = &src_p[(top + k + 2) * src_stride];
			src_ptr3 = &src_p[(top + k + 3) * src_stride];
			src_ptr4 = &src_p[(top + k + 4) * src_stride];
			src_ptr5 = &src_p[(top + k + 5) * src_stride];
			src_ptr6 = &src_p[(top + k + 6) * src_stride];
			src_ptr7 = &src_p[(top + k + 7) * src_stride];

			coeff0 = _mm256_broadcast_ss(row + k + 0);
			coeff1 = _mm256_broadcast_ss(row + k + 1);
			coeff2 = _mm256_broadcast_ss(row + k + 2);
			coeff3 = _mm256_broadcast_ss(row + k + 3);
			coeff4 = _mm256_broadcast_ss(row + k + 4);
			coeff5 = _mm256_broadcast_ss(row + k + 5);
			coeff6 = _mm256_broadcast_ss(row + k + 6);
			coeff7 = _mm256_broadcast_ss(row + k + 7);
				
			for (ptrdiff_t j = 0; j < mod(src_width, 8); j += 8) {
				x0 = _mm256_load_ps(&src_ptr0[j]);
				accum0 = _mm256_mul_ps(coeff0, x0);

				x1 = _mm256_load_ps(&src_ptr1[j]);
				accum1 = _mm256_mul_ps(coeff1, x1);

				x2 = _mm256_load_ps(&src_ptr2[j]);
				accum2 = _mm256_mul_ps(coeff2, x2);

				x3 = _mm256_load_ps(&src_ptr3[j]);
				accum3 = _mm256_mul_ps(coeff3, x3);

				x4 = _mm256_load_ps(&src_ptr4[j]);
				accum0 = _mm256_fmadd_ps(coeff4, x4, accum0);

				x5 = _mm256_load_ps(&src_ptr5[j]);
				accum1 = _mm256_fmadd_ps(coeff5, x5, accum1);

				x6 = _mm256_load_ps(&src_ptr6[j]);
				accum2 = _mm256_fmadd_ps(coeff6, x6, accum2);

				x7 = _mm256_load_ps(&src_ptr7[j]);
				accum3 = _mm256_fmadd_ps(coeff7, x7, accum3);

				accum0 = _mm256_add_ps(accum0, accum2);
				accum1 = _mm256_add_ps(accum1, accum3);
				accum0 = _mm256_add_ps(accum0, accum1);

				if (k)
					accum0 = _mm256_add_ps(accum0, _mm256_load_ps(&dst_ptr[j]));

				_mm256_store_ps(&dst_ptr[j], accum0);
			}
		}
		if (ctx.matrix_row_size % 8) {
			ptrdiff_t m = ctx.matrix_row_size % 8;
			ptrdiff_t k = ctx.matrix_row_size - m;

			coeff6 = _mm256_broadcast_ss(row + k + 6);
			coeff5 = _mm256_broadcast_ss(row + k + 5);
			coeff4 = _mm256_broadcast_ss(row + k + 4);
			coeff3 = _mm256_broadcast_ss(row + k + 3);
			coeff2 = _mm256_broadcast_ss(row + k + 2);
			coeff1 = _mm256_broadcast_ss(row + k + 1);
			coeff0 = _mm256_broadcast_ss(row + k + 0);

			src_ptr6 = &src_p[(top + k + 6) * src_stride];
			src_ptr5 = &src_p[(top + k + 5) * src_stride];
			src_ptr4 = &src_p[(top + k + 4) * src_stride];
			src_ptr3 = &src_p[(top + k + 3) * src_stride];
			src_ptr2 = &src_p[(top + k + 2) * src_stride];
			src_ptr1 = &src_p[(top + k + 1) * src_stride];
			src_ptr0 = &src_p[(top + k + 0) * src_stride];

			for (ptrdiff_t j = 0; j < mod(src_width, 8); j += 8) {
				accum0 = _mm256_setzero_ps();
				accum1 = _mm256_setzero_ps();
				accum2 = _mm256_setzero_ps();
				accum3 = _mm256_setzero_ps();

				switch (m) {
				case 7:
					x6 = _mm256_load_ps(&src_ptr6[j]);
					accum2 = _mm256_mul_ps(coeff6, x6);
				case 6:
					x5 = _mm256_load_ps(&src_ptr5[j]);
					accum1 = _mm256_mul_ps(coeff5, x5);
				case 5:
					x4 = _mm256_load_ps(&src_ptr4[j]);
					accum0 = _mm256_mul_ps(coeff4, x4);
				case 4:
					x3 = _mm256_load_ps(&src_ptr3[j]);
					accum3 = _mm256_mul_ps(coeff3, x3);
				case 3:
					x2 = _mm256_load_ps(&src_ptr2[j]);
					accum2 = _mm256_fmadd_ps(coeff2, x2, accum2);
				case 2:
					x1 = _mm256_load_ps(&src_ptr1[j]);
					accum1 = _mm256_fmadd_ps(coeff1, x1, accum1);
				case 1:
					x0 = _mm256_load_ps(&src_ptr0[j]);
					accum0 = _mm256_fmadd_ps(coeff0, x0, accum0);
				}

				accum0 = _mm256_add_ps(accum0, accum2);
				accum1 = _mm256_add_ps(accum1, accum3);
				accum0 = _mm256_add_ps(accum0, accum1);

				if (k)
					accum0 = _mm256_add_ps(accum0, _mm256_load_ps(&dst_ptr[j]));

				_mm256_store_ps(&dst_ptr[j], accum0);
			}
		}

		c = _mm256_broadcast_ss(pc + i);
		l = _mm256_broadcast_ss(pl + i);

		// Forward substitution.
		for (ptrdiff_t j = 0; j < mod(src_width, 8); j += 8) {
			z = i ? _mm256_load_ps(&dst_p[(i - 1) * dst_stride + j]) : _mm256_setzero_ps();
			f = _mm256_load_ps(&dst_p[i * src_stride + j]);

			z = _mm256_fnmadd_ps(c, z, f);
			z = _mm256_mul_ps(z, l);

			_mm256_store_ps(&dst_p[i * dst_stride + j], z);
		}
		
		filter_scanline_v_forward(ctx, src, dst, i, mod(src_width, 8), src_width);
	}

	// Back substitution.
	for (ptrdiff_t i = ctx.dst_width; i > 0; --i) {
		u = _mm256_broadcast_ss(pu + i - 1);

		for (ptrdiff_t j = 0; j < mod(src_width, 8); j += 8) {
			w = i < ctx.dst_width ? _mm256_load_ps(&dst_p[i * dst_stride + j]) : _mm256_setzero_ps();
			z = _mm256_load_ps(&dst_p[(i - 1) * dst_stride + j]);
			w = _mm256_fnmadd_ps(u, w, z);

			_mm256_store_ps(&dst_p[(i - 1) * dst_stride + j], w);
		}
		filter_scanline_v_back(ctx, dst, i, mod(src_width, 8), src_width);
	}
}

class UnresizeImplAVX2 : public UnresizeImpl {
public:
	UnresizeImplAVX2(const BilinearContext &hcontext, const BilinearContext &vcontext) : UnresizeImpl(hcontext, vcontext)
	{}

	void process_f32_h(const ImagePlane<const float> &src, const ImagePlane<float> &dst, float *tmp) const override
	{
		filter_plane_h_avx2(m_hcontext, src, dst, tmp);
	}

	void process_f32_v(const ImagePlane<const float> &src, const ImagePlane<float> &dst, float *tmp) const override
	{
		filter_plane_v_avx2(m_vcontext, src, dst);
	}
};

} // namespace


UnresizeImpl *create_unresize_impl_avx2(const BilinearContext &hcontext, const BilinearContext &vcontext)
{
	return new UnresizeImplAVX2{ hcontext, vcontext };
}

} // namespace unresize
} // namespace zimg

#endif // ZIMG_X86
