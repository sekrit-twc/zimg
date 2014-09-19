#ifdef ZIMG_X86
#include <cstddef>
#include <emmintrin.h>
#include "Common/osdep.h"
#include "unresize_impl.h"
#include "unresize_impl_x86.h"

namespace zimg {;
namespace unresize {;

namespace {;

void transpose4_ps(__m128 &x0, __m128 &x1, __m128 &x2, __m128 &x3)
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

void filter_plane_h_sse2(const BilinearContext &ctx, const float * RESTRICT src, float * RESTRICT dst, float * RESTRICT tmp,
                         ptrdiff_t src_width, ptrdiff_t src_height, ptrdiff_t src_stride, ptrdiff_t dst_stride)
{
	const float *pc = ctx.lu_c.data();
	const float *pl = ctx.lu_l.data();
	const float *pu = ctx.lu_u.data();

	for (ptrdiff_t i = 0; i < mod(src_height, 4); i += 4) {
		__m128 w, z, f, c, l;
		ptrdiff_t j;

		z = _mm_setzero_ps();

		// Input, matrix-vector product, and forward substitution loop.
		for (j = 0; j < ctx.dst_width; ++j) {
			const float *matrix_row = &ctx.matrix_coefficients[i * ctx.matrix_row_stride];
			ptrdiff_t left = ctx.matrix_row_offsets[i];

			if (left + ctx.matrix_row_stride > ctx.dst_width)
				break;

			// Matrix-vector product.
			__m128 accum0 = _mm_setzero_ps();
			__m128 accum1 = _mm_setzero_ps();
			for (ptrdiff_t k = 0; k < ctx.matrix_row_size; k += 4) {
				__m128 coeffs = _mm_loadu_ps(&matrix_row[k]);
				__m128 v0, v1, v2, v3;

				v0 = _mm_loadu_ps(&src[(i + 0) * src_stride + left + k]);
				v0 = _mm_mul_ps(coeffs, v0);

				v1 = _mm_loadu_ps(&src[(i + 1) * src_stride + left + k]);
				v1 = _mm_mul_ps(coeffs, v1);

				v2 = _mm_loadu_ps(&src[(i + 2) * src_stride + left + k]);
				v2 = _mm_mul_ps(coeffs, v2);

				v3 = _mm_loadu_ps(&src[(i + 3) * src_stride + left + k]);
				v3 = _mm_mul_ps(coeffs, v3);

				transpose4_ps(v0, v1, v2, v3);
	
				accum0 = _mm_add_ps(accum0, v0);
				accum1 = _mm_add_ps(accum1, v1);
				accum0 = _mm_add_ps(accum0, v2);
				accum1 = _mm_add_ps(accum0, v3);
			}

			// Forward substitution.
			f = _mm_add_ps(accum0, accum1);
			c = _mm_load_ps1(&pc[j]);
			l = _mm_load_ps1(&pl[j]);

			z = _mm_mul_ps(c, z);
			z = _mm_sub_ps(f, z);
			z = _mm_mul_ps(z, l);

			_mm_store_ps(&tmp[j * 4], z);
		}
		// Handle remainder of line.
		for (; j < ctx.dst_width; ++j) {
			const float *matrix_row = &ctx.matrix_coefficients[i * ctx.matrix_row_stride];
			ptrdiff_t left = ctx.matrix_row_offsets[i];

			for (ptrdiff_t ii = 0; ii < 4; ++ii) {
				float accum = 0;
				for (ptrdiff_t k = 0; k < ctx.matrix_row_size; ++k) {
					accum += matrix_row[k] * src[(i + ii) * src_stride + left + k];
				}

				tmp[j * 4 + ii] = (accum - pc[j] * tmp[(j - 1) * 4 + ii]) * pl[j];
			}
		}

		w = _mm_setzero_ps();

		// Backward substitution and output loop.
		for (ptrdiff_t j = ctx.dst_width; j > mod(ctx.dst_width, 4); --j) {
			float w_buf[4];

			_mm_storeu_ps(w_buf, w);
			for (ptrdiff_t ii = 0; ii < 4; ++ii) {
				w_buf[ii] = tmp[(j - 1) * 4 + ii] - pu[j - 1] * w_buf[ii];
				dst[(i + ii) * dst_stride + j - 1] = w_buf[ii];
			}
			w = _mm_loadu_ps(w_buf);
		}
		for (ptrdiff_t j = mod(ctx.dst_width, 4); j > 0; j -= 4) {
			__m128 u0, u1, u2, u3;
			__m128 z0, z1, z2, z3;
			__m128 w0, w1, w2, w3;

			z0 = _mm_load_ps(&tmp[(j - 4) * 4]);
			z1 = _mm_load_ps(&tmp[(j - 3) * 4]);
			z2 = _mm_load_ps(&tmp[(j - 2) * 4]);
			z3 = _mm_load_ps(&tmp[(j - 1) * 4]);

			u3 = _mm_load_ps1(&pu[j - 4]);
			w = _mm_mul_ps(u3, w);
			w = _mm_sub_ps(z3, w);
			w3 = w;

			u2 = _mm_load_ps1(&pu[j - 3]);
			w = _mm_mul_ps(u2, w);
			w = _mm_sub_ps(z2, w);
			w2 = w;

			u1 = _mm_load_ps1(&pu[j - 2]);
			w = _mm_mul_ps(u1, w);
			w = _mm_sub_ps(z1, w);
			w1 = w;

			u0 = _mm_load_ps1(&pu[j - 1]);
			w = _mm_mul_ps(u0, w);
			w = _mm_sub_ps(z0, w);
			w0 = w;

			transpose4_ps(w0, w1, w2, w3);

			_mm_store_ps(&dst[i + 0 * dst_stride + j - 4], w0);
			_mm_store_ps(&dst[i + 1 * dst_stride + j - 4], w1);
			_mm_store_ps(&dst[i + 2 * dst_stride + j - 4], w2);
			_mm_store_ps(&dst[i + 3 * dst_stride + j - 4], w3);
		}
	}
}

void filter_plane_v_sse2(const BilinearContext &ctx, const float * RESTRICT src, float * RESTRICT dst,
                         ptrdiff_t src_width, ptrdiff_t src_height, ptrdiff_t src_stride, ptrdiff_t dst_stride)
{
	const float *pc = ctx.lu_c.data();
	const float *pl = ctx.lu_l.data();
	const float *pu = ctx.lu_u.data();

	__m128 w, z, f, c, l, u;

	for (ptrdiff_t i = 0; i < ctx.dst_width; ++i) {
		__m128 coeff0, coeff1, coeff2, coeff3;
		__m128 x0, x1, x2, x3;
		__m128 accum0, accum1;

		const float *row = &ctx.matrix_coefficients.data()[i * ctx.matrix_row_stride];
		ptrdiff_t top = ctx.matrix_row_offsets[i];

		const float *src_ptr0, *src_ptr1, *src_ptr2, *src_ptr3;
		float *dst_ptr = &dst[i * dst_stride];

		// Matrix-vector product.
		for (ptrdiff_t k = 0; k < mod(ctx.matrix_row_size, 4); k += 4) {
			src_ptr0 = &src[(i + 0) * src_stride];
			src_ptr1 = &src[(i + 1) * src_stride];
			src_ptr2 = &src[(i + 2) * src_stride];
			src_ptr3 = &src[(i + 3) * src_stride];

			coeff0 = _mm_load_ss(&row[k]);
			coeff1 = _mm_load_ss(&row[k + 1]);
			coeff2 = _mm_load_ss(&row[k + 2]);
			coeff3 = _mm_load_ss(&row[k + 3]);
				
			for (ptrdiff_t j = 0; j < mod(src_width, 4); j += 4) {
				x0 = _mm_load_ps(&src_ptr0[j]);
				accum0 = _mm_mul_ps(coeff0, x0);

				x1 = _mm_load_ps(&src_ptr1[j]);
				accum1 = _mm_mul_ps(coeff1, x1);

				x2 = _mm_load_ps(&src_ptr2[j]);
				x2 = _mm_mul_ps(coeff2, x2);
				accum0 = _mm_add_ps(accum0, x2);

				x3 = _mm_load_ps(&src_ptr3[j]);
				x3 = _mm_mul_ps(coeff3, x3);
				accum1 = _mm_add_ps(accum1, x3);

				accum0 = _mm_add_ps(accum0, accum1);

				if (k)
					accum0 = _mm_add_ps(accum0, _mm_load_ps(&dst_ptr[j]));

				_mm_store_ps(&dst_ptr[j], accum0);
			}
		}
		if (ctx.matrix_row_size % 4) {
			ptrdiff_t m = ctx.matrix_row_size % 4;
			ptrdiff_t k = ctx.matrix_row_size - m;

			coeff2 = _mm_load_ps1(&row[k + 2]);
			coeff1 = _mm_load_ps1(&row[k + 1]);
			coeff0 = _mm_load_ps1(&row[k + 0]);

			src_ptr2 = &src[(top + k + 2) * src_stride];
			src_ptr1 = &src[(top + k + 1) * src_stride];
			src_ptr0 = &src[(top + k + 0) * src_stride];

			for (ptrdiff_t j = 0; j < mod(src_width, 4); j += 4) {
				accum0 = _mm_setzero_ps();
				accum1 = _mm_setzero_ps();

				switch (m) {
				case 3:
					x2 = _mm_load_ps(&src_ptr2[j]);
					accum0 = _mm_mul_ps(coeff2, x2);
				case 2:
					x1 = _mm_load_ps(&src_ptr1[j]);
					accum1 = _mm_mul_ps(coeff1, x1);
				case 1:
					x0 = _mm_load_ps(&src_ptr0[j]);
					x0 = _mm_mul_ps(coeff0, x0);
					accum0 = _mm_add_ps(accum0, x0);
				}

				accum0 = _mm_add_ps(accum0, accum1);

				if (k)
					accum0 = _mm_add_ps(accum0, _mm_load_ps(&dst_ptr[j]));

				_mm_store_ps(&dst_ptr[j], accum0);
			}
		}

		c = _mm_load_ps1(&pc[i]);
		l = _mm_load_ps1(&pl[i]);

		// Forward substitution.
		for (ptrdiff_t j = 0; j < mod(src_width, 4); j += 4) {
			z = i ? _mm_load_ps(&dst[(i - 1) * dst_stride + j]) : _mm_setzero_ps();
			f = _mm_load_ps(&dst[i * src_stride + j]);

			z = _mm_mul_ps(c, z);
			z = _mm_sub_ps(f, z);
			z = _mm_mul_ps(z, l);

			_mm_store_ps(&dst[i * dst_stride + j], z);
		}
		
		filter_scanline_v_forward(ctx, src, dst, src_stride, dst_stride, i, mod(src_width, 4), src_width);
	}

	// Back substitution.
	for (ptrdiff_t i = ctx.dst_width; i > 0; --i) {
		u = _mm_load_ps1(&pu[i]);

		for (ptrdiff_t j = 0; j < mod(src_width, 4); j += 4) {
			w = i < ctx.dst_width ? _mm_load_ps(&dst[i * dst_stride + j]) : _mm_setzero_ps();
			z = _mm_load_ps(&dst[(i - 1) * dst_stride + j]);
			w = _mm_mul_ps(u, w);
			w = _mm_sub_ps(z, w);

			_mm_store_ps(&dst[(i - 1) * dst_stride + j], w);
		}
		filter_scanline_v_back(ctx, dst, dst_stride, i, mod(src_width, 4), src_width);
	}
}

class UnresizeImplX86 : public UnresizeImpl {
public:
	UnresizeImplX86(const BilinearContext &hcontext, const BilinearContext &vcontext) : UnresizeImpl(hcontext, vcontext)
	{}

	void process_f32_h(const float *src, float *dst, float *tmp,
	                   int src_width, int src_height, int src_stride, int dst_stride) const
	{
		filter_plane_h_sse2(m_hcontext, src, dst, tmp, src_width, src_height, src_stride, dst_stride);
	}

	void process_f32_v(const float *src, float *dst, float *tmp,
	                   int src_width, int src_height, int src_stride, int dst_stride) const
	{
		filter_plane_v_sse2(m_vcontext, src, dst, src_width, src_height, src_stride, dst_stride);
	}
};


} // namespace


UnresizeImpl *create_unresize_impl_sse2(const BilinearContext &hcontext, const BilinearContext &vcontext)
{
	return new UnresizeImplX86{ hcontext, vcontext };
}

} // namespace unresize
} // namespace zimg

#endif // ZIMG_X86
