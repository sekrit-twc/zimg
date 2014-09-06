#ifdef RESIZE_X86

#include <intrin.h>
#include "osdep.h"
#include "resize_impl.h"

namespace resize {;

namespace {;

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

template <bool DoLoop>
void filter_plane_h_avx(const EvaluatedFilter &filter, const float * RESTRICT src, float * RESTRICT dst,
                        int src_width, int src_height, int src_stride, int dst_stride)
{
	for (int i = 0; i < mod(src_height, 8); i += 8) {
		int j;

		for (j = 0; j < mod(filter.height(), 8); ++j) {
			__m256 x0, x1, x2, x3, x4, x5, x6, x7;
			__m256 accum = _mm256_setzero_ps();

			int dst_i = i + j % 8;
			int dst_j = mod(j, 8);

			const float *filter_row = filter.data() + j * filter.stride();
			int left = filter.left()[j];

			if (left + filter.stride() > src_width)
				break;

			for (int k = 0; k < (DoLoop ? filter.width() : 8); k += 8) {
				__m256 coeff = _mm256_load_ps(filter_row + k);

				x0 = _mm256_loadu_ps(src + (i + 0) * src_stride + left + k);
				x0 = _mm256_mul_ps(coeff, x0);
				
				x1 = _mm256_loadu_ps(src + (i + 1) * src_stride + left + k);
				x1 = _mm256_mul_ps(coeff, x1);

				x2 = _mm256_loadu_ps(src + (i + 2) * src_stride + left + k);
				x2 = _mm256_mul_ps(coeff, x2);

				x3 = _mm256_loadu_ps(src + (i + 3) * src_stride + left + k);
				x3 = _mm256_mul_ps(coeff, x3);

				x4 = _mm256_loadu_ps(src + (i + 4) * src_stride + left + k);
				x4 = _mm256_mul_ps(coeff, x4);

				x5 = _mm256_loadu_ps(src + (i + 5) * src_stride + left + k);
				x5 = _mm256_mul_ps(coeff, x5);

				x6 = _mm256_loadu_ps(src + (i + 6) * src_stride + left + k);
				x6 = _mm256_mul_ps(coeff, x6);

				x7 = _mm256_loadu_ps(src + (i + 7) * src_stride + left + k);
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
			_mm256_store_ps(dst + dst_i * dst_stride + dst_j, accum);

			if (j % 8 == 7) {
				x0 = _mm256_load_ps(dst + (i + 0) * dst_stride + dst_j);
				x1 = _mm256_load_ps(dst + (i + 1) * dst_stride + dst_j);
				x2 = _mm256_load_ps(dst + (i + 2) * dst_stride + dst_j);
				x3 = _mm256_load_ps(dst + (i + 3) * dst_stride + dst_j);
				x4 = _mm256_load_ps(dst + (i + 4) * dst_stride + dst_j);
				x5 = _mm256_load_ps(dst + (i + 5) * dst_stride + dst_j);
				x6 = _mm256_load_ps(dst + (i + 6) * dst_stride + dst_j);
				x7 = _mm256_load_ps(dst + (i + 7) * dst_stride + dst_j);

				transpose8_ps_avx(x0, x1, x2, x3, x4, x5, x6, x7);

				_mm256_store_ps(dst + (i + 0) * dst_stride + dst_j, x0);
				_mm256_store_ps(dst + (i + 1) * dst_stride + dst_j, x1);
				_mm256_store_ps(dst + (i + 2) * dst_stride + dst_j, x2);
				_mm256_store_ps(dst + (i + 3) * dst_stride + dst_j, x3);
				_mm256_store_ps(dst + (i + 4) * dst_stride + dst_j, x4);
				_mm256_store_ps(dst + (i + 5) * dst_stride + dst_j, x5);
				_mm256_store_ps(dst + (i + 6) * dst_stride + dst_j, x6);
				_mm256_store_ps(dst + (i + 7) * dst_stride + dst_j, x7);
			}
		}

		for (int ii = i; ii < i + 8; ++ii) {
			for (j = mod(j, 8); j < filter.height(); ++j) {
				int left = filter.left()[j];
				float accum = 0.f;

				for (int k = 0; k < filter.width(); ++k) {
					float coeff = filter.data()[j * filter.stride() + k];
					float x = src[ii * src_stride + left + k];
					accum += coeff * x;
				}
				dst[ii * dst_stride + j] = accum;
			}
		}
	}
	for (int i = mod(src_height, 8); i < src_height; ++i) {
		for (int j = 0; j < filter.height(); ++j) {
			int left = filter.left()[j];
			float accum = 0.f;

			for (int k = 0; k < filter.width(); ++k) {
				float coeff = filter.data()[j * filter.stride() + k];
				float x = src[i * src_stride + left + k];
				accum += coeff * x;
			}
			dst[i * dst_stride + j] = accum;
		}
	}
}

FORCE_INLINE void filter_plane_v_avx(const EvaluatedFilter &filter, const float * RESTRICT src, float * RESTRICT dst,
                                     int src_width, int src_height, int src_stride, int dst_stride)
{
	for (int i = 0; i < filter.height(); ++i) {
		__m256 coeff0, coeff1, coeff2, coeff3, coeff4, coeff5, coeff6, coeff7;
		__m256 x0, x1, x2, x3, x4, x5, x6, x7;
		__m256 accum0, accum1, accum2, accum3;

		const float *src_ptr0, *src_ptr1, *src_ptr2, *src_ptr3, *src_ptr4, *src_ptr5, *src_ptr6, *src_ptr7;
		float *dst_ptr = dst + i * dst_stride;

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
				x0 = _mm256_load_ps(src_ptr0 + j);
				accum0 = _mm256_mul_ps(coeff0, x0);

				x1 = _mm256_load_ps(src_ptr1 + j);
				accum1 = _mm256_mul_ps(coeff1, x1);

				x2 = _mm256_load_ps(src_ptr2 + j);
				accum2 = _mm256_mul_ps(coeff2, x2);

				x3 = _mm256_load_ps(src_ptr3 + j);
				accum3 = _mm256_mul_ps(coeff3, x3);

				x4 = _mm256_load_ps(src_ptr4 + j);
				accum0 = _mm256_fmadd_ps(coeff4, x4, accum0);

				x5 = _mm256_load_ps(src_ptr5 + j);
				accum1 = _mm256_fmadd_ps(coeff5, x5, accum1);

				x6 = _mm256_load_ps(src_ptr6 + j);
				accum2 = _mm256_fmadd_ps(coeff6, x6, accum2);

				x7 = _mm256_load_ps(src_ptr7 + j);
				accum3 = _mm256_fmadd_ps(coeff7, x7, accum3);

				accum0 = _mm256_add_ps(accum0, accum2);
				accum1 = _mm256_add_ps(accum1, accum3);
				accum0 = _mm256_add_ps(accum0, accum1);

				if (k)
					accum0 = _mm256_add_ps(accum0, _mm256_load_ps(dst_ptr + j));

				_mm256_store_ps(dst_ptr + j, accum0);
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
					x6 = _mm256_load_ps(src_ptr6 + j);
					accum2 = _mm256_mul_ps(coeff6, x6);
				case 6:
					x5 = _mm256_load_ps(src_ptr5 + j);
					accum1 = _mm256_mul_ps(coeff5, x5);
				case 5:
					x4 = _mm256_load_ps(src_ptr4 + j);
					accum0 = _mm256_mul_ps(coeff4, x4);
				case 4:
					x3 = _mm256_load_ps(src_ptr3 + j);
					accum3 = _mm256_mul_ps(coeff3, x3);
				case 3:
					x2 = _mm256_load_ps(src_ptr2 + j);
					accum2 = _mm256_fmadd_ps(coeff2, x2, accum2);
				case 2:
					x1 = _mm256_load_ps(src_ptr1 + j);
					accum1 = _mm256_fmadd_ps(coeff1, x1, accum1);
				case 1:
					x0 = _mm256_load_ps(src_ptr0 + j);
					accum0 = _mm256_fmadd_ps(coeff0, x0, accum0);
				}

				accum0 = _mm256_add_ps(accum0, accum2);
				accum1 = _mm256_add_ps(accum1, accum3);
				accum0 = _mm256_add_ps(accum0, accum1);

				if (k)
					accum0 = _mm256_add_ps(accum0, _mm256_load_ps(dst_ptr + j));

				_mm256_store_ps(dst_ptr + j, accum0);
			}
		}

		for (int j = mod(src_width, 8); j < src_width; ++j) {
			int top = filter.left()[i];
			float accum = 0.f;

			for (int k = 0; k < filter.width(); ++k) {
				float coeff = filter.data()[i * filter.stride() + k];
				float x = src[(top + k) * src_stride + j];
				accum += coeff * x;
			}
			dst[i * dst_stride + j] = accum;
		}
	}
}

} // namespace


ResizeImplX86::ResizeImplX86(const EvaluatedFilter &filter_h, const EvaluatedFilter &filter_v) :
	ResizeImpl(filter_h, filter_v)
{
}

void ResizeImplX86::process_h(const float * RESTRICT src, float * RESTRICT dst, float * RESTRICT tmp,
                              int src_width, int src_height, int src_stride, int dst_stride) const
{
	if (m_filter_h.width() >= 8)
		filter_plane_h_avx<true>(m_filter_h, src, dst, src_width, src_height, src_stride, dst_stride);
	else
		filter_plane_h_avx<false>(m_filter_h, src, dst, src_width, src_height, src_stride, dst_stride);
}

void ResizeImplX86::process_v(const float * RESTRICT src, float * RESTRICT dst, float * RESTRICT tmp,
                              int src_width, int src_height, int src_stride, int dst_stride) const
{
	filter_plane_v_avx(m_filter_v, src, dst, src_width, src_height, src_stride, dst_stride);
}

} // namespace resize;

#include "resize_impl.h"

#endif // RESIZE_X86
