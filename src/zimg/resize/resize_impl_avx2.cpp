#ifdef ZIMG_X86

#include "common/ccdep.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <immintrin.h>

#include "common/align.h"
#include "common/ccdep.h"
#include "common/checked_int.h"
#include "common/except.h"

#define HAVE_CPU_SSE2
#define HAVE_CPU_AVX
  #include "common/x86util.h"
#undef HAVE_CPU_SSE2
#undef HAVE_CPU_AVX

#include "common/pixel.h"
#include "common/make_unique.h"
#include "graph/image_filter.h"
#include "filter.h"
#include "resize_impl.h"
#include "resize_impl_x86.h"

namespace zimg {
namespace resize {

namespace {

struct f16_traits {
	typedef __m128i vec8_type;
	typedef uint16_t pixel_type;

	static const PixelType type_constant = PixelType::HALF;

	static inline FORCE_INLINE vec8_type load8_raw(const pixel_type *ptr)
	{
		return _mm_load_si128((const __m128i *)ptr);
	}

	static inline FORCE_INLINE void store8_raw(pixel_type *ptr, vec8_type x)
	{
		_mm_store_si128((__m128i *)ptr, x);
	}

	static inline FORCE_INLINE __m256 load8(const pixel_type *ptr)
	{
		return _mm256_cvtph_ps(load8_raw(ptr));
	}

	static inline FORCE_INLINE void store8(pixel_type *ptr, __m256 x)
	{
		store8_raw(ptr, _mm256_cvtps_ph(x, 0));
	}

	static inline FORCE_INLINE void transpose8(vec8_type &x0, vec8_type &x1, vec8_type &x2, vec8_type &x3,
	                                           vec8_type &x4, vec8_type &x5, vec8_type &x6, vec8_type &x7)
	{
		mm_transpose8_epi16(x0, x1, x2, x3, x4, x5, x6, x7);
	}

	static inline FORCE_INLINE void scatter8(pixel_type *dst0, pixel_type *dst1, pixel_type *dst2, pixel_type *dst3,
	                                         pixel_type *dst4, pixel_type *dst5, pixel_type *dst6, pixel_type *dst7, __m256 x)
	{
		mm_scatter_epi16(dst0, dst1, dst2, dst3, dst4, dst5, dst6, dst7, _mm256_cvtps_ph(x, 0));
	}

	static inline FORCE_INLINE void store_left(pixel_type *dst, __m256 x, unsigned count)
	{
		mm_store_left_si128((__m128i *)dst, _mm256_cvtps_ph(x, 0), count * sizeof(uint16_t));
	}

	static inline FORCE_INLINE void store_right(pixel_type *dst, __m256 x, unsigned count)
	{
		mm_store_right_si128((__m128i *)dst, _mm256_cvtps_ph(x, 0), count * sizeof(uint16_t));
	}
};

struct f32_traits {
	typedef __m256 vec8_type;
	typedef float pixel_type;
	static const PixelType type_constant = PixelType::FLOAT;

	static inline FORCE_INLINE vec8_type load8_raw(const pixel_type *ptr)
	{
		return _mm256_load_ps(ptr);
	}

	static inline FORCE_INLINE void store8_raw(pixel_type *ptr, vec8_type x)
	{
		_mm256_store_ps(ptr, x);
	}

	static inline FORCE_INLINE __m256 load8(const pixel_type *ptr)
	{
		return load8_raw(ptr);
	}

	static inline FORCE_INLINE void store8(pixel_type *ptr, __m256 x)
	{
		store8_raw(ptr, x);
	}

	static inline FORCE_INLINE void transpose8(vec8_type &x0, vec8_type &x1, vec8_type &x2, vec8_type &x3,
	                                           vec8_type &x4, vec8_type &x5, vec8_type &x6, vec8_type &x7)
	{
		mm256_transpose8_ps(x0, x1, x2, x3, x4, x5, x6, x7);
	}

	static inline FORCE_INLINE void scatter8(pixel_type *dst0, pixel_type *dst1, pixel_type *dst2, pixel_type *dst3,
	                                         pixel_type *dst4, pixel_type *dst5, pixel_type *dst6, pixel_type *dst7, __m256 x)
	{
		mm256_scatter_ps(dst0, dst1, dst2, dst3, dst4, dst5, dst6, dst7, x);
	}

	static inline FORCE_INLINE void store_left(pixel_type *dst, __m256 x, unsigned count)
	{
		mm256_store_left_ps(dst, x, count * sizeof(float));
	}

	static inline FORCE_INLINE void store_right(pixel_type *dst, __m256 x, unsigned count)
	{
		mm256_store_right_ps(dst, x, count * sizeof(float));
	}
};


template <class Traits, class T>
void transpose_line_8x8(T *dst,
                        const T *src_p0, const T *src_p1, const T *src_p2, const T *src_p3,
                        const T *src_p4, const T *src_p5, const T *src_p6, const T *src_p7,
                        unsigned left, unsigned right)
{
	typedef typename Traits::vec8_type vec8_type;

	for (unsigned j = left; j < right; j += 8) {
		vec8_type x0, x1, x2, x3, x4, x5, x6, x7;

		x0 = Traits::load8_raw(src_p0 + j);
		x1 = Traits::load8_raw(src_p1 + j);
		x2 = Traits::load8_raw(src_p2 + j);
		x3 = Traits::load8_raw(src_p3 + j);
		x4 = Traits::load8_raw(src_p4 + j);
		x5 = Traits::load8_raw(src_p5 + j);
		x6 = Traits::load8_raw(src_p6 + j);
		x7 = Traits::load8_raw(src_p7 + j);

		Traits::transpose8(x0, x1, x2, x3, x4, x5, x6, x7);

		Traits::store8_raw(dst + 0, x0);
		Traits::store8_raw(dst + 8, x1);
		Traits::store8_raw(dst + 16, x2);
		Traits::store8_raw(dst + 24, x3);
		Traits::store8_raw(dst + 32, x4);
		Traits::store8_raw(dst + 40, x5);
		Traits::store8_raw(dst + 48, x6);
		Traits::store8_raw(dst + 56, x7);

		dst += 64;
	}
}

template <class Traits, unsigned FWidth, unsigned Tail>
inline FORCE_INLINE __m256 resize_line8_h_fp_avx2_xiter(unsigned j,
                                                        const unsigned *filter_left, const float * RESTRICT filter_data, unsigned filter_stride, unsigned filter_width,
                                                        const typename Traits::pixel_type * RESTRICT src_ptr, unsigned src_base)
{
	typedef typename Traits::pixel_type pixel_type;

	const float *filter_coeffs = filter_data + j * filter_stride;
	const pixel_type *src_p = src_ptr + (filter_left[j] - src_base) * 8;

	__m256 accum0 = _mm256_setzero_ps();
	__m256 accum1 = _mm256_setzero_ps();
	__m256 x, c, coeffs;

	unsigned k_end = FWidth ? FWidth - Tail : floor_n(filter_width, 4);

	for (unsigned k = 0; k < k_end; k += 4) {
		coeffs = _mm256_broadcast_ps((const __m128 *)(filter_coeffs + k));

		c = _mm256_shuffle_ps(coeffs, coeffs, _MM_SHUFFLE(0, 0, 0, 0));
		x = Traits::load8(src_p + (k + 0) * 8);
		accum0 = _mm256_fmadd_ps(c, x, accum0);

		c = _mm256_shuffle_ps(coeffs, coeffs, _MM_SHUFFLE(1, 1, 1, 1));
		x = Traits::load8(src_p + (k + 1) * 8);
		accum1 = _mm256_fmadd_ps(c, x, accum1);

		c = _mm256_shuffle_ps(coeffs, coeffs, _MM_SHUFFLE(2, 2, 2, 2));
		x = Traits::load8(src_p + (k + 2) * 8);
		accum0 = _mm256_fmadd_ps(c, x, accum0);

		c = _mm256_shuffle_ps(coeffs, coeffs, _MM_SHUFFLE(3, 3, 3, 3));
		x = Traits::load8(src_p + (k + 3) * 8);
		accum1 = _mm256_fmadd_ps(c, x, accum1);
	}

	if (Tail >= 1) {
		coeffs = _mm256_broadcast_ps((const __m128 *)(filter_coeffs + k_end));

		c = _mm256_shuffle_ps(coeffs, coeffs, _MM_SHUFFLE(0, 0, 0, 0));
		x = Traits::load8(src_p + (k_end + 0) * 8);
		accum0 = _mm256_fmadd_ps(c, x, accum0);
	}
	if (Tail >= 2) {
		c = _mm256_shuffle_ps(coeffs, coeffs, _MM_SHUFFLE(1, 1, 1, 1));
		x = Traits::load8(src_p + (k_end + 1) * 8);
		accum1 = _mm256_fmadd_ps(c, x, accum1);
	}
	if (Tail >= 3) {
		c = _mm256_shuffle_ps(coeffs, coeffs, _MM_SHUFFLE(2, 2, 2, 2));
		x = Traits::load8(src_p + (k_end + 2) * 8);
		accum0 = _mm256_fmadd_ps(c, x, accum0);
	}
	if (Tail >= 4) {
		c = _mm256_shuffle_ps(coeffs, coeffs, _MM_SHUFFLE(3, 3, 3, 3));
		x = Traits::load8(src_p + (k_end + 3) * 8);
		accum1 = _mm256_fmadd_ps(c, x, accum1);
	}

	if (!FWidth || FWidth >= 2)
		accum0 = _mm256_add_ps(accum0, accum1);

	return accum0;
}

template <class Traits, unsigned FWidth, unsigned Tail>
void resize_line8_h_fp_avx2(const unsigned *filter_left, const float * RESTRICT filter_data, unsigned filter_stride, unsigned filter_width,
							const typename Traits::pixel_type *src_ptr, typename Traits::pixel_type * const *dst_ptr, unsigned src_base, unsigned left, unsigned right)
{
	typedef typename Traits::pixel_type pixel_type;

	unsigned vec_left = ceil_n(left, 8);
	unsigned vec_right = floor_n(right, 8);

	pixel_type * RESTRICT dst_p0 = dst_ptr[0];
	pixel_type * RESTRICT dst_p1 = dst_ptr[1];
	pixel_type * RESTRICT dst_p2 = dst_ptr[2];
	pixel_type * RESTRICT dst_p3 = dst_ptr[3];
	pixel_type * RESTRICT dst_p4 = dst_ptr[4];
	pixel_type * RESTRICT dst_p5 = dst_ptr[5];
	pixel_type * RESTRICT dst_p6 = dst_ptr[6];
	pixel_type * RESTRICT dst_p7 = dst_ptr[7];
#define XITER resize_line8_h_fp_avx2_xiter<Traits, FWidth, Tail>
#define XARGS filter_left, filter_data, filter_stride, filter_width, src_ptr, src_base
	for (unsigned j = left; j < vec_left; ++j) {
		__m256 x = XITER(j, XARGS);
		Traits::scatter8(dst_p0 + j, dst_p1 + j, dst_p2 + j, dst_p3 + j, dst_p4 + j, dst_p5 + j, dst_p6 + j, dst_p7 + j, x);
	}

	for (unsigned j = vec_left; j < vec_right; j += 8) {
		__m256 x0, x1, x2, x3, x4, x5, x6, x7;

		x0 = XITER(j + 0, XARGS);
		x1 = XITER(j + 1, XARGS);
		x2 = XITER(j + 2, XARGS);
		x3 = XITER(j + 3, XARGS);
		x4 = XITER(j + 4, XARGS);
		x5 = XITER(j + 5, XARGS);
		x6 = XITER(j + 6, XARGS);
		x7 = XITER(j + 7, XARGS);

		mm256_transpose8_ps(x0, x1, x2, x3, x4, x5, x6, x7);

		Traits::store8(dst_p0 + j, x0);
		Traits::store8(dst_p1 + j, x1);
		Traits::store8(dst_p2 + j, x2);
		Traits::store8(dst_p3 + j, x3);
		Traits::store8(dst_p4 + j, x4);
		Traits::store8(dst_p5 + j, x5);
		Traits::store8(dst_p6 + j, x6);
		Traits::store8(dst_p7 + j, x7);
	}

	for (unsigned j = vec_right; j < right; ++j) {
		__m256 x = XITER(j, XARGS);
		Traits::scatter8(dst_p0 + j, dst_p1 + j, dst_p2 + j, dst_p3 + j, dst_p4 + j, dst_p5 + j, dst_p6 + j, dst_p7 + j, x);
	}
#undef XITER
#undef XARGS
}

template <class Traits>
struct resize_line8_h_fp_avx2_jt {
	typedef decltype(&resize_line8_h_fp_avx2<Traits, 0, 0>) func_type;

	static const func_type small[8];
	static const func_type large[4];
};

template <class Traits>
const typename resize_line8_h_fp_avx2_jt<Traits>::func_type resize_line8_h_fp_avx2_jt<Traits>::small[8] = {
	resize_line8_h_fp_avx2<Traits, 1, 1>,
	resize_line8_h_fp_avx2<Traits, 2, 2>,
	resize_line8_h_fp_avx2<Traits, 3, 3>,
	resize_line8_h_fp_avx2<Traits, 4, 4>,
	resize_line8_h_fp_avx2<Traits, 5, 1>,
	resize_line8_h_fp_avx2<Traits, 6, 2>,
	resize_line8_h_fp_avx2<Traits, 7, 3>,
	resize_line8_h_fp_avx2<Traits, 8, 4>
};

template <class Traits>
const typename resize_line8_h_fp_avx2_jt<Traits>::func_type resize_line8_h_fp_avx2_jt<Traits>::large[4] = {
	resize_line8_h_fp_avx2<Traits, 0, 0>,
	resize_line8_h_fp_avx2<Traits, 0, 1>,
	resize_line8_h_fp_avx2<Traits, 0, 2>,
	resize_line8_h_fp_avx2<Traits, 0, 3>
};

template <class Traits, unsigned N, bool UpdateAccum, class T = typename Traits::pixel_type>
inline FORCE_INLINE __m256 resize_line_v_fp_avx2_xiter(unsigned j,
                                                       const T * RESTRICT src_p0, const T * RESTRICT src_p1,
                                                       const T * RESTRICT src_p2, const T * RESTRICT src_p3,
                                                       const T * RESTRICT src_p4, const T * RESTRICT src_p5,
                                                       const T * RESTRICT src_p6, const T * RESTRICT src_p7, T * RESTRICT dst_p,
                                                       const __m256 &c0, const __m256 &c1, const __m256 &c2, const __m256 &c3,
                                                       const __m256 &c4, const __m256 &c5, const __m256 &c6, const __m256 &c7)
{
	typedef typename Traits::pixel_type pixel_type;
	static_assert(std::is_same<pixel_type, T>::value, "must not specify T");

	__m256 accum0 = _mm256_setzero_ps();
	__m256 accum1 = _mm256_setzero_ps();
	__m256 x;

	if (N >= 0) {
		x = Traits::load8(src_p0 + j);
		accum0 = UpdateAccum ? _mm256_fmadd_ps(c0, x, Traits::load8(dst_p + j)) : _mm256_mul_ps(c0, x);
	}
	if (N >= 1) {
		x = Traits::load8(src_p1 + j);
		accum1 = _mm256_mul_ps(c1, x);
	}
	if (N >= 2) {
		x = Traits::load8(src_p2 + j);
		accum0 = _mm256_fmadd_ps(c2, x, accum0);
	}
	if (N >= 3) {
		x = Traits::load8(src_p3 + j);
		accum1 = _mm256_fmadd_ps(c3, x, accum1);
	}
	if (N >= 4) {
		x = Traits::load8(src_p4 + j);
		accum0 = _mm256_fmadd_ps(c4, x, accum0);
	}
	if (N >= 5) {
		x = Traits::load8(src_p5 + j);
		accum1 = _mm256_fmadd_ps(c5, x, accum1);
	}
	if (N >= 6) {
		x = Traits::load8(src_p6 + j);
		accum0 = _mm256_fmadd_ps(c6, x, accum0);
	}
	if (N >= 7) {
		x = Traits::load8(src_p7 + j);
		accum1 = _mm256_fmadd_ps(c7, x, accum1);
	}

	accum0 = (N >= 1) ? _mm256_add_ps(accum0, accum1) : accum0;
	return accum0;
}

template <class Traits, unsigned N, bool UpdateAccum>
void resize_line_v_fp_avx2(const float *filter_data, const typename Traits::pixel_type * const *src_lines, typename Traits::pixel_type *dst, unsigned left, unsigned right)
{
	typedef typename Traits::pixel_type pixel_type;

	const pixel_type * RESTRICT src_p0 = src_lines[0];
	const pixel_type * RESTRICT src_p1 = src_lines[1];
	const pixel_type * RESTRICT src_p2 = src_lines[2];
	const pixel_type * RESTRICT src_p3 = src_lines[3];
	const pixel_type * RESTRICT src_p4 = src_lines[4];
	const pixel_type * RESTRICT src_p5 = src_lines[5];
	const pixel_type * RESTRICT src_p6 = src_lines[6];
	const pixel_type * RESTRICT src_p7 = src_lines[7];
	pixel_type * RESTRICT dst_p = dst;

	unsigned vec_left = ceil_n(left, 8);
	unsigned vec_right = floor_n(right, 8);

	const __m256 c0 = _mm256_broadcast_ss(filter_data + 0);
	const __m256 c1 = _mm256_broadcast_ss(filter_data + 1);
	const __m256 c2 = _mm256_broadcast_ss(filter_data + 2);
	const __m256 c3 = _mm256_broadcast_ss(filter_data + 3);
	const __m256 c4 = _mm256_broadcast_ss(filter_data + 4);
	const __m256 c5 = _mm256_broadcast_ss(filter_data + 5);
	const __m256 c6 = _mm256_broadcast_ss(filter_data + 6);
	const __m256 c7 = _mm256_broadcast_ss(filter_data + 7);

	__m256 accum;

#define XITER resize_line_v_fp_avx2_xiter<Traits, N, UpdateAccum>
#define XARGS src_p0, src_p1, src_p2, src_p3, src_p4, src_p5, src_p6, src_p7, dst_p, c0, c1, c2, c3, c4, c5, c6, c7
	if (left != vec_left) {
		accum = XITER(vec_left - 8, XARGS);
		Traits::store_left(dst_p + vec_left - 8, accum, vec_left - left);
	}

	for (unsigned j = vec_left; j < vec_right; j += 8) {
		accum = XITER(j, XARGS);
		Traits::store8(dst_p + j, accum);
	}

	if (right != vec_right) {
		accum = XITER(vec_right, XARGS);
		Traits::store_right(dst_p + vec_right, accum, right - vec_right);
	}
#undef XITER
#undef XARGS
}

template <class Traits>
struct resize_line_v_fp_avx2_jt {
	typedef decltype(&resize_line_v_fp_avx2<Traits, 0, false>) func_type;

	static const func_type table_a[8];
	static const func_type table_b[8];
};

template <class Traits>
const typename resize_line_v_fp_avx2_jt<Traits>::func_type resize_line_v_fp_avx2_jt<Traits>::table_a[8] = {
	resize_line_v_fp_avx2<Traits, 0, false>,
	resize_line_v_fp_avx2<Traits, 1, false>,
	resize_line_v_fp_avx2<Traits, 2, false>,
	resize_line_v_fp_avx2<Traits, 3, false>,
	resize_line_v_fp_avx2<Traits, 4, false>,
	resize_line_v_fp_avx2<Traits, 5, false>,
	resize_line_v_fp_avx2<Traits, 6, false>,
	resize_line_v_fp_avx2<Traits, 7, false>,
};

template <class Traits>
const typename resize_line_v_fp_avx2_jt<Traits>::func_type resize_line_v_fp_avx2_jt<Traits>::table_b[8] = {
	resize_line_v_fp_avx2<Traits, 0, true>,
	resize_line_v_fp_avx2<Traits, 1, true>,
	resize_line_v_fp_avx2<Traits, 2, true>,
	resize_line_v_fp_avx2<Traits, 3, true>,
	resize_line_v_fp_avx2<Traits, 4, true>,
	resize_line_v_fp_avx2<Traits, 5, true>,
	resize_line_v_fp_avx2<Traits, 6, true>,
	resize_line_v_fp_avx2<Traits, 7, true>,
};


template <class Traits>
class ResizeImplH_FP_AVX2 final : public ResizeImplH {
	typedef typename Traits::pixel_type pixel_type;
	typedef typename resize_line8_h_fp_avx2_jt<Traits>::func_type func_type;

	func_type m_func;
public:
	ResizeImplH_FP_AVX2(const FilterContext &filter, unsigned height) :
		ResizeImplH(filter, image_attributes{ filter.filter_rows, height, Traits::type_constant }),
		m_func{}
	{
		if (filter.filter_width <= 8)
			m_func = resize_line8_h_fp_avx2_jt<Traits>::small[filter.filter_width - 1];
		else
			m_func = resize_line8_h_fp_avx2_jt<Traits>::large[filter.filter_width % 4];
	}

	unsigned get_simultaneous_lines() const override { return 8; }

	size_t get_tmp_size(unsigned left, unsigned right) const override
	{
		auto range = get_required_col_range(left, right);

		try {
			checked_size_t size = (static_cast<checked_size_t>(range.second) - floor_n(range.first, 8) + 8) * sizeof(pixel_type) * 8;
			return size.get();
		} catch (const std::overflow_error &) {
			throw error::OutOfMemory{};
		}
	}

	void process(void *, const graph::ImageBuffer<const void> *src, const graph::ImageBuffer<void> *dst, void *tmp, unsigned i, unsigned left, unsigned right) const override
	{
		auto src_buf = graph::static_buffer_cast<const pixel_type>(*src);
		auto dst_buf = graph::static_buffer_cast<pixel_type>(*dst);
		auto range = get_required_col_range(left, right);

		const pixel_type *src_ptr[8] = { 0 };
		pixel_type *dst_ptr[8] = { 0 };
		pixel_type *transpose_buf = static_cast<pixel_type *>(tmp);
		unsigned height = get_image_attributes().height;

		src_ptr[0] = src_buf[std::min(i + 0, height - 1)];
		src_ptr[1] = src_buf[std::min(i + 1, height - 1)];
		src_ptr[2] = src_buf[std::min(i + 2, height - 1)];
		src_ptr[3] = src_buf[std::min(i + 3, height - 1)];
		src_ptr[4] = src_buf[std::min(i + 4, height - 1)];
		src_ptr[5] = src_buf[std::min(i + 5, height - 1)];
		src_ptr[6] = src_buf[std::min(i + 6, height - 1)];
		src_ptr[7] = src_buf[std::min(i + 7, height - 1)];

		transpose_line_8x8<Traits>(transpose_buf, src_ptr[0], src_ptr[1], src_ptr[2], src_ptr[3], src_ptr[4], src_ptr[5], src_ptr[6], src_ptr[7],
		                           floor_n(range.first, 8), ceil_n(range.second, 8));

		dst_ptr[0] = dst_buf[std::min(i + 0, height - 1)];
		dst_ptr[1] = dst_buf[std::min(i + 1, height - 1)];
		dst_ptr[2] = dst_buf[std::min(i + 2, height - 1)];
		dst_ptr[3] = dst_buf[std::min(i + 3, height - 1)];
		dst_ptr[4] = dst_buf[std::min(i + 4, height - 1)];
		dst_ptr[5] = dst_buf[std::min(i + 5, height - 1)];
		dst_ptr[6] = dst_buf[std::min(i + 6, height - 1)];
		dst_ptr[7] = dst_buf[std::min(i + 7, height - 1)];

		m_func(m_filter.left.data(), m_filter.data.data(), m_filter.stride, m_filter.filter_width,
		       transpose_buf, dst_ptr, floor_n(range.first, 8), left, right);
	}
};

template <class Traits>
class ResizeImplV_FP_AVX2 final : public ResizeImplV {
	typedef typename Traits::pixel_type pixel_type;
public:
	ResizeImplV_FP_AVX2(const FilterContext &filter, unsigned width) :
		ResizeImplV(filter, image_attributes{ width, filter.filter_rows, Traits::type_constant })
	{}

	void process(void *, const graph::ImageBuffer<const void> *src, const graph::ImageBuffer<void> *dst, void *, unsigned i, unsigned left, unsigned right) const override
	{
		auto src_buf = graph::static_buffer_cast<const pixel_type>(*src);
		auto dst_buf = graph::static_buffer_cast<pixel_type>(*dst);

		const float *filter_data = m_filter.data.data() + i * m_filter.stride;
		unsigned filter_width = m_filter.filter_width;
		unsigned src_height = m_filter.input_width;

		const pixel_type *src_lines[8] = { 0 };
		pixel_type *dst_line = dst_buf[i];

		for (unsigned k = 0; k < filter_width; k += 8) {
			unsigned taps_remain = std::min(filter_width - k, 8U);
			unsigned top = m_filter.left[i] + k;

			src_lines[0] = src_buf[std::min(top + 0, src_height - 1)];
			src_lines[1] = src_buf[std::min(top + 1, src_height - 1)];
			src_lines[2] = src_buf[std::min(top + 2, src_height - 1)];
			src_lines[3] = src_buf[std::min(top + 3, src_height - 1)];
			src_lines[4] = src_buf[std::min(top + 4, src_height - 1)];
			src_lines[5] = src_buf[std::min(top + 5, src_height - 1)];
			src_lines[6] = src_buf[std::min(top + 6, src_height - 1)];
			src_lines[7] = src_buf[std::min(top + 7, src_height - 1)];

			if (k == 0)
				resize_line_v_fp_avx2_jt<Traits>::table_a[taps_remain - 1](filter_data + k, src_lines, dst_line, left, right);
			else
				resize_line_v_fp_avx2_jt<Traits>::table_b[taps_remain - 1](filter_data + k, src_lines, dst_line, left, right);
		}
	}
};

} // namespace


std::unique_ptr<graph::ImageFilter> create_resize_impl_h_avx2(const FilterContext &context, unsigned height, PixelType type, unsigned depth)
{
	std::unique_ptr<graph::ImageFilter> ret;

	if (type == PixelType::HALF)
		ret = ztd::make_unique<ResizeImplH_FP_AVX2<f16_traits>>(context, height);
	else if (type == PixelType::FLOAT)
		ret = ztd::make_unique<ResizeImplH_FP_AVX2<f32_traits>>(context, height);

	return ret;
}

std::unique_ptr<graph::ImageFilter> create_resize_impl_v_avx2(const FilterContext &context, unsigned width, PixelType type, unsigned depth)
{
	std::unique_ptr<graph::ImageFilter> ret;

	if (type == PixelType::HALF)
		ret = ztd::make_unique<ResizeImplV_FP_AVX2<f16_traits>>(context, width);
	else if (type == PixelType::FLOAT)
		ret = ztd::make_unique<ResizeImplV_FP_AVX2<f32_traits>>(context, width);

	return ret;
}

} // namespace resize
} // namespace zimg

#endif // ZIMG_X86
