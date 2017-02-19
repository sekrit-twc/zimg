#ifdef ZIMG_X86

#include <algorithm>
#include <stdexcept>
#include <tuple>
#include <vector>
#include <emmintrin.h>
#include "common/align.h"
#include "common/ccdep.h"
#include "common/checked_int.h"
#include "common/except.h"
#include "common/make_unique.h"
#include "common/pixel.h"

#define HAVE_CPU_SSE2
  #include "common/x86util.h"
#undef HAVE_CPU_SSE2

#include "common/zassert.h"
#include "graph/image_buffer.h"
#include "graph/image_filter.h"
#include "dither_x86.h"
#include "quantize.h"

namespace zimg {
namespace depth {

namespace {

struct error_state {
	float err_left[4];
	float err_top_right[4];
	float err_top[4];
	float err_top_left[4];
};


template <class T>
struct error_diffusion_traits;

template <>
struct error_diffusion_traits<uint8_t> {
	static __m128 load4(const uint8_t *ptr)
	{
		__m128i x = _mm_cvtsi32_si128(*(const uint32_t *)ptr);
		x = _mm_unpacklo_epi8(x, _mm_setzero_si128());
		x = _mm_unpacklo_epi16(x, _mm_setzero_si128());
		return _mm_cvtepi32_ps(x);
	}

	static void store4(uint8_t *ptr, __m128i x)
	{
		x = _mm_packs_epi32(x, x);
		x = _mm_packus_epi16(x, x);
		*(uint32_t *)ptr = _mm_cvtsi128_si32(x);
	}
};

template <>
struct error_diffusion_traits<uint16_t> {
	static __m128 load4(const uint16_t *ptr)
	{
		__m128i x = _mm_loadl_epi64((const __m128i *)ptr);
		x = _mm_unpacklo_epi16(x, _mm_setzero_si128());
		return _mm_cvtepi32_ps(x);
	}

	static void store4(uint16_t *ptr, __m128i x)
	{
		x = mm_packus_epi32(x, x);
		_mm_storel_epi64((__m128i *)ptr, x);
	}
};

template <>
struct error_diffusion_traits<float> {
	static __m128 load4(const float *ptr) { return _mm_loadu_ps(ptr); }
};


inline FORCE_INLINE float max(float x, float y)
{
	return _mm_cvtss_f32(_mm_max_ss(_mm_set_ss(x), _mm_set_ss(y)));
}

inline FORCE_INLINE float min(float x, float y)
{
	return _mm_cvtss_f32(_mm_min_ss(_mm_set_ss(x), _mm_set_ss(y)));
}

inline FORCE_INLINE float extract_hi_ps(__m128 x)
{
	return _mm_cvtss_f32(_mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(x), 12)));
}

inline FORCE_INLINE __m128 rotate_insert_lo(__m128 x, float y)
{
	return _mm_or_ps(_mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(x), 4)), _mm_set_ss(y));
}


template <class T, class U>
void error_diffusion_scalar(const void *src, void *dst, const float * RESTRICT error_top, float * RESTRICT error_cur,
                            float scale, float offset, unsigned bits, unsigned width)
{
	const T *src_p = static_cast<const T *>(src);
	U *dst_p = static_cast<U *>(dst);

	float err_left = error_cur[0];
	float err_top_right = error_top[1 + 1];
	float err_top = error_top[0 + 1];
	float err_top_left = error_top[0];

	for (unsigned j = 0; j < width; ++j) {
		// Error array is padded by one on each side.
		unsigned j_err = j + 1;

		float x = static_cast<float>(src_p[j]) * scale + offset;
		float err;

		err = (err_left * (7.0f / 16.0f) + err_top_right * (3.0f / 16.0f)) +
		      (err_top * (5.0f / 16.0f) + err_top_left * (1.0f / 16.0f));

		x += err;
		x = min(max(x, 0.0f), static_cast<float>(1L << bits) - 1);

		U q = static_cast<U>(_mm_cvt_ss2si(_mm_set_ss(x)));
		err = x - static_cast<float>(q);

		dst_p[j] = q;
		error_cur[j_err] = err;

		err_left = err;
		err_top_left = err_top;
		err_top = err_top_right;
		err_top_right = error_top[j_err + 2];
	}
}

decltype(&error_diffusion_scalar<uint8_t, uint8_t>) select_error_diffusion_scalar_func(PixelType pixel_in, PixelType pixel_out)
{
	if (pixel_in == PixelType::HALF)
		pixel_in = PixelType::FLOAT;

	if (pixel_in == PixelType::BYTE && pixel_out == PixelType::BYTE)
		return error_diffusion_scalar<uint8_t, uint8_t>;
	else if (pixel_in == PixelType::BYTE && pixel_out == PixelType::WORD)
		return error_diffusion_scalar<uint8_t, uint16_t>;
	else if (pixel_in == PixelType::WORD && pixel_out == PixelType::BYTE)
		return error_diffusion_scalar<uint16_t, uint8_t>;
	else if (pixel_in == PixelType::WORD && pixel_out == PixelType::WORD)
		return error_diffusion_scalar<uint16_t, uint16_t>;
	else if (pixel_in == PixelType::FLOAT && pixel_out == PixelType::BYTE)
		return error_diffusion_scalar<float, uint8_t>;
	else if (pixel_in == PixelType::FLOAT && pixel_out == PixelType::WORD)
		return error_diffusion_scalar<float, uint16_t>;
	else
		throw error::InternalError{ "no conversion between pixel types" };
}


inline FORCE_INLINE void error_diffusion_wf_sse2_xiter(__m128 &v, unsigned j, const float *error_top, float *error_cur, const __m128 &max_val,
                                                       const __m128 &err_left_w, const __m128 &err_top_right_w, const __m128 &err_top_w, const __m128 &err_top_left_w,
                                                       __m128 &err_left, __m128 &err_top_right, __m128 &err_top, __m128 &err_top_left)
{
	unsigned j_err = j + 1;

	__m128 x, y, err0, err1;
	__m128i q;

	err0 = _mm_mul_ps(err_left_w, err_left);
	err0 = _mm_add_ps(err0, _mm_mul_ps(err_top_right_w, err_top_right));
	err1 = _mm_mul_ps(err_top_w, err_top);
	err1 = _mm_add_ps(err1, _mm_mul_ps(err_top_left_w, err_top_left));
	err0 = _mm_add_ps(err0, err1);

	x = _mm_add_ps(v, err0);
	x = _mm_max_ps(x, _mm_setzero_ps());
	x = _mm_min_ps(x, max_val);
	q = _mm_cvtps_epi32(x);
	v = _mm_castsi128_ps(q);

	y = _mm_cvtepi32_ps(q);
	err0 = _mm_sub_ps(x, y);

	error_cur[j_err + 0] = extract_hi_ps(err0);

	err_left = err0;
	err_top_left = err_top;
	err_top = err_top_right;
	err_top_right = rotate_insert_lo(err0, error_top[j_err + 6 + 2]);
}

template <class T, class U>
void error_diffusion_wf_sse2(const graph::ImageBuffer<const T> &src, const graph::ImageBuffer<U> &dst, unsigned i,
                             const float *error_top, float *error_cur, error_state *state, float scale, float offset, unsigned bits, unsigned width)
{
	typedef error_diffusion_traits<T> src_traits;
	typedef error_diffusion_traits<U> dst_traits;

	const T *src_p0 = src[i + 0];
	const T *src_p1 = src[i + 1];
	const T *src_p2 = src[i + 2];
	const T *src_p3 = src[i + 3];

	U *dst_p0 = dst[i + 0];
	U *dst_p1 = dst[i + 1];
	U *dst_p2 = dst[i + 2];
	U *dst_p3 = dst[i + 3];

	const __m128 err_left_w = _mm_set_ps1(7.0f / 16.0f);
	const __m128 err_top_right_w = _mm_set_ps1(3.0f / 16.0f);
	const __m128 err_top_w = _mm_set_ps1(5.0f / 16.0f);
	const __m128 err_top_left_w = _mm_set_ps1(1.0f / 16.0f);

	const __m128 scale_ps = _mm_set_ps1(scale);
	const __m128 offset_ps = _mm_set_ps1(offset);

	const __m128 max_val = _mm_set_ps1(static_cast<float>((1UL << bits) - 1));

	__m128 err_left = _mm_load_ps(state->err_left);
	__m128 err_top_right = _mm_load_ps(state->err_top_right);
	__m128 err_top = _mm_load_ps(state->err_top);
	__m128 err_top_left = _mm_load_ps(state->err_top_left);

#define XITER error_diffusion_wf_sse2_xiter
#define XARGS error_top, error_cur, max_val, err_left_w, err_top_right_w, err_top_w, err_top_left_w, err_left, err_top_right, err_top, err_top_left
	for (unsigned j = 0; j < width; j += 4) {
		__m128 v0 = src_traits::load4(src_p0 + j + 6);
		__m128 v1 = src_traits::load4(src_p1 + j + 4);
		__m128 v2 = src_traits::load4(src_p2 + j + 2);
		__m128 v3 = src_traits::load4(src_p3 + j + 0);

		v0 = _mm_add_ps(_mm_mul_ps(v0, scale_ps), offset_ps);
		v1 = _mm_add_ps(_mm_mul_ps(v1, scale_ps), offset_ps);
		v2 = _mm_add_ps(_mm_mul_ps(v2, scale_ps), offset_ps);
		v3 = _mm_add_ps(_mm_mul_ps(v3, scale_ps), offset_ps);

		_MM_TRANSPOSE4_PS(v0, v1, v2, v3);

		XITER(v0, j + 0, XARGS);
		XITER(v1, j + 1, XARGS);
		XITER(v2, j + 2, XARGS);
		XITER(v3, j + 3, XARGS);

		_MM_TRANSPOSE4_PS(v0, v1, v2, v3);

		dst_traits::store4(dst_p0 + j + 6, _mm_castps_si128(v0));
		dst_traits::store4(dst_p1 + j + 4, _mm_castps_si128(v1));
		dst_traits::store4(dst_p2 + j + 2, _mm_castps_si128(v2));
		dst_traits::store4(dst_p3 + j + 0, _mm_castps_si128(v3));
	}
#undef XITER
#undef XARGS

	_mm_store_ps(state->err_left, err_left);
	_mm_store_ps(state->err_top_right, err_top_right);
	_mm_store_ps(state->err_top, err_top);
	_mm_store_ps(state->err_top_left, err_top_left);
}

template <class T, class U>
void error_diffusion_sse2(const graph::ImageBuffer<const void> &src, const graph::ImageBuffer<void> &dst, unsigned i,
                          const float *error_top, float *error_cur, float scale, float offset, unsigned bits, unsigned width)
{
	const graph::ImageBuffer<const T> &src_buf = graph::static_buffer_cast<const T>(src);
	const graph::ImageBuffer<U> &dst_buf = graph::static_buffer_cast<U>(dst);

	error_state state alignas(16) = {};
	float error_tmp[3][12] = {};

	// Prologue.
	error_diffusion_scalar<T, U>(src_buf[i + 0], dst_buf[i + 0], error_top, error_tmp[0], scale, offset, bits, 6);
	error_diffusion_scalar<T, U>(src_buf[i + 1], dst_buf[i + 1], error_tmp[0], error_tmp[1], scale, offset, bits, 4);
	error_diffusion_scalar<T, U>(src_buf[i + 2], dst_buf[i + 2], error_tmp[1], error_tmp[2], scale, offset, bits, 2);

	// Wavefront.
	state.err_left[0] = error_tmp[0][5 + 1];
	state.err_left[1] = error_tmp[1][3 + 1];
	state.err_left[2] = error_tmp[2][1 + 1];
	state.err_left[3] = 0.0f;

	state.err_top_right[0] = error_top[7 + 1];
	state.err_top_right[1] = error_tmp[0][5 + 1];
	state.err_top_right[2] = error_tmp[1][3 + 1];
	state.err_top_right[3] = error_tmp[2][1 + 1];

	state.err_top[0] = error_top[6 + 1];
	state.err_top[1] = error_tmp[0][4 + 1];
	state.err_top[2] = error_tmp[1][2 + 1];
	state.err_top[3] = error_tmp[2][0 + 1];

	state.err_top_left[0] = error_top[5 + 1];
	state.err_top_left[1] = error_tmp[0][3 + 1];
	state.err_top_left[2] = error_tmp[1][1 + 1];
	state.err_top_left[3] = 0.0f;

	unsigned vec_count = floor_n(width - 6, 4);
	error_diffusion_wf_sse2<T, U>(src_buf, dst_buf, i, error_top, error_cur, &state, scale, offset, bits, vec_count);

	error_tmp[0][5 + 1] = state.err_top_right[1];
	error_tmp[0][4 + 1] = state.err_top[1];
	error_tmp[0][3 + 1] = state.err_top_left[1];

	error_tmp[1][3 + 1] = state.err_top_right[2];
	error_tmp[1][2 + 1] = state.err_top[2];
	error_tmp[1][1 + 1] = state.err_top_left[2];

	error_tmp[2][1 + 1] = state.err_top_right[3];
	error_tmp[2][0 + 1] = state.err_top[3];
	error_tmp[2][0] = state.err_top_left[3];

	// Epilogue.
	error_diffusion_scalar<T, U>(src_buf[i + 0] + vec_count + 6, dst_buf[i + 0] + vec_count + 6, error_top + vec_count + 6, error_tmp[0] + 6,
	                             scale, offset, bits, width - vec_count - 6);
	error_diffusion_scalar<T, U>(src_buf[i + 1] + vec_count + 4, dst_buf[i + 1] + vec_count + 4, error_tmp[0] + 4, error_tmp[1] + 4,
	                             scale, offset, bits, width - vec_count - 4);
	error_diffusion_scalar<T, U>(src_buf[i + 2] + vec_count + 2, dst_buf[i + 2] + vec_count + 2, error_tmp[1] + 2, error_tmp[2] + 2,
	                             scale, offset, bits, width - vec_count - 2);
	error_diffusion_scalar<T, U>(src_buf[i + 3] + vec_count + 0, dst_buf[i + 3] + vec_count + 0, error_tmp[2] + 0, error_cur + vec_count + 0,
	                             scale, offset, bits, width - vec_count - 0);
}

decltype(&error_diffusion_sse2<uint8_t, uint8_t>) select_error_diffusion_sse2_func(PixelType pixel_in, PixelType pixel_out)
{
	if (pixel_in == PixelType::HALF)
		pixel_in = PixelType::FLOAT;

	if (pixel_in == PixelType::BYTE && pixel_out == PixelType::BYTE)
		return error_diffusion_sse2<uint8_t, uint8_t>;
	else if (pixel_in == PixelType::BYTE && pixel_out == PixelType::WORD)
		return error_diffusion_sse2<uint8_t, uint16_t>;
	else if (pixel_in == PixelType::WORD && pixel_out == PixelType::BYTE)
		return error_diffusion_sse2<uint16_t, uint8_t>;
	else if (pixel_in == PixelType::WORD && pixel_out == PixelType::WORD)
		return error_diffusion_sse2<uint16_t, uint16_t>;
	else if (pixel_in == PixelType::FLOAT && pixel_out == PixelType::BYTE)
		return error_diffusion_sse2<float, uint8_t>;
	else if (pixel_in == PixelType::FLOAT && pixel_out == PixelType::WORD)
		return error_diffusion_sse2<float, uint16_t>;
	else
		throw error::InternalError{ "no conversion between pixel types" };
}



class ErrorDiffusionSSE2 final : public graph::ImageFilter {
	decltype(&error_diffusion_scalar<uint8_t, uint8_t>) m_scalar_func;
	decltype(&error_diffusion_sse2<uint8_t, uint8_t>) m_sse2_func;
	dither_f16c_func m_f16c;

	PixelType m_pixel_in;
	PixelType m_pixel_out;

	float m_scale;
	float m_offset;
	unsigned m_depth;

	unsigned m_width;
	unsigned m_height;

	void process_scalar(void *ctx, const void *src, void *dst, void *tmp, bool parity) const
	{
		float *ctx_a = reinterpret_cast<float *>(ctx);
		float *ctx_b = reinterpret_cast<float *>(static_cast<unsigned char *>(ctx) + get_context_size() / 2);

		float *error_top = parity ? ctx_a : ctx_b;
		float *error_cur = parity ? ctx_b : ctx_a;

		if (m_f16c) {
			m_f16c(src, tmp, 0, m_width);
			src = tmp;
		}
		m_scalar_func(src, dst, error_top, error_cur, m_scale, m_offset, m_depth, m_width);
	}

	void process_vector(void *ctx, const graph::ImageBuffer<const void> &src, const graph::ImageBuffer<void> &dst, unsigned i) const
	{
		float *ctx_a = reinterpret_cast<float *>(ctx);
		float *ctx_b = reinterpret_cast<float *>(static_cast<unsigned char *>(ctx) + get_context_size() / 2);

		float *error_top = (i / 4) % 2 ? ctx_a : ctx_b;
		float *error_cur = (i / 4) % 2 ? ctx_b : ctx_a;

		m_sse2_func(src, dst, i, error_top, error_cur, m_scale, m_offset, m_depth, m_width);
	}
public:
	ErrorDiffusionSSE2(unsigned width, unsigned height, const PixelFormat &format_in, const PixelFormat &format_out, CPUClass cpu) :
		m_scalar_func{ select_error_diffusion_scalar_func(format_in.type, format_out.type) },
		m_sse2_func{ select_error_diffusion_sse2_func(format_in.type, format_out.type) },
		m_f16c{},
		m_pixel_in{ format_in.type },
		m_pixel_out{ format_out.type },
		m_scale{},
		m_offset{},
		m_depth{ format_out.depth },
		m_width{ width },
		m_height{ height }
	{
		zassert_d(width <= pixel_max_width(format_in.type), "overflow");
		zassert_d(width <= pixel_max_width(format_out.type), "overflow");

		if (!pixel_is_integer(format_out.type))
			throw error::InternalError{ "cannot dither to non-integer format" };
		if (m_pixel_in == PixelType::HALF)
			m_f16c = select_dither_f16c_func_x86(cpu);

		std::tie(m_scale, m_offset) = get_scale_offset(format_in, format_out);
	}

	filter_flags get_flags() const override
	{
		filter_flags flags{};

		flags.has_state = true;
		flags.same_row = true;
		flags.in_place = pixel_size(m_pixel_in) == pixel_size(m_pixel_out);
		flags.entire_row = true;

		return flags;
	}

	image_attributes get_image_attributes() const override
	{
		return{ m_width, m_height, m_pixel_out };
	}

	pair_unsigned get_required_row_range(unsigned i) const override
	{
		return{ i, std::min(i + 4, m_height) };
	}

	pair_unsigned get_required_col_range(unsigned, unsigned) const override
	{
		return{ 0, get_image_attributes().width };
	}

	unsigned get_simultaneous_lines() const override { return 4; }

	unsigned get_max_buffering() const override { return 4; }

	size_t get_context_size() const override
	{
		try {
			checked_size_t size = (static_cast<checked_size_t>(m_width) + 2) * sizeof(float) * 2;
			return size.get();
		} catch (const std::overflow_error &) {
			throw error::OutOfMemory{};
		}
	}

	size_t get_tmp_size(unsigned, unsigned) const override
	{
		try {
			checked_ptrdiff_t size = m_f16c ? ceil_n(static_cast<checked_ptrdiff_t>(m_width) * sizeof(float), ALIGNMENT) * 4 : 0;
			return size.get();
		} catch (const std::overflow_error &) {
			throw error::OutOfMemory{};
		}
	}

	void init_context(void *ctx) const override
	{
		std::fill_n(static_cast<unsigned char *>(ctx), get_context_size(), 0);
	}

	void process(void *ctx, const graph::ImageBuffer<const void> *src, const graph::ImageBuffer<void> *dst, void *tmp, unsigned i, unsigned, unsigned) const override
	{
		if (m_height - i < 4) {
			bool parity = !!((i / 4) % 2);

			for (unsigned ii = i; ii < m_height; ++ii) {
				process_scalar(ctx, (*src)[ii], (*dst)[ii], tmp, parity);
				parity = !parity;
			}
		} else if (m_f16c) {
			float *tmp_p = static_cast<float *>(tmp);
			ptrdiff_t tmp_stride = ceil_n(m_width * sizeof(float), ALIGNMENT);

			for (unsigned n = 0; n < 4; ++n) {
				m_f16c((*src)[i + n], tmp_p + n * (tmp_stride / sizeof(float)), 0, m_width);
			}

			graph::ImageBuffer<const void> tmp_buf{ tmp_p, tmp_stride, 0x03 };
			process_vector(ctx, tmp_buf, *dst, i);
		} else {
			process_vector(ctx, *src, *dst, i);
		}
	}
};

} // namespace


std::unique_ptr<graph::ImageFilter> create_error_diffusion_sse2(unsigned width, unsigned height, const PixelFormat &pixel_in, const PixelFormat &pixel_out, CPUClass cpu)
{
	if (width < 6)
		return nullptr;

	return ztd::make_unique<ErrorDiffusionSSE2>(width, height, pixel_in, pixel_out, cpu);
}

} // namespace depth
} // namespace zimg

#endif // ZIMG_X86
