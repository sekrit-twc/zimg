#ifdef ZIMG_X86

#include <algorithm>
#include <cstdint>
#include <emmintrin.h>
#include "Common/align.h"
#include "Common/linebuffer.h"
#include "Common/osdep.h"
#include "Common/pixel.h"
#include "resize_impl2.h"
#include "resize_impl2_x86.h"

namespace zimg {;
namespace resize {;

namespace {;

inline FORCE_INLINE void mm_store_left_epi16(uint16_t *dst, __m128i x, unsigned count)
{
	switch (count - 1) {
	case 6:
		dst[1] = _mm_extract_epi16(x, 1);
	case 5:
		dst[2] = _mm_extract_epi16(x, 2);
	case 4:
		dst[3] = _mm_extract_epi16(x, 3);
	case 3:
		dst[4] = _mm_extract_epi16(x, 4);
	case 2:
		dst[5] = _mm_extract_epi16(x, 5);
	case 1:
		dst[6] = _mm_extract_epi16(x, 6);
	case 0:
		dst[7] = _mm_extract_epi16(x, 7);
	}
}

inline FORCE_INLINE void mm_store_right_epi16(uint16_t *dst, __m128i x, unsigned count)
{
	switch (count - 1) {
	case 6:
		dst[6] = _mm_extract_epi16(x, 6);
	case 5:
		dst[5] = _mm_extract_epi16(x, 5);
	case 4:
		dst[4] = _mm_extract_epi16(x, 4);
	case 3:
		dst[3] = _mm_extract_epi16(x, 3);
	case 2:
		dst[2] = _mm_extract_epi16(x, 2);
	case 1:
		dst[1] = _mm_extract_epi16(x, 1);
	case 0:
		dst[0] = _mm_extract_epi16(x, 0);
	}
}

inline FORCE_INLINE __m128i export_i30_u16(__m128i lo, __m128i hi, uint16_t limit)
{
	const __m128i round = _mm_set1_epi32(1 << 13);
	const __m128i i16_min = _mm_set1_epi16(INT16_MIN);
	const __m128i lim = _mm_set1_epi16(limit + INT16_MIN);

	lo = _mm_add_epi32(lo, round);
	hi = _mm_add_epi32(hi, round);

	lo = _mm_srai_epi32(lo, 14);
	hi = _mm_srai_epi32(hi, 14);

	lo = _mm_packs_epi32(lo, hi);
	lo = _mm_min_epi16(lo, lim);
	lo = _mm_sub_epi16(lo, i16_min);
	return lo;
}


template <unsigned N, bool ReadAccum, bool WriteToAccum>
inline FORCE_INLINE __m128i resize_line_v_u16_sse2_xiter(unsigned j, unsigned accum_base,
                                                         const uint16_t * RESTRICT src_p0, const uint16_t * RESTRICT src_p1, const uint16_t * RESTRICT src_p2, const uint16_t * RESTRICT src_p3,
                                                         const uint16_t * RESTRICT src_p4, const uint16_t * RESTRICT src_p5, const uint16_t * RESTRICT src_p6, const uint16_t * RESTRICT src_p7,
                                                         const uint32_t *accum_p, const __m128i &c01, const __m128i &c23, const __m128i &c45, const __m128i &c67, uint16_t limit)
{
	const __m128i i16_min = _mm_set1_epi16(INT16_MIN);

	__m128i accum_lo = _mm_setzero_si128();
	__m128i accum_hi = _mm_setzero_si128();
	__m128i x0, x1, xl, xh;

	if (N >= 0) {
		x0 = _mm_load_si128((const __m128i *)(src_p0 + j));
		x1 = _mm_load_si128((const __m128i *)(src_p1 + j));
		x0 = _mm_add_epi16(x0, i16_min);
		x1 = _mm_add_epi16(x1, i16_min);

		xl = _mm_unpacklo_epi16(x0, x1);
		xh = _mm_unpackhi_epi16(x0, x1);
		xl = _mm_madd_epi16(c01, xl);
		xh = _mm_madd_epi16(c01, xh);

		if (ReadAccum) {
			accum_lo = _mm_add_epi32(_mm_load_si128((const __m128i *)(accum_p + j - accum_base + 0)), xl);
			accum_hi = _mm_add_epi32(_mm_load_si128((const __m128i *)(accum_p + j - accum_base + 4)), xh);
		} else {
			accum_lo = xl;
			accum_hi = xh;
		}
	}
	if (N >= 2) {
		x0 = _mm_load_si128((const __m128i *)(src_p2 + j));
		x1 = _mm_load_si128((const __m128i *)(src_p3 + j));
		x0 = _mm_add_epi16(x0, i16_min);
		x1 = _mm_add_epi16(x1, i16_min);

		xl = _mm_unpacklo_epi16(x0, x1);
		xh = _mm_unpackhi_epi16(x0, x1);
		xl = _mm_madd_epi16(c23, xl);
		xh = _mm_madd_epi16(c23, xh);

		accum_lo = _mm_add_epi32(accum_lo, xl);
		accum_hi = _mm_add_epi32(accum_hi, xh);
	}
	if (N >= 4) {
		x0 = _mm_load_si128((const __m128i *)(src_p4 + j));
		x1 = _mm_load_si128((const __m128i *)(src_p5 + j));
		x0 = _mm_add_epi16(x0, i16_min);
		x1 = _mm_add_epi16(x1, i16_min);

		xl = _mm_unpacklo_epi16(x0, x1);
		xh = _mm_unpackhi_epi16(x0, x1);
		xl = _mm_madd_epi16(c45, xl);
		xh = _mm_madd_epi16(c45, xh);

		accum_lo = _mm_add_epi32(accum_lo, xl);
		accum_hi = _mm_add_epi32(accum_hi, xh);
	}
	if (N >= 6) {
		x0 = _mm_load_si128((const __m128i *)(src_p6 + j));
		x1 = _mm_load_si128((const __m128i *)(src_p7 + j));
		x0 = _mm_add_epi16(x0, i16_min);
		x1 = _mm_add_epi16(x1, i16_min);

		xl = _mm_unpacklo_epi16(x0, x1);
		xh = _mm_unpackhi_epi16(x0, x1);
		xl = _mm_madd_epi16(c67, xl);
		xh = _mm_madd_epi16(c67, xh);

		accum_lo = _mm_add_epi32(accum_lo, xl);
		accum_hi = _mm_add_epi32(accum_hi, xh);
	}

	if (WriteToAccum) {
		_mm_store_si128((__m128i *)(accum_p + j - accum_base + 0), accum_lo);
		_mm_store_si128((__m128i *)(accum_p + j - accum_base + 4), accum_hi);
		return _mm_setzero_si128();
	} else {
		return export_i30_u16(accum_lo, accum_hi, limit);
	}
}

template <unsigned N, bool ReadAccum, bool WriteToAccum>
void resize_line_v_u16_sse2(const int16_t *filter_data, const uint16_t * const *src_lines, uint16_t *dst, uint32_t *accum, unsigned left, unsigned right, uint16_t limit)
{
	const uint16_t * RESTRICT src_p0 = src_lines[0];
	const uint16_t * RESTRICT src_p1 = src_lines[1];
	const uint16_t * RESTRICT src_p2 = src_lines[2];
	const uint16_t * RESTRICT src_p3 = src_lines[3];
	const uint16_t * RESTRICT src_p4 = src_lines[4];
	const uint16_t * RESTRICT src_p5 = src_lines[5];
	const uint16_t * RESTRICT src_p6 = src_lines[6];
	const uint16_t * RESTRICT src_p7 = src_lines[7];
	uint16_t * RESTRICT dst_p = dst;
	uint32_t * RESTRICT accum_p = accum;

	unsigned vec_begin = align(left, 8);
	unsigned vec_end = mod(right, 8);
	unsigned accum_base = mod(left, 8);

	const __m128i c01 = _mm_unpacklo_epi16(_mm_set1_epi16(filter_data[0]), _mm_set1_epi16(filter_data[1]));
	const __m128i c23 = _mm_unpacklo_epi16(_mm_set1_epi16(filter_data[2]), _mm_set1_epi16(filter_data[3]));
	const __m128i c45 = _mm_unpacklo_epi16(_mm_set1_epi16(filter_data[4]), _mm_set1_epi16(filter_data[5]));
	const __m128i c67 = _mm_unpacklo_epi16(_mm_set1_epi16(filter_data[6]), _mm_set1_epi16(filter_data[7]));

	__m128i out;

#define XITER resize_line_v_u16_sse2_xiter<N, ReadAccum, WriteToAccum>
#define XARGS accum_base, src_p0, src_p1, src_p2, src_p3, src_p4, src_p5, src_p6, src_p7, accum_p, c01, c23, c45, c67, limit
	if (left != vec_begin) {
		out = XITER(vec_begin - 8, XARGS);

		if (!WriteToAccum)
			mm_store_left_epi16(dst_p + vec_begin - 8, out, vec_begin - left);
	}

	for (unsigned j = vec_begin; j < vec_end; j += 8) {
		out = XITER(j, XARGS);

		if (!WriteToAccum)
			_mm_store_si128((__m128i *)(dst_p + j), out);
	}

	if (right != vec_end) {
		out = XITER(vec_end, XARGS);

		if (!WriteToAccum)
			mm_store_right_epi16(dst_p + vec_end, out, right - vec_end);
	}
#undef XITER
#undef XARGS
}

const decltype(&resize_line_v_u16_sse2<0, false, false>) resize_line_v_u16_sse2_jt_a[] = {
	resize_line_v_u16_sse2<0, false, false>,
	resize_line_v_u16_sse2<0, false, false>,
	resize_line_v_u16_sse2<2, false, false>,
	resize_line_v_u16_sse2<2, false, false>,
	resize_line_v_u16_sse2<4, false, false>,
	resize_line_v_u16_sse2<4, false, false>,
	resize_line_v_u16_sse2<6, false, false>,
	resize_line_v_u16_sse2<6, false, false>,
};

const decltype(&resize_line_v_u16_sse2<0, false, false>) resize_line_v_u16_sse2_jt_b[] = {
	resize_line_v_u16_sse2<0, true, false>,
	resize_line_v_u16_sse2<0, true, false>,
	resize_line_v_u16_sse2<2, true, false>,
	resize_line_v_u16_sse2<2, true, false>,
	resize_line_v_u16_sse2<4, true, false>,
	resize_line_v_u16_sse2<4, true, false>,
	resize_line_v_u16_sse2<6, true, false>,
	resize_line_v_u16_sse2<6, true, false>,
};


class ResizeImplV_U16_SSE2 final : public ResizeImplV {
	uint16_t m_pixel_max;
public:
	ResizeImplV_U16_SSE2(const FilterContext &filter, unsigned width, unsigned depth) :
		ResizeImplV(filter, image_attributes{ width, filter.filter_rows, zimg::PixelType::WORD }),
		m_pixel_max{ (uint16_t)((1UL << depth) - 1) }
	{
	}

	size_t get_tmp_size(unsigned left, unsigned right) const override
	{
		if (m_filter.filter_width > 4)
			return (align(right, 8) - mod(left, 8)) * sizeof(uint32_t);
		else
			return 0;
	}

	void process(void *, const ZimgImageBufferConst &src, const ZimgImageBuffer &dst, void *tmp, unsigned i, unsigned left, unsigned right) const override
	{
		LineBuffer<const uint16_t> src_buf{ src };
		LineBuffer<uint16_t> dst_buf{ dst };

		const int16_t *filter_data = m_filter.data_i16.data() + i * m_filter.stride_i16;
		unsigned filter_width = m_filter.filter_width;
		unsigned src_height = m_filter.input_width;

		const uint16_t *src_lines[8] = { 0 };
		uint16_t *dst_line = dst_buf[i];
		uint32_t *accum_buf = reinterpret_cast<uint32_t *>(tmp);

		unsigned k_end = align(filter_width, 8) - 8;
		unsigned top = m_filter.left[i];

		for (unsigned k = 0; k < k_end; k += 8) {
			for (unsigned n = 0; n < 8; ++n) {
				src_lines[n] = src_buf[std::min(top + k + n, src_height - 1)];
			}

			if (k == 0)
				resize_line_v_u16_sse2<6, false, true>(filter_data + k, src_lines, dst_line, accum_buf, left, right, m_pixel_max);
			else
				resize_line_v_u16_sse2<6, true, true>(filter_data + k, src_lines, dst_line, accum_buf, left, right, m_pixel_max);
		}

		for (unsigned n = 0; n < 8; ++n) {
			src_lines[n] = src_buf[std::min(top + k_end + n, src_height - 1)];
		}

		if (k_end == 0)
			resize_line_v_u16_sse2_jt_a[filter_width - k_end - 1](filter_data + k_end, src_lines, dst_line, accum_buf, left, right, m_pixel_max);
		else
			resize_line_v_u16_sse2_jt_b[filter_width - k_end - 1](filter_data + k_end, src_lines, dst_line, accum_buf, left, right, m_pixel_max);
	}
};

} // namespace


IZimgFilter *create_resize_impl2_v_sse2(const FilterContext &context, unsigned width, PixelType type, unsigned depth)
{
	IZimgFilter *ret = nullptr;

	if (type == zimg::PixelType::WORD)
		ret = new ResizeImplV_U16_SSE2{ context, width, depth };

	return ret;
}

} // namespace resize
} // namespace zimg

#endif // ZIMG_X86
