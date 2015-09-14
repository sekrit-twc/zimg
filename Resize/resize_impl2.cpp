#include <algorithm>
#include <cstdint>
#include "Common/except.h"
#include "Common/linebuffer.h"
#include "Common/pixel.h"
#include "Common/zfilter.h"
#include "filter.h"
#include "resize_impl2.h"

namespace zimg {;
namespace resize {;

namespace {;

int32_t unpack_pixel_u16(uint16_t x)
{
	return (int32_t)x + INT16_MIN;
}

uint16_t pack_pixel_u16(int32_t x, int32_t pixel_max)
{
	x = ((x + (1 << 13)) >> 14) - INT16_MIN;
	x = std::max(std::min(x, pixel_max), (int32_t)0);

	return (uint16_t)x;
}

void resize_line_h_u16_c(const FilterContext &filter, const uint16_t *src, uint16_t *dst, unsigned left, unsigned right, unsigned pixel_max)
{
	for (unsigned j = left; j < right; ++j) {
		unsigned left = filter.left[j];
		int32_t accum = 0;

		for (unsigned k = 0; k < filter.filter_width; ++k) {
			int32_t coeff = filter.data_i16[j * filter.stride_i16 + k];
			int32_t x = unpack_pixel_u16(src[left + k]);

			accum += coeff * x;
		}

		dst[j] = pack_pixel_u16(accum, pixel_max);
	}
}

void resize_line_h_f32_c(const FilterContext &filter, const float *src, float *dst, unsigned left, unsigned right)
{
	for (unsigned j = left; j < right; ++j) {
		unsigned top = filter.left[j];
		float accum = 0;

		for (unsigned k = 0; k < filter.filter_width; ++k) {
			float coeff = filter.data[j * filter.stride + k];
			float x = src[top + k];

			accum += coeff * x;
		}

		dst[j] = accum;
	}
}

void resize_line_v_u16_c(const FilterContext &filter, const LineBuffer<const uint16_t> &src, LineBuffer<uint16_t> &dst, unsigned i, unsigned left, unsigned right, unsigned pixel_max)
{
	const int16_t *filter_coeffs = &filter.data_i16[i * filter.stride_i16];
	unsigned top = filter.left[i];

	for (unsigned j = left; j < right; ++j) {
		int32_t accum = 0;

		for (unsigned k = 0; k < filter.filter_width; ++k) {
			int32_t coeff = filter_coeffs[k];
			int32_t x = unpack_pixel_u16(src[top + k][j]);

			accum += coeff * x;
		}

		dst[i][j] = pack_pixel_u16(accum, pixel_max);
	}
}

void resize_line_v_f32_c(const FilterContext &filter, const LineBuffer<const float> &src, LineBuffer<float> &dst, unsigned i, unsigned left, unsigned right)
{
	const float *filter_coeffs = &filter.data[i * filter.stride];
	unsigned top = filter.left[i];

	for (unsigned j = left; j < right; ++j) {
		float accum = 0;

		for (unsigned k = 0; k < filter.filter_width; ++k) {
			float coeff = filter_coeffs[k];
			float x = src[top + k][j];

			accum += coeff * x;
		}

		dst[i][j] = accum;
	}
}


class ResizeImplH_C : public ZimgFilter {
	FilterContext m_filter;
	unsigned m_height;
	PixelType m_type;
	int32_t m_pixel_max;
	bool m_is_sorted;
public:
	ResizeImplH_C(const FilterContext &filter, unsigned height, PixelType type, unsigned depth) :
		m_filter(filter),
		m_height{ height },
		m_type{ type },
		m_pixel_max{ (int32_t)((uint32_t)1 << depth) - 1 },
		m_is_sorted{ std::is_sorted(m_filter.left.begin(), m_filter.left.end()) }
	{
		if (m_type != PixelType::WORD && m_type != PixelType::FLOAT)
			throw zimg::error::InternalError{ "pixel type not supported" };
	}

	ZimgFilterFlags get_flags() const override
	{
		ZimgFilterFlags flags{};

		flags.same_row = true;
		flags.entire_row = !m_is_sorted;

		return flags;
	}

	image_attributes get_image_attributes() const override
	{
		return{ m_filter.filter_rows, m_height, m_type };
	}

	pair_unsigned get_required_col_range(unsigned left, unsigned right) const override
	{
		if (m_is_sorted) {
			unsigned col_left = m_filter.left[left];
			unsigned col_right = m_filter.left[right - 1] + m_filter.filter_width;

			return{ col_left, col_right };
		} else {
			return{ 0, m_filter.input_width };
		}
	}

	void process(void *, const ZimgImageBufferConst &src, const ZimgImageBuffer &dst, void *, unsigned i, unsigned left, unsigned right) const override
	{
		if (m_type == PixelType::WORD) {
			LineBuffer<const uint16_t> src_buf{ src };
			LineBuffer<uint16_t> dst_buf{ dst };

			resize_line_h_u16_c(m_filter, src_buf[i], dst_buf[i], left, right, m_pixel_max);
		} else {
			LineBuffer<const float> src_buf{ src };
			LineBuffer<float> dst_buf{ dst };

			resize_line_h_f32_c(m_filter, src_buf[i], dst_buf[i], left, right);
		}
	}
};

class ResizeImplV_C : public ZimgFilter {
	FilterContext m_filter;
	unsigned m_width;
	PixelType m_type;
	int32_t m_pixel_max;
	bool m_is_sorted;
public:
	ResizeImplV_C(const FilterContext &filter, unsigned width, PixelType type, unsigned depth) :
		m_filter(filter),
		m_width{ width },
		m_type{ type },
		m_pixel_max{ (int32_t)((uint32_t)1 << depth) - 1 },
		m_is_sorted{ std::is_sorted(m_filter.left.begin(), m_filter.left.end()) }
	{
		if (m_type != PixelType::WORD && m_type != PixelType::FLOAT)
			throw zimg::error::InternalError{ "pixel type not supported" };
	}

	ZimgFilterFlags get_flags() const override
	{
		ZimgFilterFlags flags{};

		flags.entire_row = !m_is_sorted;
		flags.entire_plane = !m_is_sorted;

		return flags;
	}

	image_attributes get_image_attributes() const override
	{
		return{ m_width, m_filter.filter_rows, m_type };
	}

	pair_unsigned get_required_row_range(unsigned i) const override
	{
		if (m_is_sorted) {
			unsigned row = m_filter.left[i];

			return{ row, row + m_filter.filter_width };
		} else {
			return{ 0, m_filter.input_width };
		}
	}

	unsigned get_max_buffering() const override
	{
		return m_is_sorted ? m_filter.filter_width : -1;
	}

	void process(void *, const ZimgImageBufferConst &src, const ZimgImageBuffer &dst, void *, unsigned i, unsigned left, unsigned right) const override
	{
		if (m_type == PixelType::WORD) {
			LineBuffer<const uint16_t> src_buf{ src };
			LineBuffer<uint16_t> dst_buf{ dst };

			resize_line_v_u16_c(m_filter, src_buf, dst_buf, i, left, right, m_pixel_max);
		} else {
			LineBuffer<const float> src_buf{ src };
			LineBuffer<float> dst_buf{ dst };

			resize_line_v_f32_c(m_filter, src_buf, dst_buf, i, left, right);
		}
	}
};

} // namespace


IZimgFilter *create_resize_impl2(const Filter &f, PixelType type, bool horizontal, unsigned depth, unsigned src_width, unsigned src_height, unsigned dst_width, unsigned dst_height,
                                 double shift, double subwidth, CPUClass cpu)
{
	unsigned src_dim = horizontal ? src_width : src_height;
	unsigned dst_dim = horizontal ? dst_width : dst_height;

	if (src_width != dst_width && src_height != dst_height)
		throw zimg::error::InternalError{ "cannot resize both width and height" };

	FilterContext filter_ctx = compute_filter(f, src_dim, dst_dim, shift, subwidth);

	if (horizontal)
		return new ResizeImplH_C{ filter_ctx, dst_height, type, depth };
	else
		return new ResizeImplV_C{ filter_ctx, dst_width, type, depth };
}

} // namespace resize
} // namespace zimg
