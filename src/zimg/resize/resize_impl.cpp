#include <algorithm>
#include <cstdint>
#include "common/except.h"
#include "common/linebuffer.h"
#include "common/pixel.h"
#include "graph/zfilter.h"
#include "filter.h"
#include "resize_impl.h"

#ifdef ZIMG_X86
  #include "resize_impl_x86.h"
#endif

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


class ResizeImplH_C : public ResizeImplH {
	PixelType m_type;
	int32_t m_pixel_max;
public:
	ResizeImplH_C(const FilterContext &filter, unsigned height, PixelType type, unsigned depth) :
		ResizeImplH(filter, image_attributes{ filter.filter_rows, height, type }),
		m_type{ type },
		m_pixel_max{ (int32_t)((uint32_t)1 << depth) - 1 }
	{
		if (m_type != PixelType::WORD && m_type != PixelType::FLOAT)
			throw error::InternalError{ "pixel type not supported" };
	}

	void process(void *, const graph::ZimgImageBufferConst &src, const graph::ZimgImageBuffer &dst, void *, unsigned i, unsigned left, unsigned right) const override
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

class ResizeImplV_C : public ResizeImplV {
	PixelType m_type;
	int32_t m_pixel_max;
public:
	ResizeImplV_C(const FilterContext &filter, unsigned width, PixelType type, unsigned depth) :
		ResizeImplV(filter, image_attributes{ width, filter.filter_rows, type}),
		m_type{ type },
		m_pixel_max{ (int32_t)((uint32_t)1 << depth) - 1 }
	{
		if (m_type != PixelType::WORD && m_type != PixelType::FLOAT)
			throw error::InternalError{ "pixel type not supported" };
	}

	void process(void *, const graph::ZimgImageBufferConst &src, const graph::ZimgImageBuffer &dst, void *, unsigned i, unsigned left, unsigned right) const override
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


ResizeImplH::ResizeImplH(const FilterContext &filter, const image_attributes &attr) :
	m_filter(filter),
	m_attr(attr),
	m_is_sorted{ std::is_sorted(m_filter.left.begin(), m_filter.left.end()) }
{
}

graph::ZimgFilterFlags ResizeImplH::get_flags() const
{
	graph::ZimgFilterFlags flags{};

	flags.same_row = true;
	flags.entire_row = !m_is_sorted;

	return flags;
}

graph::IZimgFilter::image_attributes ResizeImplH::get_image_attributes() const
{
	return m_attr;
}

graph::IZimgFilter::pair_unsigned ResizeImplH::get_required_row_range(unsigned i) const
{
	return{ i, std::min(i + get_simultaneous_lines(), get_image_attributes().height) };
}

graph::IZimgFilter::pair_unsigned ResizeImplH::get_required_col_range(unsigned left, unsigned right) const
{
	if (m_is_sorted) {
		unsigned col_left = m_filter.left[left];
		unsigned col_right = m_filter.left[right - 1] + m_filter.filter_width;

		return{ col_left, col_right };
	} else {
		return{ 0, m_filter.input_width };
	}
}

unsigned ResizeImplH::get_max_buffering() const
{
	return get_simultaneous_lines();
}


ResizeImplV::ResizeImplV(const FilterContext &filter, const image_attributes &attr) :
	m_filter(filter),
	m_attr(attr),
	m_is_sorted{ std::is_sorted(m_filter.left.begin(), m_filter.left.end()) }
{
}

graph::ZimgFilterFlags ResizeImplV::get_flags() const
{
	graph::ZimgFilterFlags flags{};

	flags.entire_row = !m_is_sorted;

	return flags;
}

graph::IZimgFilter::image_attributes ResizeImplV::get_image_attributes() const
{
	return m_attr;
}

graph::IZimgFilter::pair_unsigned ResizeImplV::get_required_row_range(unsigned i) const
{
	unsigned bot = std::min(i + get_simultaneous_lines(), get_image_attributes().height);

	if (m_is_sorted) {
		unsigned row_top = m_filter.left[i];
		unsigned row_bot = m_filter.left[bot - 1];

		return{ row_top, row_bot + m_filter.filter_width };
	} else {
		return{ 0, m_filter.input_width };
	}
}

unsigned ResizeImplV::get_max_buffering() const
{
	unsigned step = get_flags().has_state ? get_simultaneous_lines() : 1;
	unsigned buffering = 0;

	for (unsigned i = 0; i < get_image_attributes().height; i += step) {
		auto range = get_required_row_range(i);
		buffering = std::max(buffering, range.second - range.first);
	}

	return buffering;
}


graph::IZimgFilter *create_resize_impl(const Filter &f, PixelType type, bool horizontal, unsigned depth, unsigned src_width, unsigned src_height, unsigned dst_width, unsigned dst_height,
                                       double shift, double subwidth, CPUClass cpu)
{
	graph::IZimgFilter *ret = nullptr;

	unsigned src_dim = horizontal ? src_width : src_height;
	unsigned dst_dim = horizontal ? dst_width : dst_height;

	if (src_width != dst_width && src_height != dst_height)
		throw error::InternalError{ "cannot resize both width and height" };

	FilterContext filter_ctx = compute_filter(f, src_dim, dst_dim, shift, subwidth);

#ifdef ZIMG_X86
	ret = horizontal ?
	      create_resize_impl_h_x86(filter_ctx, dst_height, type, depth, cpu) :
	      create_resize_impl_v_x86(filter_ctx, dst_width, type, depth, cpu);
#endif
	if (!ret)
		ret = horizontal ?
		      static_cast<graph::IZimgFilter *>(new ResizeImplH_C{ filter_ctx, dst_height, type, depth }) :
		      new ResizeImplV_C{ filter_ctx, dst_width, type, depth };

	return ret;
}

} // namespace resize
} // namespace zimg
