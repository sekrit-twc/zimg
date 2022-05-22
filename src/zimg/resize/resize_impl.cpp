#include <algorithm>
#include <climits>
#include <cstdint>
#include "common/cpuinfo.h"
#include "common/except.h"
#include "common/pixel.h"
#include "common/zassert.h"
#include "graph/image_filter.h"
#include "filter.h"
#include "resize_impl.h"

#if defined(ZIMG_X86)
  #include "x86/resize_impl_x86.h"
#elif defined(ZIMG_ARM)
  #include "arm/resize_impl_arm.h"
#endif

namespace zimg {
namespace resize {

namespace {

int32_t unpack_pixel_u16(uint16_t x) noexcept
{
	return static_cast<int32_t>(x) + INT16_MIN;
}

uint16_t pack_pixel_u16(int32_t x, int32_t pixel_max) noexcept
{
	x = ((x + (1 << 13)) >> 14) - INT16_MIN;
	x = std::max(std::min(x, pixel_max), static_cast<int32_t>(0));

	return static_cast<uint16_t>(x);
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

void resize_line_v_u16_c(const FilterContext &filter, const graph::ImageBuffer<const uint16_t> &src, const graph::ImageBuffer<uint16_t> &dst, unsigned i, unsigned left, unsigned right, unsigned pixel_max)
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

void resize_line_v_f32_c(const FilterContext &filter, const graph::ImageBuffer<const float> &src, const graph::ImageBuffer<float> &dst, unsigned i, unsigned left, unsigned right)
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


class ResizeImplH_GE_C : public ResizeImplH_GE {
	PixelType m_type;
	uint32_t m_pixel_max;
public:
	ResizeImplH_GE_C(const FilterContext &filter, unsigned height, PixelType type, unsigned depth) :
		ResizeImplH_GE(filter, height, type),
		m_type{ type },
		m_pixel_max{ static_cast<uint32_t>(1UL << depth) - 1 }
	{
		if (m_type != PixelType::WORD && m_type != PixelType::FLOAT)
			error::throw_<error::InternalError>("pixel type not supported");
	}

	void process(const graphengine::BufferDescriptor *in, const graphengine::BufferDescriptor *out,
	             unsigned i, unsigned left, unsigned right, void *, void *) const noexcept override
	{
		if (m_type == PixelType::WORD)
			resize_line_h_u16_c(m_filter, in->get_line<uint16_t>(i), out->get_line<uint16_t>(i), left, right, m_pixel_max);
		else
			resize_line_h_f32_c(m_filter, in->get_line<float>(i), out->get_line<float>(i), left, right);
	}
};

class ResizeImplV_GE_C : public ResizeImplV_GE {
	PixelType m_type;
	uint32_t m_pixel_max;
public:
	ResizeImplV_GE_C(const FilterContext &filter, unsigned width, PixelType type, unsigned depth) :
		ResizeImplV_GE(filter, width, type),
		m_type{ type },
		m_pixel_max{ static_cast<uint32_t>(1UL << depth) - 1 }
	{
		if (m_type != PixelType::WORD && m_type != PixelType::FLOAT)
			error::throw_<error::InternalError>("pixel type not supported");
	}

	void process(const graphengine::BufferDescriptor *in, const graphengine::BufferDescriptor *out,
	             unsigned i, unsigned left, unsigned right, void *, void *) const noexcept override
	{
		graph::ImageBuffer<const void> src_buf{ in->ptr, in->stride, in->mask };
		graph::ImageBuffer<void> dst_buf{ out->ptr, out->stride, out->mask };

		if (m_type == PixelType::WORD)
			resize_line_v_u16_c(m_filter, src_buf.static_buffer_cast<const uint16_t>(), dst_buf.static_buffer_cast<uint16_t>(), i, left, right, m_pixel_max);
		else
			resize_line_v_f32_c(m_filter, src_buf.static_buffer_cast<const float>(), dst_buf.static_buffer_cast<float>(), i, left, right);
	}
};


class ResizeImplH_C : public ResizeImplH {
	PixelType m_type;
	int32_t m_pixel_max;
public:
	ResizeImplH_C(const FilterContext &filter, unsigned height, PixelType type, unsigned depth) :
		ResizeImplH(filter, image_attributes{ filter.filter_rows, height, type }),
		m_type{ type },
		m_pixel_max{ static_cast<int32_t>(1UL << depth) - 1 }
	{
		if (m_type != PixelType::WORD && m_type != PixelType::FLOAT)
			error::throw_<error::InternalError>("pixel type not supported");
	}

	void process(void *, const graph::ImageBuffer<const void> *src, const graph::ImageBuffer<void> *dst, void *, unsigned i, unsigned left, unsigned right) const override
	{
		if (m_type == PixelType::WORD)
			resize_line_h_u16_c(m_filter, static_cast<const uint16_t *>((*src)[i]), static_cast<uint16_t *>((*dst)[i]), left, right, m_pixel_max);
		else
			resize_line_h_f32_c(m_filter, static_cast<const float *>((*src)[i]), static_cast<float *>((*dst)[i]), left, right);
	}
};

class ResizeImplV_C : public ResizeImplV {
	PixelType m_type;
	int32_t m_pixel_max;
public:
	ResizeImplV_C(const FilterContext &filter, unsigned width, PixelType type, unsigned depth) :
		ResizeImplV(filter, image_attributes{ width, filter.filter_rows, type}),
		m_type{ type },
		m_pixel_max{ static_cast<int32_t>(1UL << depth) - 1 }
	{
		if (m_type != PixelType::WORD && m_type != PixelType::FLOAT)
			error::throw_<error::InternalError>("pixel type not supported");
	}

	void process(void *, const graph::ImageBuffer<const void> *src, const graph::ImageBuffer<void> *dst, void *, unsigned i, unsigned left, unsigned right) const override
	{
		if (m_type == PixelType::WORD)
			resize_line_v_u16_c(m_filter, graph::static_buffer_cast<const uint16_t>(*src), graph::static_buffer_cast<uint16_t>(*dst), i, left, right, m_pixel_max);
		else
			resize_line_v_f32_c(m_filter, graph::static_buffer_cast<const float>(*src), graph::static_buffer_cast<float>(*dst), i, left, right);
	}
};

} // namespace


ResizeImplH_GE::ResizeImplH_GE(const FilterContext &filter, unsigned height, PixelType type) :
	m_desc{},
	m_filter(filter)
{
	zassert_d(m_filter.input_width <= pixel_max_width(type), "overflow");
	zassert_d(m_filter.filter_rows <= pixel_max_width(type), "overflow");

	m_desc.format = { m_filter.filter_rows, height, pixel_size(type) };
	m_desc.num_deps = 1;
	m_desc.num_planes = 1;
	m_desc.step = 1;

	m_desc.flags.entire_row = !std::is_sorted(m_filter.left.begin(), m_filter.left.end());
}

std::pair<unsigned, unsigned> ResizeImplH_GE::get_row_deps(unsigned i) const noexcept
{
	unsigned step = m_desc.step;
	unsigned last = std::min(i, UINT_MAX - step) + step;
	return{ i, std::min(last, m_desc.format.height) };
}

std::pair<unsigned, unsigned> ResizeImplH_GE::get_col_deps(unsigned left, unsigned right) const noexcept
{
	if (m_desc.flags.entire_row)
		return{ 0, m_filter.input_width };

	unsigned left_dep = m_filter.left[left];
	unsigned right_dep = m_filter.left[right - 1] + m_filter.filter_width;
	return{ left_dep, right_dep };
}


ResizeImplV_GE::ResizeImplV_GE(const FilterContext &filter, unsigned width, PixelType type) :
	m_desc{},
	m_filter(filter),
	m_unsorted{}
{
	zassert_d(width <= pixel_max_width(type), "overflow");

	m_desc.format = { width, filter.filter_rows, pixel_size(type) };
	m_desc.num_deps = 1;
	m_desc.num_planes = 1;
	m_desc.step = 1;

	m_unsorted = !std::is_sorted(m_filter.left.begin(), m_filter.left.end());
}

std::pair<unsigned, unsigned> ResizeImplV_GE::get_row_deps(unsigned i) const noexcept
{
	if (m_unsorted)
		return{ 0, m_filter.input_width };

	unsigned step = m_desc.step;
	unsigned last = std::min(std::min(i, UINT_MAX - step) + step, m_desc.format.height);
	unsigned top_dep = m_filter.left[i];
	unsigned bot_dep = m_filter.left[last - 1];

	zassert_d(bot_dep <= UINT_MAX - m_filter.filter_width, "overflow");
	return{ top_dep, bot_dep + m_filter.filter_width };
}

std::pair<unsigned, unsigned> ResizeImplV_GE::get_col_deps(unsigned left, unsigned right) const noexcept
{
	return{ left, right };
}


ResizeImplH::ResizeImplH(const FilterContext &filter, const image_attributes &attr) :
	m_filter(filter),
	m_attr(attr),
	m_is_sorted{ std::is_sorted(m_filter.left.begin(), m_filter.left.end()) }
{
	zassert_d(m_filter.input_width <= pixel_max_width(attr.type), "overflow");
	zassert_d(attr.width <= pixel_max_width(attr.type), "overflow");
}

graph::ImageFilter::filter_flags ResizeImplH::get_flags() const
{
	graph::ImageFilter::filter_flags flags{};

	flags.same_row = true;
	flags.entire_row = !m_is_sorted;

	return flags;
}

graph::ImageFilter::image_attributes ResizeImplH::get_image_attributes() const { return m_attr; }

graph::ImageFilter::pair_unsigned ResizeImplH::get_required_row_range(unsigned i) const
{
	unsigned lines = get_simultaneous_lines();
	unsigned last = std::min(i, UINT_MAX - lines) + lines;

	return{ i, std::min(last, get_image_attributes().height) };
}

graph::ImageFilter::pair_unsigned ResizeImplH::get_required_col_range(unsigned left, unsigned right) const
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
	zassert_d(attr.width <= pixel_max_width(attr.type), "overflow");
}

graph::ImageFilter::filter_flags ResizeImplV::get_flags() const
{
	graph::ImageFilter::filter_flags flags{};

	flags.entire_row = !m_is_sorted;

	return flags;
}

graph::ImageFilter::image_attributes ResizeImplV::get_image_attributes() const { return m_attr; }

graph::ImageFilter::pair_unsigned ResizeImplV::get_required_row_range(unsigned i) const
{
	if (m_is_sorted) {
		unsigned lines = get_simultaneous_lines();
		unsigned last = std::min(i, UINT_MAX - lines) + lines;
		unsigned bot = std::min(last, get_image_attributes().height);

		unsigned row_top = m_filter.left[i];
		unsigned row_bot = m_filter.left[bot - 1];

		zassert_d(row_bot <= UINT_MAX - m_filter.filter_width, "overflow");

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


ResizeImplBuilder::ResizeImplBuilder(unsigned src_width, unsigned src_height, PixelType type) :
	src_width{ src_width },
	src_height{ src_height },
	type{ type },
	horizontal{},
	dst_dim{},
	depth{},
	filter{},
	shift{},
	subwidth{},
	cpu{ CPUClass::NONE }
{}

std::unique_ptr<graph::ImageFilter> ResizeImplBuilder::create() const
{
	std::unique_ptr<graph::ImageFilter> ret;

	unsigned src_dim = horizontal ? src_width : src_height;
	FilterContext filter_ctx = compute_filter(*filter, src_dim, dst_dim, shift, subwidth);

#if defined(ZIMG_X86)
	ret = horizontal ?
		create_resize_impl_h_x86(filter_ctx, src_height, type, depth, cpu) :
		create_resize_impl_v_x86(filter_ctx, src_width, type, depth, cpu);
#elif defined(ZIMG_ARM)
	ret = horizontal ?
		create_resize_impl_h_arm(filter_ctx, src_height, type, depth, cpu) :
		create_resize_impl_v_arm(filter_ctx, src_width, type, depth, cpu);
#endif
	if (!ret && horizontal)
		ret = std::make_unique<ResizeImplH_C>(filter_ctx, src_height, type, depth);
	if (!ret && !horizontal)
		ret = std::make_unique<ResizeImplV_C>(filter_ctx, src_width, type, depth);

	return ret;
}

std::unique_ptr<graphengine::Filter> ResizeImplBuilder::create_ge() const
{
	std::unique_ptr<graphengine::Filter> ret;

	unsigned src_dim = horizontal ? src_width : src_height;
	FilterContext filter_ctx = compute_filter(*filter, src_dim, dst_dim, shift, subwidth);

#if defined(ZIMG_X86)
	ret = nullptr /*horizontal ?
		create_resize_impl_h_x86(filter_ctx, src_height, type, depth, cpu) :
		create_resize_impl_v_x86(filter_ctx, src_width, type, depth, cpu)*/;
#elif defined(ZIMG_ARM)
	ret = nullptr /*horizontal ?
		create_resize_impl_h_arm(filter_ctx, src_height, type, depth, cpu) :
		create_resize_impl_v_arm(filter_ctx, src_width, type, depth, cpu)*/;
#endif
	if (!ret && horizontal)
		ret = std::make_unique<ResizeImplH_GE_C>(filter_ctx, src_height, type, depth);
	if (!ret && !horizontal)
		ret = std::make_unique<ResizeImplV_GE_C>(filter_ctx, src_width, type, depth);

	return ret;
}

} // namespace resize
} // namespace zimg
