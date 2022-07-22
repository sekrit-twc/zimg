#include <algorithm>
#include <climits>
#include <cstdint>
#include "common/cpuinfo.h"
#include "common/except.h"
#include "common/pixel.h"
#include "common/zassert.h"
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

template <class T>
struct Buffer {
	const graphengine::BufferDescriptor &buffer;

	Buffer(const graphengine::BufferDescriptor &buffer) : buffer{ buffer } {}

	T *operator[](unsigned i) const { return buffer.get_line<T>(i); }
};


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

void resize_line_v_u16_c(const FilterContext &filter, const Buffer<const uint16_t> &src, const Buffer<uint16_t> &dst, unsigned i, unsigned left, unsigned right, unsigned pixel_max)
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

void resize_line_v_f32_c(const FilterContext &filter, const Buffer<const float> &src, const Buffer<float> &dst, unsigned i, unsigned left, unsigned right)
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
	uint32_t m_pixel_max;
public:
	ResizeImplH_C(const FilterContext &filter, unsigned height, PixelType type, unsigned depth) :
		ResizeImplH(filter, height, type),
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

class ResizeImplV_C : public ResizeImplV {
	PixelType m_type;
	uint32_t m_pixel_max;
public:
	ResizeImplV_C(const FilterContext &filter, unsigned width, PixelType type, unsigned depth) :
		ResizeImplV(filter, width, type),
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
			resize_line_v_u16_c(m_filter, *in, *out, i, left, right, m_pixel_max);
		else
			resize_line_v_f32_c(m_filter, *in, *out, i, left, right);
	}
};

} // namespace


ResizeImplH::ResizeImplH(const FilterContext &filter, unsigned height, PixelType type) :
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

auto ResizeImplH::get_row_deps(unsigned i) const noexcept -> pair_unsigned
{
	unsigned step = m_desc.step;
	unsigned last = std::min(i, UINT_MAX - step) + step;
	return{ i, std::min(last, m_desc.format.height) };
}

auto ResizeImplH::get_col_deps(unsigned left, unsigned right) const noexcept -> pair_unsigned
{
	if (m_desc.flags.entire_row)
		return{ 0, m_filter.input_width };

	unsigned left_dep = m_filter.left[left];
	unsigned right_dep = m_filter.left[right - 1] + m_filter.filter_width;
	return{ left_dep, right_dep };
}


ResizeImplV::ResizeImplV(const FilterContext &filter, unsigned width, PixelType type) :
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

auto ResizeImplV::get_row_deps(unsigned i) const noexcept -> pair_unsigned
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

auto ResizeImplV::get_col_deps(unsigned left, unsigned right) const noexcept -> pair_unsigned
{
	return{ left, right };
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

std::unique_ptr<graphengine::Filter> ResizeImplBuilder::create() const
{
	std::unique_ptr<graphengine::Filter> ret;

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

} // namespace resize
} // namespace zimg
