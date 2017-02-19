#include <algorithm>
#include <cstdint>
#include "common/except.h"
#include "common/make_unique.h"
#include "common/pixel.h"
#include "common/zassert.h"
#include "graph/image_filter.h"
#include "depth_convert.h"
#include "depth_convert_x86.h"
#include "quantize.h"

namespace zimg {
namespace depth {

namespace {

template <class T, class U>
void integer_to_integer(const void *src, void *dst, unsigned shift, unsigned left, unsigned right)
{
	const T *src_p = static_cast<const T *>(src);
	U *dst_p = static_cast<U *>(dst);

	std::transform(src_p + left, src_p + right, dst_p + left, [=](T x) { return static_cast<U>(static_cast<unsigned>(x) << shift); });
}

template <class T>
void integer_to_float(const void *src, void *dst, float scale, float offset, unsigned left, unsigned right)
{
	const T *src_p = static_cast<const T *>(src);
	float *dst_p = static_cast<float *>(dst);

	std::transform(src_p + left, src_p + right, dst_p + left, [=](T x){ return static_cast<float>(x) * scale + offset; });
}

void half_to_float_n(const void *src, void *dst, unsigned left, unsigned right)
{
	const uint16_t *src_p = static_cast<const uint16_t *>(src);
	float *dst_p = static_cast<float *>(dst);

	std::transform(src_p + left, src_p + right, dst_p + left, half_to_float);
}

void float_to_half_n(const void *src, void *dst, unsigned left, unsigned right)
{
	const float *src_p = static_cast<const float *>(src);
	uint16_t *dst_p = static_cast<uint16_t *>(dst);

	std::transform(src_p + left, src_p + right, dst_p + left, float_to_half);
}


left_shift_func select_left_shift_func(PixelType pixel_in, PixelType pixel_out)
{
	if (pixel_in == PixelType::BYTE && pixel_out == PixelType::BYTE)
		return integer_to_integer<uint8_t, uint8_t>;
	else if (pixel_in == PixelType::BYTE && pixel_out == PixelType::WORD)
		return integer_to_integer<uint8_t, uint16_t>;
	else if (pixel_in == PixelType::WORD && pixel_out == PixelType::BYTE)
		return integer_to_integer<uint16_t, uint8_t>;
	else if (pixel_in == PixelType::WORD && pixel_out == PixelType::WORD)
		return integer_to_integer<uint16_t, uint16_t>;
	else
		throw error::InternalError{ "no conversion between pixel types" };
}

depth_convert_func select_depth_convert_func(PixelType type_in, PixelType type_out)
{
	if (type_in == PixelType::HALF)
		type_in = PixelType::FLOAT;
	if (type_out == PixelType::HALF)
		type_out = PixelType::FLOAT;

	if (type_in == PixelType::BYTE && type_out == PixelType::FLOAT)
		return integer_to_float<uint8_t>;
	else if (type_in == PixelType::WORD && type_out == PixelType::FLOAT)
		return integer_to_float<uint16_t>;
	else if (type_in == PixelType::FLOAT && type_out == PixelType::FLOAT)
		return nullptr;
	else
		throw error::InternalError{ "no conversion between pixel types" };
}


class IntegerLeftShift final : public graph::ImageFilterBase {
	left_shift_func m_func;

	PixelType m_pixel_in;
	PixelType m_pixel_out;
	unsigned m_shift;

	unsigned m_width;
	unsigned m_height;
public:
	IntegerLeftShift(left_shift_func func, unsigned width, unsigned height, const PixelFormat &pixel_in, const PixelFormat &pixel_out) :
		m_func{ func },
		m_pixel_in{ pixel_in.type },
		m_pixel_out{ pixel_out.type },
		m_shift{},
		m_width{ width },
		m_height{ height }
	{
		zassert_d(width <= pixel_max_width(pixel_in.type), "overflow");
		zassert_d(width <= pixel_max_width(pixel_out.type), "overflow");

		if (!pixel_is_integer(pixel_in.type) || !pixel_is_integer(pixel_out.type))
			throw error::InternalError{ "cannot left shift floating point types" };
		if (pixel_in.fullrange || pixel_out.fullrange)
			throw error::InternalError{ "cannot left shift full-range format" };
		if (pixel_in.chroma != pixel_out.chroma)
			throw error::InternalError{ "cannot convert between luma and chroma" };
		if (pixel_in.depth > pixel_out.depth)
			throw error::InternalError{ "cannot reduce depth by left shifting" };
		if (pixel_out.depth - pixel_in.depth > 15)
			throw error::InternalError{ "too much shifting" };

		m_shift = pixel_out.depth - pixel_in.depth;
	}

	filter_flags get_flags() const override
	{
		filter_flags flags{};

		flags.same_row = true;
		flags.in_place = (pixel_size(m_pixel_in) == pixel_size(m_pixel_out));

		return flags;
	}

	image_attributes get_image_attributes() const override
	{
		return{ m_width, m_height, m_pixel_out };
	}

	void process(void *, const graph::ImageBuffer<const void> src[], const graph::ImageBuffer<void> dst[], void *, unsigned i, unsigned left, unsigned right) const override
	{
		const char *src_line = graph::static_buffer_cast<const char>(*src)[i];
		char *dst_line = graph::static_buffer_cast<char>(*dst)[i];

		unsigned pixel_align = std::max(pixel_alignment(m_pixel_in), pixel_alignment(m_pixel_out));
		unsigned line_base = floor_n(left, pixel_align);

		src_line += pixel_size(m_pixel_in) * line_base;
		dst_line += pixel_size(m_pixel_out) * line_base;

		left -= line_base;
		right -= line_base;

		m_func(src_line, dst_line, m_shift, left, right);
	}
};


class ConvertToFloat final : public graph::ImageFilterBase {
	depth_convert_func m_func;
	depth_f16c_func m_f16c;

	PixelType m_pixel_in;
	PixelType m_pixel_out;
	float m_scale;
	float m_offset;

	unsigned m_width;
	unsigned m_height;
public:
	ConvertToFloat(depth_convert_func func, depth_f16c_func f16c, unsigned width, unsigned height,
	               const PixelFormat &pixel_in, const PixelFormat &pixel_out) :
		m_func{ func },
		m_f16c{ f16c },
		m_pixel_in{ pixel_in.type },
		m_pixel_out{ pixel_out.type },
		m_scale{},
		m_offset{},
		m_width{ width },
		m_height{ height }
	{
		zassert_d(width <= pixel_max_width(pixel_in.type), "overflow");
		zassert_d(width <= pixel_max_width(pixel_out.type), "overflow");

		if (pixel_in == pixel_out)
			throw error::InternalError{ "cannot perform no-op conversion" };
		if (f16c && pixel_in.type != PixelType::HALF && pixel_out.type != PixelType::HALF)
			throw error::InternalError{ "cannot provide f16c function for non-HALF types" };
		if (!pixel_is_float(pixel_out.type))
			throw error::InternalError{ "DepthConvert only converts to floating point types" };

		int32_t range = integer_range(pixel_in);
		int32_t offset = integer_offset(pixel_in);

		m_scale = static_cast<float>(1.0 / range);
		m_offset = static_cast<float>(-offset * (1.0 / range));
	}

	filter_flags get_flags() const override
	{
		filter_flags flags{};

		flags.same_row = true;
		flags.in_place = (pixel_size(m_pixel_in) == pixel_size(m_pixel_out));

		return flags;
	}

	image_attributes get_image_attributes() const override
	{
		return{ m_width, m_height, m_pixel_out };
	}

	size_t get_tmp_size(unsigned left, unsigned right) const override
	{
		size_t size = 0;

		if (m_func && m_f16c) {
			unsigned pixel_align = std::max(pixel_alignment(m_pixel_in), pixel_alignment(m_pixel_out));

			left = floor_n(left, pixel_align);
			right = ceil_n(right, pixel_align);

			size = (right - left) * sizeof(float);
		}

		return size;
	}

	void process(void *, const graph::ImageBuffer<const void> *src, const graph::ImageBuffer<void> *dst, void *tmp, unsigned i, unsigned left, unsigned right) const override
	{
		const char *src_line = graph::static_buffer_cast<const char>(*src)[i];
		char *dst_line = graph::static_buffer_cast<char>(*dst)[i];

		unsigned pixel_align = std::max(pixel_alignment(m_pixel_in), pixel_alignment(m_pixel_out));
		unsigned line_base = floor_n(left, pixel_align);

		src_line += pixel_size(m_pixel_in) * line_base;
		dst_line += pixel_size(m_pixel_out) * line_base;

		left -= line_base;
		right -= line_base;

		if (m_func && m_f16c) {
			m_func(src_line, tmp, m_scale, m_offset, left, right);
			m_f16c(tmp, dst_line, left, right);
		} else if (m_func) {
			m_func(src_line, dst_line, m_scale, m_offset, left, right);
		} else {
			m_f16c(src_line, dst_line, left, right);
		}
	}
};

} // namespace


std::unique_ptr<graph::ImageFilter> create_left_shift(unsigned width, unsigned height, const PixelFormat &pixel_in, const PixelFormat &pixel_out, CPUClass cpu)
{
	left_shift_func func = nullptr;

#ifdef ZIMG_X86
	func = select_left_shift_func_x86(pixel_in.type, pixel_out.type, cpu);
#endif
	if (!func)
		func = select_left_shift_func(pixel_in.type, pixel_out.type);

	return ztd::make_unique<IntegerLeftShift>(func, width, height, pixel_in, pixel_out);
}


std::unique_ptr<graph::ImageFilter> create_convert_to_float(unsigned width, unsigned height, const PixelFormat &pixel_in, const PixelFormat &pixel_out, CPUClass cpu)
{
	depth_convert_func func = nullptr;
	depth_f16c_func f16c = nullptr;
	bool needs_f16c = (pixel_in.type == PixelType::HALF || pixel_out.type == PixelType::HALF);

#ifdef ZIMG_X86
	func = select_depth_convert_func_x86(pixel_in, pixel_out, cpu);
	needs_f16c = needs_f16c && needs_depth_f16c_func_x86(pixel_in, pixel_out, cpu);

	if (needs_f16c)
		f16c = select_depth_f16c_func_x86(pixel_out.type == PixelType::HALF, cpu);
#endif
	if (!func)
		func = select_depth_convert_func(pixel_in.type, pixel_out.type);

	if (needs_f16c) {
		if (!f16c && pixel_in.type == zimg::PixelType::HALF)
			f16c = half_to_float_n;
		if (!f16c && pixel_out.type == zimg::PixelType::HALF)
			f16c = float_to_half_n;
	}

	return ztd::make_unique<ConvertToFloat>(func, f16c, width, height, pixel_in, pixel_out);
}

} // namespace depth
} // namespace zimg
