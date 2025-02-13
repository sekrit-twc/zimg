#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <tuple>
#include "common/checked_int.h"
#include "common/except.h"
#include "common/pixel.h"
#include "common/zassert.h"
#include "graph/filter_base.h"
#include "depth_convert.h"
#include "quantize.h"

#if defined(ZIMG_X86)
  #include "x86/depth_convert_x86.h"
#elif defined(ZIMG_ARM)
  #include "arm/depth_convert_arm.h"
#endif

namespace zimg::depth {

namespace {

template <class T, class U>
void integer_to_integer(const void *src, void *dst, unsigned shift, unsigned left, unsigned right)
{
	const T *src_p = static_cast<const T *>(src);
	U *dst_p = static_cast<U *>(dst);

	std::transform(src_p + left, src_p + right, dst_p + left, [=](T x) { return static_cast<U>(static_cast<unsigned>(x) << shift); });
}

template <class T>
void integer_to_half(const void *src, void *dst, float scale, float offset, unsigned left, unsigned right)
{
	const T *src_p = static_cast<const T *>(src);
	uint16_t *dst_p = static_cast<uint16_t *>(dst);

	std::transform(src_p + left, src_p + right, dst_p + left, [=](T x) { return float_to_half(static_cast<float>(x) * scale + offset); });
}

template <class T>
void integer_to_float(const void *src, void *dst, float scale, float offset, unsigned left, unsigned right)
{
	const T *src_p = static_cast<const T *>(src);
	float *dst_p = static_cast<float *>(dst);

	std::transform(src_p + left, src_p + right, dst_p + left, [=](T x){ return static_cast<float>(x) * scale + offset; });
}

void half_to_float_n(const void *src, void *dst, float, float, unsigned left, unsigned right)
{
	const uint16_t *src_p = static_cast<const uint16_t *>(src);
	float *dst_p = static_cast<float *>(dst);

	std::transform(src_p + left, src_p + right, dst_p + left, half_to_float);
}

void float_to_half_n(const void *src, void *dst, float, float, unsigned left, unsigned right)
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
		error::throw_<error::InternalError>("no conversion between pixel types");
}

depth_convert_func select_depth_convert_func(PixelType type_in, PixelType type_out)
{
	if (type_in == PixelType::BYTE && type_out == PixelType::HALF)
		return integer_to_half<uint8_t>;
	else if (type_in == PixelType::BYTE && type_out == PixelType::FLOAT)
		return integer_to_float<uint8_t>;
	else if (type_in == PixelType::WORD && type_out == PixelType::HALF)
		return integer_to_half<uint16_t>;
	else if (type_in == PixelType::WORD && type_out == PixelType::FLOAT)
		return integer_to_float<uint16_t>;
	else if (type_in == PixelType::HALF && type_out == PixelType::HALF)
		return nullptr;
	else if (type_in == PixelType::HALF && type_out == PixelType::FLOAT)
		return half_to_float_n;
	else if (type_in == PixelType::FLOAT && type_out == PixelType::HALF)
		return float_to_half_n;
	else if (type_in == PixelType::FLOAT && type_out == PixelType::FLOAT)
		return nullptr;
	else
		error::throw_<error::InternalError>("no conversion between pixel types");
}


class IntegerLeftShift : public graph::PointFilter {
	left_shift_func m_func;
	unsigned m_shift;

	void check_preconditions(unsigned width, const PixelFormat &pixel_in, const PixelFormat &pixel_out)
	{
		zassert_d(width <= pixel_max_width(pixel_in.type), "overflow");
		zassert_d(width <= pixel_max_width(pixel_out.type), "overflow");

		if (!pixel_is_integer(pixel_in.type) || !pixel_is_integer(pixel_out.type))
			error::throw_<error::InternalError>("cannot left shift floating point types");
		if (pixel_in.fullrange || pixel_out.fullrange)
			error::throw_<error::InternalError>("cannot left shift full-range format");
		if (pixel_in.chroma != pixel_out.chroma)
			error::throw_<error::InternalError>("cannot convert between luma and chroma");
		if (pixel_in.depth > pixel_out.depth)
			error::throw_<error::InternalError>("cannot reduce depth by left shifting");
		if (pixel_out.depth - pixel_in.depth > 15)
			error::throw_<error::InternalError>("too much shifting");
	}
public:
	IntegerLeftShift(left_shift_func func, unsigned width, unsigned height, const PixelFormat &pixel_in, const PixelFormat &pixel_out) :
		PointFilter(width, height, pixel_out.type),
		m_func{ func },
		m_shift{}
	{
		check_preconditions(width, pixel_in, pixel_out);

		m_desc.num_deps = 1;
		m_desc.num_planes = 1;
		m_desc.flags.in_place = pixel_size(pixel_in.type) == pixel_size(pixel_out.type);

		m_shift = pixel_out.depth - pixel_in.depth;
	}

	void process(const graphengine::BufferDescriptor *in, const graphengine::BufferDescriptor *out,
	             unsigned i, unsigned left, unsigned right, void *, void *) const noexcept override
	{
		m_func(in->get_line(i), out->get_line(i), m_shift, left, right);
	}
};


class ConvertToFloat : public graph::PointFilter {
	depth_convert_func m_func;
	float m_scale;
	float m_offset;

	void check_preconditions(unsigned width, const PixelFormat &pixel_in, const PixelFormat &pixel_out)
	{
		zassert_d(width <= pixel_max_width(pixel_in.type), "overflow");
		zassert_d(width <= pixel_max_width(pixel_out.type), "overflow");

		if (pixel_in == pixel_out)
			error::throw_<error::InternalError>("cannot perform no-op conversion");
		if (!pixel_is_float(pixel_out.type))
			error::throw_<error::InternalError>("DepthConvert only converts to floating point types");
	}
public:
	ConvertToFloat(depth_convert_func func, unsigned width, unsigned height, const PixelFormat &pixel_in, const PixelFormat &pixel_out) :
		PointFilter(width, height, pixel_out.type),
		m_func{ func },
		m_scale{},
		m_offset{}
	{
		check_preconditions(width, pixel_in, pixel_out);

		m_desc.num_deps = 1;
		m_desc.num_planes = 1;
		m_desc.flags.in_place = pixel_size(pixel_in.type) == pixel_size(pixel_out.type);

		std::tie(m_scale, m_offset) = get_scale_offset(pixel_in, pixel_out);
	}

	void process(const graphengine::BufferDescriptor *in, const graphengine::BufferDescriptor *out,
	             unsigned i, unsigned left, unsigned right, void *, void *tmp) const noexcept override
	{
		m_func(in->get_line(i), out->get_line(i), m_scale, m_offset, left, right);
	}
};

} // namespace


std::unique_ptr<graphengine::Filter> create_left_shift(unsigned width, unsigned height, const PixelFormat &pixel_in, const PixelFormat &pixel_out, CPUClass cpu)
{
	left_shift_func func = nullptr;

#if defined(ZIMG_X86)
	func = select_left_shift_func_x86(pixel_in.type, pixel_out.type, cpu);
#elif defined(ZIMG_ARM)
	func = select_left_shift_func_arm(pixel_in.type, pixel_out.type, cpu);
#endif
	if (!func)
		func = select_left_shift_func(pixel_in.type, pixel_out.type);

	return std::make_unique<IntegerLeftShift>(func, width, height, pixel_in, pixel_out);
}

std::unique_ptr<graphengine::Filter> create_convert_to_float(unsigned width, unsigned height, const PixelFormat &pixel_in, const PixelFormat &pixel_out, CPUClass cpu)
{
	depth_convert_func func = nullptr;

#if defined(ZIMG_X86)
	func = select_depth_convert_func_x86(pixel_in, pixel_out, cpu);
#elif defined(ZIMG_ARM)
	func = select_depth_convert_func_arm(pixel_in, pixel_out, cpu);
#endif
	if (!func)
		func = select_depth_convert_func(pixel_in.type, pixel_out.type);


	return std::make_unique<ConvertToFloat>(func, width, height, pixel_in, pixel_out);
}

} // namespace zimg::depth
