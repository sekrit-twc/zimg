#include "common/cpuinfo.h"
#include "common/except.h"
#include "common/make_unique.h"
#include "common/pixel.h"
#include "graph/basic_filter.h"
#include "graph/image_filter.h"
#include "depth.h"
#include "depth_convert.h"
#include "dither.h"

namespace zimg {
namespace depth {

namespace {

bool is_lossless_conversion(const PixelFormat &pixel_in, const PixelFormat &pixel_out)
{
	return pixel_is_integer(pixel_in.type) &&
	       pixel_is_integer(pixel_out.type) &&
	       !pixel_in.fullrange &&
	       !pixel_out.fullrange &&
	       pixel_in.chroma == pixel_out.chroma &&
	       pixel_out.depth >= pixel_in.depth;
}

} // namespace


DepthConversion::DepthConversion(unsigned width, unsigned height) :
	width{ width },
	height{ height },
	pixel_in{},
	pixel_out{},
	dither_type{ DitherType::NONE },
	cpu{ CPUClass::NONE }
{}

std::unique_ptr<graph::ImageFilter> DepthConversion::create() const try
{
	if (width > pixel_max_width(pixel_in.type) || width > pixel_max_width(pixel_out.type))
		throw error::OutOfMemory{};

	if (pixel_in == pixel_out)
		return ztd::make_unique<graph::CopyFilter>(width, height, pixel_in.type);
	else if (is_lossless_conversion(pixel_in, pixel_out))
		return create_left_shift(width, height, pixel_in, pixel_out, cpu);
	else if (pixel_is_float(pixel_out.type))
		return create_convert_to_float(width, height, pixel_in, pixel_out, cpu);
	else
		return create_dither(dither_type, width, height, pixel_in, pixel_out, cpu);
} catch (const std::bad_alloc &) {
	throw error::OutOfMemory{};
}

} // namespace depth
} // namespace zimg
