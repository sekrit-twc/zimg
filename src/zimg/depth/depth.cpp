#include "common/cpuinfo.h"
#include "common/except.h"
#include "common/pixel.h"
#include "graph/copy_filter.h"
#include "graph/image_filter.h"
#include "depth.h"
#include "depth_convert.h"
#include "dither.h"

namespace zimg {;
namespace depth {;

namespace {;

bool is_lossless_conversion(const PixelFormat &pixel_in, const PixelFormat &pixel_out)
{
	return (pixel_in.type == PixelType::BYTE || pixel_in.type == PixelType::WORD) &&
	       (pixel_out.type == PixelType::BYTE || pixel_out.type == PixelType::WORD) &&
	       (!pixel_in.fullrange && !pixel_out.fullrange) &&
	       (pixel_in.chroma == pixel_out.chroma) &&
	       (pixel_out.depth >= pixel_in.depth);
}

} // namespace


DepthConversion::DepthConversion(unsigned width, unsigned height) :
	width{ width },
	height{ height },
	pixel_in{},
	pixel_out{},
	dither_type{ DitherType::DITHER_NONE },
	cpu{ CPUClass::CPU_NONE }
{
}

std::unique_ptr<graph::ImageFilter> DepthConversion::create() const
{
	if (pixel_in == pixel_out)
		return std::unique_ptr<graph::ImageFilter>{ new graph::CopyFilter{ width, height, pixel_in.type } };
	else if (is_lossless_conversion(pixel_in, pixel_out))
		return std::unique_ptr<graph::ImageFilter>{ create_left_shift(width, height, pixel_in, pixel_out, cpu) };
	else if (pixel_out.type == PixelType::HALF || pixel_out.type == PixelType::FLOAT)
		return std::unique_ptr<graph::ImageFilter>{ create_convert_to_float(width, height, pixel_in, pixel_out, cpu) };
	else
		return std::unique_ptr<graph::ImageFilter>{ create_dither(dither_type, width, height, pixel_in, pixel_out, cpu) };
}

} // namespace depth
} // namespace zimg
