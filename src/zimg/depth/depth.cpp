#include "graph/copy_filter.h"
#include "common/except.h"
#include "common/pixel.h"
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


IZimgFilter *create_depth(DitherType type, unsigned width, unsigned height, const PixelFormat &pixel_in, const PixelFormat &pixel_out, CPUClass cpu)
{
	try
	{
		if (pixel_in == pixel_out)
			return new CopyFilter{ width, height, pixel_in.type };
		else if (is_lossless_conversion(pixel_in, pixel_out))
			return create_left_shift(width, height, pixel_in, pixel_out, cpu);
		else if (pixel_out.type == PixelType::HALF || pixel_out.type == PixelType::FLOAT)
			return create_convert_to_float(width, height, pixel_in, pixel_out, cpu);
		else
			return create_dither(type, width, height, pixel_in, pixel_out, cpu);
	} catch (const std::bad_alloc &) {
		throw error::OutOfMemory{};
	}
}

} // namespace depth
} // namespace zimg
