#include "Common/except.h"
#include "Common/pixel.h"
#include "depth2.h"
#include "depth_convert2.h"
#include "dither2.h"

namespace zimg {;
namespace depth {;

IZimgFilter *create_depth2(DitherType type, unsigned width, unsigned height, const PixelFormat &pixel_in, const PixelFormat &pixel_out, CPUClass cpu)
{
	try
	{
		if (pixel_out.type == PixelType::HALF || pixel_out.type == PixelType::FLOAT)
			return new DepthConvert2{ width, height, pixel_in, pixel_out, cpu };
		else
			return create_dither_convert2(type, width, height, pixel_in, pixel_out, cpu);
	} catch (const std::bad_alloc &) {
		throw ZimgOutOfMemory{};
	}
}

} // namespace depth
} // namespace zimg
