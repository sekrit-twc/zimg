#include <memory>
#include <utility>
#include "Common/except.h"
#include "depth.h"
#include "depth_convert.h"
#include "dither.h"
#include "dither_impl.h"
#include "error_diffusion.h"

namespace zimg {;
namespace depth {;

DitherConvert::~DitherConvert()
{
}

DitherConvert *create_dither_convert(DitherType type, CPUClass cpu)
{
	switch (type) {
	case DitherType::DITHER_NONE:
	case DitherType::DITHER_ORDERED:
	case DitherType::DITHER_RANDOM:
		return create_ordered_dither(type, cpu);
	case DitherType::DITHER_ERROR_DIFFUSION:
		return create_error_diffusion(cpu);
	default:
		throw ZimgIllegalArgument{ "unrecognized dither type" };
	}
}

} // namespace depth
} // namespace zimg
