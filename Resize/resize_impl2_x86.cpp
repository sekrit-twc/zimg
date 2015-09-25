#ifdef ZIMG_X86

#include "Common/cpuinfo.h"
#include "resize_impl2_x86.h"

namespace zimg {;
namespace resize {;

IZimgFilter *create_resize_impl2_h_x86(const FilterContext &context, unsigned height, PixelType type, unsigned depth, CPUClass cpu)
{
	return nullptr;
}

IZimgFilter *create_resize_impl2_v_x86(const FilterContext &context, unsigned width, PixelType type, unsigned depth, CPUClass cpu)
{
	return nullptr;
}

} // namespace resize
} // namespace zimg

#endif // ZIMG_X86
