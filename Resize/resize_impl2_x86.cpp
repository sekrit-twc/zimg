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
	X86Capabilities caps = query_x86_capabilities();
	IZimgFilter *ret = nullptr;

	if (cpu == CPUClass::CPU_AUTO) {
		if (caps.sse)
			ret = create_resize_impl2_v_sse(context, width, type, depth);
	} else if (cpu >= CPUClass::CPU_X86_SSE) {
		ret = create_resize_impl2_v_sse(context, width, type, depth);
	}

	return ret;
}

} // namespace resize
} // namespace zimg

#endif // ZIMG_X86
