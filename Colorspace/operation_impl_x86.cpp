#ifdef ZIMG_X86

#include "Common/cpuinfo.h"
#include "operation_impl_x86.h"

namespace zimg {;
namespace colorspace {;

PixelAdapter *create_pixel_adapter_x86(CPUClass cpu)
{
	X86Capabilities caps = query_x86_capabilities();
	PixelAdapter *ret;

	if (cpu == CPUClass::CPU_X86_AUTO) {
		if (caps.f16c)
			ret = create_pixel_adapter_f16c();
		else
			ret = nullptr;
	} else if (cpu >= CPUClass::CPU_X86_F16C) {
		ret = create_pixel_adapter_f16c();
	} else {
		ret = nullptr;
	}

	return ret;
}

} // namespace colorspace
} // namespace zimg

#endif // ZIMG_X86
