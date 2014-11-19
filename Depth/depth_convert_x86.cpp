#ifdef ZIMG_X86

#include "Common/cpuinfo.h"
#include "depth_convert_x86.h"

namespace zimg {;
namespace depth {;

DepthConvert *create_depth_convert_x86(CPUClass cpu)
{
	X86Capabilities caps = query_x86_capabilities();
	DepthConvert *ret;

	if (cpu == CPUClass::CPU_X86_AUTO) {
		if (caps.avx2)
			ret = create_depth_convert_avx2();
		else if (caps.sse2)
			ret = create_depth_convert_sse2();
		else
			ret = nullptr;
	} else if (cpu >= CPUClass::CPU_X86_AVX2) {
		ret = create_depth_convert_avx2();
	} else if (cpu >= CPUClass::CPU_X86_SSE2) {
		ret = create_depth_convert_sse2();
	} else {
		ret = nullptr;
	}

	return ret;
}

} // namespace depth
} // namespace zimg

#endif // ZIMG_X86
