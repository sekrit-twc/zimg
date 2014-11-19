#ifdef ZIMG_X86

#include "Common/cpuinfo.h"
#include "dither_impl_x86.h"

namespace zimg {;
namespace depth {;

DitherConvert *create_ordered_dither_x86(const float *dither, CPUClass cpu)
{
	X86Capabilities caps = query_x86_capabilities();
	DitherConvert *ret;

	if (cpu == CPUClass::CPU_X86_AUTO) {
		if (caps.avx2)
			ret = create_ordered_dither_avx2(dither);
		else if (caps.sse2)
			ret = create_ordered_dither_sse2(dither);
		else
			ret = nullptr;
	} else if (cpu >= CPUClass::CPU_X86_AVX2) {
		ret = create_ordered_dither_avx2(dither);
	} else if (cpu >= CPUClass::CPU_X86_SSE2) {
		ret = create_ordered_dither_sse2(dither);
	} else {
		ret = nullptr;
	}

	return ret;
}

} // namespace depth
} // namespace zimg

#endif // ZIMG_X86
