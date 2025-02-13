#ifdef ZIMG_X86

#include "common/cpuinfo.h"
#include "common/x86/cpuinfo_x86.h"
#include "graphengine/filter.h"
#include "unresize_impl_x86.h"

namespace zimg::unresize {

std::unique_ptr<graphengine::Filter> create_unresize_impl_h_x86(const BilinearContext &context, unsigned height, PixelType type, CPUClass cpu)
{
	X86Capabilities caps = query_x86_capabilities();
	std::unique_ptr<graphengine::Filter> ret;

	if (cpu_is_autodetect(cpu)) {
		if (!ret && caps.avx2)
			return create_unresize_impl_h_avx2(context, height, type);
	} else {
		if (!ret && cpu >= CPUClass::X86_AVX2)
			return create_unresize_impl_h_avx2(context, height, type);
	}

	return ret;
}

std::unique_ptr<graphengine::Filter> create_unresize_impl_v_x86(const BilinearContext &context, unsigned width, PixelType type, CPUClass cpu)
{
	X86Capabilities caps = query_x86_capabilities();
	std::unique_ptr<graphengine::Filter> ret;

	if (cpu_is_autodetect(cpu)) {
		if (!ret && caps.avx2)
			return create_unresize_impl_v_avx2(context, width, type);
	} else {
		if (cpu >= CPUClass::X86_AVX2)
			return create_unresize_impl_v_avx2(context, width, type);
	}

	return ret;
}

} // namespace zimg::unresize

#endif // ZIMG_X86
