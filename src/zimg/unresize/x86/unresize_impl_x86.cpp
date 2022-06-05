#ifdef ZIMG_X86

#include "common/cpuinfo.h"
#include "common/x86/cpuinfo_x86.h"
#include "graph/image_filter.h"
#include "graphengine/filter.h"
#include "unresize_impl_x86.h"

namespace zimg {
namespace unresize {

std::unique_ptr<graph::ImageFilter> create_unresize_impl_h_x86(const BilinearContext &context, unsigned height, PixelType type, CPUClass cpu)
{
	X86Capabilities caps = query_x86_capabilities();
	std::unique_ptr<graph::ImageFilter> ret;

	if (cpu_is_autodetect(cpu)) {
		if (!ret && caps.sse)
			return create_unresize_impl_h_sse(context, height, type);
	} else {
		if (!ret && cpu >= CPUClass::X86_SSE)
			return create_unresize_impl_h_sse(context, height, type);
	}

	return ret;
}

std::unique_ptr<graphengine::Filter> create_unresize_impl_h_ge_x86(const BilinearContext &context, unsigned height, PixelType type, CPUClass cpu)
{
	X86Capabilities caps = query_x86_capabilities();
	std::unique_ptr<graphengine::Filter> ret;

	if (cpu_is_autodetect(cpu)) {
		if (!ret && caps.sse)
			return create_unresize_impl_h_ge_sse(context, height, type);
	} else {
		if (!ret && cpu >= CPUClass::X86_SSE)
			return create_unresize_impl_h_ge_sse(context, height, type);
	}

	return ret;
}

std::unique_ptr<graph::ImageFilter> create_unresize_impl_v_x86(const BilinearContext &context, unsigned width, PixelType type, CPUClass cpu)
{
	X86Capabilities caps = query_x86_capabilities();
	std::unique_ptr<graph::ImageFilter> ret;

	if (cpu_is_autodetect(cpu)) {
		if (!ret && caps.sse)
			return create_unresize_impl_v_sse(context, width, type);
	} else {
		if (cpu >= CPUClass::X86_SSE)
			return create_unresize_impl_v_sse(context, width, type);
	}

	return ret;
}

std::unique_ptr<graphengine::Filter> create_unresize_impl_v_ge_x86(const BilinearContext &context, unsigned width, PixelType type, CPUClass cpu)
{
	X86Capabilities caps = query_x86_capabilities();
	std::unique_ptr<graphengine::Filter> ret;

	if (cpu_is_autodetect(cpu)) {
		if (!ret && caps.sse)
			return create_unresize_impl_v_ge_sse(context, width, type);
	} else {
		if (cpu >= CPUClass::X86_SSE)
			return create_unresize_impl_v_ge_sse(context, width, type);
	}

	return ret;
}

} // namespace unresize
} // namespace zimg

#endif // ZIMG_X86
