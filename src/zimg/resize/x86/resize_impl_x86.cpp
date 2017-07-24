#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
  #undef ZIMG_X86_AVX512
#endif

#ifdef ZIMG_X86

#include "common/cpuinfo.h"
#include "common/x86/cpuinfo_x86.h"
#include "graph/image_filter.h"
#include "resize_impl_x86.h"

namespace zimg {
namespace resize {

std::unique_ptr<graph::ImageFilter> create_resize_impl_h_x86(const FilterContext &context, unsigned height, PixelType type, unsigned depth, CPUClass cpu)
{
	X86Capabilities caps = query_x86_capabilities();
	std::unique_ptr<graph::ImageFilter> ret;

	if (cpu_is_autodetect(cpu)) {
#ifdef ZIMG_X86_AVX512
		if (!ret && cpu == CPUClass::AUTO_64B && caps.avx512f && caps.avx512dq && caps.avx512bw && caps.avx512vl)
			ret = create_resize_impl_h_avx512(context, height, type, depth);
#endif
		if (!ret && caps.avx2)
			ret = create_resize_impl_h_avx2(context, height, type, depth);
		if (!ret && caps.avx)
			ret = create_resize_impl_h_avx(context, height, type, depth);
		if (!ret && caps.sse2)
			ret = create_resize_impl_h_sse2(context, height, type, depth);
		if (!ret && caps.sse)
			ret = create_resize_impl_h_sse(context, height, type, depth);
	} else {
#ifdef ZIMG_X86_AVX512
		if (!ret && cpu >= CPUClass::X86_AVX512)
			ret = create_resize_impl_h_avx512(context, height, type, depth);
#endif
		if (!ret && cpu >= CPUClass::X86_AVX2)
			ret = create_resize_impl_h_avx2(context, height, type, depth);
		if (!ret && cpu >= CPUClass::X86_AVX)
			ret = create_resize_impl_h_avx(context, height, type, depth);
		if (!ret && cpu >= CPUClass::X86_SSE2)
			ret = create_resize_impl_h_sse2(context, height, type, depth);
		if (!ret && cpu >= CPUClass::X86_SSE)
			ret = create_resize_impl_h_sse(context, height, type, depth);
	}

	return ret;
}

std::unique_ptr<graph::ImageFilter> create_resize_impl_v_x86(const FilterContext &context, unsigned width, PixelType type, unsigned depth, CPUClass cpu)
{
	X86Capabilities caps = query_x86_capabilities();
	std::unique_ptr<graph::ImageFilter> ret;

	if (cpu_is_autodetect(cpu)) {
#ifdef ZIMG_X86_AVX512
		if (!ret && cpu == CPUClass::AUTO_64B && caps.avx512f && caps.avx512dq && caps.avx512bw && caps.avx512vl)
			ret = create_resize_impl_v_avx512(context, width, type, depth);
#endif
		if (!ret && caps.avx2)
			ret = create_resize_impl_v_avx2(context, width, type, depth);
		if (!ret && caps.avx)
			ret = create_resize_impl_v_avx(context, width, type, depth);
		if (!ret && caps.sse2)
			ret = create_resize_impl_v_sse2(context, width, type, depth);
		if (!ret && caps.sse)
			ret = create_resize_impl_v_sse(context, width, type, depth);
	} else {
#ifdef ZIMG_X86_AVX512
		if (!ret && cpu >= CPUClass::X86_AVX512)
			ret = create_resize_impl_v_avx512(context, width, type, depth);
#endif
		if (!ret && cpu >= CPUClass::X86_AVX2)
			ret = create_resize_impl_v_avx2(context, width, type, depth);
		if (!ret && cpu >= CPUClass::X86_AVX)
			ret = create_resize_impl_v_avx(context, width, type, depth);
		if (!ret && cpu >= CPUClass::X86_SSE2)
			ret = create_resize_impl_v_sse2(context, width, type, depth);
		if (!ret && cpu >= CPUClass::X86_SSE)
			ret = create_resize_impl_v_sse(context, width, type, depth);
	}

	return ret;
}

} // namespace resize
} // namespace zimg

#endif // ZIMG_X86
