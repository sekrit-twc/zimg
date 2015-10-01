#ifdef ZIMG_X86

#include "common/cpuinfo.h"
#include "common/pixel.h"
#include "dither_x86.h"

namespace zimg {;
namespace depth {;

namespace {;

dither_convert_func select_ordered_dither_func_sse2(PixelType pixel_in, PixelType pixel_out)
{
	if (pixel_in == PixelType::HALF)
		pixel_in = PixelType::FLOAT;

	if (pixel_in == PixelType::BYTE && pixel_out == PixelType::BYTE)
		return ordered_dither_b2b_sse2;
	else if (pixel_in == PixelType::BYTE && pixel_out == PixelType::WORD)
		return ordered_dither_b2w_sse2;
	else if (pixel_in == PixelType::WORD && pixel_out == PixelType::BYTE)
		return ordered_dither_w2b_sse2;
	else if (pixel_in == PixelType::WORD && pixel_out == PixelType::WORD)
		return ordered_dither_w2w_sse2;
	else if (pixel_in == PixelType::FLOAT && pixel_out == PixelType::BYTE)
		return ordered_dither_f2b_sse2;
	else if (pixel_in == PixelType::FLOAT && pixel_out == PixelType::WORD)
		return ordered_dither_f2w_sse2;
	else
		return nullptr;
}

} // namespace


dither_convert_func select_ordered_dither_func_x86(const PixelFormat &pixel_in, const PixelFormat &pixel_out, CPUClass cpu)
{
	X86Capabilities caps = query_x86_capabilities();
	dither_convert_func func = nullptr;

	if (cpu == CPUClass::CPU_AUTO) {
		if (!func && caps.sse2)
			func = select_ordered_dither_func_sse2(pixel_in.type, pixel_out.type);
	} else {
		if (!func && cpu >= CPUClass::CPU_X86_SSE2)
			func = select_ordered_dither_func_sse2(pixel_in.type, pixel_out.type);
	}

	return func;
}

dither_f16c_func select_dither_f16c_func_x86(CPUClass cpu)
{
	return nullptr;
}

} // namespace depth
} // namespace zimg

#endif // ZIMG_X86
