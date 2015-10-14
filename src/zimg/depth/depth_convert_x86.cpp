#ifdef ZIMG_X86

#include "common/cpuinfo.h"
#include "common/pixel.h"
#include "depth_convert_x86.h"

namespace zimg {;
namespace depth {;

namespace {;

left_shift_func select_left_shift_func_sse2(PixelType pixel_in, PixelType pixel_out)
{
	if (pixel_in == PixelType::BYTE && pixel_out == PixelType::BYTE)
		return left_shift_b2b_sse2;
	else if (pixel_in == PixelType::BYTE && pixel_out == PixelType::WORD)
		return left_shift_b2w_sse2;
	else if (pixel_in == PixelType::WORD && pixel_out == PixelType::BYTE)
		return left_shift_w2b_sse2;
	else if (pixel_in == PixelType::WORD && pixel_out == PixelType::WORD)
		return left_shift_w2w_sse2;
	else
		return nullptr;
}

depth_convert_func select_depth_convert_func_sse2(PixelType pixel_in, PixelType pixel_out)
{
	if (pixel_out == PixelType::HALF)
		pixel_out = PixelType::FLOAT;

	if (pixel_in == PixelType::BYTE && pixel_out == PixelType::FLOAT)
		return depth_convert_b2f_sse2;
	else if (pixel_in == PixelType::WORD && pixel_out == PixelType::FLOAT)
		return depth_convert_w2f_sse2;
	else
		return nullptr;
}

} // namespace


left_shift_func select_left_shift_func_x86(PixelType pixel_in, PixelType pixel_out, CPUClass cpu)
{
	X86Capabilities caps = query_x86_capabilities();
	left_shift_func func = nullptr;

	if (cpu == CPUClass::CPU_AUTO) {
		if (!func && caps.sse2)
			func = select_left_shift_func_sse2(pixel_in, pixel_out);
	} else {
		if (!func && cpu >= CPUClass::CPU_X86_SSE2)
			func = select_left_shift_func_sse2(pixel_in, pixel_out);
	}

	return func;
}

depth_convert_func select_depth_convert_func_x86(const PixelFormat &format_in, const PixelFormat &format_out, CPUClass cpu)
{
	X86Capabilities caps = query_x86_capabilities();
	depth_convert_func func = nullptr;

	if (cpu == CPUClass::CPU_AUTO) {
		if (!func && caps.sse2)
			func = select_depth_convert_func_sse2(format_in.type, format_out.type);
	} else {
		if (!func && cpu >= CPUClass::CPU_X86_SSE2)
			func = select_depth_convert_func_sse2(format_in.type, format_out.type);
	}

	return func;
}

depth_f16c_func select_depth_f16c_func_x86(bool to_half, CPUClass cpu)
{
	return nullptr;
}

} // namespace depth
} // namespace zimg

#endif // ZIMG_X86
