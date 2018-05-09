#ifdef ZIMG_X86

#include "common/cpuinfo.h"
#include "common/x86/cpuinfo_x86.h"
#include "common/pixel.h"
#include "depth_convert_x86.h"
#include "f16c_x86.h"

namespace zimg {
namespace depth {

namespace {

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

depth_convert_func select_depth_convert_func_avx2(PixelType pixel_in, PixelType pixel_out)
{
	if (pixel_in == PixelType::BYTE && pixel_out == PixelType::HALF)
		return depth_convert_b2h_avx2;
	else if (pixel_in == PixelType::BYTE && pixel_out == PixelType::FLOAT)
		return depth_convert_b2f_avx2;
	else if (pixel_in == PixelType::WORD && pixel_out == PixelType::HALF)
		return depth_convert_w2h_avx2;
	else if (pixel_in == PixelType::WORD && pixel_out == PixelType::FLOAT)
		return depth_convert_w2f_avx2;
	else
		return nullptr;
}

#ifdef ZIMG_X86_AVX512
depth_convert_func select_depth_convert_func_avx512(PixelType pixel_in, PixelType pixel_out)
{
	if (pixel_in == PixelType::BYTE && pixel_out == PixelType::HALF)
		return depth_convert_b2h_avx512;
	else if (pixel_in == PixelType::BYTE && pixel_out == PixelType::FLOAT)
		return depth_convert_b2f_avx512;
	else if (pixel_in == PixelType::WORD && pixel_out == PixelType::HALF)
		return depth_convert_w2h_avx512;
	else if (pixel_in == PixelType::WORD && pixel_out == PixelType::FLOAT)
		return depth_convert_w2f_avx512;
	else
		return nullptr;
}
#endif // ZIMG_X86_AVX512

} // namespace


left_shift_func select_left_shift_func_x86(PixelType pixel_in, PixelType pixel_out, CPUClass cpu)
{
	X86Capabilities caps = query_x86_capabilities();
	left_shift_func func = nullptr;

	if (cpu_is_autodetect(cpu)) {
		if (!func && caps.sse2)
			func = select_left_shift_func_sse2(pixel_in, pixel_out);
	} else {
		if (!func && cpu >= CPUClass::X86_SSE2)
			func = select_left_shift_func_sse2(pixel_in, pixel_out);
	}

	return func;
}

depth_convert_func select_depth_convert_func_x86(const PixelFormat &format_in, const PixelFormat &format_out, CPUClass cpu)
{
	X86Capabilities caps = query_x86_capabilities();
	depth_convert_func func = nullptr;

	if (cpu_is_autodetect(cpu)) {
#ifdef ZIMG_X86_AVX512
		if (!func && cpu == CPUClass::AUTO_64B && caps.avx512f && caps.avx512bw && caps.avx512vl)
			func = select_depth_convert_func_avx512(format_in.type, format_out.type);
#endif
		if (!func && caps.avx2 && caps.fma)
			func = select_depth_convert_func_avx2(format_in.type, format_out.type);
		if (!func && caps.sse2)
			func = select_depth_convert_func_sse2(format_in.type, format_out.type);
	} else {
#ifdef ZIMG_X86_AVX512
		if (!func && cpu >= CPUClass::X86_AVX512)
			func = select_depth_convert_func_avx512(format_in.type, format_out.type);
#endif
		if (!func && cpu >= CPUClass::X86_AVX2)
			func = select_depth_convert_func_avx2(format_in.type, format_out.type);
		if (!func && cpu >= CPUClass::X86_SSE2)
			func = select_depth_convert_func_sse2(format_in.type, format_out.type);
	}

	return func;
}

depth_f16c_func select_depth_f16c_func_x86(bool to_half, CPUClass cpu)
{
	X86Capabilities caps = query_x86_capabilities();
	depth_f16c_func func = nullptr;

	if (cpu_is_autodetect(cpu)) {
		if (!func && caps.avx && caps.f16c)
			func = to_half ? f16c_float_to_half_ivb : f16c_half_to_float_ivb;
		if (!func && caps.sse2)
			func = to_half ? f16c_float_to_half_sse2 : f16c_half_to_float_sse2;
	} else {
		if (!func && cpu >= CPUClass::X86_F16C)
			func = to_half ? f16c_float_to_half_ivb : f16c_half_to_float_ivb;
		if (!func && cpu >= CPUClass::X86_SSE2)
			func = to_half ? f16c_float_to_half_sse2 : f16c_half_to_float_sse2;
	}

	return func;
}

bool needs_depth_f16c_func_x86(const PixelFormat &format_in, const PixelFormat &format_out, CPUClass cpu)
{
	X86Capabilities caps = query_x86_capabilities();
	bool value = format_in.type == PixelType::HALF || format_out.type == PixelType::HALF;

	if ((cpu_is_autodetect(cpu) && caps.avx2) || cpu >= CPUClass::X86_AVX2)
		value = value && pixel_is_float(format_in.type) && pixel_is_float(format_out.type);

	return value;
}

} // namespace depth
} // namespace zimg

#endif // ZIMG_X86
