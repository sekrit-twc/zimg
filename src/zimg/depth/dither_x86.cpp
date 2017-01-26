#ifdef ZIMG_X86

#include "common/cpuinfo.h"
#include "common/pixel.h"
#include "graph/image_filter.h"
#include "dither_x86.h"
#include "f16c_x86.h"

namespace zimg {
namespace depth {

namespace {

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

dither_convert_func select_ordered_dither_func_avx2(PixelType pixel_in, PixelType pixel_out)
{
	if (pixel_in == PixelType::BYTE && pixel_out == PixelType::BYTE)
		return ordered_dither_b2b_avx2;
	else if (pixel_in == PixelType::BYTE && pixel_out == PixelType::WORD)
		return ordered_dither_b2w_avx2;
	else if (pixel_in == PixelType::WORD && pixel_out == PixelType::BYTE)
		return ordered_dither_w2b_avx2;
	else if (pixel_in == PixelType::WORD && pixel_out == PixelType::WORD)
		return ordered_dither_w2w_avx2;
	else if (pixel_in == PixelType::HALF && pixel_out == PixelType::BYTE)
		return ordered_dither_h2b_avx2;
	else if (pixel_in == PixelType::HALF && pixel_out == PixelType::WORD)
		return ordered_dither_h2w_avx2;
	else if (pixel_in == PixelType::FLOAT && pixel_out == PixelType::BYTE)
		return ordered_dither_f2b_avx2;
	else if (pixel_in == PixelType::FLOAT && pixel_out == PixelType::WORD)
		return ordered_dither_f2w_avx2;
	else
		return nullptr;
}

} // namespace


dither_convert_func select_ordered_dither_func_x86(const PixelFormat &pixel_in, const PixelFormat &pixel_out, CPUClass cpu)
{
	X86Capabilities caps = query_x86_capabilities();
	dither_convert_func func = nullptr;

	if (cpu == CPUClass::AUTO) {
		if (!func && caps.avx2 && caps.fma)
			func = select_ordered_dither_func_avx2(pixel_in.type, pixel_out.type);
		if (!func && caps.sse2)
			func = select_ordered_dither_func_sse2(pixel_in.type, pixel_out.type);
	} else {
		if (!func && cpu >= CPUClass::X86_AVX2)
			func = select_ordered_dither_func_avx2(pixel_in.type, pixel_out.type);
		if (!func && cpu >= CPUClass::X86_SSE2)
			func = select_ordered_dither_func_sse2(pixel_in.type, pixel_out.type);
	}

	return func;
}

dither_f16c_func select_dither_f16c_func_x86(CPUClass cpu)
{
	X86Capabilities caps = query_x86_capabilities();
	dither_f16c_func func = nullptr;

	if (cpu == CPUClass::AUTO) {
		if (!func && caps.avx && caps.f16c)
			func = f16c_half_to_float_ivb;
		if (!func && caps.sse2)
			func = f16c_half_to_float_sse2;
	} else {
		if (!func && cpu >= CPUClass::X86_F16C)
			func = f16c_half_to_float_ivb;
		if (!func && cpu >= CPUClass::X86_SSE2)
			func = f16c_half_to_float_sse2;
	}

	return func;
}

bool needs_dither_f16c_func_x86(CPUClass cpu)
{
	X86Capabilities caps = query_x86_capabilities();
	return (cpu == CPUClass::AUTO && !caps.avx2) || cpu < CPUClass::X86_AVX2;
}

std::unique_ptr<graph::ImageFilter> create_error_diffusion_x86(unsigned width, unsigned height, const PixelFormat &pixel_in, const PixelFormat &pixel_out, CPUClass cpu)
{
	X86Capabilities caps = query_x86_capabilities();
	std::unique_ptr<graph::ImageFilter> ret;

	if (cpu == CPUClass::AUTO) {
		if (!ret && caps.avx2 && caps.f16c && caps.fma)
			ret = create_error_diffusion_avx2(width, height, pixel_in, pixel_out);
		if (!ret && caps.sse2)
			ret = create_error_diffusion_sse2(width, height, pixel_in, pixel_out, cpu);
	} else {
		if (!ret && cpu >= CPUClass::X86_AVX2)
			ret = create_error_diffusion_avx2(width, height, pixel_in, pixel_out);
		if (!ret && cpu >= CPUClass::X86_SSE2)
			ret = create_error_diffusion_sse2(width, height, pixel_in, pixel_out, cpu);
	}

	return ret;
}

} // namespace depth
} // namespace zimg

#endif // ZIMG_X86
