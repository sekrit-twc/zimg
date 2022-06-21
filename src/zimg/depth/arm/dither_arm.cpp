#ifdef ZIMG_ARM

#include "common/cpuinfo.h"
#include "common/pixel.h"
#include "common/arm/cpuinfo_arm.h"
#include "dither_arm.h"
#include "f16c_arm.h"

namespace zimg {
namespace depth {

namespace {

dither_convert_func select_ordered_dither_func_neon(PixelType pixel_in, PixelType pixel_out)
{
#if defined(_MSC_VER) && !defined(_M_ARM64)
	if (pixel_in == PixelType::HALF)
		pixel_in = PixelType::FLOAT;
#endif

	if (pixel_in == PixelType::BYTE && pixel_out == PixelType::BYTE)
		return ordered_dither_b2b_neon;
	else if (pixel_in == PixelType::BYTE && pixel_out == PixelType::WORD)
		return ordered_dither_b2w_neon;
	else if (pixel_in == PixelType::WORD && pixel_out == PixelType::BYTE)
		return ordered_dither_w2b_neon;
	else if (pixel_in == PixelType::WORD && pixel_out == PixelType::WORD)
		return ordered_dither_w2w_neon;
#if !defined(_MSC_VER) || defined(_M_ARM64)
	else if (pixel_in == PixelType::HALF && pixel_out == PixelType::BYTE)
		return ordered_dither_h2b_neon;
	else if (pixel_in == PixelType::HALF && pixel_out == PixelType::WORD)
		return ordered_dither_h2w_neon;
#endif
	else if (pixel_in == PixelType::FLOAT && pixel_out == PixelType::BYTE)
		return ordered_dither_f2b_neon;
	else if (pixel_in == PixelType::FLOAT && pixel_out == PixelType::WORD)
		return ordered_dither_f2w_neon;
	else
		return nullptr;
}

} // namespace


dither_convert_func select_ordered_dither_func_arm(const PixelFormat &pixel_in, const PixelFormat &pixel_out, CPUClass cpu)
{
	ARMCapabilities caps = query_arm_capabilities();
	dither_convert_func func = nullptr;

	if (cpu_is_autodetect(cpu)) {
		if (!func && caps.neon && caps.vfpv4)
			func = select_ordered_dither_func_neon(pixel_in.type, pixel_out.type);
	} else {
		if (!func && cpu >= CPUClass::ARM_NEON)
			func = select_ordered_dither_func_neon(pixel_in.type, pixel_out.type);
	}

	return func;
}

dither_f16c_func select_dither_f16c_func_arm(CPUClass cpu)
{
	ARMCapabilities caps = query_arm_capabilities();
	dither_f16c_func func = nullptr;

#if !defined(_MSC_VER) || defined(_M_ARM64)
	if (cpu_is_autodetect(cpu)) {
		if (!func && caps.neon && caps.vfpv4)
			func = f16c_half_to_float_neon;
	} else {
		if (!func && cpu >= CPUClass::ARM_NEON)
			func = f16c_half_to_float_neon;
	}
#endif

	return func;
}

bool needs_dither_f16c_func_arm(CPUClass cpu)
{
#if defined(_MSC_VER) && !defined(_M_ARM64)
	return true;
#else
	ARMCapabilities caps = query_arm_capabilities();

	if (cpu_is_autodetect(cpu))
		return !caps.neon || !caps.vfpv4;
	else
		return cpu < CPUClass::ARM_NEON;
#endif
}

} // namespace depth
} // namespace zimg

#endif // ZIMG_ARM
