#ifdef ZIMG_ARM

#include "common/cpuinfo.h"
#include "common/arm/cpuinfo_arm.h"
#include "common/pixel.h"
#include "depth_convert_arm.h"
#include "f16c_arm.h"

namespace zimg {
namespace depth {

namespace {

left_shift_func select_left_shift_func_neon(PixelType pixel_in, PixelType pixel_out)
{
	if (pixel_in == PixelType::BYTE && pixel_out == PixelType::BYTE)
		return left_shift_b2b_neon;
	else if (pixel_in == PixelType::BYTE && pixel_out == PixelType::WORD)
		return left_shift_b2w_neon;
	else if (pixel_in == PixelType::WORD && pixel_out == PixelType::BYTE)
		return left_shift_w2b_neon;
	else if (pixel_in == PixelType::WORD && pixel_out == PixelType::WORD)
		return left_shift_w2w_neon;
	else
		return nullptr;
}

depth_convert_func select_depth_convert_func_neon(PixelType pixel_in, PixelType pixel_out)
{
#if defined(_MSC_VER) && !defined(_M_ARM64)
	if (pixel_out == PixelType::HALF)
		pixel_out = PixelType::FLOAT;
#endif

#if !defined(_MSC_VER) || defined(_M_ARM64)
	if (pixel_in == PixelType::BYTE && pixel_out == PixelType::HALF)
		return depth_convert_b2h_neon;
	else
#endif
	if (pixel_in == PixelType::BYTE && pixel_out == PixelType::FLOAT)
		return depth_convert_b2f_neon;
#if !defined(_MSC_VER) || defined(_M_ARM64)
	else if (pixel_in == PixelType::WORD && pixel_out == PixelType::HALF)
		return depth_convert_w2h_neon;
#endif
	else if (pixel_in == PixelType::WORD && pixel_out == PixelType::FLOAT)
		return depth_convert_w2f_neon;
	else
		return nullptr;
}

} // namespace


left_shift_func select_left_shift_func_arm(PixelType pixel_in, PixelType pixel_out, CPUClass cpu)
{
	ARMCapabilities caps = query_arm_capabilities();
	left_shift_func func = nullptr;

	if (cpu_is_autodetect(cpu)) {
		if (!func && caps.neon)
			func = select_left_shift_func_neon(pixel_in, pixel_out);
	} else {
		if (!func && cpu >= CPUClass::ARM_NEON)
			func = select_left_shift_func_neon(pixel_in, pixel_out);
	}

	return func;
}

depth_convert_func select_depth_convert_func_arm(const PixelFormat &format_in, const PixelFormat &format_out, CPUClass cpu)
{
	ARMCapabilities caps = query_arm_capabilities();
	depth_convert_func func = nullptr;

	if (cpu_is_autodetect(cpu)) {
		if (!func && caps.neon && caps.vfpv4)
			func = select_depth_convert_func_neon(format_in.type, format_out.type);
	} else {
		if (!func && cpu >= CPUClass::ARM_NEON)
			func = select_depth_convert_func_neon(format_in.type, format_out.type);
	}

	return func;
}

depth_f16c_func select_depth_f16c_func_arm(bool to_half, CPUClass cpu)
{
	ARMCapabilities caps = query_arm_capabilities();
	depth_f16c_func func = nullptr;

#if !defined(_MSC_VER) || defined(_M_ARM64)
	if (cpu_is_autodetect(cpu)) {
		if (!func && caps.neon && caps.vfpv4)
			func = to_half ? f16c_float_to_half_neon : f16c_half_to_float_neon;
	} else {
		if (!func && cpu >= CPUClass::ARM_NEON)
			func = to_half ? f16c_float_to_half_neon : f16c_half_to_float_neon;
	}
#endif

	return func;
}

bool needs_depth_f16c_func_arm(const PixelFormat &format_in, const PixelFormat &format_out, CPUClass cpu)
{
	ARMCapabilities caps = query_arm_capabilities();
	bool value = format_in.type == PixelType::HALF || format_out.type == PixelType::HALF;

#if !defined(_MSC_VER) || defined(_M_ARM64)
	if ((cpu_is_autodetect(cpu) && caps.neon && caps.vfpv4) || cpu >= CPUClass::ARM_NEON)
		value = value && pixel_is_float(format_in.type) && pixel_is_float(format_out.type);
#endif

	return value;
}

} // namespace depth
} // namespace zimg

#endif // ZIMG_ARM
