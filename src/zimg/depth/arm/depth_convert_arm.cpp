#ifdef ZIMG_ARM

#include "common/cpuinfo.h"
#include "common/arm/cpuinfo_arm.h"
#include "common/pixel.h"
#include "depth_convert_arm.h"

namespace zimg::depth {

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
	if (pixel_in == PixelType::BYTE && pixel_out == PixelType::HALF)
		return depth_convert_b2h_neon;
	else if (pixel_in == PixelType::BYTE && pixel_out == PixelType::FLOAT)
		return depth_convert_b2f_neon;
	else if (pixel_in == PixelType::WORD && pixel_out == PixelType::HALF)
		return depth_convert_w2h_neon;
	else if (pixel_in == PixelType::WORD && pixel_out == PixelType::FLOAT)
		return depth_convert_w2f_neon;
	else if (pixel_in == PixelType::HALF && pixel_out == PixelType::FLOAT)
		return half_to_float_neon;
	else if (pixel_in == PixelType::FLOAT && pixel_out == PixelType::HALF)
		return float_to_half_neon;
	else
		return nullptr;
}

} // namespace


left_shift_func select_left_shift_func_arm(PixelType pixel_in, PixelType pixel_out, CPUClass cpu)
{
	ARMCapabilities caps = query_arm_capabilities();
	left_shift_func func = nullptr;

	if (cpu_is_autodetect(cpu)) {
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
		func = select_depth_convert_func_neon(format_in.type, format_out.type);
	} else {
		if (!func && cpu >= CPUClass::ARM_NEON)
			func = select_depth_convert_func_neon(format_in.type, format_out.type);
	}

	return func;
}

} // namespace zimg::depth

#endif // ZIMG_ARM
