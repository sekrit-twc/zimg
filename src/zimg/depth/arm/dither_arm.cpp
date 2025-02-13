#ifdef ZIMG_ARM

#include "common/cpuinfo.h"
#include "common/pixel.h"
#include "common/arm/cpuinfo_arm.h"
#include "dither_arm.h"

namespace zimg::depth {

namespace {

dither_convert_func select_ordered_dither_func_neon(PixelType pixel_in, PixelType pixel_out)
{
	if (pixel_in == PixelType::BYTE && pixel_out == PixelType::BYTE)
		return ordered_dither_b2b_neon;
	else if (pixel_in == PixelType::BYTE && pixel_out == PixelType::WORD)
		return ordered_dither_b2w_neon;
	else if (pixel_in == PixelType::WORD && pixel_out == PixelType::BYTE)
		return ordered_dither_w2b_neon;
	else if (pixel_in == PixelType::WORD && pixel_out == PixelType::WORD)
		return ordered_dither_w2w_neon;
	else if (pixel_in == PixelType::HALF && pixel_out == PixelType::BYTE)
		return ordered_dither_h2b_neon;
	else if (pixel_in == PixelType::HALF && pixel_out == PixelType::WORD)
		return ordered_dither_h2w_neon;
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
		func = select_ordered_dither_func_neon(pixel_in.type, pixel_out.type);
	} else {
		if (!func && cpu >= CPUClass::ARM_NEON)
			func = select_ordered_dither_func_neon(pixel_in.type, pixel_out.type);
	}

	return func;
}

} // namespace zimg::depth

#endif // ZIMG_ARM
