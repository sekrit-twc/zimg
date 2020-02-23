#ifdef ZIMG_ARM

#include "common/cpuinfo.h"
#include "common/arm/cpuinfo_arm.h"
#include "common/pixel.h"
#include "depth_convert_arm.h"

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

} // namespace depth
} // namespace zimg

#endif // ZIMG_ARM
