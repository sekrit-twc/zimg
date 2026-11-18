#ifdef ZIMG_LOONGARCH

#include "common/cpuinfo.h"
#include "common/loongarch/cpuinfo_loongarch.h"
#include "common/pixel.h"
#include "depth_convert_loongarch.h"

namespace zimg::depth {

namespace {

left_shift_func select_left_shift_func_lsx(
	PixelType pixel_in, PixelType pixel_out)
{
	if (pixel_in == PixelType::BYTE && pixel_out == PixelType::BYTE)
		return left_shift_b2b_lsx;
	else if (pixel_in == PixelType::BYTE && pixel_out == PixelType::WORD)
		return left_shift_b2w_lsx;
	else if (pixel_in == PixelType::WORD && pixel_out == PixelType::BYTE)
		return left_shift_w2b_lsx;
	else if (pixel_in == PixelType::WORD && pixel_out == PixelType::WORD)
		return left_shift_w2w_lsx;
	else
		return nullptr;
}

depth_convert_func select_depth_convert_func_lsx(
	PixelType pixel_in, PixelType pixel_out)
{
	if (pixel_in == PixelType::BYTE && pixel_out == PixelType::HALF)
		return depth_convert_b2h_lsx;
	else if (pixel_in == PixelType::BYTE && pixel_out == PixelType::FLOAT)
		return depth_convert_b2f_lsx;
	else if (pixel_in == PixelType::WORD && pixel_out == PixelType::HALF)
		return depth_convert_w2h_lsx;
	else if (pixel_in == PixelType::WORD && pixel_out == PixelType::FLOAT)
		return depth_convert_w2f_lsx;
	else if (pixel_in == PixelType::HALF && pixel_out == PixelType::FLOAT)
		return half_to_float_lsx;
	else if (pixel_in == PixelType::FLOAT && pixel_out == PixelType::HALF)
		return float_to_half_lsx;
	else
		return nullptr;
}

} // namespace


left_shift_func select_left_shift_func_loongarch(
	PixelType pixel_in, PixelType pixel_out, CPUClass cpu)
{
	LoongArchCapabilities caps = query_loongarch_capabilities();
	left_shift_func func = nullptr;

	if (cpu_is_autodetect(cpu)) {
		if (!func && caps.lsx)
			func = select_left_shift_func_lsx(pixel_in, pixel_out);
	} else {
		if (!func && cpu >= CPUClass::LOONGARCH_LSX)
			func = select_left_shift_func_lsx(pixel_in, pixel_out);
	}

	return func;
}

depth_convert_func select_depth_convert_func_loongarch(
	const PixelFormat &format_in, const PixelFormat &format_out,
	CPUClass cpu)
{
	LoongArchCapabilities caps = query_loongarch_capabilities();
	depth_convert_func func = nullptr;

	if (cpu_is_autodetect(cpu)) {
		if (!func && caps.lsx)
			func = select_depth_convert_func_lsx(
				format_in.type, format_out.type);
	} else {
		if (!func && cpu >= CPUClass::LOONGARCH_LSX)
			func = select_depth_convert_func_lsx(
				format_in.type, format_out.type);
	}

	return func;
}

} // namespace zimg::depth

#endif // ZIMG_LOONGARCH
