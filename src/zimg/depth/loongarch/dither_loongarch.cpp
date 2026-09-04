#ifdef ZIMG_LOONGARCH

#include "common/cpuinfo.h"
#include "common/loongarch/cpuinfo_loongarch.h"
#include "common/pixel.h"
#include "graphengine/filter.h"
#include "dither_loongarch.h"

namespace zimg::depth {

namespace {

dither_convert_func select_ordered_dither_func_lsx(
	PixelType pixel_in, PixelType pixel_out)
{
	if (pixel_in == PixelType::BYTE && pixel_out == PixelType::BYTE)
		return ordered_dither_b2b_lsx;
	else if (pixel_in == PixelType::BYTE && pixel_out == PixelType::WORD)
		return ordered_dither_b2w_lsx;
	else if (pixel_in == PixelType::WORD && pixel_out == PixelType::BYTE)
		return ordered_dither_w2b_lsx;
	else if (pixel_in == PixelType::WORD && pixel_out == PixelType::WORD)
		return ordered_dither_w2w_lsx;
	else if (pixel_in == PixelType::HALF && pixel_out == PixelType::BYTE)
		return ordered_dither_h2b_lsx;
	else if (pixel_in == PixelType::HALF && pixel_out == PixelType::WORD)
		return ordered_dither_h2w_lsx;
	else if (pixel_in == PixelType::FLOAT && pixel_out == PixelType::BYTE)
		return ordered_dither_f2b_lsx;
	else if (pixel_in == PixelType::FLOAT && pixel_out == PixelType::WORD)
		return ordered_dither_f2w_lsx;
	else
		return nullptr;
}

} // namespace


dither_convert_func select_ordered_dither_func_loongarch(
	const PixelFormat &pixel_in, const PixelFormat &pixel_out, CPUClass cpu)
{
	LoongArchCapabilities caps = query_loongarch_capabilities();
	dither_convert_func func = nullptr;

	if (cpu_is_autodetect(cpu)) {
		if (!func && caps.lsx)
			func = select_ordered_dither_func_lsx(
				pixel_in.type, pixel_out.type);
	} else {
		if (!func && cpu >= CPUClass::LOONGARCH_LSX)
			func = select_ordered_dither_func_lsx(
				pixel_in.type, pixel_out.type);
	}

	return func;
}

std::unique_ptr<graphengine::Filter> create_error_diffusion_loongarch(
	unsigned width, unsigned height, const PixelFormat &pixel_in,
	const PixelFormat &pixel_out, CPUClass cpu)
{
	LoongArchCapabilities caps = query_loongarch_capabilities();
	std::unique_ptr<graphengine::Filter> ret;

	if (cpu_is_autodetect(cpu)) {
		if (!ret && caps.lsx)
			ret = create_error_diffusion_lsx(
				width, height, pixel_in, pixel_out);
	} else {
		if (!ret && cpu >= CPUClass::LOONGARCH_LSX)
			ret = create_error_diffusion_lsx(
				width, height, pixel_in, pixel_out);
	}

	return ret;
}

} // namespace zimg::depth

#endif // ZIMG_LOONGARCH
