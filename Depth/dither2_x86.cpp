#ifdef ZIMG_X86

#include "dither2_x86.h"

namespace zimg {;
namespace depth {;

dither_convert_func select_ordered_dither_func_x86(const PixelFormat &pixel_in, const PixelFormat &pixel_out, CPUClass cpu)
{
	return nullptr;
}

dither_f16c_func select_dither_f16c_func_x86(CPUClass cpu)
{
	return nullptr;
}

} // namespace depth
} // namespace zimg

#endif // ZIMG_X86
