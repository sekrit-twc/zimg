#ifdef ZIMG_X86

#include "depth_convert2_x86.h"

namespace zimg {;
namespace depth {;

left_shift_func select_left_shift_func_x86(PixelType pixel_in, PixelType pixel_out, CPUClass cpu)
{
	return nullptr;
}

depth_convert_func select_depth_convert_func_x86(const PixelFormat &format_in, const PixelFormat &format_out, CPUClass cpu)
{
	return nullptr;
}

depth_f16c_func select_depth_f16c_func_x86(bool to_half, CPUClass cpu)
{
	return nullptr;
}

} // namespace depth
} // namespace zimg

#endif // ZIMG_X86
