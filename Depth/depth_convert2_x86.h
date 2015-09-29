#pragma once

#ifdef ZIMG_X86

#ifndef ZIMG_DEPTH_DEPTH_CONVERT2_X86_H_
#define ZIMG_DEPTH_DEPTH_CONVERT2_X86_H_

#include "depth_convert2.h"

namespace zimg {;
namespace depth {;

left_shift_func select_left_shift_func_x86(PixelType pixel_in, PixelType pixel_out, CPUClass cpu);

depth_convert_func select_depth_convert_func_x86(const PixelFormat &format_in, const PixelFormat &format_out, CPUClass cpu);

depth_f16c_func select_depth_f16c_func_x86(bool to_half, CPUClass cpu);

} // depth
} // zimg

#endif // ZIMG_DEPTH_DEPTH_CONVERT2_X86_H_

#endif // ZIMG_X86
