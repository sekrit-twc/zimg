#pragma once

#ifdef ZIMG_X86

#ifndef ZIMG_DEPTH_DITHER2_X86_H_
#define ZIMG_DEPTH_DITHER2_X86_H_

#include "dither2.h"

namespace zimg {;
namespace depth {;

dither_convert_func select_ordered_dither_func_x86(const PixelFormat &pixel_in, const PixelFormat &pixel_out, CPUClass cpu);

dither_f16c_func select_dither_f16c_func_x86(CPUClass cpu);

} // depth
} // zimg

#endif // ZIMG_DEPTH_DITHER2_X86_H_

#endif // ZIMG_X86
