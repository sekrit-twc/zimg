#pragma once

#ifdef ZIMG_ARM

#ifndef ZIMG_DEPTH_ARM_DEPTH_CONVERT_ARM_H_
#define ZIMG_DEPTH_ARM_DEPTH_CONVERT_ARM_H_

#include "depth/depth_convert.h"

namespace zimg::depth {

#define DECLARE_LEFT_SHIFT(x, cpu) \
void left_shift_##x##_##cpu(const void *src, void *dst, unsigned shift, unsigned left, unsigned right)
#define DECLARE_DEPTH_CONVERT(x, cpu) \
void depth_convert_##x##_##cpu(const void *src, void *dst, float scale, float offset, unsigned left, unsigned right)

DECLARE_LEFT_SHIFT(b2b, neon);
DECLARE_LEFT_SHIFT(b2w, neon);
DECLARE_LEFT_SHIFT(w2b, neon);
DECLARE_LEFT_SHIFT(w2w, neon);

DECLARE_DEPTH_CONVERT(b2h, neon);
DECLARE_DEPTH_CONVERT(b2f, neon);
DECLARE_DEPTH_CONVERT(w2h, neon);
DECLARE_DEPTH_CONVERT(w2f, neon);

#undef DECLARE_LEFT_SHIFT
#undef DECLARE_DEPTH_CONVERT

left_shift_func select_left_shift_func_arm(PixelType pixel_in, PixelType pixel_out, CPUClass cpu);

depth_convert_func select_depth_convert_func_arm(const PixelFormat &format_in, const PixelFormat &format_out, CPUClass cpu);

depth_f16c_func select_depth_f16c_func_arm(bool to_half, CPUClass cpu);

bool needs_depth_f16c_func_arm(const PixelFormat &format_in, const PixelFormat &format_out, CPUClass cpu);

} // namespace zimg::depth

#endif // ZIMG_DEPTH_ARM_DEPTH_CONVERT_ARM_H_

#endif // ZIMG_ARM
