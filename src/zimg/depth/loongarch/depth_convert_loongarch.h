#pragma once

#ifdef ZIMG_LOONGARCH

#ifndef ZIMG_DEPTH_LOONGARCH_DEPTH_CONVERT_LOONGARCH_H_
#define ZIMG_DEPTH_LOONGARCH_DEPTH_CONVERT_LOONGARCH_H_

#include "depth/depth_convert.h"

namespace zimg::depth {

#define DECLARE_LEFT_SHIFT(x, cpu) \
void left_shift_##x##_##cpu(const void *src, void *dst, unsigned shift, unsigned left, unsigned right)
#define DECLARE_DEPTH_CONVERT(x, cpu) \
void depth_convert_##x##_##cpu(const void *src, void *dst, float scale, float offset, unsigned left, unsigned right)

DECLARE_LEFT_SHIFT(b2b, lsx);
DECLARE_LEFT_SHIFT(b2w, lsx);
DECLARE_LEFT_SHIFT(w2b, lsx);
DECLARE_LEFT_SHIFT(w2w, lsx);

DECLARE_DEPTH_CONVERT(b2h, lsx);
DECLARE_DEPTH_CONVERT(b2f, lsx);
DECLARE_DEPTH_CONVERT(w2h, lsx);
DECLARE_DEPTH_CONVERT(w2f, lsx);

#undef DECLARE_LEFT_SHIFT
#undef DECLARE_DEPTH_CONVERT

void half_to_float_lsx(const void *src, void *dst, float, float, unsigned left, unsigned right);
void float_to_half_lsx(const void *src, void *dst, float, float, unsigned left, unsigned right);

left_shift_func select_left_shift_func_loongarch(PixelType pixel_in, PixelType pixel_out, CPUClass cpu);

depth_convert_func select_depth_convert_func_loongarch(const PixelFormat &format_in, const PixelFormat &format_out, CPUClass cpu);

} // namespace zimg::depth

#endif // ZIMG_DEPTH_LOONGARCH_DEPTH_CONVERT_LOONGARCH_H_

#endif // ZIMG_LOONGARCH
