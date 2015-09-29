#pragma once

#ifdef ZIMG_X86

#ifndef ZIMG_DEPTH_DITHER2_X86_H_
#define ZIMG_DEPTH_DITHER2_X86_H_

#include "dither2.h"

namespace zimg {;
namespace depth {;

void ordered_dither_b2b_sse2(const float *dither, unsigned dither_offset, unsigned dither_mask,
                             const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right);

void ordered_dither_b2w_sse2(const float *dither, unsigned dither_offset, unsigned dither_mask,
                             const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right);

void ordered_dither_w2b_sse2(const float *dither, unsigned dither_offset, unsigned dither_mask,
                             const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right);

void ordered_dither_w2w_sse2(const float *dither, unsigned dither_offset, unsigned dither_mask,
                             const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right);

void ordered_dither_f2b_sse2(const float *dither, unsigned dither_offset, unsigned dither_mask,
                             const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right);

void ordered_dither_f2w_sse2(const float *dither, unsigned dither_offset, unsigned dither_mask,
                             const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right);

dither_convert_func select_ordered_dither_func_x86(const PixelFormat &pixel_in, const PixelFormat &pixel_out, CPUClass cpu);

dither_f16c_func select_dither_f16c_func_x86(CPUClass cpu);

} // depth
} // zimg

#endif // ZIMG_DEPTH_DITHER2_X86_H_

#endif // ZIMG_X86
