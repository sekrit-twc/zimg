#pragma once

#ifdef ZIMG_X86

#ifndef ZIMG_DEPTH_DEPTH_CONVERT_X86_H_
#define ZIMG_DEPTH_DEPTH_CONVERT_X86_H_

#include "depth_convert.h"

namespace zimg {
namespace depth {

void left_shift_b2b_sse2(const void *src, void *dst, unsigned shift, unsigned left, unsigned right);

void left_shift_b2w_sse2(const void *src, void *dst, unsigned shift, unsigned left, unsigned right);

void left_shift_w2b_sse2(const void *src, void *dst, unsigned shift, unsigned left, unsigned right);

void left_shift_w2w_sse2(const void *src, void *dst, unsigned shift, unsigned left, unsigned right);

void depth_convert_b2f_sse2(const void *src, void *dst, float scale, float offset, unsigned left, unsigned right);

void depth_convert_w2f_sse2(const void *src, void *dst, float scale, float offset, unsigned left, unsigned right);

void depth_convert_b2h_avx2(const void *src, void *dst, float scale, float offset, unsigned left, unsigned right);

void depth_convert_b2f_avx2(const void *src, void *dst, float scale, float offset, unsigned left, unsigned right);

void depth_convert_w2h_avx2(const void *src, void *dst, float scale, float offset, unsigned left, unsigned right);

void depth_convert_w2f_avx2(const void *src, void *dst, float scale, float offset, unsigned left, unsigned right);

left_shift_func select_left_shift_func_x86(PixelType pixel_in, PixelType pixel_out, CPUClass cpu);

depth_convert_func select_depth_convert_func_x86(const PixelFormat &format_in, const PixelFormat &format_out, CPUClass cpu);

depth_f16c_func select_depth_f16c_func_x86(bool to_half, CPUClass cpu);

bool needs_depth_f16c_func_x86(const PixelFormat &format_in, const PixelFormat &format_out, CPUClass cpu);

} // namespace depth
} // namespace zimg

#endif // ZIMG_DEPTH_DEPTH_CONVERT_X86_H_

#endif // ZIMG_X86
