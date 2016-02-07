#pragma once

#ifdef ZIMG_X86

#ifndef ZIMG_DEPTH_F16C_X86_H_

namespace zimg {
namespace depth {

void f16c_half_to_float_sse2(const void *src, void *dst, unsigned left, unsigned right);
void f16c_float_to_half_sse2(const void *src, void *dst, unsigned left, unsigned right);

void f16c_half_to_float_ivb(const void *src, void *dst, unsigned left, unsigned right);
void f16c_float_to_half_ivb(const void *src, void *dst, unsigned left, unsigned right);

} // namespace depth
} // namespace zimg

#endif // ZIMG_DEPTH_F16C_X86_H_

#endif // ZIMG_X86
