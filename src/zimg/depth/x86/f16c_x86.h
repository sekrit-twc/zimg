#pragma once

#ifdef ZIMG_X86

#ifndef ZIMG_DEPTH_X86_F16C_X86_H_

namespace zimg::depth {

void f16c_half_to_float_sse2(const void *src, void *dst, unsigned left, unsigned right);
void f16c_float_to_half_sse2(const void *src, void *dst, unsigned left, unsigned right);

void f16c_half_to_float_avx2(const void *src, void *dst, unsigned left, unsigned right);
void f16c_float_to_half_avx2(const void *src, void *dst, unsigned left, unsigned right);

} // namespace zimg::depth

#endif // ZIMG_DEPTH_X86_F16C_X86_H_

#endif // ZIMG_X86
