#pragma once

#ifdef ZIMG_ARM

#ifndef ZIMG_DEPTH_ARM_F16C_ARM_H_

namespace zimg {
namespace depth {

void f16c_half_to_float_neon(const void *src, void *dst, unsigned left, unsigned right);
void f16c_float_to_half_neon(const void *src, void *dst, unsigned left, unsigned right);

} // namespace depth
} // namespace zimg

#endif // ZIMG_DEPTH_ARM_F16C_ARM_H_

#endif // ZIMG_ARM
