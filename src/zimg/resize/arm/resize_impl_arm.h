#pragma once

#ifdef ZIMG_ARM

#ifndef ZIMG_RESIZE_ARM_RESIZE_IMPL_ARM_H_
#define ZIMG_RESIZE_ARM_RESIZE_IMPL_ARM_H_

#include <memory>

namespace zimg {

enum class CPUClass;
enum class PixelType;

namespace graph {

class ImageFilter;

} // namespace graph


namespace resize {

struct FilterContext;

#define DECLARE_IMPL_H(cpu) \
std::unique_ptr<graph::ImageFilter> create_resize_impl_h_##cpu(const FilterContext &context, unsigned height, PixelType type, unsigned depth)
#define DECLARE_IMPL_V(cpu) \
std::unique_ptr<graph::ImageFilter> create_resize_impl_v_##cpu(const FilterContext &context, unsigned width, PixelType type, unsigned depth)

DECLARE_IMPL_H(neon);

DECLARE_IMPL_V(neon);

#undef DECLARE_IMPL_H
#undef DECLARE_IMPL_V

std::unique_ptr<graph::ImageFilter> create_resize_impl_h_arm(const FilterContext &context, unsigned height, PixelType type, unsigned depth, CPUClass cpu);

std::unique_ptr<graph::ImageFilter> create_resize_impl_v_arm(const FilterContext &context, unsigned width, PixelType type, unsigned depth, CPUClass cpu);

} // namespace resize
} // namespace zimg

#endif // ZIMG_RESIZE_ARM_RESIZE_IMPL_ARM_H_

#endif // ZIMG_ARM
