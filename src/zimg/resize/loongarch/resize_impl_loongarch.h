#pragma once

#ifdef ZIMG_LOONGARCH

#ifndef ZIMG_RESIZE_LOONGARCH_RESIZE_IMPL_LOONGARCH_H_
#define ZIMG_RESIZE_LOONGARCH_RESIZE_IMPL_LOONGARCH_H_

#include <memory>

namespace graphengine {
class Filter;
}

namespace zimg {
enum class CPUClass;
enum class PixelType;
}

namespace zimg::resize {

struct FilterContext;

#define DECLARE_IMPL_H(cpu) \
std::unique_ptr<graphengine::Filter> create_resize_impl_h_##cpu(const FilterContext &context, unsigned height, PixelType type, unsigned depth)
#define DECLARE_IMPL_V(cpu) \
std::unique_ptr<graphengine::Filter> create_resize_impl_v_##cpu(const FilterContext &context, unsigned width, PixelType type, unsigned depth)

DECLARE_IMPL_H(lsx);

DECLARE_IMPL_V(lsx);

#undef DECLARE_IMPL_H
#undef DECLARE_IMPL_V

std::unique_ptr<graphengine::Filter> create_resize_impl_h_loongarch(const FilterContext &context, unsigned height, PixelType type, unsigned depth, CPUClass cpu);

std::unique_ptr<graphengine::Filter> create_resize_impl_v_loongarch(const FilterContext &context, unsigned width, PixelType type, unsigned depth, CPUClass cpu);

} // namespace zimg::resize

#endif // ZIMG_RESIZE_LOONGARCH_RESIZE_IMPL_LOONGARCH_H_

#endif // ZIMG_LOONGARCH
