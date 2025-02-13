#pragma once

#ifdef ZIMG_X86

#ifndef ZIMG_RESIZE_X86_RESIZE_IMPL_X86_H_
#define ZIMG_RESIZE_X86_RESIZE_IMPL_X86_H_

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
std::unique_ptr<graphengine::Filter> create_resize_impl_h_##cpu(const FilterContext &context, unsigned height, PixelType type, unsigned depth);
#define DECLARE_IMPL_V(cpu) \
std::unique_ptr<graphengine::Filter> create_resize_impl_v_##cpu(const FilterContext &context, unsigned width, PixelType type, unsigned depth);

DECLARE_IMPL_H(avx2)
DECLARE_IMPL_H(avx512)
DECLARE_IMPL_H(avx512_vnni)

DECLARE_IMPL_V(avx2)
DECLARE_IMPL_V(avx512)
DECLARE_IMPL_V(avx512_vnni)

#undef DECLARE_IMPL_H
#undef DECLARE_IMPL_V

std::unique_ptr<graphengine::Filter> create_resize_impl_h_x86(const FilterContext &context, unsigned height, PixelType type, unsigned depth, CPUClass cpu);
std::unique_ptr<graphengine::Filter> create_resize_impl_v_x86(const FilterContext &context, unsigned width, PixelType type, unsigned depth, CPUClass cpu);

} // namespace zimg::resize

#endif // ZIMG_RESIZE_X86_RESIZE_IMPL_X86_H_

#endif // ZIMG_X86
