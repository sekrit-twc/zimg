#pragma once

#ifdef ZIMG_X86

#ifndef ZIMG_UNRESIZE_X86_UNRESIZE_IMPL_X86_H_
#define ZIMG_UNRESIZE_X86_UNRESIZE_IMPL_X86_H_

#include <memory>

namespace graphengine {
class Filter;
}

namespace zimg {
enum class CPUClass;
enum class PixelType;
}

namespace zimg::unresize {

struct BilinearContext;

#define DECLARE_IMPL_H(cpu) \
std::unique_ptr<graphengine::Filter> create_unresize_impl_h_##cpu(const BilinearContext &context, unsigned height, PixelType type);
#define DECLARE_IMPL_V(cpu) \
std::unique_ptr<graphengine::Filter> create_unresize_impl_v_##cpu(const BilinearContext &context, unsigned width, PixelType type);

DECLARE_IMPL_H(avx2)
DECLARE_IMPL_V(avx2)

#undef DECLARE_IMPL_H
#undef DECLARE_IMPL_V

std::unique_ptr<graphengine::Filter> create_unresize_impl_h_x86(const BilinearContext &context, unsigned height, PixelType type, CPUClass cpu);
std::unique_ptr<graphengine::Filter> create_unresize_impl_v_x86(const BilinearContext &context, unsigned width, PixelType type, CPUClass cpu);

} // namespace zimg::unresize

#endif // ZIMG_UNRESIZE_X86_UNRESIZE_IMPL_X86_H_

#endif // ZIMG_X86
