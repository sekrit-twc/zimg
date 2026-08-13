#pragma once

#ifdef ZIMG_LOONGARCH

#ifndef ZIMG_UNRESIZE_LOONGARCH_UNRESIZE_IMPL_LOONGARCH_H_
#define ZIMG_UNRESIZE_LOONGARCH_UNRESIZE_IMPL_LOONGARCH_H_

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

DECLARE_IMPL_H(lsx)
DECLARE_IMPL_V(lsx)

#undef DECLARE_IMPL_H
#undef DECLARE_IMPL_V

std::unique_ptr<graphengine::Filter> create_unresize_impl_h_loongarch(const BilinearContext &context, unsigned height, PixelType type, CPUClass cpu);
std::unique_ptr<graphengine::Filter> create_unresize_impl_v_loongarch(const BilinearContext &context, unsigned width, PixelType type, CPUClass cpu);

} // namespace zimg::unresize

#endif // ZIMG_UNRESIZE_LOONGARCH_UNRESIZE_IMPL_LOONGARCH_H_

#endif // ZIMG_LOONGARCH
