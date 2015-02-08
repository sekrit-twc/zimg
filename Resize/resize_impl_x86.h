#pragma once

#ifdef ZIMG_X86

#ifndef ZIMG_RESIZE_RESIZE_IMPL_X86_H_
#define ZIMG_RESIZE_RESIZE_IMPL_X86_H_

namespace zimg {;

enum class CPUClass;

namespace resize {;

struct FilterContext;
class ResizeImpl;

ResizeImpl *create_resize_impl_sse2(const FilterContext &filter, bool horizontal);

ResizeImpl *create_resize_impl_avx2(const FilterContext &filter, bool horizontal);

/**
 * Create an appropriate x86 optimized ResizeImpl for the given CPU.
 *
 * @see create_resize_impl
 */
ResizeImpl *create_resize_impl_x86(const FilterContext &filter, bool horizontal, CPUClass cpu);

} // namespace resize
} // namespace zimg

#endif // ZIMG_RESIZE_RESIZE_IMPL_X86_H_

#endif // ZIMG_X86
