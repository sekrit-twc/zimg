#ifdef RESIZE_X86

#ifndef RESIZE_IMPL_X86_H_
#define RESIZE_IMPL_X86_H_

#include "resize_impl.h"

namespace resize {

ResizeImpl *create_resize_impl_x86(const EvaluatedFilter &filter_h, const EvaluatedFilter &filter_v);

}; // namespace resize

#endif // RESIZE_IMPL_X86_H_

#endif // RESIZE_X86
