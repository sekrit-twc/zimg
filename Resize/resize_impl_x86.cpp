#ifdef ZIMG_X86

#include "resize_impl_x86.h"

namespace zimg {;
namespace resize {;

const bool HAVE_AVX2 = true;
const bool HAVE_SSE2 = true;

ResizeImpl *create_resize_impl_x86(const EvaluatedFilter &filter_h, const EvaluatedFilter &filter_v)
{
	ResizeImpl *ret;

	if (HAVE_AVX2)
		ret = create_resize_impl_avx2(filter_h, filter_v);
	else if (HAVE_SSE2)
		ret = create_resize_impl_sse2(filter_h, filter_v);
	else
		ret = nullptr;

	return ret;
}

} // namespace resize
} // namespace zimg

#endif // ZIMG_X86
