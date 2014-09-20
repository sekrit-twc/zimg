#ifdef ZIMG_X86

#include "Common/cpuinfo.h"
#include "unresize_impl_x86.h"

namespace zimg {;
namespace unresize {;

UnresizeImpl *create_unresize_impl_x86(const BilinearContext &hcontext, const BilinearContext &vcontext)
{
	X86Capabilities caps = query_x86_capabilities();
	UnresizeImpl *ret = nullptr;

	if (caps.sse2) {
		ret = create_unresize_impl_sse2(hcontext, vcontext);
	}

	return ret;
}

} // namespace unresize
} // namespace zimg

#endif // ZIMG_X86
