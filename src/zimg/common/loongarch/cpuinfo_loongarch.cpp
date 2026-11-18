#ifdef ZIMG_LOONGARCH

#include <cstdint>
#include "cpuinfo_loongarch.h"

namespace zimg {

namespace {

LoongArchCapabilities do_query_loongarch_capabilities() noexcept
{
	LoongArchCapabilities caps = {};
	caps.lsx  = 1;
	caps.lasx = 1;

	return caps;
}

} // namespace


LoongArchCapabilities query_loongarch_capabilities() noexcept
{
	static const LoongArchCapabilities caps = do_query_loongarch_capabilities();
	return caps;
}

} // namespace zimg

#endif // ZIMG_LOONGARCH
