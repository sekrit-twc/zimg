#ifdef ZIMG_LOONGARCH

#include "cpuinfo_loongarch.h"

namespace zimg {

namespace {

LoongArchCapabilities do_query_loongarch_capabilities() noexcept
{
	return{};
}

} // namespace


LoongArchCapabilities query_loongarch_capabilities() noexcept
{
	static const LoongArchCapabilities caps = do_query_loongarch_capabilities();
	return caps;
}

} // namespace zimg

#endif // ZIMG_LOONGARCH
