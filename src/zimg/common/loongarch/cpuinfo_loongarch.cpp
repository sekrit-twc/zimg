#ifdef ZIMG_LOONGARCH

#include <cstdint>
#include "cpuinfo_loongarch.h"

#if defined(__linux__)
#include <sys/auxv.h>
#endif

namespace zimg {

namespace {

#if defined(__linux__)
#ifndef HWCAP_LOONGARCH_LSX
#define HWCAP_LOONGARCH_LSX (1 << 4)
#endif
#endif

LoongArchCapabilities do_query_loongarch_capabilities() noexcept
{
	LoongArchCapabilities caps = {};

#if defined(__linux__)
	unsigned long hwcap = getauxval(AT_HWCAP);

	caps.lsx = (hwcap & HWCAP_LOONGARCH_LSX) != 0;
#endif

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
