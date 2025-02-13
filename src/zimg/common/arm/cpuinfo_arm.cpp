#ifdef ZIMG_ARM

#if defined(_WIN32)
  #define NOMINMAX
  #define STRICT
  #define WIN32_LEAN_AND_MEAN
  #include <Windows.h>
#elif defined(__linux__)
  #include <sys/auxv.h>
  #include <asm/hwcap.h>
#endif

#include "cpuinfo_arm.h"

namespace zimg {

namespace {

ARMCapabilities do_query_arm_capabilities() noexcept
{
	return{};
}

} // namespace


ARMCapabilities query_arm_capabilities() noexcept
{
	static const ARMCapabilities caps = do_query_arm_capabilities();
	return caps;
}

} // namespace zimg

#endif // ZIMG_ARM
