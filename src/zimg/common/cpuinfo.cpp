#include "cpuinfo.h"

#ifdef ZIMG_X86
  #include "x86/cpuinfo_x86.h"
#endif

namespace zimg {

unsigned long cpu_cache_size() noexcept
{
	unsigned long ret = 0;
#ifdef ZIMG_X86
	ret = cpu_cache_size_x86();
#endif
	return ret ? ret : 1024 * 1024UL;
}

bool cpu_has_fast_f16(CPUClass cpu) noexcept
{
	bool ret = false;
#ifdef ZIMG_X86
	ret = cpu_has_fast_f16_x86(cpu);
#endif
	return ret;
}

bool cpu_requires_64b_alignment(CPUClass cpu) noexcept
{
	bool ret = false;
#ifdef ZIMG_X86
	ret = cpu_requires_64b_alignment_x86(cpu);
#endif
	return ret;
}

} // namespace zimg
