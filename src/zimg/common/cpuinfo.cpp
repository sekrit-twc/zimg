#include "cpuinfo.h"

#ifdef ZIMG_X86
  #include "x86/cpuinfo_x86.h"
#endif

namespace zimg {

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
