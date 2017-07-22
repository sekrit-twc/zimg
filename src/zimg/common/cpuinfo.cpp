#include "cpuinfo.h"

#ifdef ZIMG_X86

#if defined(_MSC_VER)
  #include <intrin.h>
#elif defined(__GNUC__)
  #include <cpuid.h>
#endif

namespace zimg {
namespace {

/**
 * Execute the CPUID instruction.
 *
 * @param regs array to receive eax, ebx, ecx, edx
 * @param eax argument to instruction
 * @param ecx argument to instruction
 */
void do_cpuid(int regs[4], int eax, int ecx)
{
#if defined(_MSC_VER)
	__cpuidex(regs, eax, ecx);
#elif defined(__GNUC__)
	__cpuid_count(eax, ecx, regs[0], regs[1], regs[2], regs[3]);
#else
	regs[0] = 0;
	regs[1] = 0;
	regs[2] = 0;
	regs[3] = 0;
#endif
}


/**
 * Execute the XGETBV instruction.
 *
 * @param ecx argument to instruction
 * @return (edx << 32) | eax
 */
unsigned long long do_xgetbv(unsigned ecx)
{
#if defined(_MSC_VER)
	return _xgetbv(ecx);
#elif defined(__GNUC__)
	unsigned eax, edx;
	__asm("xgetbv" : "=a"(eax), "=d"(edx) : "c"(ecx) : );
	return (static_cast<unsigned long long>(edx) << 32) | eax;
#else
	return 0;
#endif
}

X86Capabilities do_query_x86_capabilities() noexcept
{
	X86Capabilities caps = { 0 };
	unsigned long long xcr0 = 0;
	int regs[4] = { 0 };

	do_cpuid(regs, 1, 0);
	caps.sse      = !!(regs[3] & (1U << 25));
	caps.sse2     = !!(regs[3] & (1U << 26));
	caps.sse3     = !!(regs[2] & (1U << 0));
	caps.ssse3    = !!(regs[2] & (1U << 9));
	caps.fma      = !!(regs[2] & (1U << 12));
	caps.sse41    = !!(regs[2] & (1U << 19));
	caps.sse42    = !!(regs[2] & (1U << 20));

	// osxsave
	if (regs[2] & (1U << 27))
		xcr0 = do_xgetbv(0);

	// XMM and YMM state.
	if ((xcr0 & 0x06) != 0x06)
		return caps;

	caps.avx      = !!(regs[2] & (1U << 28));
	caps.f16c     = !!(regs[2] & (1U << 29));

	do_cpuid(regs, 7, 0);
	caps.avx2     = !!(regs[1] & (1U << 5));

	// ZMM state.
	if ((xcr0 & 0xE0) != 0xE0)
		return caps;

	caps.avx512f  = !!(regs[1] & (1U << 16));
	caps.avx512dq = !!(regs[1] & (1U << 17));
	caps.avx512cd = !!(regs[1] & (1U << 28));
	caps.avx512bw = !!(regs[1] & (1U << 30));
	caps.avx512vl = !!(regs[1] & (1U << 31));

	return caps;
}

} // namespace


X86Capabilities query_x86_capabilities() noexcept
{
	static const X86Capabilities caps = do_query_x86_capabilities();
	return caps;
}

bool cpu_has_fast_f16(CPUClass cpu) noexcept
{
	// Although F16C is supported on Ivy Bridge, the latency penalty is too great before Haswell.
	if (cpu_is_autodetect(cpu)) {
		X86Capabilities caps = query_x86_capabilities();
		return caps.fma && caps.f16c && caps.avx2;
	} else {
		return cpu >= CPUClass::X86_AVX2;
	}
}

bool cpu_requires_64b_alignment(CPUClass cpu) noexcept
{
	if (cpu == CPUClass::AUTO_64B) {
		X86Capabilities caps = query_x86_capabilities();
		return !!caps.avx512f;
	} else {
		return cpu >= CPUClass::X86_AVX512;
	}
}

} // namespace zimg

#else // ZIMG_X86

namespace zimg {

bool cpu_has_fast_f16(CPUClass) noexcept { return false; }

bool cpu_requires_64b_alignment(CPUClass) noexcept { return false; }

} // namespace zimg

#endif // ZIMG_X86
