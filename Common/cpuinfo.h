#pragma once

#ifndef ZIMG_CPUINFO_H_
#define ZIMG_CPUINFO_H_

#ifdef _WIN32
#include <intrin.h>
#endif // _WIN32

namespace zimg {;

/**
 * Enum for CPU type.
 */
enum class CPUClass {
	CPU_NONE,
#ifdef ZIMG_X86
	CPU_X86_AUTO,
	CPU_X86_SSE2,
	CPU_X86_F16C,
	CPU_X86_AVX2
#endif // ZIMG_X86
};

#ifdef ZIMG_X86

#ifdef __GNUC__
#include <cpuid.h>
#endif

/**
 * Bitfield of selected x86 feature flags.
 */
struct X86Capabilities {
	unsigned sse   : 1;
	unsigned sse2  : 1;
	unsigned sse3  : 1;
	unsigned ssse3 : 1;
	unsigned fma   : 1;
	unsigned sse41 : 1;
	unsigned sse42 : 1;
	unsigned avx   : 1;
	unsigned f16c  : 1;
	unsigned avx2  : 1;
};

/**
 * Execute the CPUID instruction.
 *
 * @param regs array to receive eax, ebx, ecx, edx
 * @param eax argument to instruction
 * @param ecx argument to instruction
 */
inline void do_cpuid(int regs[4], int eax, int ecx)
{
#if defined(_WIN32)
	__cpuidex(regs, eax, ecx);
#elif defined(__GNUC__)
	__get_cpuid(eax, (unsigned int *)&regs[0], (unsigned int *)&regs[1], (unsigned int *)&regs[2], (unsigned int *)&regs[3]);
#else
	regs[0] = 0;
	regs[1] = 0;
	regs[2] = 0;
	regs[3] = 0;
#endif // _WIN32
}

/**
 * Get the x86 feature flags on the current CPU.
 *
 * @return capabilities
 */
inline X86Capabilities query_x86_capabilities()
{
	X86Capabilities caps = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	int regs[4] = { 0 };

	do_cpuid(regs, 1, 0);
	caps.sse   = !!(regs[3] & (1 << 25));
	caps.sse2  = !!(regs[3] & (1 << 26));
	caps.sse3  = !!(regs[2] & (1 << 0));
	caps.ssse3 = !!(regs[2] & (1 << 9));
	caps.fma   = !!(regs[2] & (1 << 12));
	caps.sse41 = !!(regs[2] & (1 << 19));
	caps.sse42 = !!(regs[2] & (1 << 20));
	caps.avx   = !!(regs[2] & (1 << 28));
	caps.f16c  = !!(regs[2] & (1 << 29));

	do_cpuid(regs, 7, 0);
	caps.avx2 = !!(regs[1] & (1 << 5));

	return caps;
}

#endif // ZIMG_X86

} // namespace zimg

#endif // ZIMG_CPUINFO_H_
