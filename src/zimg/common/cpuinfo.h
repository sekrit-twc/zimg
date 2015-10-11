#pragma once

#ifndef ZIMG_CPUINFO_H_
#define ZIMG_CPUINFO_H_

namespace zimg {;

/**
 * Enum for CPU type.
 */
enum class CPUClass {
	CPU_NONE,
	CPU_AUTO,
#ifdef ZIMG_X86
	CPU_X86_SSE,
	CPU_X86_SSE2,
#endif // ZIMG_X86
};


#ifdef ZIMG_X86

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
 * Get the x86 feature flags on the current CPU.
 *
 * @return capabilities
 */
X86Capabilities query_x86_capabilities();

#endif // ZIMG_X86

} // namespace zimg

#endif // ZIMG_CPUINFO_H_
