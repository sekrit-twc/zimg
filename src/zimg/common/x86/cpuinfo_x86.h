#pragma once

#ifdef ZIMG_X86

#ifndef ZIMG_X86_CPUINFO_X86_H_
#define ZIMG_X86_CPUINFO_X86_H_

namespace zimg {

enum class CPUClass;

/**
 * Bitfield of selected x86 feature flags.
 */
struct X86Capabilities {
	unsigned sse      : 1;
	unsigned sse2     : 1;
	unsigned sse3     : 1;
	unsigned ssse3    : 1;
	unsigned fma      : 1;
	unsigned sse41    : 1;
	unsigned sse42    : 1;
	unsigned avx      : 1;
	unsigned f16c     : 1;
	unsigned avx2     : 1;
	unsigned avx512f  : 1;
	unsigned avx512dq : 1;
	unsigned avx512cd : 1;
	unsigned avx512bw : 1;
	unsigned avx512vl : 1;
};

/**
 * Get the x86 feature flags on the current CPU.
 *
 * @return capabilities
 */
X86Capabilities query_x86_capabilities() noexcept;

unsigned long cpu_cache_size_x86() noexcept;

bool cpu_has_fast_f16_x86(CPUClass cpu) noexcept;
bool cpu_requires_64b_alignment_x86(CPUClass cpu) noexcept;

} // namespace zimg

#endif // ZIMG_X86_CPUINFO_X86_H_

#endif // ZIMG_X86
