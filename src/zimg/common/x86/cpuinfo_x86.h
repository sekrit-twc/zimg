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
	unsigned sse                : 1;
	unsigned sse2               : 1;
	unsigned sse3               : 1;
	unsigned ssse3              : 1;
	unsigned fma                : 1;
	unsigned sse41              : 1;
	unsigned sse42              : 1;
	unsigned avx                : 1;
	unsigned f16c               : 1;
	unsigned avx2               : 1;
	unsigned avxvnni            : 1;
	unsigned avx512f            : 1;
	unsigned avx512dq           : 1;
	unsigned avx512ifma         : 1;
	unsigned avx512cd           : 1;
	unsigned avx512bw           : 1;
	unsigned avx512vl           : 1;
	unsigned avx512vbmi         : 1;
	unsigned avx512vbmi2        : 1;
	unsigned avx512vnni         : 1;
	unsigned avx512bitalg       : 1;
	unsigned avx512vpopcntdq    : 1;
	unsigned avx512vp2intersect : 1;
	unsigned avx512fp16         : 1;
	unsigned avx512bf16         : 1;
	/* AMD architectures needing workarounds. */
	unsigned xop : 1;
	unsigned zen1 : 1;
	unsigned zen2 : 1;
	unsigned zen3 : 1;
};

/* 2+ cycles per value on AMD. Still >1 cycle on Zen3, but usable. */
constexpr bool cpu_has_slow_gather(const X86Capabilities &caps) { return caps.xop || caps.zen1 || caps.zen2; }
/* 4 cycles per vpermd on Zen. Higher throughput on Zen3, but still long latency. */
constexpr bool cpu_has_slow_permute(const X86Capabilities &caps) { return caps.zen1 || caps.zen2 || caps.zen3; }

constexpr bool cpu_has_avx512_f_dq_bw_vl(const X86Capabilities &caps) { return caps.avx512f && caps.avx512dq && caps.avx512bw && caps.avx512vl; }

/**
 * Representation of processor cache topology.
 */
struct X86CacheHierarchy {
	unsigned long l1d;
	unsigned long l1d_threads;
	unsigned long l2;
	unsigned long l2_threads;
	unsigned long l3;
	unsigned long l3_threads;
	bool l2_inclusive;
	bool l3_inclusive;
	bool valid;
};

/**
 * Get the x86 feature flags on the current CPU.
 *
 * @return capabilities
 */
X86Capabilities query_x86_capabilities() noexcept;

/**
 * Get the cache topology of the current CPU.
 *
 * On a multi-processor system, the returned topology corresponds to the first
 * processor package on which the function is called. The behaviour is
 * undefined if the platform contains non-identical processors.
 *
 * @return cache hierarchy
 */
X86CacheHierarchy query_x86_cache_hierarchy() noexcept;

unsigned long cpu_cache_per_thread_x86() noexcept;

bool cpu_has_fast_f16_x86(CPUClass cpu) noexcept;
bool cpu_requires_64b_alignment_x86(CPUClass cpu) noexcept;

} // namespace zimg

#endif // ZIMG_X86_CPUINFO_X86_H_

#endif // ZIMG_X86
