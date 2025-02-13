#pragma once

#ifndef ZIMG_CPUINFO_H_
#define ZIMG_CPUINFO_H_

namespace zimg {

/**
 * Enum for CPU type.
 */
enum class CPUClass {
	NONE,
	AUTO,
	AUTO_64B,
#if defined(ZIMG_X86)
	X86_AVX2,
	X86_AVX512, // F, CD, BW, DQ, VL
	X86_AVX512_CLX, // VNNI
#elif defined(ZIMG_ARM)
	ARM_NEON,
#endif
};

constexpr bool cpu_is_autodetect(CPUClass cpu) noexcept
{
	return cpu == CPUClass::AUTO || cpu == CPUClass::AUTO_64B;
}

unsigned long cpu_cache_per_thread() noexcept;

bool cpu_has_fast_f16(CPUClass cpu) noexcept;
bool cpu_requires_64b_alignment(CPUClass cpu) noexcept;

} // namespace zimg

#endif // ZIMG_CPUINFO_H_
