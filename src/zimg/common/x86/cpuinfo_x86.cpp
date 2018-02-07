#ifdef ZIMG_X86

#if defined(_MSC_VER)
  #include <intrin.h>
#elif defined(__GNUC__)
  #include <cpuid.h>
#endif

#include "common/cpuinfo.h"
#include "cpuinfo_x86.h"

namespace zimg {
namespace {

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

X86CacheHierarchy do_query_x86_cache_hierarchy_intel(int max_feature) noexcept
{
	X86CacheHierarchy cache = { 0 };
	int regs[4];

	if (max_feature < 2)
		return cache;

	// Detect cache size of single-threaded CPU from flags.
	if (max_feature >= 2 && max_feature < 4)
		return cache;

	// Detect cache hierarchy.
	if (max_feature >= 4) {
		for (int i = 0; i < 8; ++i) {
			unsigned threads;
			unsigned long line_size;
			unsigned long partitions;
			unsigned long ways;
			unsigned long sets;
			unsigned long cache_size;
			int cache_type;
			bool inclusive;

			do_cpuid(regs, 4, i);
			cache_type = regs[0] & 0x1FU;

			// No more caches.
			if (cache_type == 0)
				break;

			// Not data or unified cache.
			if (cache_type != 1 && cache_type != 3)
				continue;

			threads    = ((static_cast<unsigned>(regs[0]) >> 14) & 0x0FFFU) + 1;
			line_size  = ((static_cast<unsigned>(regs[1]) >> 0) & 0x0FFFU) + 1;
			partitions = ((static_cast<unsigned>(regs[1]) >> 12) & 0x03FFU) + 1;
			ways       = ((static_cast<unsigned>(regs[1]) >> 22) & 0x03FFU) + 1;
			sets       = static_cast<unsigned>(regs[2]) + 1;

			cache_size = line_size * partitions * ways * sets;
			inclusive = regs[3] & (1U << 1);

			// Cache level.
			switch ((static_cast<unsigned>(regs[0]) >> 5) & 0x07U) {
			case 1:
				cache.l1d = cache_size;
				cache.l1d_threads = threads;
				break;
			case 2:
				cache.l2 = cache_size;
				cache.l2_threads = threads;
				cache.l2_inclusive = inclusive;
				break;
			case 3:
				cache.l3 = cache_size;
				cache.l3_threads = threads;
				cache.l3_inclusive = inclusive;
				break;
			default:
				break;
			}
		}
	}

	// Detect logical processor count on x2APIC systems.
	if (max_feature >= 0x0B) {
		unsigned l1d_threads = cache.l1d_threads;
		unsigned l2_threads = cache.l2_threads;
		unsigned l3_threads = cache.l3_threads;

		for (int i = 0; i < 8; ++i) {
			unsigned logical_processors;

			do_cpuid(regs, 0x0B, i);

			if (((regs[2] >> 8) & 0xFF) == 0)
				break;

			logical_processors = regs[1] & 0xFFFFU;
			if (logical_processors <= cache.l1d_threads)
				l1d_threads = logical_processors;
			if (logical_processors <= cache.l2_threads)
				l2_threads = logical_processors;
			if (logical_processors <= cache.l3_threads)
				l3_threads = logical_processors;
		}

		cache.l1d_threads = l1d_threads;
		cache.l2_threads = l2_threads;
		cache.l3_threads = l3_threads;
	}

	cache.valid = cache.l1d && cache.l1d_threads && !(cache.l2 && !cache.l2_threads) && !(cache.l3 && !cache.l3_threads);
	return cache;
}

X86CacheHierarchy do_query_x86_cache_hierarchy() noexcept
{
	enum { GENUINEINTEL, AUTHENTICAMD, OTHER } vendor;

	X86CacheHierarchy cache = { 0 };
	int regs[4] = { 0 };
	int max_feature;

	do_cpuid(regs, 0, 1);
	max_feature = regs[0] & 0xFF;

	if (regs[1] == 0x756E6547U && regs[3] == 0x49656E69U && regs[2] == 0x6C65746EU)
		vendor = GENUINEINTEL;
	else if (regs[1] == 0x68747541U && regs[3] == 0x69746E65U && regs[2] == 0x444D4163U)
		vendor = AUTHENTICAMD;
	else
		vendor = OTHER;

	if (vendor == GENUINEINTEL)
		return do_query_x86_cache_hierarchy_intel(max_feature);
	else if (vendor == AUTHENTICAMD)
		return cache;
	else
		return cache;
}

} // namespace


X86Capabilities query_x86_capabilities() noexcept
{
	static const X86Capabilities caps = do_query_x86_capabilities();
	return caps;
}

unsigned long cpu_cache_size_x86() noexcept
{
	static const X86CacheHierarchy cache = do_query_x86_cache_hierarchy();

	if (!cache.valid)
		return 0;

	// Detect Skylake-SP cache hierarchy and report L2 size instead of L3.
	if (cache.l3 && !cache.l3_inclusive && cache.l2 >= 1024 * 1024U && cache.l2_threads <= 2)
		return cache.l2 / cache.l2_threads;

	if (cache.l3)
		return cache.l3 / cache.l3_threads;
	else if (cache.l2)
		return cache.l2 / cache.l2_threads;
	else
		return cache.l1d / cache.l1d_threads;
}

bool cpu_has_fast_f16_x86(CPUClass cpu) noexcept
{
	// Although F16C is supported on Ivy Bridge, the latency penalty is too great before Haswell.
	if (cpu_is_autodetect(cpu)) {
		X86Capabilities caps = query_x86_capabilities();
		return caps.fma && caps.f16c && caps.avx2;
	} else {
		return cpu >= CPUClass::X86_AVX2;
	}
}

bool cpu_requires_64b_alignment_x86(CPUClass cpu) noexcept
{
	if (cpu == CPUClass::AUTO_64B) {
		X86Capabilities caps = query_x86_capabilities();
		return !!caps.avx512f;
	} else {
		return cpu >= CPUClass::X86_AVX512;
	}
}

} // namespace zimg

#endif // ZIMG_X86
