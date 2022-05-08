#ifdef ZIMG_X86

#if 0
  #include <stdio.h>
  #define TRACE(fmt, ...) fprintf(stderr, "[cpuinfo] " fmt, __VA_ARGS__)
#else
  #define TRACE(fmt, ...) do {} while (0)
#endif

#if defined(_MSC_VER)
  #include <intrin.h>
#elif defined(__GNUC__)
  #include <cpuid.h>
#endif

#include "common/cpuinfo.h"
#include "cpuinfo_x86.h"

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

enum { GENUINEINTEL, AUTHENTICAMD, OTHER };

struct X86BasicInfo {
	unsigned vendor;
	unsigned max_feature;
	unsigned max_extended_feature;
	unsigned family;
	unsigned model;
	unsigned stepping;
};

X86BasicInfo do_query_x86_basic_info() noexcept
{
	X86BasicInfo info = { 0 };
	int regs[4] = { 0 };

	do_cpuid(regs, 0, 1);
	info.max_feature = regs[0] & 0xFFU;
	TRACE("max feature #: 0x%x\n", info.max_feature);

	if (regs[1] == 0x756E6547U && regs[3] == 0x49656E69U && regs[2] == 0x6C65746EU) {
		info.vendor = GENUINEINTEL;
		TRACE("%s\n", "GenuineIntel");
	} else if (regs[1] == 0x68747541U && regs[3] == 0x69746E65U && regs[2] == 0x444D4163U) {
		info.vendor = AUTHENTICAMD;
		TRACE("%s\n", "AuthenticAMD");
	} else {
		info.vendor = OTHER;
		TRACE("vendor %08x-%08x-%08x\n", regs[0], regs[2], regs[1]);
	}

	do_cpuid(regs, 1, 0);
	info.family = (regs[0] >> 8) & 0x0FU;
	info.model = (regs[0] >> 4) & 0x0FU;
	info.stepping = regs[0] & 0x0FU;

	if (info.family == 0x0F) {
		unsigned extended_family = (regs[0] >> 20) & 0xFFU;
		info.family += extended_family;
	}
	if (info.family == 0x06 || info.family == 0x0F) {
		unsigned extended_model = (regs[0] >> 16) & 0x0FU;
		info.model += (extended_model) << 4;
	}
	TRACE("model %xh family %xh stepping %u\n", info.model, info.family, info.stepping);

	do_cpuid(regs, 0x80000000, 0);
	info.max_extended_feature = static_cast<unsigned>(regs[0]);
	TRACE("max extended feature #: 0x%x\n", info.max_extended_feature);

	return info;
}

X86BasicInfo query_x86_basic_info() noexcept
{
	static const X86BasicInfo info = do_query_x86_basic_info();
	return info;
}

X86Capabilities do_query_x86_capabilities() noexcept
{
	X86BasicInfo info = query_x86_basic_info();

	X86Capabilities caps = { 0 };
	unsigned long long xcr0 = 0;
	int regs[4] = { 0 };
	int xmmymm = 0;
	int zmm = 0;

	do_cpuid(regs, 1, 0);
	caps.sse      = !!(regs[3] & (1U << 25));
	caps.sse2     = !!(regs[3] & (1U << 26));
	caps.sse3     = !!(regs[2] & (1U << 0));
	caps.ssse3    = !!(regs[2] & (1U << 9));
	caps.fma      = !!(regs[2] & (1U << 12));
	caps.sse41    = !!(regs[2] & (1U << 19));
	caps.sse42    = !!(regs[2] & (1U << 20));

	// osxsave
	if (regs[2] & (1U << 27)) {
		xcr0 = do_xgetbv(0);
		xmmymm = (xcr0 & 0x06) == 0x06;
		zmm = (xcr0 & 0xE0) == 0xE0;
	}

	// XMM and YMM state.
	if (xmmymm) {
		caps.avx  = !!(regs[2] & (1U << 28));
		caps.f16c = !!(regs[2] & (1U << 29));
	}

	do_cpuid(regs, 7, 0);
	if (xmmymm) {
		caps.avx2 = !!(regs[1] & (1U << 5));
	}

	// ZMM state.
	if (zmm) {
		caps.avx512f            = !!(regs[1] & (1U << 16));
		caps.avx512dq           = !!(regs[1] & (1U << 17));
		caps.avx512ifma         = !!(regs[1] & (1U << 21));
		caps.avx512cd           = !!(regs[1] & (1U << 28));
		caps.avx512bw           = !!(regs[1] & (1U << 30));
		caps.avx512vl           = !!(regs[1] & (1U << 31));
		caps.avx512vbmi         = !!(regs[2] & (1U << 1));
		caps.avx512vbmi2        = !!(regs[2] & (1U << 6));
		caps.avx512vnni         = !!(regs[2] & (1U << 11));
		caps.avx512bitalg       = !!(regs[2] & (1U << 12));
		caps.avx512vpopcntdq    = !!(regs[2] & (1U << 14));
		caps.avx512vp2intersect = !!(regs[3] & (1U << 8));
		caps.avx512fp16         = !!(regs[3] & (1U << 23));
	}

	do_cpuid(regs, 7, 1);
	if (zmm) {
		caps.avxvnni            = !!(regs[0] & (1U << 4));
		caps.avx512bf16         = !!(regs[0] & (1U << 5));
	}

	// Extended processor info.
	if (info.max_extended_feature >= 0x80000001U) {
		do_cpuid(regs, 0x80000001U, 0);
		caps.xop = !!(regs[2] & (1U << 11));
	}

	// Zen1 vs Zen2.
	if (info.vendor == AUTHENTICAMD) {
		unsigned model = info.model;
		unsigned family = info.family;

		caps.piledriver = family == 0x15 && model == 0x02;
		caps.zen1 = family == 0x17 && model <= 0x2F;
		caps.zen2 = family == 0x17 && model >= 0x30;
		caps.zen3 = family == 0x19;
	}

	return caps;
}

// Query leaf 4h (Intel) or leaf 8000001Dh (AMD).
void do_query_x86_deterministic_cache_parameters(X86CacheHierarchy &cache, int leaf) noexcept
{
	int regs[4];

	for (int i = 0; i < 8; ++i) {
		unsigned threads;
		unsigned long line_size;
		unsigned long partitions;
		unsigned long ways;
		unsigned long sets;
		unsigned long cache_size;
		int cache_type;
		bool inclusive;

		do_cpuid(regs, leaf, i);
		cache_type = regs[0] & 0x1FU;
		TRACE("L%u cache, type %d\n", (static_cast<unsigned>(regs[0]) >> 5) & 0x07U, cache_type);

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
		TRACE("%u threads, %lu bytes, %s\n", threads, cache_size, inclusive ? "inclusive" : "non-inclusive");

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

X86CacheHierarchy do_query_x86_cache_hierarchy_intel() noexcept
{
	X86BasicInfo info = query_x86_basic_info();
	X86CacheHierarchy cache = { 0 };
	int regs[4];

	if (info.max_feature < 0x2)
		return cache;

	// Detect cache size of single-threaded CPU from flags.
	if (info.max_feature >= 0x2 && info.max_feature < 0x4)
		return cache;

	// Detect cache hierarchy.
	if (info.max_feature >= 0x4)
		do_query_x86_deterministic_cache_parameters(cache, 4);

	// Detect logical processor count on x2APIC systems.
	if (info.max_feature >= 0x0B) {
		unsigned l1d_threads = cache.l1d_threads;
		unsigned l2_threads = cache.l2_threads;
		unsigned l3_threads = cache.l3_threads;

		for (int i = 0; i < 8; ++i) {
			unsigned logical_processors;

			do_cpuid(regs, 0x0B, i);
			TRACE("APIC level %u\n", (static_cast<unsigned>(regs[2]) >> 8) & 0xFFU);

			if (((regs[2] >> 8) & 0xFFU) == 0)
				break;

			logical_processors = regs[1] & 0xFFFFU;
			TRACE("logical processors: %u\n", logical_processors);

			l1d_threads = logical_processors <= cache.l1d_threads ? logical_processors : l1d_threads;
			l2_threads = logical_processors <= cache.l2_threads ? logical_processors : l2_threads;
			l3_threads = logical_processors <= cache.l3_threads ? logical_processors : l3_threads;
			TRACE("updated cache sharing: %u %u %u\n", l1d_threads, l2_threads, l3_threads);
		}

		cache.l1d_threads = l1d_threads;
		cache.l2_threads = l2_threads;
		cache.l3_threads = l3_threads;
	}

	return cache;
}

X86CacheHierarchy do_query_x86_cache_hierarchy_amd() noexcept
{
	X86BasicInfo info = query_x86_basic_info();
	X86CacheHierarchy cache = { 0 };
	int regs[4];

	if (info.max_extended_feature >= 0x80000005U) {
		do_cpuid(regs, 0x80000005U, 0);
		cache.l1d = ((static_cast<unsigned>(regs[2]) >> 24) & 0xFFU) * 1024U;
		cache.l1d_threads = cache.l1d ? 1 : cache.l1d_threads;
		TRACE("L1d: %lu\n", cache.l1d);
	}

	if (info.max_extended_feature >= 0x80000006U) {
		do_cpuid(regs, 0x80000006U, 0);
		cache.l2 = ((static_cast<unsigned>(regs[2]) >> 16) & 0xFFFFU) * 1024U;
		cache.l3 = ((static_cast<unsigned>(regs[3]) >> 18) & 0x3FFFU) * 512U * 1024U;
		cache.l2_threads = cache.l2 ? 1 : cache.l2_threads;
		cache.l3_threads = cache.l3 ? 1 : cache.l3_threads;
		TRACE("L2: %lu\n", cache.l2);
		TRACE("L3: %lu\n", cache.l3);
	}

	if (info.max_extended_feature >= 0x80000008U) {
		unsigned threads;

		do_cpuid(regs, 0x80000008U, 0);
		threads = (regs[2] & 0xFFU) + 1;
		cache.l3_threads = cache.l3 ? threads : cache.l3_threads;
		TRACE("package threads: %u\n", threads);

		if (info.family == 0x15)
			cache.l2_threads = 2; // Bulldozer shared L2 cache.
		else if (info.family == 0x16)
			cache.l2_threads = threads; // Jaguar L2 LLC.
	}

	if (info.max_extended_feature >= 0x8000001DU)
		do_query_x86_deterministic_cache_parameters(cache, 0x8000001DU);

	return cache;
}

X86CacheHierarchy do_query_x86_cache_hierarchy() noexcept
{
	X86BasicInfo info = query_x86_basic_info();
	X86CacheHierarchy cache = { 0 };

	if (info.vendor == GENUINEINTEL)
		cache = do_query_x86_cache_hierarchy_intel();
	else if (info.vendor == AUTHENTICAMD)
		cache = do_query_x86_cache_hierarchy_amd();

	TRACE("final hierarchy: L1 %lu / %lu, L2: %lu / %lu, L3: %lu / %lu\n",
	      cache.l1d, cache.l1d_threads, cache.l2, cache.l2_threads, cache.l3, cache.l3_threads);

	cache.valid = cache.l1d && cache.l1d_threads && !(cache.l2 && !cache.l2_threads) && !(cache.l3 && !cache.l3_threads);
	return cache;
}

} // namespace


X86Capabilities query_x86_capabilities() noexcept
{
	static const X86Capabilities caps = do_query_x86_capabilities();
	return caps;
}

X86CacheHierarchy query_x86_cache_hierarchy() noexcept
{
	static const X86CacheHierarchy cache = do_query_x86_cache_hierarchy();
	return cache;
}

unsigned long cpu_cache_size_x86() noexcept
{
	const X86CacheHierarchy cache = query_x86_cache_hierarchy();

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
	if (cpu_is_autodetect(cpu)) {
		// Transparent F16C support is only implemented in AVX2+FMA code paths.
		// Excavator and Zen1 are also excluded, because of long instruction latency.
		X86Capabilities caps = query_x86_capabilities();
		return caps.fma && caps.f16c && caps.avx2 && !caps.xop && !caps.zen1;
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
