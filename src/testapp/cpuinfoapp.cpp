#include <iostream>
#include "common/cpuinfo.h"

#ifdef ZIMG_X86
  #include "common/x86/cpuinfo_x86.h"
#endif

#include "apps.h"

namespace {

template <class T>
constexpr const char *yes_no(const T &b) noexcept { return b ? "yes" : "no"; }

#ifdef ZIMG_X86
void show_x86_info()
{
	const zimg::X86Capabilities caps = zimg::query_x86_capabilities();

	std::cout << "Supported instruction set extensions:\n";
	std::cout << "SSE:    " << yes_no(caps.sse) << '\n';
	std::cout << "SSE2:   " << yes_no(caps.sse2) << '\n';
	std::cout << "SSE3:   " << yes_no(caps.sse3) << '\n';
	std::cout << "SSSE3:  " << yes_no(caps.ssse3) << '\n';
	std::cout << "FMA:    " << yes_no(caps.fma) << '\n';
	std::cout << "SSE4.1: " << yes_no(caps.sse41) << '\n';
	std::cout << "SSE4.2: " << yes_no(caps.sse42) << '\n';
	std::cout << "AVX:    " << yes_no(caps.avx) << '\n';
	std::cout << "F16C:   " << yes_no(caps.f16c) << '\n';
	std::cout << "AVX2:   " << yes_no(caps.avx2) << '\n';
	std::cout << "AVX-512 F:  " << yes_no(caps.avx512f) << '\n';
	std::cout << "AVX-512 DQ: " << yes_no(caps.avx512dq) << '\n';
	std::cout << "AVX-512 CD: " << yes_no(caps.avx512dq) << '\n';
	std::cout << "AVX-512 BW: " << yes_no(caps.avx512bw) << '\n';
	std::cout << "AVX-512 VL: " << yes_no(caps.avx512vl) << '\n';
	std::cout << "XOP:    " << yes_no(caps.xop) << '\n';
	std::cout << "Zen1:   " << yes_no(caps.zen1) << '\n';
	std::cout << '\n';

	const zimg::X86CacheHierarchy cache = zimg::query_x86_cache_hierarchy();
	if (cache.valid) {
		std::cout << "Processor cache topology:\n";

		if (cache.l1d)
			std::cout << "L1d: " << cache.l1d << " bytes / " << cache.l1d_threads << " threads\n";
		if (cache.l2) {
			std::cout << "L2: " << cache.l2 << " bytes / " << cache.l2_threads << " threads (";
			std::cout << (cache.l2_inclusive ? "inclusive" : "non-inclusive") << ")\n";
		}
		if (cache.l3) {
			std::cout << "L3: " << cache.l3 << " bytes / " << cache.l3_threads << " threads (";
			std::cout << (cache.l3_inclusive ? "inclusive" : "non-inclusive") << ")\n";
		}

		std::cout << '\n';
	} else {
		std::cout << "Cache detection failed\n";
		return;
	}
}
#endif

void show_generic_info()
{
	std::cout << "Per-thread effective cache size: " << zimg::cpu_cache_size() << '\n';
	std::cout << "Fast fp16 support: " << yes_no(zimg::cpu_has_fast_f16(zimg::CPUClass::AUTO)) << '\n';
	std::cout << "64-byte (512-bit) instructions: " << yes_no(zimg::cpu_requires_64b_alignment(zimg::CPUClass::AUTO_64B)) << '\n';
}

} // namespace


int cpuinfo_main(int, char **)
{
#ifdef ZIMG_X86
	show_x86_info();
#endif

	show_generic_info();
	return 0;
}
