#include <iostream>
#include "Common/cpuinfo.h"
#include "Common/except.h"
#include "Common/static_map.h"

#include "apps.h"

namespace {;

zimg::CPUClass parse_cpu(const char *cpu)
{
	static const zimg::static_string_map<zimg::CPUClass, 4> map{
		{ "none", zimg::CPUClass::CPU_NONE },
		{ "auto", zimg::CPUClass::CPU_AUTO },
#ifdef ZIMG_X86
		{ "sse2", zimg::CPUClass::CPU_X86_SSE2 },
		{ "avx2", zimg::CPUClass::CPU_X86_AVX2 },
#endif
	};
	auto it = map.find(cpu);
	return it == map.end() ? throw std::invalid_argument{ "bad CPU type" } : it->second;
}


typedef int (*main_func)(int, char **);

void usage()
{
	std::cout << "TestApp subapp [args]\n";
	//std::cout << "    colorspace - change colorspace\n";
	//std::cout << "    depth      - change depth\n";
	//std::cout << "    resize     - resize images\n";
	//std::cout << "    unresize   - unresize images\n";
}

main_func lookup_app(const char *name)
{
	static const zimg::static_string_map<main_func, 1> map{
		{ "", nullptr }
	};

	auto it = map.find(name);
	return it == map.end() ? nullptr : it->second;
}

} // namespace


int arg_decode_cpu(const struct ArgparseOption *, void *out, int argc, char **argv)
{
	if (argc < 1)
		return -1;

	zimg::CPUClass *cpu = reinterpret_cast<zimg::CPUClass *>(out);

	try {
		*cpu = parse_cpu(*argv);
	} catch (const std::exception &e) {
		std::cerr << e.what() << '\n';
		return -1;
	}

	return 1;
}

int main(int argc, char **argv)
{
	if (argc < 2) {
		usage();
		return 1;
	}

	try {
		main_func func = lookup_app(argv[1]);

		if (!func) {
			usage();
			return 1;
		}

		return func(argc - 1, argv + 1);
	} catch (const zimg::error::Exception &e) {
		std::cerr << e.what() << '\n';
		return 2;
	} catch (const std::exception &e) {
		std::cerr << e.what() << '\n';
		return 2;
	}
}
