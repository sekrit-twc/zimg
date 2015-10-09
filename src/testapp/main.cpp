#include <iostream>
#include <regex>
#include <string>
#include "common/cpuinfo.h"
#include "common/except.h"
#include "common/pixel.h"
#include "common/static_map.h"

#include "apps.h"

namespace {;

zimg::CPUClass parse_cpu(const char *cpu)
{
	static const zimg::static_string_map<zimg::CPUClass, 4> map{
		{ "none", zimg::CPUClass::CPU_NONE },
		{ "auto", zimg::CPUClass::CPU_AUTO },
#ifdef ZIMG_X86
		{ "sse",  zimg::CPUClass::CPU_X86_SSE },
		{ "sse2", zimg::CPUClass::CPU_X86_SSE2 },
#endif
	};
	auto it = map.find(cpu);
	return it == map.end() ? throw std::invalid_argument{ "bad CPU type" } : it->second;
}

zimg::PixelType parse_pixel_type(const char *type)
{
	static const zimg::static_string_map<zimg::PixelType, 4> map{
		{ "byte", zimg::PixelType::BYTE },
		{ "word", zimg::PixelType::WORD },
		{ "half", zimg::PixelType::HALF },
		{ "float", zimg::PixelType::FLOAT },
	};
	auto it = map.find(type);
	return it == map.end() ? throw std::invalid_argument{ "bad pixel type" } : it->second;
}


typedef int (*main_func)(int, char **);

void usage()
{
	std::cout << "TestApp subapp [args]\n";
	std::cout << "    colorspace - change colorspace\n";
	std::cout << "    depth      - change depth\n";
	std::cout << "    resize     - resize images\n";
	std::cout << "    unresize   - unresize images\n";
}

main_func lookup_app(const char *name)
{
	static const zimg::static_string_map<main_func, 4> map{
		{ "colorspace", colorspace_main },
		{ "depth",      depth_main },
		{ "resize",     resize_main },
		{ "unresize",   unresize_main }
	};

	auto it = map.find(name);
	return it == map.end() ? nullptr : it->second;
}

} // namespace


int arg_decode_cpu(const struct ArgparseOption *, void *out, int argc, char **argv)
{
	if (argc < 1)
		return -1;

	zimg::CPUClass *cpu = static_cast<zimg::CPUClass *>(out);

	try {
		*cpu = parse_cpu(*argv);
	} catch (const std::exception &e) {
		std::cerr << e.what() << '\n';
		return -1;
	}

	return 1;
}

int arg_decode_pixfmt(const struct ArgparseOption *, void *out, int argc, char **argv)
{
	if (argc < 1)
		return -1;

	zimg::PixelFormat *format = static_cast<zimg::PixelFormat *>(out);

	try {
		std::regex format_regex{ R"(^(byte|word|half|float)(?::(f|l)(c|l)?(?::(\d+))?)?$)" };
		std::cmatch match;

		if (!std::regex_match(*argv, match, format_regex))
			throw std::runtime_error{ "bad format string" };

		*format = zimg::default_pixel_format(parse_pixel_type(match[1].str().c_str()));

		if (match.size() >= 2 && match[2].length())
			format->fullrange = (match[2] == "f");
		if (match.size() >= 3 && match[3].length())
			format->chroma = (match[3] == "c");
		if (match.size() >= 4 && match[4].length())
			format->depth = std::stoi(match[4]);
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
