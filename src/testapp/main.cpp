#include <iostream>
#include <regex>
#include <string>
#include "common/except.h"
#include "common/pixel.h"

#include "apps.h"
#include "table.h"

namespace {

typedef int (*main_func)(int, char **);

void usage()
{
	std::cout << "TestApp subapp [args]\n";
	std::cout << "    colorspace - change colorspace\n";
	std::cout << "    depth      - change depth\n";
	std::cout << "    graph      - benchmark filter graph\n";
	std::cout << "    resize     - resize images\n";
	std::cout << "    unresize   - unresize images\n";
}

main_func lookup_app(const char *name)
{
	static const zimg::static_string_map<main_func, 5> map{
		{ "colorspace", colorspace_main },
		{ "depth",      depth_main },
		{ "graph",      graph_main },
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

	try {
		zimg::CPUClass *cpu = static_cast<zimg::CPUClass *>(out);

		*cpu = g_cpu_table[*argv];
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

	try {
		zimg::PixelFormat *format = static_cast<zimg::PixelFormat *>(out);
		std::regex format_regex{ R"(^(byte|word|half|float)(?::(f|l)(c|l)?(?::(\d+))?)?$)" };
		std::cmatch match;

		if (!std::regex_match(*argv, match, format_regex))
			throw std::runtime_error{ "bad format string" };

		*format = g_pixel_table[match[1].str().c_str()];

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

	main_func func = lookup_app(argv[1]);

	if (!func) {
		usage();
		return 1;
	}

	return func(argc - 1, argv + 1);
}
