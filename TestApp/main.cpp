#include <iostream>
#include "Common/except.h"
#include "Common/static_map.h"

#include "apps.h"

namespace {;

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
