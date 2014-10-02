#include <cstring>
#include <iostream>
#include "Common/except.h"
#include "apps.h"

using namespace zimg;

namespace {;

void usage()
{
	std::cout << "TestApp subapp [args]\n";
	std::cout << "    resize     - resize images\n";
	std::cout << "    unresize   - unresize images\n";
	std::cout << "    colorspace - change colorspace\n";
}

} // namespace


int main(int argc, const char **argv)
{
	if (argc < 2) {
		usage();
		return -1;
	}

	try {
		if (!strcmp(argv[1], "resize")) {
			return resize_main(argc - 1, argv + 1);
		} else if (!strcmp(argv[1], "unresize")) {
			return unresize_main(argc - 1, argv + 1);
		} else if (!strcmp(argv[1], "colorspace")) {
			return colorspace_main(argc - 1, argv + 1);
		} else {
			usage();
			return -1;
		}
	} catch (const ZimgException &e) {
		std::cerr << e.what() << '\n';
	} catch (const std::exception &e) {
		std::cerr << e.what() << '\n';
	}
}
