#include <cstring>
#include <iostream>
#include "Common/except.h"
#include "apps.h"

namespace {;

void usage()
{
	std::cout << "TestApp subapp [args]\n";
	//std::cout << "    colorspace - change colorspace\n";
	//std::cout << "    depth      - change depth\n";
	//std::cout << "    resize     - resize images\n";
	//std::cout << "    unresize   - unresize images\n";
}

} // namespace


int main(int argc, const char **argv)
{
	if (argc < 2) {
		usage();
		return -1;
	}

	try {
#if 0
		if (!strcmp(argv[1], "colorspace")) {
			return colorspace_main(argc - 1, argv + 1);
		} else if (!strcmp(argv[1], "depth")) {
			return depth_main(argc - 1, argv + 1);
		} else if (!strcmp(argv[1], "resize")) {
			return resize_main(argc - 1, argv + 1);
		} else if (!strcmp(argv[1], "unresize")) {
			return unresize_main(argc - 1, argv + 1);
		} else {
			usage();
			return -1;
		}
#endif
		return 0;
	} catch (const zimg::ZimgException &e) {
		std::cerr << e.what() << '\n';
	} catch (const std::exception &e) {
		std::cerr << e.what() << '\n';
	}
}
