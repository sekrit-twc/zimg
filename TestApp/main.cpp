#include <iostream>
#include "apps.h"

namespace {;

void usage()
{
	std::cout << "TestApp subapp [args]\n";
	std::cout << "    resize   - resize images\n";
	std::cout << "    unresize - unresize images\n";
}

} // namespace


int main(int argc, const char **argv)
{
	if (argc < 2) {
		usage();
		return -1;
	}

	if (!strcmp(argv[1], "resize")) {
		return resize_main(argc - 1, argv + 1);
	} else if (!strcmp(argv[1], "unresize")) {
		return unresize_main(argc - 1, argv + 1);
	} else {
		usage();
		return -1;
	}
}
