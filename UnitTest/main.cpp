#include <cstdio>
#include <cstdlib>
#include "gtest/gtest.h"

int main(int argc, char **argv)
{
	int ret;

	::testing::InitGoogleTest(&argc, argv);
	ret = RUN_ALL_TESTS();

	if (getenv("INTERACTIVE") != nullptr) {
		printf("Press any key to continue...");
		getc(stdin);
	}

	return ret;
}
