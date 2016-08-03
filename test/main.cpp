#include <cstdio>
#include <cstdlib>
#include "common/libm_wrapper.h"

#include "gtest/gtest.h"
#include "musl-libm/mymath.h"

namespace {

void wrap_libm()
{
	zimg_x_sin = mysin;
	zimg_x_cos = mycos;
	zimg_x_pow = mypow;
	zimg_x_powf = mypowf;
}

} // namespace


int main(int argc, char **argv)
{
	int ret;

	wrap_libm();

	::testing::InitGoogleTest(&argc, argv);
	ret = RUN_ALL_TESTS();

	if (getenv("INTERACTIVE") != nullptr) {
		puts("Press any key to continue...");
		getc(stdin);
	}

	return ret;
}
