#include <cstdio>
#include <cstdlib>
#include "common/libm_wrapper.h"

#include "gtest/gtest.h"
#include "musl-libm/mymath.h"

namespace {

void wrap_libm()
{
	_zimg_sin = mysin;
	_zimg_cos = mycos;
	_zimg_pow = mypow;
	_zimg_powf = mypowf;
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
