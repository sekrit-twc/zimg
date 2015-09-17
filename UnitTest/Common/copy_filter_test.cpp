#include "Common/copy_filter.h"
#include "Common/pixel.h"

#include "gtest/gtest.h"
#include "Common/filter_validator.h"

TEST(CopyFilterTest, test)
{
	const zimg::PixelType types[] = { zimg::PixelType::BYTE, zimg::PixelType::WORD, zimg::PixelType::HALF, zimg::PixelType::FLOAT };
	const unsigned w = 591;
	const unsigned h = 333;

	const char *expected_sha1[][3] = {
		{ "b7399d798c5f96b4c9ac4c6cccd4c979468bdc7a" },
		{ "43362943f1de4b51f45679a0c460f55c8bd8d2f2" },
		{ "43362943f1de4b51f45679a0c460f55c8bd8d2f2" },
		{ "50ef40fa46a93bfeb6adb4c684add74e1f0bbd49" }
	};

	for (unsigned x = 0; x < 4; ++x) {
		SCOPED_TRACE(static_cast<int>(types[x]));

		zimg::CopyFilter copy{ w, h, types[x] };
		validate_filter(&copy, w, h, types[x], expected_sha1[x]);
	}
}
