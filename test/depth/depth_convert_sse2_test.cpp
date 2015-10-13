#ifdef ZIMG_X86

#include <cmath>
#include "common/cpuinfo.h"
#include "common/pixel.h"
#include "graph/image_filter.h"
#include "depth/depth_convert.h"

#include "gtest/gtest.h"
#include "graph/filter_validator.h"

namespace {;

void test_case_left_shift(const zimg::PixelFormat &pixel_in, const zimg::PixelFormat &pixel_out, const char * const expected_sha1[3], double expected_snr)
{
	const unsigned w = 640;
	const unsigned h = 480;

	if (!zimg::query_x86_capabilities().sse2) {
		SUCCEED() << "sse2 not available, skipping";
		return;
	}

	auto filter_c = zimg::depth::create_left_shift(w, h, pixel_in, pixel_out, zimg::CPUClass::CPU_NONE);
	auto filter_sse2 = zimg::depth::create_left_shift(w, h, pixel_in, pixel_out, zimg::CPUClass::CPU_X86_SSE2);

	validate_filter(filter_sse2.get(), w, h, pixel_in, expected_sha1);
	validate_filter_reference(filter_c.get(), filter_sse2.get(), w, h, pixel_in, expected_snr);
}

TEST(DepthConvertSSE2Test, test_left_shift_b2b)
{
	zimg::PixelFormat pixel_in{ zimg::PixelType::BYTE, 4 };
	zimg::PixelFormat pixel_out{ zimg::PixelType::BYTE, 8 };

	const char *expected_sha1[3] = {
		"09f66fc9d2221b4fad52b3e18b9b31585ebd2b61"
	};

	test_case_left_shift(pixel_in, pixel_out, expected_sha1, INFINITY);
}

TEST(DepthConvertSSE2Test, test_left_shift_b2w)
{
	zimg::PixelFormat pixel_in{ zimg::PixelType::BYTE, 8 };
	zimg::PixelFormat pixel_out{ zimg::PixelType::WORD, 16 };

	const char *expected_sha1[3] = {
		"d5794ead078fee72fd10fc396aef511c96f8279c"
	};

	test_case_left_shift(pixel_in, pixel_out, expected_sha1, INFINITY);
}

TEST(DepthConvertSSE2Test, test_left_shift_w2b)
{
	zimg::PixelFormat pixel_in{ zimg::PixelType::WORD, 4 };
	zimg::PixelFormat pixel_out{ zimg::PixelType::BYTE, 8 };

	const char *expected_sha1[3] = {
		"09f66fc9d2221b4fad52b3e18b9b31585ebd2b61"
	};

	test_case_left_shift(pixel_in, pixel_out, expected_sha1, INFINITY);
}

TEST(DepthConvertSSE2Test, test_left_shift_w2w)
{
	zimg::PixelFormat pixel_in{ zimg::PixelType::WORD, 10 };
	zimg::PixelFormat pixel_out{ zimg::PixelType::WORD, 16 };

	const char *expected_sha1[3] = {
		"1fa20cfbaa8c2de073d5a9569e474c164c4d3ec6"
	};

	test_case_left_shift(pixel_in, pixel_out, expected_sha1, INFINITY);
}

} // namespace

#endif // ZIMG_X86
