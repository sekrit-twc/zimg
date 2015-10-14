#ifdef ZIMG_X86

#include <cmath>
#include "common/cpuinfo.h"
#include "common/pixel.h"
#include "graph/image_filter.h"
#include "depth/depth_convert.h"

#include "gtest/gtest.h"
#include "graph/filter_validator.h"

namespace {;

void test_case(zimg::PixelType pixel_in, zimg::PixelType pixel_out, const char * const expected_sha1[3], double expected_snr)
{
	const unsigned w = 640;
	const unsigned h = 480;

	if (!zimg::query_x86_capabilities().sse2) {
		SUCCEED() << "sse2 not available, skipping";
		return;
	}

	auto filter_c = zimg::depth::create_convert_to_float(w, h, pixel_in, pixel_out, zimg::CPUClass::CPU_NONE);
	auto filter_sse2 = zimg::depth::create_convert_to_float(w, h, pixel_in, pixel_out, zimg::CPUClass::CPU_X86_SSE2);

	validate_filter(filter_sse2.get(), w, h, pixel_in, expected_sha1);
	validate_filter_reference(filter_c.get(), filter_sse2.get(), w, h, pixel_in, expected_snr);
}

} // namespace


TEST(F16CSSE2Test, test_half_to_float)
{
	const char *expected_sha1[3] = {
		"68442b2c5704fd2792d92b15fa2e259a51c601dc"
	};

	test_case(zimg::PixelType::HALF, zimg::PixelType::FLOAT, expected_sha1, INFINITY);
}

TEST(F16CSSE2Test, test_float_to_half)
{
	const char *expected_sha1[3] = {
		"4184caae2bd2a3f54722cba1d561cc8720b117ce"
	};

	// The SSE2 approximation does not implement correct rounding.
	test_case(zimg::PixelType::FLOAT, zimg::PixelType::HALF, expected_sha1, 90.0);
}

#endif // ZIMG_X86
