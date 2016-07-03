#ifdef ZIMG_X86

#include <cmath>
#include "common/cpuinfo.h"
#include "common/pixel.h"
#include "graph/image_filter.h"
#include "depth/depth_convert.h"

#include "gtest/gtest.h"
#include "graph/filter_validator.h"

namespace {

void test_case(zimg::PixelType pixel_in, zimg::PixelType pixel_out, const char * const expected_sha1[3], double expected_snr)
{
	const unsigned w = 640;
	const unsigned h = 480;

	if (!zimg::query_x86_capabilities().f16c) {
		SUCCEED() << "f16c not available, skipping";
		return;
	}

	auto filter_c = zimg::depth::create_convert_to_float(w, h, pixel_in, pixel_out, zimg::CPUClass::NONE);
	auto filter_f16c = zimg::depth::create_convert_to_float(w, h, pixel_in, pixel_out, zimg::CPUClass::X86_F16C);

	validate_filter(filter_f16c.get(), w, h, pixel_in, expected_sha1);
	validate_filter_reference(filter_c.get(), filter_f16c.get(), w, h, pixel_in, expected_snr);
}

} // namespace


TEST(F16CIVBTest, test_half_to_float)
{
	const char *expected_sha1[3] = {
		"68442b2c5704fd2792d92b15fa2e259a51c601dc"
	};

	test_case(zimg::PixelType::HALF, zimg::PixelType::FLOAT, expected_sha1, INFINITY);
}

TEST(F16CIVBTest, test_float_to_half)
{
	const char *expected_sha1[3] = {
		"8907defd10af0b7c71abfb9c20147adc1b0a1f70"
	};

	test_case(zimg::PixelType::FLOAT, zimg::PixelType::HALF, expected_sha1, INFINITY);
}

#endif // ZIMG_X86
