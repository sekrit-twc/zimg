#ifdef ZIMG_ARM

#include <cmath>
#include "common/cpuinfo.h"
#include "common/pixel.h"
#include "common/arm/cpuinfo_arm.h"
#include "graph/image_filter.h"
#include "depth/depth_convert.h"

#include "gtest/gtest.h"
#include "graph/filter_validator.h"

namespace {

void test_case(zimg::PixelType pixel_in, zimg::PixelType pixel_out, const char * const expected_sha1[3], double expected_snr)
{
	const unsigned w = 640;
	const unsigned h = 480;

	if (!zimg::query_arm_capabilities().neon) {
		SUCCEED() << "neon not available, skipping";
		return;
	}

	auto filter_c = zimg::depth::create_convert_to_float(w, h, pixel_in, pixel_out, zimg::CPUClass::NONE);
	auto filter_neon = zimg::depth::create_convert_to_float(w, h, pixel_in, pixel_out, zimg::CPUClass::ARM_NEON);

	FilterValidator validator{ filter_neon.get(), w, h, pixel_in };
	validator.set_sha1(expected_sha1)
	         .set_ref_filter(filter_c.get(), expected_snr)
	         .validate();
}

} // namespace


TEST(F16CNeonTest, test_half_to_float)
{
	const char *expected_sha1[3] = {
		"68442b2c5704fd2792d92b15fa2e259a51c601dc"
	};

	test_case(zimg::PixelType::HALF, zimg::PixelType::FLOAT, expected_sha1, INFINITY);
}

TEST(F16CNeonTest, test_float_to_half)
{
	const char *expected_sha1[3] = {
		"8907defd10af0b7c71abfb9c20147adc1b0a1f70"
	};

	test_case(zimg::PixelType::FLOAT, zimg::PixelType::HALF, expected_sha1, INFINITY);
}

#endif // ZIMG_ARM
