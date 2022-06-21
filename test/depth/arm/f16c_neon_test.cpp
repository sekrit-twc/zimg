#ifdef ZIMG_ARM

#include <cmath>
#include "common/cpuinfo.h"
#include "common/pixel.h"
#include "common/arm/cpuinfo_arm.h"
#include "depth/depth_convert.h"
#include "graphengine/filter.h"

#include "gtest/gtest.h"
#include "graphengine/filter_validation.h"

namespace {

void test_case(zimg::PixelType pixel_in, zimg::PixelType pixel_out, const char *expected_sha1, double expected_snr)
{
	const unsigned w = 640;
	const unsigned h = 480;

	if (!zimg::query_arm_capabilities().neon) {
		SUCCEED() << "f16c not available, skipping";
		return;
	}

	auto filter_c = zimg::depth::create_convert_to_float(w, h, pixel_in, pixel_out, zimg::CPUClass::NONE);
	auto filter_neon = zimg::depth::create_convert_to_float(w, h, pixel_in, pixel_out, zimg::CPUClass::ARM_NEON);

	graphengine::FilterValidation(filter_neon.get(), { w, h, zimg::pixel_size(pixel_in) })
		.set_reference_filter(filter_c.get(), expected_snr)
		.set_input_pixel_format({ zimg::pixel_depth(pixel_in), zimg::pixel_is_float(pixel_in), false })
		.set_output_pixel_format({ zimg::pixel_depth(pixel_out), zimg::pixel_is_float(pixel_out), false })
		.set_sha1(0, expected_sha1)
		.run();
}

} // namespace


TEST(F16CNeonTest, test_half_to_float)
{
	const char *expected_sha1 = "68442b2c5704fd2792d92b15fa2e259a51c601dc";

	test_case(zimg::PixelType::HALF, zimg::PixelType::FLOAT, expected_sha1, INFINITY);
}

TEST(F16CNeonTest, test_float_to_half)
{
	const char *expected_sha1 = "8907defd10af0b7c71abfb9c20147adc1b0a1f70";

	test_case(zimg::PixelType::FLOAT, zimg::PixelType::HALF, expected_sha1, INFINITY);
}

#endif // ZIMG_ARM
