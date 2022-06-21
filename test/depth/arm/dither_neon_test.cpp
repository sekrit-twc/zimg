#ifdef ZIMG_ARM

#include <cmath>
#include "common/cpuinfo.h"
#include "common/pixel.h"
#include "common/arm/cpuinfo_arm.h"
#include "depth/depth.h"
#include "depth/dither.h"
#include "graphengine/filter.h"

#include "gtest/gtest.h"
#include "graphengine/filter_validation.h"

// ARMv7a vs ARMv8a numerics:
//     ARMv7a NEON does not support round-half-to-even mode (i.e. IEEE-754).
// For performance reasons, the ARM 32-bit kernels use floor(x + 0.5) instead.
//     ARMv8a implements fused-multiply-add, which produces the same result as
// the x86 AVX2 kernels, but slightly different output from C.

namespace {

void test_case(const zimg::PixelFormat &pixel_in, const zimg::PixelFormat &pixel_out, const char *expected_sha1, double expected_snr)
{
	const unsigned w = 640;
	const unsigned h = 480;
	const zimg::depth::DitherType dither = zimg::depth::DitherType::ORDERED;

	if (!zimg::query_arm_capabilities().neon) {
		SUCCEED() << "sse2 not available, skipping";
		return;
	}

	bool planes[] = { true, false, false, false };
	auto result_c = zimg::depth::create_dither(dither, w, h, pixel_in, pixel_out, planes, zimg::CPUClass::NONE);
	auto result_neon = zimg::depth::create_dither(dither, w, h, pixel_in, pixel_out, planes, zimg::CPUClass::ARM_NEON);

	graphengine::FilterValidation(result_neon.filter_refs[0], { w, h, zimg::pixel_size(pixel_in.type) })
		.set_reference_filter(result_c.filter_refs[0], expected_snr)
		.set_input_pixel_format({ pixel_in.depth, zimg::pixel_is_float(pixel_in.type), pixel_in.chroma })
		.set_output_pixel_format({ pixel_out.depth, zimg::pixel_is_float(pixel_out.type), pixel_out.chroma })
		.set_sha1(0, expected_sha1)
		.run();
}

} // namespace


TEST(DitherNeonTest, test_ordered_dither_b2b)
{
	zimg::PixelFormat pixel_in{ zimg::PixelType::BYTE, 8, true, false };
	zimg::PixelFormat pixel_out{ zimg::PixelType::BYTE, 1, true, false };

	const char *expected_sha1 = "85ac9596d3e91f4f52c4b66c611509fbf891064d";

	test_case(pixel_in, pixel_out, expected_sha1, INFINITY);
}

TEST(DitherNeonTest, test_ordered_dither_b2w)
{

	zimg::PixelFormat pixel_in{ zimg::PixelType::BYTE, 8, true, false };
	zimg::PixelFormat pixel_out{ zimg::PixelType::WORD, 9, true, false };

#if defined(_M_ARM64) || defined(__aarch64__)
	const char *expected_sha1 = "267b1039372fab31c14ebf09911da9493ecea95e";
#else
	const char *expected_sha1 = "5cf2428d1af1e316f292fa71cc8a78a658acb546";
#endif

#if defined(_M_ARM64) || defined(__aarch64__)
	test_case(pixel_in, pixel_out, expected_sha1, INFINITY);
#else
	test_case(pixel_in, pixel_out, expected_sha1, 90.0);
#endif
}

TEST(DitherNeonTest, test_ordered_dither_w2b)
{
	zimg::PixelFormat pixel_in = zimg::PixelType::WORD;
	zimg::PixelFormat pixel_out = zimg::PixelType::BYTE;

	const char *expected_sha1 = "49bb64a45e15aa87f7f85e6f9b4940ef97308c1b";

	test_case(pixel_in, pixel_out, expected_sha1, INFINITY);
}

TEST(DitherNeonTest, test_ordered_dither_w2w)
{
	zimg::PixelFormat pixel_in{ zimg::PixelType::WORD, 16, false, false };
	zimg::PixelFormat pixel_out{ zimg::PixelType::WORD, 10, false, false };

	const char *expected_sha1 = "0495169ad8e289cf171553f1cf4f2c0599bce986";

	test_case(pixel_in, pixel_out, expected_sha1, INFINITY);
}

TEST(DitherNeonTest, test_ordered_dither_h2b)
{
	zimg::PixelFormat pixel_in = zimg::PixelType::HALF;
	zimg::PixelFormat pixel_out = zimg::PixelType::BYTE;

#if defined(_M_ARM64) || defined(__aarch64__)
	const char *expected_sha1 = "708482e7450ab5b770bc820b08810b98df2f4b98";
#else
	const char *expected_sha1 = "865ca02e1b3bcd29f237a5698ce005116e1f992a";
#endif

#if defined(_M_ARM64) || defined(__aarch64__)
	test_case(pixel_in, pixel_out, expected_sha1, INFINITY);
#else
	test_case(pixel_in, pixel_out, expected_sha1, 90.0);
#endif
}

TEST(DitherNeonTest, test_ordered_dither_h2w)
{
	zimg::PixelFormat pixel_in = zimg::PixelType::HALF;
	zimg::PixelFormat pixel_out = zimg::PixelType::WORD;

#if defined(_M_ARM64) || defined(__aarch64__)
	const char *expected_sha1 = "8db2cf8d8ffa46eb351e5615bd6d684801431bf9";
#else
	const char *expected_sha1 = "f4ea5e3bbe05b668180a74a8701c5fdcb01d78ae";
#endif

#if defined(_M_ARM64) || defined(__aarch64__)
	test_case(pixel_in, pixel_out, expected_sha1, INFINITY);
#else
	test_case(pixel_in, pixel_out, expected_sha1, 90.0);
#endif
}

TEST(DitherNeonTest, test_ordered_dither_f2b)
{
	zimg::PixelFormat pixel_in = zimg::PixelType::FLOAT;
	zimg::PixelFormat pixel_out = zimg::PixelType::BYTE;

	const char *expected_sha1 = "3bee9485fd5258fbd5e6ba1a361660bf9aaeaa3f";

	test_case(pixel_in, pixel_out, expected_sha1, INFINITY);
}

TEST(DitherNeonTest, test_ordered_dither_f2w)
{
	zimg::PixelFormat pixel_in = zimg::PixelType::FLOAT;
	zimg::PixelFormat pixel_out = zimg::PixelType::WORD;

#if defined(_M_ARM64) || defined(__aarch64__)
	const char *expected_sha1 = "126fbca2f0d0027ba7a98d7e86b58596288655c6";
#else
	const char *expected_sha1 = "dc8d5ed5955003310eb9d74c0baa48fbe6bd5135";
#endif

#if defined(_M_ARM64) || defined(__aarch64__)
	test_case(pixel_in, pixel_out, expected_sha1, 120.0);
#else
	test_case(pixel_in, pixel_out, expected_sha1, 110.0);
#endif
}

#endif // ZIMG_ARM
