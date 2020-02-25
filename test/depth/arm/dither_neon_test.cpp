#ifdef ZIMG_ARM

#include <cmath>
#include "common/cpuinfo.h"
#include "common/pixel.h"
#include "common/arm/cpuinfo_arm.h"
#include "graph/image_filter.h"
#include "depth/depth.h"
#include "depth/dither.h"

#include "gtest/gtest.h"
#include "graph/filter_validator.h"

// ARMv7a vs ARMv8a numerics:
//     ARMv7a NEON does not support round-half-to-even mode (i.e. IEEE-754).
// For performance reasons, the ARM 32-bit kernels use floor(x + 0.5) instead.
//     ARMv8a implements fused-multiply-add, which produces the same result as
// the x86 AVX2 kernels, but slightly different output from C.

namespace {

void test_case(const zimg::PixelFormat &pixel_in, const zimg::PixelFormat &pixel_out, const char * const expected_sha1[3], double expected_snr)
{
	const unsigned w = 640;
	const unsigned h = 480;
	const zimg::depth::DitherType dither = zimg::depth::DitherType::ORDERED;

	if (!zimg::query_arm_capabilities().neon) {
		SUCCEED() << "neon not available, skipping";
		return;
	}

	auto filter_c = zimg::depth::create_dither(dither, w, h, pixel_in, pixel_out, zimg::CPUClass::NONE);
	auto filter_neon = zimg::depth::create_dither(dither, w, h, pixel_in, pixel_out, zimg::CPUClass::ARM_NEON);

	FilterValidator validator{ filter_neon.get(), w, h, pixel_in };
	validator.set_sha1(expected_sha1)
	         .set_ref_filter(filter_c.get(), expected_snr)
	         .validate();
}

} // namespace


TEST(DitherNeonTest, test_ordered_dither_b2b)
{
	zimg::PixelFormat pixel_in{ zimg::PixelType::BYTE, 8, true, false };
	zimg::PixelFormat pixel_out{ zimg::PixelType::BYTE, 1, true, false };

	const char *expected_sha1[3] = {
		"85ac9596d3e91f4f52c4b66c611509fbf891064d"
	};

	test_case(pixel_in, pixel_out, expected_sha1, INFINITY);
}

TEST(DitherNeonTest, test_ordered_dither_b2w)
{

	zimg::PixelFormat pixel_in{ zimg::PixelType::BYTE, 8, true, false };
	zimg::PixelFormat pixel_out{ zimg::PixelType::WORD, 9, true, false };

	const char *expected_sha1[3] = {
#if defined(_M_ARM64) || defined(__aarch64__)
		"267b1039372fab31c14ebf09911da9493ecea95e"
#else
		"5cf2428d1af1e316f292fa71cc8a78a658acb546"
#endif
	};

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

	const char *expected_sha1[3] = {
		"49bb64a45e15aa87f7f85e6f9b4940ef97308c1b"
	};

	test_case(pixel_in, pixel_out, expected_sha1, INFINITY);
}

TEST(DitherNeonTest, test_ordered_dither_w2w)
{
	zimg::PixelFormat pixel_in{ zimg::PixelType::WORD, 16, false, false };
	zimg::PixelFormat pixel_out{ zimg::PixelType::WORD, 10, false, false };

	const char *expected_sha1[3] = {
		"0495169ad8e289cf171553f1cf4f2c0599bce986"
	};

	test_case(pixel_in, pixel_out, expected_sha1, INFINITY);
}

TEST(DitherNeonTest, test_ordered_dither_f2b)
{
	zimg::PixelFormat pixel_in = zimg::PixelType::FLOAT;
	zimg::PixelFormat pixel_out = zimg::PixelType::BYTE;

	const char *expected_sha1[3] = {
		"3bee9485fd5258fbd5e6ba1a361660bf9aaeaa3f"
	};

	test_case(pixel_in, pixel_out, expected_sha1, INFINITY);
}

TEST(DitherNeonTest, test_ordered_dither_f2w)
{
	zimg::PixelFormat pixel_in = zimg::PixelType::FLOAT;
	zimg::PixelFormat pixel_out = zimg::PixelType::WORD;

	const char *expected_sha1[3] = {
#if defined(_M_ARM64) || defined(__aarch64__)
		"126fbca2f0d0027ba7a98d7e86b58596288655c6"
#else
		"dc8d5ed5955003310eb9d74c0baa48fbe6bd5135"
#endif
	};

#if defined(_M_ARM64) || defined(__aarch64__)
	test_case(pixel_in, pixel_out, expected_sha1, 120.0);
#else
	test_case(pixel_in, pixel_out, expected_sha1, 110.0);
#endif
}

#endif // ZIMG_ARM
