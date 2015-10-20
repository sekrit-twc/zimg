#ifdef ZIMG_X86

#include <cmath>
#include "common/cpuinfo.h"
#include "common/pixel.h"
#include "graph/image_filter.h"
#include "depth/depth.h"
#include "depth/dither.h"

#include "gtest/gtest.h"
#include "graph/filter_validator.h"

namespace {;

void test_case(const zimg::PixelFormat &pixel_in, const zimg::PixelFormat &pixel_out, const char * const expected_sha1[3], double expected_snr)
{
	const unsigned w = 640;
	const unsigned h = 480;
	const zimg::depth::DitherType dither = zimg::depth::DitherType::DITHER_RANDOM;

	if (!zimg::query_x86_capabilities().avx2) {
		SUCCEED() << "avx2 not available, skipping";
		return;
	}

	auto filter_c = zimg::depth::create_dither(dither, w, h, pixel_in, pixel_out, zimg::CPUClass::CPU_NONE);
	auto filter_avx2 = zimg::depth::create_dither(dither, w, h, pixel_in, pixel_out, zimg::CPUClass::CPU_X86_AVX2);

	validate_filter(filter_avx2.get(), w, h, pixel_in, expected_sha1);
	validate_filter_reference(filter_c.get(), filter_avx2.get(), w, h, pixel_in, expected_snr);
}

} // namespace


TEST(DitherAVX2Test, test_ordered_dither_b2b)
{
	zimg::PixelFormat pixel_in{ zimg::PixelType::BYTE, 8, true, false };
	zimg::PixelFormat pixel_out{ zimg::PixelType::BYTE, 1, true, false };

	const char *expected_sha1[3] = {
		"e53ddd497f8cc5e881518dcb0b800b872c9231c1"
	};

	test_case(pixel_in, pixel_out, expected_sha1, INFINITY);
}

TEST(DitherAVX2Test, test_ordered_dither_b2w)
{

	zimg::PixelFormat pixel_in{ zimg::PixelType::BYTE, 8, true, false };
	zimg::PixelFormat pixel_out{ zimg::PixelType::WORD, 9, true, false };

	const char *expected_sha1[3] = {
		"6ade59cee43eb811941a8aafed11fca2feb28557"
	};

	test_case(pixel_in, pixel_out, expected_sha1, INFINITY);
}

TEST(DitherAVX2Test, test_ordered_dither_w2b)
{
	zimg::PixelFormat pixel_in = zimg::PixelType::WORD;
	zimg::PixelFormat pixel_out = zimg::PixelType::BYTE;

	const char *expected_sha1[3] = {
		"ab2d2b92cd836f28080b28474b66faab21f2f372"
	};

	test_case(pixel_in, pixel_out, expected_sha1, INFINITY);
}

TEST(DitherAVX2Test, test_ordered_dither_w2w)
{
	zimg::PixelFormat pixel_in{ zimg::PixelType::WORD, 16, false, false };
	zimg::PixelFormat pixel_out{ zimg::PixelType::WORD, 10, false, false };

	const char *expected_sha1[3] = {
		"8c70968a3c9bd9ed508aee81dd06aa27f64fd0ff"
	};

	test_case(pixel_in, pixel_out, expected_sha1, INFINITY);
}

TEST(DitherAVX2Test, test_ordered_dither_h2b)
{
	zimg::PixelFormat pixel_in = zimg::PixelType::HALF;
	zimg::PixelFormat pixel_out = zimg::PixelType::BYTE;

	const char *expected_sha1[3] = {
		"71e8c03d502ce33036e4d73478aa2932a17c7599"
	};

	test_case(pixel_in, pixel_out, expected_sha1, INFINITY);
}

TEST(DitherAVX2Test, test_ordered_dither_h2w)
{
	zimg::PixelFormat pixel_in = zimg::PixelType::HALF;
	zimg::PixelFormat pixel_out = zimg::PixelType::WORD;

	const char *expected_sha1[3] = {
		"9f782faa689eeff1400880e4bc166a4a66ac229e"
	};

	test_case(pixel_in, pixel_out, expected_sha1, INFINITY);
}

TEST(DitherAVX2Test, test_ordered_dither_f2b)
{
	zimg::PixelFormat pixel_in = zimg::PixelType::FLOAT;
	zimg::PixelFormat pixel_out = zimg::PixelType::BYTE;

	const char *expected_sha1[3] = {
		"23e5c6cb1fdd7f9c4dcbbe66178f5a47a5d4ed3e"
	};

	test_case(pixel_in, pixel_out, expected_sha1, INFINITY);
}

TEST(DitherAVX2Test, test_ordered_dither_f2w)
{
	zimg::PixelFormat pixel_in = zimg::PixelType::FLOAT;
	zimg::PixelFormat pixel_out = zimg::PixelType::WORD;

	const char *expected_sha1[3] = {
		"ea75546d27adb26c4fbf829113081cda1173c817"
	};

	// The use of FMA changes the rounding of the result at 16-bits.
	test_case(pixel_in, pixel_out, expected_sha1, 120.0);
}

#endif // ZIMG_X86
