#ifdef ZIMG_X86

#include <cmath>
#include <memory>
#include <typeinfo>
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

	if (!zimg::query_x86_capabilities().sse2) {
		SUCCEED() << "sse2 not available, skipping";
		return;
	}

	std::unique_ptr<zimg::graph::ImageFilter> filter_c{ zimg::depth::create_dither(dither, w, h, pixel_in, pixel_out, zimg::CPUClass::CPU_NONE) };
	std::unique_ptr<zimg::graph::ImageFilter> filter_sse2{ zimg::depth::create_dither(dither, w, h, pixel_in, pixel_out, zimg::CPUClass::CPU_X86_SSE2) };

	validate_filter(filter_sse2.get(), w, h, pixel_in, expected_sha1);
	validate_filter_reference(filter_c.get(), filter_sse2.get(), w, h, pixel_in, expected_snr);
}

} // namespace


TEST(DitherSSE2Test, test_ordered_dither_b2b)
{
	zimg::PixelFormat pixel_in{ zimg::PixelType::BYTE, 8, true, false };
	zimg::PixelFormat pixel_out{ zimg::PixelType::BYTE, 1, true, false };

	const char *expected_sha1[3] = {
		"e53ddd497f8cc5e881518dcb0b800b872c9231c1"
	};

	test_case(pixel_in, pixel_out, expected_sha1, INFINITY);
}

TEST(DitherSSE2Test, test_ordered_dither_b2w)
{

	zimg::PixelFormat pixel_in{ zimg::PixelType::BYTE, 8, true, false };
	zimg::PixelFormat pixel_out{ zimg::PixelType::WORD, 9, true, false };

	const char *expected_sha1[3] = {
		"6ade59cee43eb811941a8aafed11fca2feb28557"
	};

	test_case(pixel_in, pixel_out, expected_sha1, INFINITY);
}

TEST(DitherSSE2Test, test_ordered_dither_w2b)
{
	zimg::PixelFormat pixel_in = zimg::default_pixel_format(zimg::PixelType::WORD);
	zimg::PixelFormat pixel_out = zimg::default_pixel_format(zimg::PixelType::BYTE);

	const char *expected_sha1[3] = {
		"ab2d2b92cd836f28080b28474b66faab21f2f372"
	};

	test_case(pixel_in, pixel_out, expected_sha1, INFINITY);
}

TEST(DitherSSE2Test, test_ordered_dither_w2w)
{
	zimg::PixelFormat pixel_in{ zimg::PixelType::WORD, 16, false, false };
	zimg::PixelFormat pixel_out{ zimg::PixelType::WORD, 10, false, false };

	const char *expected_sha1[3] = {
		"8c70968a3c9bd9ed508aee81dd06aa27f64fd0ff"
	};

	test_case(pixel_in, pixel_out, expected_sha1, INFINITY);
}

TEST(DitherSSE2Test, test_ordered_dither_f2b)
{
	zimg::PixelFormat pixel_in = zimg::default_pixel_format(zimg::PixelType::FLOAT);
	zimg::PixelFormat pixel_out = zimg::default_pixel_format(zimg::PixelType::BYTE);

	const char *expected_sha1[3] = {
		"23e5c6cb1fdd7f9c4dcbbe66178f5a47a5d4ed3e"
	};

	test_case(pixel_in, pixel_out, expected_sha1, INFINITY);
}

TEST(DitherSSE2Test, test_ordered_dither_f2w)
{
	zimg::PixelFormat pixel_in = zimg::default_pixel_format(zimg::PixelType::FLOAT);
	zimg::PixelFormat pixel_out = zimg::default_pixel_format(zimg::PixelType::WORD);

	const char *expected_sha1[3] = {
		"57ecf8fc00fc554ba932237ea830e303ed559b19"
	};

	test_case(pixel_in, pixel_out, expected_sha1, INFINITY);
}

#endif // ZIMG_X86
