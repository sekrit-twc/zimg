#ifdef ZIMG_X86

#include <cmath>
#include "common/cpuinfo.h"
#include "common/pixel.h"
#include "graph/image_filter.h"
#include "depth/depth.h"
#include "depth/dither.h"

#include "gtest/gtest.h"
#include "graph/filter_validator.h"

namespace {

void test_case(const zimg::PixelFormat &pixel_in, const zimg::PixelFormat &pixel_out, const char * const expected_sha1[3], double expected_snr)
{
	const unsigned w = 640;
	const unsigned h = 480;
	const zimg::depth::DitherType dither = zimg::depth::DitherType::ERROR_DIFFUSION;

	if (!zimg::query_x86_capabilities().avx2) {
		SUCCEED() << "avx2 not available, skipping";
		return;
	}

	auto filter_c = zimg::depth::create_dither(dither, w, h, pixel_in, pixel_out, zimg::CPUClass::NONE);
	auto filter_avx2 = zimg::depth::create_dither(dither, w, h, pixel_in, pixel_out, zimg::CPUClass::X86_AVX2);
	ASSERT_FALSE(assert_different_dynamic_type(filter_c.get(), filter_avx2.get()));

	validate_filter(filter_avx2.get(), w, h, pixel_in, expected_sha1);
	validate_filter_reference(filter_c.get(), filter_avx2.get(), w, h, pixel_in, expected_snr);
}

} // namespace


TEST(ErrorDiffusionAVX2Test, test_error_diffusion_b2b)
{
	zimg::PixelFormat pixel_in{ zimg::PixelType::BYTE, 8, true, false };
	zimg::PixelFormat pixel_out{ zimg::PixelType::BYTE, 1, true, false };

	const char *expected_sha1[3] = {
		"7f88314679a06f74d8f361b7eec07a87768ac9f4"
	};

	test_case(pixel_in, pixel_out, expected_sha1, INFINITY);
}

TEST(ErrorDiffusionAVX2Test, test_error_diffusion_b2w)
{

	zimg::PixelFormat pixel_in{ zimg::PixelType::BYTE, 8, true, false };
	zimg::PixelFormat pixel_out{ zimg::PixelType::WORD, 9, true, false };

	const char *expected_sha1[3] = {
		"db9fe2d13b97bf9f7a717f37985d88ba7b025ae0"
	};

	test_case(pixel_in, pixel_out, expected_sha1, INFINITY);
}

TEST(ErrorDiffusionAVX2Test, test_error_diffusion_w2b)
{
	zimg::PixelFormat pixel_in = zimg::PixelType::WORD;
	zimg::PixelFormat pixel_out = zimg::PixelType::BYTE;

	const char *expected_sha1[3] = {
		"e78edb136329d34c7f0a7263506351f89912bc4b"
	};

	test_case(pixel_in, pixel_out, expected_sha1, INFINITY);
}

TEST(ErrorDiffusionAVX2Test, test_error_diffusion_w2w)
{
	zimg::PixelFormat pixel_in{ zimg::PixelType::WORD, 16, false, false };
	zimg::PixelFormat pixel_out{ zimg::PixelType::WORD, 10, false, false };

	const char *expected_sha1[3] = {
		"86397c91f37ec9a671feac8cce2508a6b67181f4"
	};

	test_case(pixel_in, pixel_out, expected_sha1, INFINITY);
}

TEST(ErrorDiffusionAVX2Test, test_error_diffusion_h2b)
{
	zimg::PixelFormat pixel_in = zimg::PixelType::HALF;
	zimg::PixelFormat pixel_out = zimg::PixelType::BYTE;

	const char *expected_sha1[3] = {
		"17ffbdc53895e2576f02f8279264d7c54f723671"
	};

	test_case(pixel_in, pixel_out, expected_sha1, INFINITY);
}

TEST(ErrorDiffusionAVX2Test, test_error_diffusion_h2w)
{
	zimg::PixelFormat pixel_in = zimg::PixelType::HALF;
	zimg::PixelFormat pixel_out = zimg::PixelType::WORD;

	const char *expected_sha1[3] = {
		"cf92073110b1752ac6a1059229660457c4a9deef"
	};

	test_case(pixel_in, pixel_out, expected_sha1, INFINITY);
}

TEST(ErrorDiffusionAVX2Test, test_error_diffusion_f2b)
{
	zimg::PixelFormat pixel_in = zimg::PixelType::FLOAT;
	zimg::PixelFormat pixel_out = zimg::PixelType::BYTE;

	const char *expected_sha1[3] = {
		"4ed3a75693d507e93a1cf3550fbad51bfff17c3b"
	};

	// With single-precision input, the error calculation difference from FMA becomes apparent.
	test_case(pixel_in, pixel_out, expected_sha1, 50.0);
}

TEST(ErrorDiffusionAVX2Test, test_error_diffusion_f2w)
{
	zimg::PixelFormat pixel_in = zimg::PixelType::FLOAT;
	zimg::PixelFormat pixel_out = zimg::PixelType::WORD;

	const char *expected_sha1[3] = {
		"c512b5d7e29e6bd073d2ae194cdd1738539b6166"
	};

	// With single-precision input, the error calculation difference from FMA becomes apparent.
	test_case(pixel_in, pixel_out, expected_sha1, 50.0);
}

#endif // ZIMG_X86
