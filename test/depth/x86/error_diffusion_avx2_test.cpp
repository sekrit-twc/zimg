#ifdef ZIMG_X86

#include <cmath>
#include "common/cpuinfo.h"
#include "common/pixel.h"
#include "common/x86/cpuinfo_x86.h"
#include "graph/image_filter.h"
#include "graphengine/filter.h"
#include "depth/depth.h"
#include "depth/dither.h"

#include "gtest/gtest.h"
#include "graph/filter_validator.h"
#include "graphengine/filter_validation.h"

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

	FilterValidator validator{ filter_avx2.get(), w, h, pixel_in };
	validator.set_sha1(expected_sha1)
	         .set_ref_filter(filter_c.get(), expected_snr)
	         .validate();
}

void test_case_ge(const zimg::PixelFormat &pixel_in, const zimg::PixelFormat &pixel_out, const char *expected_sha1, double expected_snr)
{
	const unsigned w = 640;
	const unsigned h = 480;
	const zimg::depth::DitherType dither = zimg::depth::DitherType::ERROR_DIFFUSION;

	if (!zimg::query_x86_capabilities().avx2) {
		SUCCEED() << "avx2 not available, skipping";
		return;
	}

	bool planes[] = { true, false, false, false };
	zimg::depth::DepthConversion::result result_c = zimg::depth::create_dither_ge(dither, w, h, pixel_in, pixel_out, planes, zimg::CPUClass::NONE);
	zimg::depth::DepthConversion::result result_avx2 = zimg::depth::create_dither_ge(dither, w, h, pixel_in, pixel_out, planes, zimg::CPUClass::X86_AVX2);
	ASSERT_TRUE(result_c.filter_refs[0]);
	ASSERT_TRUE(result_avx2.filter_refs[0]);
	ASSERT_FALSE(assert_different_dynamic_type(result_c.filter_refs[0], result_avx2.filter_refs[0]));

	graphengine::FilterValidation(result_avx2.filter_refs[0], { w, h, zimg::pixel_size(pixel_in.type) })
		.set_input_pixel_format({ pixel_in.depth, zimg::pixel_is_float(pixel_in.type), pixel_in.chroma })
		.set_output_pixel_format({ pixel_out.depth, zimg::pixel_is_float(pixel_out.type), pixel_out.chroma })
		.set_reference_filter(result_c.filter_refs[0], expected_snr)
		.set_sha1(0, expected_sha1)
		.run();
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

TEST(ErrorDiffusionAVX2Test, test_error_diffusion_b2b_ge)
{
	zimg::PixelFormat pixel_in{ zimg::PixelType::BYTE, 8, true, false };
	zimg::PixelFormat pixel_out{ zimg::PixelType::BYTE, 1, true, false };

	const char *expected_sha1 = "7f88314679a06f74d8f361b7eec07a87768ac9f4";

	test_case_ge(pixel_in, pixel_out, expected_sha1, INFINITY);
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

TEST(ErrorDiffusionAVX2Test, test_error_diffusion_b2w_ge)
{
	zimg::PixelFormat pixel_in{ zimg::PixelType::BYTE, 8, true, false };
	zimg::PixelFormat pixel_out{ zimg::PixelType::WORD, 9, true, false };

	const char *expected_sha1 = "db9fe2d13b97bf9f7a717f37985d88ba7b025ae0";

	test_case_ge(pixel_in, pixel_out, expected_sha1, INFINITY);
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

TEST(ErrorDiffusionAVX2Test, test_error_diffusion_w2b_ge)
{
	zimg::PixelFormat pixel_in = zimg::PixelType::WORD;
	zimg::PixelFormat pixel_out = zimg::PixelType::BYTE;

	const char *expected_sha1 = "e78edb136329d34c7f0a7263506351f89912bc4b";

	test_case_ge(pixel_in, pixel_out, expected_sha1, INFINITY);
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

TEST(ErrorDiffusionAVX2Test, test_error_diffusion_w2w_ge)
{
	zimg::PixelFormat pixel_in{ zimg::PixelType::WORD, 16, false, false };
	zimg::PixelFormat pixel_out{ zimg::PixelType::WORD, 10, false, false };

	const char *expected_sha1 = "86397c91f37ec9a671feac8cce2508a6b67181f4";

	test_case_ge(pixel_in, pixel_out, expected_sha1, INFINITY);
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

TEST(ErrorDiffusionAVX2Test, test_error_diffusion_h2b_ge)
{
	zimg::PixelFormat pixel_in = zimg::PixelType::HALF;
	zimg::PixelFormat pixel_out = zimg::PixelType::BYTE;

	const char *expected_sha1 = "17ffbdc53895e2576f02f8279264d7c54f723671";

	test_case_ge(pixel_in, pixel_out, expected_sha1, INFINITY);
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

TEST(ErrorDiffusionAVX2Test, test_error_diffusion_h2w_ge)
{
	zimg::PixelFormat pixel_in = zimg::PixelType::HALF;
	zimg::PixelFormat pixel_out = zimg::PixelType::WORD;

	const char *expected_sha1 = "cf92073110b1752ac6a1059229660457c4a9deef";

	test_case_ge(pixel_in, pixel_out, expected_sha1, INFINITY);
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

TEST(ErrorDiffusionAVX2Test, test_error_diffusion_f2b_ge)
{
	zimg::PixelFormat pixel_in = zimg::PixelType::FLOAT;
	zimg::PixelFormat pixel_out = zimg::PixelType::BYTE;

	const char *expected_sha1 = "4ed3a75693d507e93a1cf3550fbad51bfff17c3b";

	// With single-precision input, the error calculation difference from FMA becomes apparent.
	test_case_ge(pixel_in, pixel_out, expected_sha1, 50.0);
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

TEST(ErrorDiffusionAVX2Test, test_error_diffusion_f2w_ge)
{
	zimg::PixelFormat pixel_in = zimg::PixelType::FLOAT;
	zimg::PixelFormat pixel_out = zimg::PixelType::WORD;

	const char *expected_sha1 = "c512b5d7e29e6bd073d2ae194cdd1738539b6166";

	// With single-precision input, the error calculation difference from FMA becomes apparent.
	test_case_ge(pixel_in, pixel_out, expected_sha1, 90.0);
}

#endif // ZIMG_X86
