#ifdef ZIMG_X86_AVX512

#include <cmath>
#include "common/cpuinfo.h"
#include "common/pixel.h"
#include "common/x86/cpuinfo_x86.h"
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
	const zimg::depth::DitherType dither = zimg::depth::DitherType::ORDERED;

	if (!zimg::query_x86_capabilities().avx512f) {
		SUCCEED() << "avx512 not available, skipping";
		return;
	}

	auto filter_c = zimg::depth::create_dither(dither, w, h, pixel_in, pixel_out, zimg::CPUClass::NONE);
	auto filter_avx512 = zimg::depth::create_dither(dither, w, h, pixel_in, pixel_out, zimg::CPUClass::X86_AVX512);

	FilterValidator validator{ filter_avx512.get(), w, h, pixel_in };
	validator.set_sha1(expected_sha1)
	         .set_ref_filter(filter_c.get(), expected_snr)
	         .validate();
}

} // namespace


TEST(DitherAVX512Test, test_ordered_dither_b2b)
{
	zimg::PixelFormat pixel_in{ zimg::PixelType::BYTE, 8, true, false };
	zimg::PixelFormat pixel_out{ zimg::PixelType::BYTE, 1, true, false };

	const char *expected_sha1[3] = {
		"85ac9596d3e91f4f52c4b66c611509fbf891064d"
	};

	test_case(pixel_in, pixel_out, expected_sha1, INFINITY);
}

TEST(DitherAVX512Test, test_ordered_dither_b2w)
{

	zimg::PixelFormat pixel_in{ zimg::PixelType::BYTE, 8, true, false };
	zimg::PixelFormat pixel_out{ zimg::PixelType::WORD, 9, true, false };

	const char *expected_sha1[3] = {
		"267b1039372fab31c14ebf09911da9493ecea95e"
	};

	test_case(pixel_in, pixel_out, expected_sha1, INFINITY);
}

TEST(DitherAVX512Test, test_ordered_dither_w2b)
{
	zimg::PixelFormat pixel_in = zimg::PixelType::WORD;
	zimg::PixelFormat pixel_out = zimg::PixelType::BYTE;

	const char *expected_sha1[3] = {
		"49bb64a45e15aa87f7f85e6f9b4940ef97308c1b"
	};

	test_case(pixel_in, pixel_out, expected_sha1, INFINITY);
}

TEST(DitherAVX512Test, test_ordered_dither_w2w)
{
	zimg::PixelFormat pixel_in{ zimg::PixelType::WORD, 16, false, false };
	zimg::PixelFormat pixel_out{ zimg::PixelType::WORD, 10, false, false };

	const char *expected_sha1[3] = {
		"0495169ad8e289cf171553f1cf4f2c0599bce986"
	};

	test_case(pixel_in, pixel_out, expected_sha1, INFINITY);
}

TEST(DitherAVX512Test, test_ordered_dither_h2b)
{
	zimg::PixelFormat pixel_in = zimg::PixelType::HALF;
	zimg::PixelFormat pixel_out = zimg::PixelType::BYTE;

	const char *expected_sha1[3] = {
		"708482e7450ab5b770bc820b08810b98df2f4b98"
	};

	test_case(pixel_in, pixel_out, expected_sha1, INFINITY);
}

TEST(DitherAVX512Test, test_ordered_dither_h2w)
{
	zimg::PixelFormat pixel_in = zimg::PixelType::HALF;
	zimg::PixelFormat pixel_out = zimg::PixelType::WORD;

	const char *expected_sha1[3] = {
		"8db2cf8d8ffa46eb351e5615bd6d684801431bf9"
	};

	test_case(pixel_in, pixel_out, expected_sha1, INFINITY);
}

TEST(DitherAVX512Test, test_ordered_dither_f2b)
{
	zimg::PixelFormat pixel_in = zimg::PixelType::FLOAT;
	zimg::PixelFormat pixel_out = zimg::PixelType::BYTE;

	const char *expected_sha1[3] = {
		"3bee9485fd5258fbd5e6ba1a361660bf9aaeaa3f"
	};

	test_case(pixel_in, pixel_out, expected_sha1, INFINITY);
}

TEST(DitherAVX512Test, test_ordered_dither_f2w)
{
	zimg::PixelFormat pixel_in = zimg::PixelType::FLOAT;
	zimg::PixelFormat pixel_out = zimg::PixelType::WORD;

	const char *expected_sha1[3] = {
		"126fbca2f0d0027ba7a98d7e86b58596288655c6"
	};

	// The use of FMA changes the rounding of the result at 16-bits.
	test_case(pixel_in, pixel_out, expected_sha1, 120.0);
}

#endif // ZIMG_X86_AVX512
