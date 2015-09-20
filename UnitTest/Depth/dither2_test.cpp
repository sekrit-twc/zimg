#include "Common/cpuinfo.h"
#include "Common/pixel.h"
#include "Depth/dither2.h"

#include "gtest/gtest.h"
#include "Common/filter_validator.h"

namespace {;

template <class T>
void test_case(bool fullrange, bool chroma, const char *(*expected_sha1)[3])
{
	const unsigned w = 640;
	const unsigned h = 480;

	zimg::PixelType pixel_in[] = { zimg::PixelType::BYTE, zimg::PixelType::WORD, zimg::PixelType::HALF, zimg::PixelType::FLOAT };
	zimg::PixelType pixel_out[] = { zimg::PixelType::BYTE, zimg::PixelType::WORD };
	unsigned sha1_idx = 0;

	for (zimg::PixelType pxin : pixel_in) {
		for (zimg::PixelType pxout : pixel_out) {
			SCOPED_TRACE(static_cast<int>(pxin));
			SCOPED_TRACE(static_cast<int>(pxout));

			zimg::PixelFormat fmt_in = zimg::default_pixel_format(pxin);
			fmt_in.fullrange = fullrange;
			fmt_in.chroma = chroma;

			zimg::PixelFormat fmt_out = zimg::default_pixel_format(pxout);
			fmt_out.fullrange = fullrange;
			fmt_out.chroma = chroma;

			T dither{ w, h, fmt_in, fmt_out, zimg::CPUClass::CPU_NONE };
			validate_filter(&dither, w, h, pxin, expected_sha1[sha1_idx++]);
		}
	}
}

} // namespace


TEST(DitherTest, test_limited_luma)
{
	const char *expected_sha1[][3] = {
		{ "02c0adca6d301444ac4bf717fa691fe2758752a5" },
		{ "d5794ead078fee72fd10fc396aef511c96f8279c" },
		{ "fcf6b89037c3adc6db0b11ea94e9c10e092d5abf" },
		{ "8ba35ed1784cb6d7903a9092abefb3d9afd7a683" },

		{ "99847f90502e63a05e0dd1c99dfa7757a6992d5e" },
		{ "bf3756e250110a245bcaef45695fe0e4965fd935" },
		{ "d87f5283ffb352d42f6a82779f68e1e19c0c29d8" },
		{ "24696c9239932a3279d69f835da7e3cc4a7f1d31" },
	};

	test_case<zimg::depth::NoneDither>(false, false, expected_sha1);
}

TEST(DitherTest, test_limited_chroma)
{
	const char *expected_sha1[][3] = {
		{ "02c0adca6d301444ac4bf717fa691fe2758752a5" },
		{ "d5794ead078fee72fd10fc396aef511c96f8279c" },
		{ "fcf6b89037c3adc6db0b11ea94e9c10e092d5abf" },
		{ "8ba35ed1784cb6d7903a9092abefb3d9afd7a683" },

		{ "f4f3c22c23a4efdb685a3a707d2f38d04b1dff4a" },
		{ "92570de5e963d41dad11e8ee8dd1cdbb9fba7a0c" },
		{ "b5f02444cc800fae5cf3440bd0fa3ea5b002e0c1" },
		{ "37af27e56e4c9a1d42909f243df7f10a6b4497ae" },
	};

	test_case<zimg::depth::NoneDither>(false, true, expected_sha1);
}

TEST(DitherTest, test_full_luma)
{
	const char *expected_sha1[][3] = {
		{ "02c0adca6d301444ac4bf717fa691fe2758752a5" },
		{ "e9587ac91d9b275d2d0cd5fbf86d603d676b2b42" },
		{ "f9256d89a978d49f1966b6e6cc4d88f5981dc90f" },
		{ "8ba35ed1784cb6d7903a9092abefb3d9afd7a683" },

		{ "07a322ec8acfb191f65fa33ee96ff791e1a6e6ab" },
		{ "95920abbf13f10c0452e82f64e1811499586cafa" },
		{ "2f0dad832822b5e6265d05474c6f51f840de9815" },
		{ "0279ff5c5e08bb2a8d0b288af8de5a5cf596011f" },
	};

	test_case<zimg::depth::NoneDither>(true, false, expected_sha1);
}

TEST(DitherTest, test_full_chroma)
{
	const char *expected_sha1[][3] = {
		{ "02c0adca6d301444ac4bf717fa691fe2758752a5" },
		{ "5e1ed7da45389f9001c0944c9cd26d050ff28cda" },
		{ "997e708b6c0ec8861bace71884852984512494cb" },
		{ "8ba35ed1784cb6d7903a9092abefb3d9afd7a683" },

		{ "65486aaa893bc373b510168b0211b03d07b9ebd6" },
		{ "e2f90b55a5b0c16d887ca72b80be143ba507307c" },
		{ "5c39d701be631b48c3de701e7c9420a87df18f0d" },
		{ "0ffac3dfd3e5641c662be38274ca978c23fd6f32" },
	};

	test_case<zimg::depth::NoneDither>(true, true, expected_sha1);
}

TEST(DitherTest, test_bayer_dither)
{
	const char *expected_sha1[][3] = {
		{ "02c0adca6d301444ac4bf717fa691fe2758752a5" },
		{ "473cdca62d5881a638579446b8f131d6f3e98166" },
		{ "a331ce3f1601d81e099ffd7fb2fbefa896eac0a9" },
		{ "8ba35ed1784cb6d7903a9092abefb3d9afd7a683" },

		{ "0d97c00e747a38d0a9c09de37feda5d26dbfd90c" },
		{ "707fb8e61b184ba14ae8716a85145f622c266474" },
		{ "b5d4bfb1fc8b8ae428572980d19e29a52934675f" },
		{ "43ef7e249783745dcd13538d16d1a5b2bb3803f4" },
	};

	test_case<zimg::depth::BayerDither>(false, false, expected_sha1);
}

TEST(DitherTest, test_random_dither)
{
	const char *expected_sha1[][3] = {
		{ "02c0adca6d301444ac4bf717fa691fe2758752a5" },
		{ "ba84edf6d0a5c0764a533bd60a303b60ee7f8256" },
		{ "0d1917010343c276219e9f65a7df961e49610f4d" },
		{ "8ba35ed1784cb6d7903a9092abefb3d9afd7a683" },

		{ "d63b3965b28a65a21274aeb28f9efbbf37d784d5" },
		{ "26927ea3d6860861ae28ca1d4fa96b8b4b9b909f" },
		{ "7897550b2db75d7024d99f17b65d4d3639abead9" },
		{ "8da9a67b0a75fa7353774dcb142fa54c8035f0d0" },
	};

	test_case<zimg::depth::RandomDither>(false, false, expected_sha1);
}

TEST(DitherTest, test_error_diffusion)
{
	const char *expected_sha1[][3] = {
		{ "02c0adca6d301444ac4bf717fa691fe2758752a5" },
		{ "d5794ead078fee72fd10fc396aef511c96f8279c" },
		{ "e78edb136329d34c7f0a7263506351f89912bc4b" },
		{ "8ba35ed1784cb6d7903a9092abefb3d9afd7a683" },

		{ "7c51826f1d8e4bb582c5f41498ee84c2d31fa53b" },
		{ "bdeb871ef207a4fc380e7137ed66247404e9541e" },
		{ "984788d7da55d911af5cac87f3a2fea8ded6f6e3" },
		{ "e1c935dccda65147066a9e01999cc4e681ba4563" },
	};

	test_case<zimg::depth::ErrorDiffusion>(false, false, expected_sha1);
}
