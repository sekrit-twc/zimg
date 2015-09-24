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

		{ "6b4f540bab559ae1ef9b352a187980c32869dccb" },
		{ "8ba35ed1784cb6d7903a9092abefb3d9afd7a683" },

		{ "99847f90502e63a05e0dd1c99dfa7757a6992d5e" },
		{ "396a1fb8f918a455323f0c13ba9d53a9be765c46" },

		{ "5e7391e61888bc96684b4fe7e08bedd677eb6233" },
		{ "1170f2c7b4ad7c7d76ae79490d97ae0ce5b4a929" },
	};

	test_case<zimg::depth::NoneDither>(false, false, expected_sha1);
}

TEST(DitherTest, test_limited_chroma)
{
	const char *expected_sha1[][3] = {
		{ "02c0adca6d301444ac4bf717fa691fe2758752a5" },
		{ "d5794ead078fee72fd10fc396aef511c96f8279c" },

		{ "6b4f540bab559ae1ef9b352a187980c32869dccb" },
		{ "8ba35ed1784cb6d7903a9092abefb3d9afd7a683" },

		{ "48f75a7f35a862d266269f9851447ed7d25a17e2" },
		{ "59adfa4848949f5a0722cee18f043ca837ca9981" },

		{ "b5f02444cc800fae5cf3440bd0fa3ea5b002e0c1" },
		{ "02a87cf37521da2e674c3d89491a932614a0e3aa" },
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

		{ "e17c0242b746b63684213f21c4290fc2f123e595" },
		{ "b48655a46858eff58b51b266d842fa5d5600e032" },
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
		{ "c32d9bb3ca16e69d2a5bc50471e9a58969a65547" },

		{ "88d248b9f51fc73e07f87f43c86eaff483ba3f50" },
		{ "2b449e40db3a8d4d5a4ca4c2cd74ef38e237ac57" },
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

		{ "e6dae1f43345aad233767ddcebd61936d1cefe23" },
		{ "fcdce35121ed4a57776ef82689b00be4523c4283" },

		{ "904ebf27c49a482f4b6cd949b9dc66c0f00a0993" },
		{ "072c15f28913d1c84c3e19e36d152be754c148bb" },
	};

	test_case<zimg::depth::BayerDither>(false, false, expected_sha1);
}

TEST(DitherTest, test_random_dither)
{
	const char *expected_sha1[][3] = {
		{ "02c0adca6d301444ac4bf717fa691fe2758752a5" },
		{ "d5794ead078fee72fd10fc396aef511c96f8279c" },

		{ "d5e73a6ba9358e8fab7f53c3eafe184590dd6fb2" },
		{ "8ba35ed1784cb6d7903a9092abefb3d9afd7a683" },

		{ "e706b6a9cfa7510568bc0e2c82092ee3fbf337c0" },
		{ "f0650684a16fbf387e5848c15c39888905d288a7" },

		{ "7897550b2db75d7024d99f17b65d4d3639abead9" },
		{ "7e0024f528edce64731273d704575ce0587f1da5" },
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
		{ "7f75933cbdb6349dafeccb385170d8125eadd061" },

		{ "c739ba4bed041192bc6167e03975fc5a709561b4" },
		{ "834f918f24da72a31bb6deb7b1e398446cf052a2" },
	};

	test_case<zimg::depth::ErrorDiffusion>(false, false, expected_sha1);
}
