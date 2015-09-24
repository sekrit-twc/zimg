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
		{ "0c8c7f18e5ee4ae4a3344a855ad31902ea4445e4" },

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

		{ "e86412979136ccf931dff6b71621e7883767466e" },
		{ "0bb320e7da86bec4a97849a97326d4309d6f3f2d" },

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

		{ "1d8189624d3e4c7e41e2b5d8992cf74df34086d3" },
		{ "34cde4d903c8eb626df6c8185e207146be7c1607" },

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
		{ "1374dbe6d9574768d7ee156aee92928dfa74c1ea" },

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

		{ "e0dc8bcbb17937fd2c573221a2057752f434f637" },
		{ "f899faf6f14cff8e84f7bdaf2e4cc415457f46b3" },

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

		{ "ba9697be1dc62391630b5cab0093662d2f0e282b" },
		{ "e4578bcc777bfa3782c19382b4f7e636d0d67dec" },

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

		{ "17ffbdc53895e2576f02f8279264d7c54f723671" },
		{ "cf92073110b1752ac6a1059229660457c4a9deef" },

		{ "c739ba4bed041192bc6167e03975fc5a709561b4" },
		{ "834f918f24da72a31bb6deb7b1e398446cf052a2" },
	};

	test_case<zimg::depth::ErrorDiffusion>(false, false, expected_sha1);
}
