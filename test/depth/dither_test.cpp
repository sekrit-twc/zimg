#include "common/cpuinfo.h"
#include "common/pixel.h"
#include "graph/image_filter.h"
#include "depth/depth.h"
#include "depth/dither.h"

#include "gtest/gtest.h"
#include "graph/filter_validator.h"

namespace {

void test_case(zimg::depth::DitherType type, bool fullrange, bool chroma, const char *(*expected_sha1)[3])
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

			zimg::PixelFormat fmt_in = pxin;
			fmt_in.fullrange = fullrange;
			fmt_in.chroma = chroma;

			zimg::PixelFormat fmt_out = pxout;
			fmt_out.fullrange = fullrange;
			fmt_out.chroma = chroma;

			auto dither = zimg::depth::create_dither(type, w, h, fmt_in, fmt_out, zimg::CPUClass::NONE);

			FilterValidator validator{ dither.get(), w, h, fmt_in };
			validator.set_sha1(expected_sha1[sha1_idx++]);
			validator.validate();
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

	test_case(zimg::depth::DitherType::NONE, false, false, expected_sha1);
}

TEST(DitherTest, test_limited_chroma)
{
	const char *expected_sha1[][3] = {
		{ "02c0adca6d301444ac4bf717fa691fe2758752a5" },
		{ "d5794ead078fee72fd10fc396aef511c96f8279c" },

		{ "6b4f540bab559ae1ef9b352a187980c32869dccb" },
		{ "8ba35ed1784cb6d7903a9092abefb3d9afd7a683" },

		{ "060f6e8e7210c498427cef730c3602a0b5f31240" },
		{ "459af9f2faa1bc11ff8cd7e1d5995194ca7330d1" },

		{ "3c1832c2c1cfdd9907de1367b3ebdf0eca6d8b31" },
		{ "0686f300af1924d6592d6b422595616bfbc7e63e" },
	};

	test_case(zimg::depth::DitherType::NONE, false, true, expected_sha1);
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

	test_case(zimg::depth::DitherType::NONE, true, false, expected_sha1);
}

TEST(DitherTest, test_full_chroma)
{
	const char *expected_sha1[][3] = {
		{ "02c0adca6d301444ac4bf717fa691fe2758752a5" },
		{ "5e1ed7da45389f9001c0944c9cd26d050ff28cda" },

		{ "997e708b6c0ec8861bace71884852984512494cb" },
		{ "8ba35ed1784cb6d7903a9092abefb3d9afd7a683" },

		{ "1dedc59f9d9fd1680aa42fdba8465f4114bc50f2" },
		{ "9602dd577a97f1d1e7939b1a9e00c4db1110b8db" },

		{ "a0680c0b28fbc02561fea7746adc430354db472a" },
		{ "1bda9505b332e08f9810d75b9858cacec4e50201" },
	};

	test_case(zimg::depth::DitherType::NONE, true, true, expected_sha1);
}

TEST(DitherTest, test_bayer_dither)
{
	const char *expected_sha1[][3] = {
		{ "02c0adca6d301444ac4bf717fa691fe2758752a5" },
		{ "d5794ead078fee72fd10fc396aef511c96f8279c" },

		{ "49bb64a45e15aa87f7f85e6f9b4940ef97308c1b" },
		{ "8ba35ed1784cb6d7903a9092abefb3d9afd7a683" },

		{ "708482e7450ab5b770bc820b08810b98df2f4b98" },
		{ "8db2cf8d8ffa46eb351e5615bd6d684801431bf9" },

		{ "3bee9485fd5258fbd5e6ba1a361660bf9aaeaa3f" },
		{ "5312234ac7d6198f138b2cded18b5bf48b6af568" },
	};

	test_case(zimg::depth::DitherType::ORDERED, false, false, expected_sha1);
}

TEST(DitherTest, test_random_dither)
{
	const char *expected_sha1[][3] = {
		{ "02c0adca6d301444ac4bf717fa691fe2758752a5" },
		{ "d5794ead078fee72fd10fc396aef511c96f8279c" },

		{ "6da260ba1fec3eda99d83739ad3ae9eaeba15df2" },
		{ "8ba35ed1784cb6d7903a9092abefb3d9afd7a683" },

		{ "30ef22fa882d6f4bc0dbb7224a1dffdedf108734" },
		{ "661fad247819b453a77fffb32aa4da95c0e7024e" },

		{ "6bf40dea8fb17f035be0e4fa14a303b3cde45dc5" },
		{ "04df32587f1e6222c1a3979ccf77e0c1862db4bf" },
	};

	test_case(zimg::depth::DitherType::RANDOM, false, false, expected_sha1);
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

	test_case(zimg::depth::DitherType::ERROR_DIFFUSION, false, false, expected_sha1);
}
