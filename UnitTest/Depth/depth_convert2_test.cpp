#include "Common/cpuinfo.h"
#include "Common/pixel.h"
#include "Depth/depth_convert2.h"

#include "gtest/gtest.h"
#include "Common/filter_validator.h"

namespace {;

void test_case(bool fullrange, bool chroma, const char *(*expected_sha1)[3])
{
	const unsigned w = 640;
	const unsigned h = 480;

	zimg::PixelType pixel_in[] = { zimg::PixelType::BYTE, zimg::PixelType::WORD, zimg::PixelType::HALF, zimg::PixelType::FLOAT };
	zimg::PixelType pixel_out[] = { zimg::PixelType::HALF, zimg::PixelType::FLOAT };
	unsigned sha1_idx = 0;

	for (zimg::PixelType pxin : pixel_in) {
		for (zimg::PixelType pxout : pixel_out) {
			SCOPED_TRACE(static_cast<int>(pxin));
			SCOPED_TRACE(static_cast<int>(pxout));

			zimg::PixelFormat fmt_in = zimg::default_pixel_format(pxin);
			fmt_in.fullrange = fullrange;
			fmt_in.chroma = chroma;

			zimg::PixelFormat fmt_out = zimg::default_pixel_format(pxout);
			fmt_out.chroma = chroma;

			zimg::depth::DepthConvert2 convert{ w, h, fmt_in, fmt_out, zimg::CPUClass::CPU_NONE };
			validate_filter(&convert, w, h, fmt_in, expected_sha1[sha1_idx++]);
		}
	}
}

} // namespace


TEST(DepthConvert2Test, test_limited_luma)
{
	const char *expected_sha1[][3] = {
		{ "a7096d8251091eb2188bb2bec9fee9d0495faf2c" },
		{ "705050fb0e56681004ede72126bd264d6c4268d9" },
		{ "5d71c3c35d4444b51d31c1739e0cf89f8071af94" },
		{ "5c813b8fda21c1dd3505f165ea0e718eb9e8a427" },

		{ "4184caae2bd2a3f54722cba1d561cc8720b117ce" },
		{ "6b8aafcdd5ea647b85fd83653086432419db6956" },
		{ "4184caae2bd2a3f54722cba1d561cc8720b117ce" },
		{ "483b6bdf608afbf1fba6bbca9657a8ca3822eef1" },
	};

	test_case(false, false, expected_sha1);
}

TEST(DepthConvert2Test, test_limited_chroma)
{
	const char *expected_sha1[][3] = {
		{ "7c84d9bc8a271e543d8e6a503ffdeb651d9b60d9" },
		{ "f0ab1d4b4fa8a87006137918c5bfb423a1020079" },
		{ "84a629953a61dc057e02f3a7af444075d27475db" },
		{ "19c80a38e459f74fbfa53055aff6aeddbddd0f79" },

		{ "8889f7eaa2c3a10bea6827c8e5cc3acfa025fb04" },
		{ "10613af29901e78809cfa4f419aaa32e8055ca2b" },
		{ "8889f7eaa2c3a10bea6827c8e5cc3acfa025fb04" },
		{ "39ba7172306b4c4a16089265e1839d80010ec14f" },
	};

	test_case(false, true, expected_sha1);
}

TEST(DepthConvert2Test, test_full_luma)
{
	const char *expected_sha1[][3] = {
		{ "f0e4a68158eab0ab350c7161498a8eed3196c233" },
		{ "20c77820ff7d4443a0de7991218e2f8eee551e8d" },
		{ "9919fde3704a99ebe5de33f040189ed1cef15add" },
		{ "7ad2bc4ba1be92699ec22f489ae93a8b0dc89821" },

		{ "4184caae2bd2a3f54722cba1d561cc8720b117ce" },
		{ "6b8aafcdd5ea647b85fd83653086432419db6956" },
		{ "4184caae2bd2a3f54722cba1d561cc8720b117ce" },
		{ "483b6bdf608afbf1fba6bbca9657a8ca3822eef1" },
	};

	test_case(true, false, expected_sha1);
}

TEST(DepthConvert2Test, test_full_chroma)
{
	const char *expected_sha1[][3] = {
		{ "333be81b7364a126a2a6167522b539cfad599814" },
		{ "48a95801578c440f7180c799bdc344a873c6d8d6" },
		{ "a4a5448b98ab83e68afde9582dec61d37eb610bb" },
		{ "ad93453f70f4d010049bb6c9e29307ddeee4fa5b" },

		{ "8889f7eaa2c3a10bea6827c8e5cc3acfa025fb04" },
		{ "10613af29901e78809cfa4f419aaa32e8055ca2b" },
		{ "8889f7eaa2c3a10bea6827c8e5cc3acfa025fb04" },
		{ "39ba7172306b4c4a16089265e1839d80010ec14f" },
	};

	test_case(true, true, expected_sha1);
}

TEST(DepthConvert2Test, test_non_full_integer)
{
	const unsigned w = 640;
	const unsigned h = 480;

	zimg::PixelFormat src_format[] = {
		{ zimg::PixelType::BYTE, 1,  true, false },
		{ zimg::PixelType::BYTE, 7,  true, true },
		{ zimg::PixelType::WORD, 9,  false, false },
		{ zimg::PixelType::WORD, 15, false, true },
		{ zimg::PixelType::WORD, 9,  true, false },
		{ zimg::PixelType::WORD, 15, true, true }
	};
	const char *expected_sha1[][3] = {
		{ "6a6a49a71b307303c68ec76e1e196736acc41730" },
		{ "1ebf85f96a3d3cc00cfa6da71edbd9030e0d371e" },
		{ "1451f96e1221f3194dce3b972dfc40fad74c5f80" },
		{ "f28cfe65453b2c1bcb4988d1db9dd3c512d9bdb1" },
		{ "62ca57c53cab818046b537449ad5418e7988cc68" },
		{ "422e55781a043f20738685f22a8c8c3c116810dd" },
	};

	unsigned idx = 0;

	for (const zimg::PixelFormat &format : src_format) {
		SCOPED_TRACE(static_cast<int>(format.type));
		SCOPED_TRACE(format.depth);
		SCOPED_TRACE(format.fullrange);
		SCOPED_TRACE(format.chroma);

		zimg::PixelFormat dst_format = zimg::default_pixel_format(zimg::PixelType::FLOAT);
		dst_format.chroma = format.chroma;

		zimg::depth::DepthConvert2 convert{ w, h, format, dst_format, zimg::CPUClass::CPU_NONE };
		validate_filter(&convert, w, h, format, expected_sha1[idx++]);
	}
}
