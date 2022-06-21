#include <utility>
#include "common/pixel.h"
#include "graphengine/filter.h"
#include "depth/depth.h"

#include "gtest/gtest.h"
#include "graphengine/filter_validation.h"	

namespace {

void test_case(bool fullrange, bool chroma, bool ycgco, const char * const *expected_sha1)
{
	const unsigned w = 640;
	const unsigned h = 480;

	zimg::PixelType pixel_in[] = { zimg::PixelType::BYTE, zimg::PixelType::WORD };
	zimg::PixelType pixel_out[] = { zimg::PixelType::HALF, zimg::PixelType::FLOAT };
	unsigned sha1_idx = 0;

	SCOPED_TRACE(fullrange);
	SCOPED_TRACE(chroma);
	SCOPED_TRACE(ycgco);

	for (zimg::PixelType pxin : pixel_in) {
		for (zimg::PixelType pxout : pixel_out) {
			SCOPED_TRACE(static_cast<int>(pxin));
			SCOPED_TRACE(static_cast<int>(pxout));

			zimg::PixelFormat fmt_in = pxin;
			fmt_in.fullrange = fullrange;
			fmt_in.chroma = chroma;
			fmt_in.ycgco = ycgco;

			zimg::PixelFormat fmt_out = pxout;
			fmt_out.chroma = chroma;
			fmt_out.ycgco = ycgco;

			zimg::depth::DepthConversion::result convert = zimg::depth::DepthConversion{ w, h }
				.set_pixel_in(fmt_in)
				.set_pixel_out(fmt_out)
				.create();
			ASSERT_TRUE(convert.filters[0]);
			ASSERT_TRUE(convert.filter_refs[0]);

			graphengine::FilterValidation(convert.filter_refs[0], { w, h, pixel_size(pxin) })
				.set_input_pixel_format({ fmt_in.depth, zimg::pixel_is_float(pxin), chroma })
				.set_output_pixel_format({ fmt_out.depth, zimg::pixel_is_float(pxout), chroma })
				.set_sha1(0, expected_sha1[sha1_idx++])
				.run();
		}
	}
}

} // namespace


TEST(DepthConvertTest, test_nop)
{
	zimg::depth::DepthConversion::result convert = zimg::depth::DepthConversion{ 640, 480 }
		.set_pixel_in(zimg::PixelType::FLOAT)
		.set_pixel_out(zimg::PixelType::FLOAT)
		.create();
	ASSERT_FALSE(convert.filters[0]);
	ASSERT_FALSE(convert.filter_refs[0]);
}

TEST(DepthConvertTest, test_half_to_float)
{
	static const char *expected_sha1[] = {
		"68442b2c5704fd2792d92b15fa2e259a51c601dc",
		"ef824bbfe3dc3f9cc4094d50d48091fa5f5fec7e",
	};
	unsigned sha1_idx = 0;

	for (bool chroma : { false, true }) {
		SCOPED_TRACE(chroma);

		zimg::depth::DepthConversion::result convert = zimg::depth::DepthConversion{ 640, 480 }
			.set_pixel_in({ zimg::PixelType::HALF, zimg::pixel_depth(zimg::PixelType::HALF), false, chroma })
			.set_pixel_out({ zimg::PixelType::FLOAT, zimg::pixel_depth(zimg::PixelType::FLOAT), false, chroma })
			.create();
		ASSERT_TRUE(convert.filters[0]);
		ASSERT_TRUE(convert.filter_refs[0]);

		graphengine::FilterValidation(convert.filter_refs[0], { 640, 480, zimg::pixel_size(zimg::PixelType::HALF) })
			.set_input_pixel_format({ zimg::pixel_depth(zimg::PixelType::HALF), zimg::pixel_is_float(zimg::PixelType::HALF), chroma })
			.set_output_pixel_format({ zimg::pixel_depth(zimg::PixelType::FLOAT), zimg::pixel_is_float(zimg::PixelType::FLOAT), chroma })
			.set_sha1(0, expected_sha1[sha1_idx++])
			.run();
	}
}

TEST(DepthConvertTest, test_float_to_half)
{
	static const char *expected_sha1[] = {
		"8907defd10af0b7c71abfb9c20147adc1b0a1f70",
		"4da423338093bef435e64b494bf13f40ec6c0ae6",
	};
	unsigned sha1_idx = 0;

	for (bool chroma : { false, true }) {
		SCOPED_TRACE(chroma);

		zimg::depth::DepthConversion::result convert = zimg::depth::DepthConversion{ 640, 480 }
			.set_pixel_in({ zimg::PixelType::FLOAT, zimg::pixel_depth(zimg::PixelType::FLOAT), false, chroma })
			.set_pixel_out({ zimg::PixelType::HALF, zimg::pixel_depth(zimg::PixelType::HALF), false, chroma })
			.create();
		ASSERT_TRUE(convert.filters[0]);
		ASSERT_TRUE(convert.filter_refs[0]);

		graphengine::FilterValidation(convert.filter_refs[0], { 640, 480, zimg::pixel_size(zimg::PixelType::FLOAT) })
			.set_input_pixel_format({ zimg::pixel_depth(zimg::PixelType::FLOAT), zimg::pixel_is_float(zimg::PixelType::FLOAT), chroma })
			.set_output_pixel_format({ zimg::pixel_depth(zimg::PixelType::HALF), zimg::pixel_is_float(zimg::PixelType::HALF), chroma })
			.set_sha1(0, expected_sha1[sha1_idx++])
			.run();
	}
}

TEST(DepthConvertTest, test_limited_luma)
{
	static const char *expected_sha1[] = {
		{ "a7096d8251091eb2188bb2bec9fee9d0495faf2c" },
		{ "705050fb0e56681004ede72126bd264d6c4268d9" },

		{ "f8c1a8d19a442a5fb480d8b77a347a6326f3e640" },
		{ "5c813b8fda21c1dd3505f165ea0e718eb9e8a427" },
	};

	test_case(false, false, false, expected_sha1);
	test_case(false, false, true, expected_sha1);
}

TEST(DepthConvertTest, test_limited_chroma)
{
	static const char *expected_sha1[] = {
		{ "7c84d9bc8a271e543d8e6a503ffdeb651d9b60d9" },
		{ "f0ab1d4b4fa8a87006137918c5bfb423a1020079" },

		{ "d87519b6603f966aef1b01e014c18909e3814918" },
		{ "19c80a38e459f74fbfa53055aff6aeddbddd0f79" },
	};

	test_case(false, true, false, expected_sha1);
}

TEST(DepthConvertTest, test_limited_chroma_ycgco)
{
	static const char *expected_sha1[] = {
		{ "3eafedbfa0fcd9d99ad374db94d38f632015123e" },
		{ "7f7f511df52314e078bc9b059fb7c6ae83926b7f" },

		{ "76b68d711e84b6c9a207c9534b8dd8fb93dbbe52" },
		{ "2d517680820b26ac60d0bb4d2021cd8999dfbce5" },
	};

	test_case(false, true, true, expected_sha1);
}

TEST(DepthConvertTest, test_full_luma)
{
	static const char *expected_sha1[] = {
		{ "f0e4a68158eab0ab350c7161498a8eed3196c233" },
		{ "20c77820ff7d4443a0de7991218e2f8eee551e8d" },

		{ "07b6aebbfe48004c8acb12a3c76137db57ba9a0b" },
		{ "7ad2bc4ba1be92699ec22f489ae93a8b0dc89821" },
	};

	test_case(true, false, false, expected_sha1);
	test_case(true, false, true, expected_sha1);
}

TEST(DepthConvertTest, test_full_chroma)
{
	static const char *expected_sha1[] = {
		{ "333be81b7364a126a2a6167522b539cfad599814" },
		{ "48a95801578c440f7180c799bdc344a873c6d8d6" },

		{ "a4a5448b98ab83e68afde9582dec61d37eb610bb" },
		{ "ad93453f70f4d010049bb6c9e29307ddeee4fa5b" },
	};

	test_case(true, true, false, expected_sha1);
	test_case(true, true, true, expected_sha1);
}

TEST(DepthConvertTest, test_padding_bits)
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
	static const char *expected_sha1[] = {
		{ "6a6a49a71b307303c68ec76e1e196736acc41730" },
		{ "1ebf85f96a3d3cc00cfa6da71edbd9030e0d371e" },
		{ "1451f96e1221f3194dce3b972dfc40fad74c5f80" },
		{ "f28cfe65453b2c1bcb4988d1db9dd3c512d9bdb1" },
		{ "62ca57c53cab818046b537449ad5418e7988cc68" },
		{ "422e55781a043f20738685f22a8c8c3c116810dd" },
	};

	unsigned sha1_idx = 0;

	for (const zimg::PixelFormat &format : src_format) {
		SCOPED_TRACE(static_cast<int>(format.type));
		SCOPED_TRACE(format.depth);
		SCOPED_TRACE(format.fullrange);
		SCOPED_TRACE(format.chroma);

		zimg::PixelFormat dst_format = zimg::PixelType::FLOAT;
		dst_format.chroma = format.chroma;

		zimg::depth::DepthConversion::result convert = zimg::depth::DepthConversion{ w, h, }
			.set_pixel_in(format)
			.set_pixel_out(dst_format)
			.create();
		ASSERT_TRUE(convert.filters[0]);
		ASSERT_TRUE(convert.filter_refs[0]);

		graphengine::FilterValidation(convert.filter_refs[0], { w, h, zimg::pixel_size(format.type) })
			.set_input_pixel_format({ format.depth, zimg::pixel_is_float(format.type), format.chroma })
			.set_output_pixel_format({ dst_format.depth, zimg::pixel_is_float(dst_format.type), dst_format.chroma })
			.set_sha1(0, expected_sha1[sha1_idx++])
			.run();
	}
}
