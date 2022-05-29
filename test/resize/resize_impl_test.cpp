#include <cmath>
#include "common/cpuinfo.h"
#include "common/pixel.h"
#include "graph/image_filter.h"
#include "graphengine/filter.h"
#include "resize/filter.h"
#include "resize/resize_impl.h"

#include "gtest/gtest.h"
#include "graph/filter_validator.h"
#include "graphengine/filter_validation.h"

namespace {

void test_case(const zimg::PixelFormat &format, bool horizontal, double scale_factor, double shift, double subwidth_factor, const char *(*expected_sha1)[3])
{
	const unsigned src_w = 640;
	const unsigned src_h = 480;

	const zimg::resize::PointFilter point{};
	const zimg::resize::BilinearFilter bilinear{};
	const zimg::resize::Spline36Filter spline36{};
	const zimg::resize::LanczosFilter lanczos4{ 4 };

	const zimg::resize::Filter *resample_filters[] = { &point, &bilinear, &spline36, &lanczos4 };
	unsigned sha1_idx = 0;

	for (const zimg::resize::Filter *resample_filter : resample_filters) {
		SCOPED_TRACE(resample_filter->support());

		auto filter = zimg::resize::ResizeImplBuilder{ src_w, src_h, format.type }
			.set_horizontal(horizontal)
			.set_dst_dim(static_cast<unsigned>(std::lrint(scale_factor * (horizontal ? src_w : src_h))))
			.set_depth(format.depth)
			.set_filter(resample_filter)
			.set_shift(shift)
			.set_subwidth(subwidth_factor * (horizontal ? src_w : src_h))
			.create();

		ASSERT_TRUE(filter);

		FilterValidator validator{ filter.get(), src_w, src_h, format };
		validator.set_sha1(expected_sha1[sha1_idx++]);
		validator.validate();
	}
}

void test_case_ge(const zimg::PixelFormat &format, bool horizontal, double scale_factor, double shift, double subwidth_factor, const char * const *expected_sha1)
{
	const unsigned src_w = 640;
	const unsigned src_h = 480;

	const zimg::resize::PointFilter point{};
	const zimg::resize::BilinearFilter bilinear{};
	const zimg::resize::Spline36Filter spline36{};
	const zimg::resize::LanczosFilter lanczos4{ 4 };

	const zimg::resize::Filter *resample_filters[] = { &point, &bilinear, &spline36, &lanczos4 };
	unsigned sha1_idx = 0;

	for (const zimg::resize::Filter *resample_filter : resample_filters) {
		SCOPED_TRACE(resample_filter->support());

		auto filter = zimg::resize::ResizeImplBuilder{ src_w, src_h, format.type }
			.set_horizontal(horizontal)
			.set_dst_dim(static_cast<unsigned>(std::lrint(scale_factor * (horizontal ? src_w : src_h))))
			.set_depth(format.depth)
			.set_filter(resample_filter)
			.set_shift(shift)
			.set_subwidth(subwidth_factor * (horizontal ? src_w : src_h))
			.create_ge();

		ASSERT_TRUE(filter);

		graphengine::FilterValidation(filter.get(), { src_w, src_h, zimg::pixel_size(format.type) })
			.set_input_pixel_format({ format.depth, zimg::pixel_is_float(format.type), format.chroma })
			.set_output_pixel_format({ format.depth, zimg::pixel_is_float(format.type), format.chroma })
			.set_sha1(0, expected_sha1[sha1_idx++])
			.run();
	}
}

} // namespace

TEST(ResizeImplTest, test_nop)
{
	const char *expected_sha1_u16[][3] = {
		{ "8ba35ed1784cb6d7903a9092abefb3d9afd7a683" },
		{ "8ba35ed1784cb6d7903a9092abefb3d9afd7a683" },
		{ "8ba35ed1784cb6d7903a9092abefb3d9afd7a683" },
		{ "8ba35ed1784cb6d7903a9092abefb3d9afd7a683" }
	};
	const char *expected_sha1_f32[][3] = {
		{ "483b6bdf608afbf1fba6bbca9657a8ca3822eef1" },
		{ "483b6bdf608afbf1fba6bbca9657a8ca3822eef1" },
		{ "483b6bdf608afbf1fba6bbca9657a8ca3822eef1" },
		{ "483b6bdf608afbf1fba6bbca9657a8ca3822eef1" }
	};

	SCOPED_TRACE("word-h");
	test_case(zimg::PixelType::WORD, true, 1.0, 0.0, 1.0, expected_sha1_u16);
	SCOPED_TRACE("word-v");
	test_case(zimg::PixelType::WORD, false, 1.0, 0.0, 1.0, expected_sha1_u16);
	SCOPED_TRACE("float-h");
	test_case(zimg::PixelType::FLOAT, true, 1.0, 0.0, 1.0, expected_sha1_f32);
	SCOPED_TRACE("float-v");
	test_case(zimg::PixelType::FLOAT, false, 1.0, 0.0, 1.0, expected_sha1_f32);
}

TEST(ResizeImplTest, test_nop_ge)
{
	static const char *expected_sha1_u16[] = {
		"8ba35ed1784cb6d7903a9092abefb3d9afd7a683",
		"8ba35ed1784cb6d7903a9092abefb3d9afd7a683",
		"8ba35ed1784cb6d7903a9092abefb3d9afd7a683",
		"8ba35ed1784cb6d7903a9092abefb3d9afd7a683"
	};
	static const char *expected_sha1_f32[] = {
		"483b6bdf608afbf1fba6bbca9657a8ca3822eef1",
		"483b6bdf608afbf1fba6bbca9657a8ca3822eef1",
		"483b6bdf608afbf1fba6bbca9657a8ca3822eef1",
		"483b6bdf608afbf1fba6bbca9657a8ca3822eef1"
	};

	{
		SCOPED_TRACE("word-h");
		test_case_ge(zimg::PixelType::WORD, true, 1.0, 0.0, 1.0, expected_sha1_u16);
	}
	{
		SCOPED_TRACE("word-v");
		test_case_ge(zimg::PixelType::WORD, false, 1.0, 0.0, 1.0, expected_sha1_u16);
	}
	{
		SCOPED_TRACE("float-h");
		test_case_ge(zimg::PixelType::FLOAT, true, 1.0, 0.0, 1.0, expected_sha1_f32);
	}
	{
		SCOPED_TRACE("float-v");
		test_case_ge(zimg::PixelType::FLOAT, false, 1.0, 0.0, 1.0, expected_sha1_f32);
	}
}

TEST(ResizeImplTest, test_horizontal_up)
{
	const char *expected_sha1_u16[][3] = {
		{ "9f37efd7adc0570ad9bab87abedea0e83601a207" },
		{ "c9f3368bc3a15079abd56df2dd6f0be7f8d92fba" },
		{ "2fc694f3e219a1089af15470b426544bbcb38833" },
		{ "5d3097e5fc7a7d59f463058f5464aab335176560" }
	};
	const char *expected_sha1_f32[][3] = {
		{ "982dbdbb4c8b4d35f4f77fd9107b6a7f306b0a0a" },
		{ "13930fc807841314134711dc02097ae51a33dd89" },
		{ "a7e24724ab081d7075b229d5aec98369fe8b21f4" },
		{ "c0c934c797bec140747421c465ec77d67e3132a6" }
	};

	SCOPED_TRACE("word");
	test_case(zimg::PixelType::WORD, true, 2.1, 0.0, 1.0, expected_sha1_u16);
	SCOPED_TRACE("float");
	test_case(zimg::PixelType::FLOAT, true, 2.1, 0.0, 1.0, expected_sha1_f32);
}

TEST(ResizeImplTest, test_horizontal_up_ge)
{
	static const char *expected_sha1_u16[] = {
		"9f37efd7adc0570ad9bab87abedea0e83601a207",
		"c9f3368bc3a15079abd56df2dd6f0be7f8d92fba",
		"2fc694f3e219a1089af15470b426544bbcb38833",
		"5d3097e5fc7a7d59f463058f5464aab335176560"
	};
	static const char *expected_sha1_f32[] = {
		"982dbdbb4c8b4d35f4f77fd9107b6a7f306b0a0a",
		"13930fc807841314134711dc02097ae51a33dd89",
		"a7e24724ab081d7075b229d5aec98369fe8b21f4",
		"c0c934c797bec140747421c465ec77d67e3132a6"
	};

	{
		SCOPED_TRACE("word");
		test_case_ge(zimg::PixelType::WORD, true, 2.1, 0.0, 1.0, expected_sha1_u16);
	}
	{
		SCOPED_TRACE("float");
		test_case_ge(zimg::PixelType::FLOAT, true, 2.1, 0.0, 1.0, expected_sha1_f32);
	}
}

TEST(ResizeImplTest, test_horizontal_down)
{
	const char *expected_sha1_u16[][3] = {
		{ "71c866436f2df395111d43ac1f10fc0dcfd4bd11" },
		{ "2ed0eda0e5fdcdb416703344ae190c82a96dfa3f" },
		{ "2df4b623345cbc1f50a0829c6697d4c1be9254c6" },
		{ "7e915933acfd1c757b5fe93bba226ff78a0c9d9c" }
	};
	const char *expected_sha1_f32[][3] = {
		{ "0f09d60179f3b4771dd800c9650dd3acee2b6360" },
		{ "0e9af71cb7067663207e6e1ab07a338ecb3db596" },
		{ "9559036ba9cb649af0f679520ac652871a5f3b94" },
		{ "7cb55ec9b5894c48aabb373ca98026202b5b7be9" }
	};

	SCOPED_TRACE("word");
	test_case(zimg::PixelType::WORD, true, 1.0 / 2.1, 0.0, 1.0, expected_sha1_u16);
	SCOPED_TRACE("float");
	test_case(zimg::PixelType::FLOAT, true, 1.0 / 2.1, 0.0, 1.0, expected_sha1_f32);
}

TEST(ResizeImplTest, test_horizontal_down_ge)
{
	static const char *expected_sha1_u16[] = {
		"71c866436f2df395111d43ac1f10fc0dcfd4bd11",
		"2ed0eda0e5fdcdb416703344ae190c82a96dfa3f",
		"2df4b623345cbc1f50a0829c6697d4c1be9254c6",
		"7e915933acfd1c757b5fe93bba226ff78a0c9d9c"
	};
	static const char *expected_sha1_f32[] = {
		"0f09d60179f3b4771dd800c9650dd3acee2b6360",
		"0e9af71cb7067663207e6e1ab07a338ecb3db596",
		"9559036ba9cb649af0f679520ac652871a5f3b94",
		"7cb55ec9b5894c48aabb373ca98026202b5b7be9"
	};

	{
		SCOPED_TRACE("word");
		test_case_ge(zimg::PixelType::WORD, true, 1.0 / 2.1, 0.0, 1.0, expected_sha1_u16);
	}
	{
		SCOPED_TRACE("float");
		test_case_ge(zimg::PixelType::FLOAT, true, 1.0 / 2.1, 0.0, 1.0, expected_sha1_f32);
	}
}

TEST(ResizeImplTest, test_vertical_up)
{
	const char *expected_sha1_u16[][3] = {
		{ "0ceeec49fef9ff273d1159701b9e2496b0fbb6de" },
		{ "dea6c6833de29cd297e9d8dfddcfb7602deb3e2e" },
		{ "05d2c85e9f83f36fa54af9e472013c2cb9eb3d7d" },
		{ "4d1436afa4de24c26fc12f07dfc1881c5ea20045" }
	};
	const char *expected_sha1_f32[][3] = {
		{ "c080343cb22d40e31be08cdada3985099a3bff5c" },
		{ "6bd2b845598bc297b7d2dc7ca9c136a40377573d" },
		{ "9ccaf807b5a6c88b484dc8187c0fe1108521438d" },
		{ "378824fb29098507c59c19c5565d983d9e96a95d" }
	};

	SCOPED_TRACE("word");
	test_case(zimg::PixelType::WORD, false, 2.1, 0.0, 1.0, expected_sha1_u16);
	SCOPED_TRACE("float");
	test_case(zimg::PixelType::FLOAT, false, 2.1, 0.0, 1.0, expected_sha1_f32);
}

TEST(ResizeImplTest, test_vertical_up_ge)
{
	static const char *expected_sha1_u16[] = {
		"0ceeec49fef9ff273d1159701b9e2496b0fbb6de",
		"dea6c6833de29cd297e9d8dfddcfb7602deb3e2e",
		"05d2c85e9f83f36fa54af9e472013c2cb9eb3d7d",
		"4d1436afa4de24c26fc12f07dfc1881c5ea20045"
	};
	static const char *expected_sha1_f32[] = {
		"c080343cb22d40e31be08cdada3985099a3bff5c",
		"6bd2b845598bc297b7d2dc7ca9c136a40377573d",
		"9ccaf807b5a6c88b484dc8187c0fe1108521438d",
		"378824fb29098507c59c19c5565d983d9e96a95d"
	};

	{
		SCOPED_TRACE("word");
		test_case_ge(zimg::PixelType::WORD, false, 2.1, 0.0, 1.0, expected_sha1_u16);
	}
	{
		SCOPED_TRACE("float");
		test_case_ge(zimg::PixelType::FLOAT, false, 2.1, 0.0, 1.0, expected_sha1_f32);
	}
}

TEST(ResizeImplTest, test_vertical_down)
{
	const char *expected_sha1_u16[][3] = {
		{ "abe8cf7a2949798936156d05153c3f736a991d72" },
		{ "c7670c929410997adc96615169141ea00829fe65" },
		{ "a787c9c11e037a637524150fbe3af5fc90b750b4)" },
		{ "c6d55c942d03c72ec6f88ff81ee6d70d20d49809" }
	};
	const char *expected_sha1_f32[][3] = {
		{ "211cf4deb08dedf90674641baea1f8338da319cd" },
		{ "b72f929d4250aa2ff071d4937d2942f90f573b58" },
		{ "94d06430d3adc570b65df0de5cf3662e60308a0d" },
		{ "7f4266dc8d82d343f24e5370e1fb8f714a0a05d1" }
	};

	SCOPED_TRACE("word");
	test_case(zimg::PixelType::WORD, false, 1.0 / 2.1, 0.0, 1.0, expected_sha1_u16);
	SCOPED_TRACE("float");
	test_case(zimg::PixelType::FLOAT, false, 1.0 / 2.1, 0.0, 1.0, expected_sha1_f32);
}

TEST(ResizeImplTest, test_vertical_down_ge)
{
	static const char *expected_sha1_u16[] = {
		"abe8cf7a2949798936156d05153c3f736a991d72",
		"c7670c929410997adc96615169141ea00829fe65",
		"a787c9c11e037a637524150fbe3af5fc90b750b4",
		"c6d55c942d03c72ec6f88ff81ee6d70d20d49809"
	};
	static const char *expected_sha1_f32[] = {
		"211cf4deb08dedf90674641baea1f8338da319cd",
		"b72f929d4250aa2ff071d4937d2942f90f573b58",
		"94d06430d3adc570b65df0de5cf3662e60308a0d",
		"7f4266dc8d82d343f24e5370e1fb8f714a0a05d1"
	};

	{
		SCOPED_TRACE("word");
		test_case_ge(zimg::PixelType::WORD, false, 1.0 / 2.1, 0.0, 1.0, expected_sha1_u16);
	}
	{
		SCOPED_TRACE("float");
		test_case_ge(zimg::PixelType::FLOAT, false, 1.0 / 2.1, 0.0, 1.0, expected_sha1_f32);
	}
}

TEST(ResizeImplTest, test_horizontal_nonfull)
{
	zimg::PixelFormat format = zimg::PixelType::WORD;
	format.depth = 7;

	const char *expected_sha1_up[][3] = {
		{ "347ce215408bd64c243e869022a416731ee18800" },
		{ "09714bcb863ab71b0820ef50dc2c1590c4c19e85" },
		{ "88f23bea4409ec09e11246b142b3ed2dab30a1f9" },
		{ "d6eb428faecfe36823e81ad2b72d0a1e69662297" }
	};
	const char *expected_sha1_down[][3] = {
		{ "d8c79763c48ba719a2e5ed3465234eebbc628efa" },
		{ "50b5ac4f91b969cab79acdf6008fa386b54d0369" },
		{ "2a836efb2e60c321d3e1c1983f6b814ea2141f27" },
		{ "b5486824a4575bec849cfffec16eceee03c337cb" }
	};

	SCOPED_TRACE("up");
	test_case(format, true, 2.1, 0.0, 1.0, expected_sha1_up);
	SCOPED_TRACE("down");
	test_case(format, true, 1.0 / 2.1, 0.0, 1.0, expected_sha1_down);
}

TEST(ResizeImplTest, test_horizontal_padding_bits_ge)
{
	zimg::PixelFormat format = zimg::PixelType::WORD;
	format.depth = 7;

	static const char *expected_sha1_up[] = {
		"347ce215408bd64c243e869022a416731ee18800",
		"09714bcb863ab71b0820ef50dc2c1590c4c19e85",
		"88f23bea4409ec09e11246b142b3ed2dab30a1f9",
		"d6eb428faecfe36823e81ad2b72d0a1e69662297"
	};
	static const char *expected_sha1_down[] = {
		"d8c79763c48ba719a2e5ed3465234eebbc628efa",
		"50b5ac4f91b969cab79acdf6008fa386b54d0369",
		"2a836efb2e60c321d3e1c1983f6b814ea2141f27",
		"b5486824a4575bec849cfffec16eceee03c337cb"
	};

	{
		SCOPED_TRACE("up");
		test_case_ge(format, true, 2.1, 0.0, 1.0, expected_sha1_up);
	}
	{
		SCOPED_TRACE("down");
		test_case_ge(format, true, 1.0 / 2.1, 0.0, 1.0, expected_sha1_down);
	}
}

TEST(ResizeImplTest, test_vertical_nonfull)
{
	zimg::PixelFormat format = zimg::PixelType::WORD;
	format.depth = 7;

	const char *expected_sha1_up[][3] = {
		{ "0824364bbd7c22b89b309827709fd2b08a73af7e" },
		{ "0a47cfac7a03c958449f27cd1bc8432eeb281b68" },
		{ "13dc90a9fad168d60f3bbdd5e3196fb8a2fc1e26" },
		{ "da37648e741402a0d9159edc598012ca0ac85d89" }
	};
	const char *expected_sha1_down[][3] = {
		{ "d3b1bbeb2c99182738e87e18ecf47bb4cbf4c80a" },
		{ "4cb76e447e5b17ac43749c3bb2a32fd486321fe1" },
		{ "665f43f90c5e253afc9e89134020274cb47829dd" },
		{ "5572cee34dd9d08e3d292c9f0e9be7f1645e9f84" }
	};

	SCOPED_TRACE("up");
	test_case(format, false, 2.1, 0.0, 1.0, expected_sha1_up);
	SCOPED_TRACE("down");
	test_case(format, false, 1.0 / 2.1, 0.0, 1.0, expected_sha1_down);
}

TEST(ResizeImplTest, test_vertical_padding_bits_ge)
{
	zimg::PixelFormat format = zimg::PixelType::WORD;
	format.depth = 7;

	static const char *expected_sha1_up[] = {
		"0824364bbd7c22b89b309827709fd2b08a73af7e",
		"0a47cfac7a03c958449f27cd1bc8432eeb281b68",
		"13dc90a9fad168d60f3bbdd5e3196fb8a2fc1e26",
		"da37648e741402a0d9159edc598012ca0ac85d89"
	};
	static const char *expected_sha1_down[] = {
		"d3b1bbeb2c99182738e87e18ecf47bb4cbf4c80a",
		"4cb76e447e5b17ac43749c3bb2a32fd486321fe1",
		"665f43f90c5e253afc9e89134020274cb47829dd",
		"5572cee34dd9d08e3d292c9f0e9be7f1645e9f84"
	};

	{
		SCOPED_TRACE("up");
		test_case_ge(format, false, 2.1, 0.0, 1.0, expected_sha1_up);
	}
	{
		SCOPED_TRACE("down");
		test_case_ge(format, false, 1.0 / 2.1, 0.0, 1.0, expected_sha1_down);
	}
}

TEST(ResizeImplTest, test_horizontal_shift)
{
	const double shift = 3.3;
	const double subwidth_factor = 0.75;

	const char *expected_sha1_up[][3] = {
		{ "2bb7dd3279f287f104e095f76153ea02e585d761" },
		{ "c7d57dd2ca4044f3371d62e3ffc4f83754e87ac9" },
		{ "af402fba042a6e9ca47337424bce547f576a99b3" },
		{ "9a764c13e5bb7d05efb6da0aceea01d9ec425fe3" }
	};
	const char *expected_sha1_down[][3] = {
		{ "7c644bacfb8559df29f7f26f22c5fbcf86e0071d" },
		{ "d27f31b72dee7166e29e645995c1993988e6a5f7" },
		{ "008ddf68e679777cdb45fc812f9084914fed28d4" },
		{ "de7941b240b2869fba9b618a4408e5e3180c7696" }
	};

	SCOPED_TRACE("up");
	test_case(zimg::PixelType::FLOAT, false, 2.1, shift, subwidth_factor, expected_sha1_up);
	SCOPED_TRACE("down");
	test_case(zimg::PixelType::FLOAT, false, 1.0 / 2.1, shift, subwidth_factor, expected_sha1_down);
}

TEST(ResizeImplTest, test_horizontal_shift_ge)
{
	const double shift = 3.3;
	const double subwidth_factor = 0.75;

	static const char *expected_sha1_up[] = {
		"2bb7dd3279f287f104e095f76153ea02e585d761",
		"c7d57dd2ca4044f3371d62e3ffc4f83754e87ac9",
		"af402fba042a6e9ca47337424bce547f576a99b3",
		"9a764c13e5bb7d05efb6da0aceea01d9ec425fe3"
	};
	static const char *expected_sha1_down[] = {
		"7c644bacfb8559df29f7f26f22c5fbcf86e0071d",
		"d27f31b72dee7166e29e645995c1993988e6a5f7",
		"008ddf68e679777cdb45fc812f9084914fed28d4",
		"de7941b240b2869fba9b618a4408e5e3180c7696"
	};

	{
		SCOPED_TRACE("up");
		test_case_ge(zimg::PixelType::FLOAT, false, 2.1, shift, subwidth_factor, expected_sha1_up);
	}
	{
		SCOPED_TRACE("down");
		test_case_ge(zimg::PixelType::FLOAT, false, 1.0 / 2.1, shift, subwidth_factor, expected_sha1_down);
	}
}

TEST(ResizeImplTest, test_vertical_shift)
{
	const double shift = -3.3;
	const double subwidth_factor = 1.05;

	const char *expected_sha1_up[][3] = {
		{ "3946bc1fb25485db67b58bf44cfba6725785f1c3" },
		{ "e240ce8b14c6ffab60cdb1fe9f923f03862dc499" },
		{ "bef00c22c2aaa7b9bd36ea26a66f5e49dbf66c45" },
		{ "74db6b4f83e1fa7ecc7557763e0fe8b3e70839a5" }
	};
	const char *expected_sha1_down[][3] = {
		{ "0a4fed4d808564de9837b7b7525974a8d1d80994" },
		{ "fdc144f85a6a2bf270f57376167160a3f320ce33" },
		{ "bdcca5438cae8a2809445ddff66016755e2dfe88" },
		{ "6237a961d5836dc6e03f67eeaa9a218203844f26" }
	};

	SCOPED_TRACE("up");
	test_case(zimg::PixelType::FLOAT, false, 2.1, shift, subwidth_factor, expected_sha1_up);
	SCOPED_TRACE("down");
	test_case(zimg::PixelType::FLOAT, false, 1.0 / 2.1, shift, subwidth_factor, expected_sha1_down);
}

TEST(ResizeImplTest, test_vertical_shift_ge)
{
	const double shift = -3.3;
	const double subwidth_factor = 1.05;

	static const char *expected_sha1_up[] = {
		"3946bc1fb25485db67b58bf44cfba6725785f1c3",
		"e240ce8b14c6ffab60cdb1fe9f923f03862dc499",
		"bef00c22c2aaa7b9bd36ea26a66f5e49dbf66c45",
		"74db6b4f83e1fa7ecc7557763e0fe8b3e70839a5"
	};
	static const char *expected_sha1_down[] = {
		"0a4fed4d808564de9837b7b7525974a8d1d80994",
		"fdc144f85a6a2bf270f57376167160a3f320ce33",
		"bdcca5438cae8a2809445ddff66016755e2dfe88",
		"6237a961d5836dc6e03f67eeaa9a218203844f26"
	};

	{
		SCOPED_TRACE("up");
		test_case_ge(zimg::PixelType::FLOAT, false, 2.1, shift, subwidth_factor, expected_sha1_up);
	}
	{
		SCOPED_TRACE("down");
		test_case_ge(zimg::PixelType::FLOAT, false, 1.0 / 2.1, shift, subwidth_factor, expected_sha1_down);
	}
}
