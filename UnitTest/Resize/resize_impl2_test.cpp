#include <cmath>
#include <memory>
#include "Common/cpuinfo.h"
#include "Common/pixel.h"
#include "Common/zfilter.h"
#include "Resize/filter.h"
#include "Resize/resize_impl2.h"

#include "gtest/gtest.h"
#include "Common/filter_validator.h"

namespace {;

void test_case(const zimg::PixelFormat &format, bool horizontal, double scale_factor, double shift, double subwidth_factor, const char *(*expected_sha1)[3])
{
	const unsigned src_w = 640;
	const unsigned src_h = 480;

	const unsigned dst_w = horizontal ? (unsigned)std::round(scale_factor * src_w) : src_w ;
	const unsigned dst_h = horizontal ? src_h : (unsigned)std::round(scale_factor * src_h);

	double subwidth = (horizontal ? src_w : src_h) * subwidth_factor;

	const zimg::resize::PointFilter point{};
	const zimg::resize::BilinearFilter bilinear{};
	const zimg::resize::Spline36Filter spline36{};
	const zimg::resize::LanczosFilter lanczos4{ 4 };

	const zimg::resize::Filter *resample_filters[] = { &point, &bilinear, &spline36, &lanczos4 };
	unsigned sha1_idx = 0;

	for (const zimg::resize::Filter *resample_filter : resample_filters) {
		SCOPED_TRACE(resample_filter->support());

		std::unique_ptr<zimg::IZimgFilter> filter;
		filter.reset(zimg::resize::create_resize_impl2(*resample_filter, format.type, horizontal, format.depth,
		                                               src_w, src_h, dst_w, dst_h, shift, subwidth, zimg::CPUClass::CPU_NONE));

		ASSERT_TRUE(filter);
		validate_filter(filter.get(), src_w, src_h, format, expected_sha1[sha1_idx++]);
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
	test_case(zimg::default_pixel_format(zimg::PixelType::WORD), true, 1.0, 0.0, 1.0, expected_sha1_u16);
	SCOPED_TRACE("word-v");
	test_case(zimg::default_pixel_format(zimg::PixelType::WORD), false, 1.0, 0.0, 1.0, expected_sha1_u16);
	SCOPED_TRACE("float-h");
	test_case(zimg::default_pixel_format(zimg::PixelType::FLOAT), true, 1.0, 0.0, 1.0, expected_sha1_f32);
	SCOPED_TRACE("float-v");
	test_case(zimg::default_pixel_format(zimg::PixelType::FLOAT), false, 1.0, 0.0, 1.0, expected_sha1_f32);
}

TEST(ResizeImplTest, test_horizontal_up)
{
	const char *expected_sha1_u16[][3] = {
		{ "9f37efd7adc0570ad9bab87abedea0e83601a207" },
		{ "c9f3368bc3a15079abd56df2dd6f0be7f8d92fba" },
		{ "9b7e9585d298b165e1fd78a5762c03f2be9750be" },
		{ "19b0ae79289a47d2795e937519cda28f31352d6f" }
	};
	const char *expected_sha1_f32[][3] = {
		{ "982dbdbb4c8b4d35f4f77fd9107b6a7f306b0a0a" },
		{ "558b5ad491be4e009064cb3a458f8d9da0fbe515" },
		{ "b10e63586f6b155ae3aa7935af0d2bb92fbd5b1b" },
		{ "f8a06d162e5a00b2b47dbaa76a24f1f4c077a7bf" }
	};

	SCOPED_TRACE("word");
	test_case(zimg::default_pixel_format(zimg::PixelType::WORD), true, 2.1, 0.0, 1.0, expected_sha1_u16);
	SCOPED_TRACE("float");
	test_case(zimg::default_pixel_format(zimg::PixelType::FLOAT), true, 2.1, 0.0, 1.0, expected_sha1_f32);
}

TEST(ResizeImplTest, test_horizontal_down)
{
	const char *expected_sha1_u16[][3] = {
		{ "35f664a086caaa5823a8dd031e06f91cdffa47d0" },
		{ "7ddad53e36e73b724bf28db0ae09a5f4e515c146" },
		{ "4fe960333b212bfdf226224e5b399e421ec10d5c" },
		{ "3d6023567154014b605b826fda17346561e15970" }
	};
	const char *expected_sha1_f32[][3] = {
		{ "e381b28a3ad48f385a596d37ee7dd68add0ddafc" },
		{ "9b501da8d9d279d26ac1293fa120c4578fe36708" },
		{ "dd37c2f6ad6fe9b2afafaf4d9598d822ece22426" },
		{ "b8c8d10c0e9e5f4df0eead282a4279866492d6ee" }
	};

	SCOPED_TRACE("word");
	test_case(zimg::default_pixel_format(zimg::PixelType::WORD), true, 1.0 / 2.1, 0.0, 1.0, expected_sha1_u16);
	SCOPED_TRACE("float");
	test_case(zimg::default_pixel_format(zimg::PixelType::FLOAT), true, 1.0 / 2.1, 0.0, 1.0, expected_sha1_f32);
}

TEST(ResizeImplTest, test_vertical_up)
{
	const char *expected_sha1_u16[][3] = {
		{ "0ceeec49fef9ff273d1159701b9e2496b0fbb6de" },
		{ "dea6c6833de29cd297e9d8dfddcfb7602deb3e2e" },
		{ "e23d8162a237adf1d0a5a6745a9e1209847f3cbf" },
		{ "7f988ff9f373ecb1f9bd8d195366c1b5d34180a5" }
	};
	const char *expected_sha1_f32[][3] = {
		{ "c080343cb22d40e31be08cdada3985099a3bff5c" },
		{ "110ed0b2979b36c6f590c6315558dbd5a9a027bd" },
		{ "a371d641dadccb5098cfc6e1763781e728615064" },
		{ "69cf86bd0cc6a04ac14025721849f31714a28b58" }
	};

	SCOPED_TRACE("word");
	test_case(zimg::default_pixel_format(zimg::PixelType::WORD), false, 2.1, 0.0, 1.0, expected_sha1_u16);
	SCOPED_TRACE("float");
	test_case(zimg::default_pixel_format(zimg::PixelType::FLOAT), false, 2.1, 0.0, 1.0, expected_sha1_f32);
}

TEST(ResizeImplTest, test_vertical_down)
{
	const char *expected_sha1_u16[][3] = {
		{ "aaef348f13b54c47b75c364a4c5db9348387753d" },
		{ "8e8da56422e90bf16e1b3c335db4daabef8983f3" },
		{ "637ad69f36083bff0abf2923a96e3ffe09c2bbdf" },
		{ "e1a9d1422bf5f2f62fcae435eb44358640c54831" }
	};
	const char *expected_sha1_f32[][3] = {
		{ "efa5092501e0069902209849b0b38538b82ced88" },
		{ "ba6e942326feed52c173443fdb5b594fadcfbeba" },
		{ "a624753a6386e1c296408e617afabd85424f1cc0" },
		{ "b6e0cf62c008a995c314dd8aef3c36317b75aebf" }
	};

	SCOPED_TRACE("word");
	test_case(zimg::default_pixel_format(zimg::PixelType::WORD), false, 1.0 / 2.1, 0.0, 1.0, expected_sha1_u16);
	SCOPED_TRACE("float");
	test_case(zimg::default_pixel_format(zimg::PixelType::FLOAT), false, 1.0 / 2.1, 0.0, 1.0, expected_sha1_f32);
}

TEST(ResizeImplTest, test_horizontal_nonfull)
{
	zimg::PixelFormat format = zimg::default_pixel_format(zimg::PixelType::WORD);
	format.depth = 7;

	const char *expected_sha1_up[][3] = {
		{ "347ce215408bd64c243e869022a416731ee18800" },
		{ "09714bcb863ab71b0820ef50dc2c1590c4c19e85" },
		{ "439d9a91d11fb94c28a67c79942f9276d10f7b30" },
		{ "699d7d740af90cf2edbb74e4f60bc898961d0906" }
	};
	const char *expected_sha1_down[][3] = {
		{ "dec8c699dbd69e5be6e37b348565dcb3e019c638" },
		{ "7eb060e033795a8a566993629bbeb82e92838d5a" },
		{ "c96e9563d72a0b4b4aa2b5e2d05273afd2517db3" },
		{ "055c27a8eb6360bfdcd6358e13283ea3ed93a236" }
	};

	SCOPED_TRACE("up");
	test_case(format, true, 2.1, 0.0, 1.0, expected_sha1_up);
	SCOPED_TRACE("down");
	test_case(format, true, 1.0 / 2.1, 0.0, 1.0, expected_sha1_down);
}

TEST(ResizeImplTest, test_vertical_nonfull)
{
	zimg::PixelFormat format = zimg::default_pixel_format(zimg::PixelType::WORD);
	format.depth = 7;

	const char *expected_sha1_up[][3] = {
		{ "0824364bbd7c22b89b309827709fd2b08a73af7e" },
		{ "0a47cfac7a03c958449f27cd1bc8432eeb281b68" },
		{ "cc847cd8cbf1b4f1163847a9bbb9e1496088236b" },
		{ "bbbafbe8b4d378ecdc58fe677f7ef7075ab5073b" }
	};
	const char *expected_sha1_down[][3] = {
		{ "49bd4c39366f7d0fe28d8c3b47ef6fc1ce27a7fb" },
		{ "69b037df9adcb0cbe8520839ce2655ee5d80ca77" },
		{ "4e7d4ee01e49e303cde28d481efdea7ff420affc" },
		{ "e91ab472ba88556bc4ed9fb190bfb3beef58e779" }
	};

	SCOPED_TRACE("up");
	test_case(format, false, 2.1, 0.0, 1.0, expected_sha1_up);
	SCOPED_TRACE("down");
	test_case(format, false, 1.0 / 2.1, 0.0, 1.0, expected_sha1_down);
}

TEST(ResizeImplTest, test_horizontal_shift)
{
	const double shift = 3.3;
	const double subwidth_factor = 0.75;

	const char *expected_sha1_up[][3] = {
		{ "2bb7dd3279f287f104e095f76153ea02e585d761" },
		{ "f1efc2eceaf9960509f7a3041f20a868599d6e29" },
		{ "9df872da217fe455fef2ce395a940ddd7c9e43b3" },
		{ "e3210f005fb478b27596cc4286ddfb5ee43a8f29" }
	};
	const char *expected_sha1_down[][3] = {
		{ "6d4a88b20d0a64b579ba02aa752d7c9069f3a06d" },
		{ "fa9f9935c3444281f9c6c70db0636c37cdbe6fd3" },
		{ "a81b85cd94a38ef48f2426e8a3827e28b06a25d4" },
		{ "4e93fb2a9b11308426692041d8977d9c03b84f06" }
	};

	SCOPED_TRACE("up");
	test_case(zimg::default_pixel_format(zimg::PixelType::FLOAT), false, 2.1, shift, subwidth_factor, expected_sha1_up);
	SCOPED_TRACE("down");
	test_case(zimg::default_pixel_format(zimg::PixelType::FLOAT), false, 1.0 / 2.1, shift, subwidth_factor, expected_sha1_down);
}

TEST(ResizeImplTest, test_vertical_shift)
{
	const double shift = -3.3;
	const double subwidth_factor = 1.05;

	const char *expected_sha1_up[][3] = {
		{ "3946bc1fb25485db67b58bf44cfba6725785f1c3" },
		{ "5a278cf5d1a25d6cb20edb89c7e4ef71524bb2d8" },
		{ "86be80e039385e89503d8209be97f3e00817f69b" },
		{ "b87a33751c801863f98a09203175622eab4f22e7" }
	};
	const char *expected_sha1_down[][3] = {
		{ "4b040ed496db4a607db8f52dad2f9b7620af5e68" },
		{ "602ed07944861e627f4bfec9c43b4eb963b986d6" },
		{ "451f8c7bd5e7ee616f53abe91c5a30d5c2abb052" },
		{ "b2b5504733adeea43f7a4569e1564abb61b37f79" }
	};

	SCOPED_TRACE("up");
	test_case(zimg::default_pixel_format(zimg::PixelType::FLOAT), false, 2.1, shift, subwidth_factor, expected_sha1_up);
	SCOPED_TRACE("down");
	test_case(zimg::default_pixel_format(zimg::PixelType::FLOAT), false, 1.0 / 2.1, shift, subwidth_factor, expected_sha1_down);
}
