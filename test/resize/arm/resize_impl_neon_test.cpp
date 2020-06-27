#ifdef ZIMG_ARM

#include <cmath>
#include "common/cpuinfo.h"
#include "common/pixel.h"
#include "common/arm/cpuinfo_arm.h"
#include "resize/filter.h"
#include "resize/resize_impl.h"

#include "gtest/gtest.h"
#include "graph/filter_validator.h"

namespace {

void test_case(const zimg::resize::Filter &filter, bool horizontal, unsigned src_w, unsigned src_h, unsigned dst_w, unsigned dst_h,
               const zimg::PixelFormat &format, const char * const expected_sha1[3], double expected_snr)
{
	if (!zimg::query_arm_capabilities().neon) {
		SUCCEED() << "neon not available, skipping";
		return;
	}

	SCOPED_TRACE(filter.support());
	SCOPED_TRACE(horizontal ? static_cast<double>(dst_w) / src_w : static_cast<double>(dst_h) / src_h);

	auto builder = zimg::resize::ResizeImplBuilder{ src_w, src_h, format.type }
		.set_horizontal(horizontal)
		.set_dst_dim(horizontal ? dst_w : dst_h)
		.set_depth(format.depth)
		.set_filter(&filter)
		.set_shift(0.0)
		.set_subwidth(horizontal ? src_w : src_h);

	std::unique_ptr<zimg::graph::ImageFilter> filter_neon = builder.set_cpu(zimg::CPUClass::ARM_NEON).create();
	std::unique_ptr<zimg::graph::ImageFilter> filter_c = builder.set_cpu(zimg::CPUClass::NONE).create();

	ASSERT_FALSE(assert_different_dynamic_type(filter_c.get(), filter_neon.get()));

	FilterValidator validator{ filter_neon.get(), src_w, src_h, format };
	validator.set_sha1(expected_sha1);
	validator.set_ref_filter(filter_c.get(), expected_snr);
	validator.validate();
}

} // namespace


TEST(ResizeImplNeonTest, test_resize_h_u10)
{
	const unsigned src_w = 640;
	const unsigned dst_w = 960;
	const unsigned h = 480;
	const zimg::PixelFormat format{ zimg::PixelType::WORD, 10 };

	const char *expected_sha1[][3] = {
		{ "8d7d269168aed9b332ccd79e2b46d661fe391642" },
		{ "842da71bbfe74cabcff24ff269e7dfd1584f544f" },
		{ "4daefef8cf500bf8a907a6f715f5c619fc8562b2" },
		{ "3ab59686bc6c5a7c748ddff214d25333e2f80011" }
	};
	const double expected_snr = INFINITY;

	test_case(zimg::resize::BilinearFilter{}, true, src_w, h, dst_w, h, format, expected_sha1[0], expected_snr);
	test_case(zimg::resize::Spline16Filter{}, true, src_w, h, dst_w, h, format, expected_sha1[1], expected_snr);
	test_case(zimg::resize::LanczosFilter{ 4 }, true, src_w, h, dst_w, h, format, expected_sha1[2], expected_snr);
	test_case(zimg::resize::LanczosFilter{ 4 }, true, dst_w, h, src_w, h, format, expected_sha1[3], expected_snr);
}

TEST(ResizeImplNeonTest, test_resize_h_u16)
{
	const unsigned src_w = 640;
	const unsigned dst_w = 960;
	const unsigned h = 480;
	const zimg::PixelFormat format{ zimg::PixelType::WORD, 16 };

	const char *expected_sha1[][3] = {
		{ "a6b7fea8f8de785248f520f605bd7c8da66f59d5" },
		{ "810c906d2b2b5e17703b220d64f9d3c10690cc16" },
		{ "b74758c6d844da2d1acf48bbc75459533f47eb9f" },
		{ "779236bf9e1d646caa8b384b283c6dfea1e12dff" }
	};
	const double expected_snr = INFINITY;

	test_case(zimg::resize::BilinearFilter{}, true, src_w, h, dst_w, h, format, expected_sha1[0], expected_snr);
	test_case(zimg::resize::Spline16Filter{}, true, src_w, h, dst_w, h, format, expected_sha1[1], expected_snr);
	test_case(zimg::resize::LanczosFilter{ 4 }, true, src_w, h, dst_w, h, format, expected_sha1[2], expected_snr);
	test_case(zimg::resize::LanczosFilter{ 4 }, true, dst_w, h, src_w, h, format, expected_sha1[3], expected_snr);
}

TEST(ResizeImplNeonTest, test_resize_v_u10)
{
	const unsigned w = 640;
	const unsigned src_h = 480;
	const unsigned dst_h = 720;
	const zimg::PixelFormat format{ zimg::PixelType::WORD, 10 };

	const char *expected_sha1[][3] = {
		{ "41ac207d1e61c7222a77532134d39dc182e78222" },
		{ "7d75acf35753b20cc48a04fad8966ecc82105a0c" },
		{ "450d1cf4ee91656026b00da583181224475c1b70" },
		{ "8231b3b149106a06acd1bbcfa56398423d27a579" }
	};
	const double expected_snr = INFINITY;

	test_case(zimg::resize::BilinearFilter{}, false, w, src_h, w, dst_h, format, expected_sha1[0], expected_snr);
	test_case(zimg::resize::Spline16Filter{}, false, w, src_h, w, dst_h, format, expected_sha1[1], expected_snr);
	test_case(zimg::resize::LanczosFilter{ 4 }, false, w, src_h, w, dst_h, format, expected_sha1[2], expected_snr);
	test_case(zimg::resize::LanczosFilter{ 4 }, false, w, dst_h, w, src_h, format, expected_sha1[3], expected_snr);
}

TEST(ResizeImplNeonTest, test_resize_v_u16)
{
	const unsigned w = 640;
	const unsigned src_h = 480;
	const unsigned dst_h = 720;
	const zimg::PixelFormat format{ zimg::PixelType::WORD, 16 };

	const char *expected_sha1[][3] = {
		{ "fbde3fbb93720f073dcc8579bc17edf6c2cab982" },
		{ "2e0b375e7014b842016e7db4fb62ecf96bb230d7" },
		{ "5f9d6c73f468d1cbfb2bc850828dd0ac9f05193d" },
		{ "9747a61169a63015fd8491b566c5f3e577f7e93e" }
	};
	const double expected_snr = INFINITY;

	test_case(zimg::resize::BilinearFilter{}, false, w, src_h, w, dst_h, format, expected_sha1[0], expected_snr);
	test_case(zimg::resize::Spline16Filter{}, false, w, src_h, w, dst_h, format, expected_sha1[1], expected_snr);
	test_case(zimg::resize::LanczosFilter{ 4 }, false, w, src_h, w, dst_h, format, expected_sha1[2], expected_snr);
	test_case(zimg::resize::LanczosFilter{ 4 }, false, w, dst_h, w, src_h, format, expected_sha1[3], expected_snr);
}

TEST(ResizeImplNeonTest, test_resize_h_f32)
{
	const unsigned src_w = 640;
	const unsigned dst_w = 960;
	const unsigned h = 480;
	const zimg::PixelType format = zimg::PixelType::FLOAT;

	const char *expected_sha1[][3] = {
#if defined(_M_ARM64) || defined(__aarch64__)
		{ "1b2e37a345d315b0fa4d11e3532c70cb57b1e569" },
		{ "2d0582a2f6af8a480e8f053fbd89eac0668b33f3" },
		{ "967f921dc3fd2b3d166a276fe671105c3fac0756" },
		{ "166dfd1881724fe546571c2d7ac959e6433623be" }
#else
		{ "1b2e37a345d315b0fa4d11e3532c70cb57b1e569" },
		{ "df391f7157d8c283abd408b35894139ca1903872" },
		{ "81fcfbdb9a3b31c625a3cdff1cf46da06f8af735" },
		{ "389b609ac62a8b9276e00fdcd39b921535196a07" }
#endif
	};
	const double expected_snr = 120.0;

	test_case(zimg::resize::BilinearFilter{}, true, src_w, h, dst_w, h, format, expected_sha1[0], expected_snr);
	test_case(zimg::resize::Spline16Filter{}, true, src_w, h, dst_w, h, format, expected_sha1[1], expected_snr);
	test_case(zimg::resize::LanczosFilter{ 4 }, true, src_w, h, dst_w, h, format, expected_sha1[2], expected_snr);
	test_case(zimg::resize::LanczosFilter{ 4 }, true, dst_w, h, src_w, h, format, expected_sha1[3], expected_snr);
}


TEST(ResizeImplNeonTest, test_resize_v_f32)
{
	const unsigned w = 640;
	const unsigned src_h = 480;
	const unsigned dst_h = 720;
	const zimg::PixelType type = zimg::PixelType::FLOAT;

	const char *expected_sha1[][3] = {
#if defined(_M_ARM64) || defined(__aarch64__)
		{ "6b7507617dc89d5d3077f9cc4c832b261dea2be0" },
		{ "46283014e580fa47deacae5e0cec1ce952973f51" },
		{ "47946b5a3aba5e9ee6967659e8aeb26070ae80d6" },
		{ "bcedc16781dc7781557d744b75ccac510a98a3ac" }
#else
		{ "6b7507617dc89d5d3077f9cc4c832b261dea2be0" },
		{ "d07a8c6f3452ada7bd865a3283dc308176541db3" },
		{ "6a15c26ad08e7576b415b70a3681f2a38667b301" },
		{ "f1cc0dea71ca9fa3d9090ecdf21369aa5e0fb0be" }
#endif
	};
	const double expected_snr = 120.0;

	test_case(zimg::resize::BilinearFilter{}, false, w, src_h, w, dst_h, type, expected_sha1[0], expected_snr);
	test_case(zimg::resize::Spline16Filter{}, false, w, src_h, w, dst_h, type, expected_sha1[1], expected_snr);
	test_case(zimg::resize::LanczosFilter{ 4 }, false, w, src_h, w, dst_h, type, expected_sha1[2], expected_snr);
	test_case(zimg::resize::LanczosFilter{ 4 }, false, w, dst_h, w, src_h, type, expected_sha1[3], expected_snr);
}

#endif // ZIMG_ARM
