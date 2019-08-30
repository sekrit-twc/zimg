#ifdef ZIMG_X86_AVX512

#include <cmath>
#include "common/cpuinfo.h"
#include "common/pixel.h"
#include "common/x86/cpuinfo_x86.h"
#include "resize/filter.h"
#include "resize/resize_impl.h"

#include "gtest/gtest.h"
#include "graph/filter_validator.h"

namespace {

void test_case(const zimg::resize::Filter &filter, bool horizontal, unsigned src_w, unsigned src_h, unsigned dst_w, unsigned dst_h,
               const zimg::PixelFormat &format, const char * const expected_sha1[3], double expected_snr)
{
	if (!zimg::query_x86_capabilities().avx512vnni) {
		SUCCEED() << "avx512 not available, skipping";
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

	std::unique_ptr<zimg::graph::ImageFilter> filter_avx512 = builder.set_cpu(zimg::CPUClass::X86_AVX512_CLX).create();
	std::unique_ptr<zimg::graph::ImageFilter> filter_c;

	FilterValidator validator{ filter_avx512.get(), src_w, src_h, format };
	validator.set_sha1(expected_sha1);

	validator.validate();
}

} // namespace


TEST(ResizeImplAVX512VNNITest, test_resize_h_u10)
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

TEST(ResizeImplAVX512VNNITest, test_resize_h_u16)
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

TEST(ResizeImplAVX512VNNITest, test_resize_v_u10)
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

TEST(ResizeImplAVX512VNNITest, test_resize_v_u16)
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

#endif // ZIMG_X86_AVX512
