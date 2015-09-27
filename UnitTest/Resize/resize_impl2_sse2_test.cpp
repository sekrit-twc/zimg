#ifdef ZIMG_X86

#include <memory>
#include "Common/cpuinfo.h"
#include "Common/pixel.h"
#include "Resize/resize_impl2.h"

#include "gtest/gtest.h"
#include "Common/filter_validator.h"

namespace {;

void test_case(const zimg::resize::Filter &filter, bool horizontal, unsigned src_w, unsigned src_h, unsigned dst_w, unsigned dst_h,
               const zimg::PixelFormat &format, const char * const expected_sha1[3], double expected_snr)
{
	if (!zimg::query_x86_capabilities().sse2) {
		SUCCEED() << "sse not available, skipping";
		return;
	}

	SCOPED_TRACE(filter.support());
	SCOPED_TRACE(horizontal ? (double)dst_w / src_w : (double)dst_h / src_h);

	std::unique_ptr<zimg::IZimgFilter> filter_c{
		zimg::resize::create_resize_impl2(filter, format.type, horizontal, format.depth, src_w, src_h, dst_w, dst_h,
	                                      0.0, horizontal ? src_w : src_h, zimg::CPUClass::CPU_NONE)
	};
	std::unique_ptr<zimg::IZimgFilter> filter_sse2{
		zimg::resize::create_resize_impl2(filter, format.type, horizontal, format.depth, src_w, src_h, dst_w, dst_h,
		                                  0.0, horizontal ? src_w : src_h, zimg::CPUClass::CPU_X86_SSE2)
	};

	validate_filter(filter_sse2.get(), src_w, src_h, format, expected_sha1);
	validate_filter_reference(filter_c.get(), filter_sse2.get(), src_w, src_h, format, expected_snr);
}

} // namespace


TEST(ResizeImplSSE2Test, test_resize_v_u10)
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

TEST(ResizeImplSSE2Test, test_resize_v_u16)
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

#endif // ZIMG_X86
