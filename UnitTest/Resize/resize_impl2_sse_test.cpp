#ifdef ZIMG_X86

#include <cmath>
#include <memory>
#include "Common/cpuinfo.h"
#include "Common/pixel.h"
#include "Resize/filter.h"
#include "Resize/resize_impl2.h"

#include "gtest/gtest.h"
#include "Common/filter_validator.h"

namespace {;

void test_case(const zimg::resize::Filter &filter, bool horizontal, unsigned src_w, unsigned src_h, unsigned dst_w, unsigned dst_h,
               zimg::PixelType type, const char * const expected_sha1[3], double expected_snr)
{
	if (!zimg::query_x86_capabilities().sse) {
		SUCCEED() << "sse not available, skipping";
		return;
	}

	SCOPED_TRACE(filter.support());
	SCOPED_TRACE(horizontal ? (double)dst_w / src_w : (double)dst_h / src_h);

	std::unique_ptr<zimg::IZimgFilter> filter_c{
		zimg::resize::create_resize_impl2(filter, type, horizontal, 0, src_w, src_h, dst_w, dst_h, 0.0, horizontal ? src_w : src_h, zimg::CPUClass::CPU_NONE)
	};
	std::unique_ptr<zimg::IZimgFilter> filter_sse{
		zimg::resize::create_resize_impl2(filter, type, horizontal, 0, src_w, src_h, dst_w, dst_h, 0.0, horizontal ? src_w : src_h, zimg::CPUClass::CPU_X86_SSE)
	};

	validate_filter(filter_sse.get(), src_w, src_h, type, expected_sha1);
	validate_filter_reference(filter_c.get(), filter_sse.get(), src_w, src_h, type, expected_snr);
}

} // namespace


TEST(ResizeImplSSETest, test_resize_v_f32)
{
	const unsigned w = 640;
	const unsigned src_h = 480;
	const unsigned dst_h = 720;
	const zimg::PixelType type = zimg::PixelType::FLOAT;

	const char *expected_sha1[][3] = {
		{ "6b7507617dc89d5d3077f9cc4c832b261dea2be0" },
		{ "d07a8c6f3452ada7bd865a3283dc308176541db3" },
		{ "bda98bc253213d2e28a54c6ccb7496f0ca5a3b7d" },
		{ "6ba3876cd08a5b11ee646954b52b379a3d8b1228" }
	};
	const double expected_snr = 120.0;

	test_case(zimg::resize::BilinearFilter{}, false, w, src_h, w, dst_h, type, expected_sha1[0], expected_snr);
	test_case(zimg::resize::Spline16Filter{}, false, w, src_h, w, dst_h, type, expected_sha1[1], expected_snr);
	test_case(zimg::resize::LanczosFilter{ 4 }, false, w, src_h, w, dst_h, type, expected_sha1[2], expected_snr);
	test_case(zimg::resize::LanczosFilter{ 4 }, false, w, dst_h, w, src_h, type, expected_sha1[3], expected_snr);
}

#endif
