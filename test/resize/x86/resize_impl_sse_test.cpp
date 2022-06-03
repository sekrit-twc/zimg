#ifdef ZIMG_X86

#include "common/cpuinfo.h"
#include "common/pixel.h"
#include "common/x86/cpuinfo_x86.h"
#include "resize/filter.h"
#include "resize/resize_impl.h"

#include "gtest/gtest.h"
#include "graph/filter_validator.h"
#include "graphengine/filter_validation.h"

namespace {

void test_case(const zimg::resize::Filter &filter, bool horizontal, unsigned src_w, unsigned src_h, unsigned dst_w, unsigned dst_h,
               zimg::PixelType type, const char * const expected_sha1[3], double expected_snr)
{
	if (!zimg::query_x86_capabilities().sse) {
		SUCCEED() << "sse not available, skipping";
		return;
	}

	SCOPED_TRACE(filter.support());
	SCOPED_TRACE(horizontal ? static_cast<double>(dst_w) / src_w : static_cast<double>(dst_h) / src_h);

	auto builder = zimg::resize::ResizeImplBuilder{ src_w, src_h, type }
		.set_horizontal(horizontal)
		.set_dst_dim(horizontal ? dst_w : dst_h)
		.set_filter(&filter)
		.set_shift(0.0)
		.set_subwidth(horizontal ? src_w : src_h);

	auto filter_c = builder.set_cpu(zimg::CPUClass::NONE).create();
	auto filter_sse = builder.set_cpu(zimg::CPUClass::X86_SSE).create();

	ASSERT_FALSE(assert_different_dynamic_type(filter_c.get(), filter_sse.get()));

	FilterValidator validator{ filter_sse.get(), src_w, src_h, type };
	validator.set_sha1(expected_sha1)
	         .set_ref_filter(filter_c.get(), expected_snr)
	         .validate();
}

void test_case_ge(const zimg::resize::Filter &filter, bool horizontal, unsigned src_w, unsigned src_h, unsigned dst_w, unsigned dst_h,
               zimg::PixelType type, const char *expected_sha1, double expected_snr)
{
	if (!zimg::query_x86_capabilities().sse) {
		SUCCEED() << "sse not available, skipping";
		return;
	}

	SCOPED_TRACE(filter.support());
	SCOPED_TRACE(horizontal ? static_cast<double>(dst_w) / src_w : static_cast<double>(dst_h) / src_h);

	auto builder = zimg::resize::ResizeImplBuilder{ src_w, src_h, type }
		.set_horizontal(horizontal)
		.set_dst_dim(horizontal ? dst_w : dst_h)
		.set_filter(&filter)
		.set_shift(0.0)
		.set_subwidth(horizontal ? src_w : src_h);

	auto filter_c = builder.set_cpu(zimg::CPUClass::NONE).create_ge();
	auto filter_sse = builder.set_cpu(zimg::CPUClass::X86_SSE).create_ge();

	ASSERT_FALSE(assert_different_dynamic_type(filter_c.get(), filter_sse.get()));

	graphengine::FilterValidation(filter_sse.get(), { src_w, src_h, zimg::pixel_size(type) })
		.set_input_pixel_format({ zimg::pixel_depth(type), zimg::pixel_is_float(type), false })
		.set_output_pixel_format({ zimg::pixel_depth(type), zimg::pixel_is_float(type), false })
		.set_reference_filter(filter_c.get(), expected_snr)
		.set_sha1(0, expected_sha1)
		.run();
}

} // namespace


TEST(ResizeImplSSETest, test_resize_h_f32)
{
	const unsigned src_w = 640;
	const unsigned dst_w = 960;
	const unsigned h = 480;
	const zimg::PixelType format = zimg::PixelType::FLOAT;

	const char *expected_sha1[][3] = {
		{ "1b2e37a345d315b0fa4d11e3532c70cb57b1e569" },
		{ "df391f7157d8c283abd408b35894139ca1903872" },
		{ "81fcfbdb9a3b31c625a3cdff1cf46da06f8af735" },
		{ "389b609ac62a8b9276e00fdcd39b921535196a07" }
	};
	const double expected_snr = 120.0;

	test_case(zimg::resize::BilinearFilter{}, true, src_w, h, dst_w, h, format, expected_sha1[0], expected_snr);
	test_case(zimg::resize::Spline16Filter{}, true, src_w, h, dst_w, h, format, expected_sha1[1], expected_snr);
	test_case(zimg::resize::LanczosFilter{ 4 }, true, src_w, h, dst_w, h, format, expected_sha1[2], expected_snr);
	test_case(zimg::resize::LanczosFilter{ 4 }, true, dst_w, h, src_w, h, format, expected_sha1[3], expected_snr);
}

TEST(ResizeImplSSETest, test_resize_h_f32_ge)
{
	const unsigned src_w = 640;
	const unsigned dst_w = 960;
	const unsigned h = 480;
	const zimg::PixelType format = zimg::PixelType::FLOAT;

	static const char *expected_sha1[] = {
		"1b2e37a345d315b0fa4d11e3532c70cb57b1e569",
		"df391f7157d8c283abd408b35894139ca1903872",
		"81fcfbdb9a3b31c625a3cdff1cf46da06f8af735",
		"389b609ac62a8b9276e00fdcd39b921535196a07"
	};
	const double expected_snr = 120.0;

	test_case_ge(zimg::resize::BilinearFilter{}, true, src_w, h, dst_w, h, format, expected_sha1[0], expected_snr);
	test_case_ge(zimg::resize::Spline16Filter{}, true, src_w, h, dst_w, h, format, expected_sha1[1], expected_snr);
	test_case_ge(zimg::resize::LanczosFilter{ 4 }, true, src_w, h, dst_w, h, format, expected_sha1[2], expected_snr);
	test_case_ge(zimg::resize::LanczosFilter{ 4 }, true, dst_w, h, src_w, h, format, expected_sha1[3], expected_snr);
}

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

TEST(ResizeImplSSETest, test_resize_v_f32_ge)
{
	const unsigned w = 640;
	const unsigned src_h = 480;
	const unsigned dst_h = 720;
	const zimg::PixelType type = zimg::PixelType::FLOAT;

	static const char *expected_sha1[] = {
		"6b7507617dc89d5d3077f9cc4c832b261dea2be0",
		"d07a8c6f3452ada7bd865a3283dc308176541db3",
		"bda98bc253213d2e28a54c6ccb7496f0ca5a3b7d",
		"6ba3876cd08a5b11ee646954b52b379a3d8b1228"
	};
	const double expected_snr = 120.0;

	test_case_ge(zimg::resize::BilinearFilter{}, false, w, src_h, w, dst_h, type, expected_sha1[0], expected_snr);
	test_case_ge(zimg::resize::Spline16Filter{}, false, w, src_h, w, dst_h, type, expected_sha1[1], expected_snr);
	test_case_ge(zimg::resize::LanczosFilter{ 4 }, false, w, src_h, w, dst_h, type, expected_sha1[2], expected_snr);
	test_case_ge(zimg::resize::LanczosFilter{ 4 }, false, w, dst_h, w, src_h, type, expected_sha1[3], expected_snr);
}

#endif
