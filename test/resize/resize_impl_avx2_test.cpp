#ifdef ZIMG_X86

#include <typeinfo>
#include "common/cpuinfo.h"
#include "common/pixel.h"
#include "resize/filter.h"
#include "resize/resize_impl.h"

#include "gtest/gtest.h"
#include "graph/filter_validator.h"

namespace {

void test_case(const zimg::resize::Filter &filter, bool horizontal, unsigned src_w, unsigned src_h, unsigned dst_w, unsigned dst_h,
               zimg::PixelType type, const char * const expected_sha1[3], double expected_snr)
{
	if (!zimg::query_x86_capabilities().avx2) {
		SUCCEED() << "avx2 not available, skipping";
		return;
	}

	SCOPED_TRACE(filter.support());
	SCOPED_TRACE(horizontal ? (double)dst_w / src_w : (double)dst_h / src_h);

	auto builder = zimg::resize::ResizeImplBuilder{ src_w, src_h, type }.
		set_horizontal(horizontal).
		set_dst_dim(horizontal ? dst_w : dst_h).
		set_filter(&filter).
		set_shift(0.0).
		set_subwidth(horizontal ? src_w : src_h);

	auto filter_avx2 = builder.set_cpu(zimg::CPUClass::CPU_X86_AVX2).create();
	validate_filter(filter_avx2.get(), src_w, src_h, type, expected_sha1);

	// No half-precision implementation is available in C. Make sure to visually check results if they differ from hash.
	if (type != zimg::PixelType::HALF) {
		auto filter_c = builder.set_cpu(zimg::CPUClass::CPU_NONE).create();
		ASSERT_NE(typeid(*filter_c), typeid(*filter_avx2)) << typeid(*filter_c).name() << " " << typeid(*filter_avx2).name();
		validate_filter_reference(filter_c.get(), filter_avx2.get(), src_w, src_h, type, expected_snr);
	}
}

} // namespace


TEST(ResizeImplAVX2Test, test_resize_h_f16)
{
	const unsigned src_w = 640;
	const unsigned dst_w = 960;
	const unsigned h = 480;
	const zimg::PixelType format = zimg::PixelType::HALF;

	const char *expected_sha1[][3] = {
		{ "7fc9f1b457b7c9d76df16597832cfca33cac934b" },
		{ "f65b7ac6105c8f1744bc9cf6fbb85cc4f10e7e00" },
		{ "a1d899272a411f3fb954b3b82f9d16a8f975a5f8" },
		{ "4b9f47f282a606b2086354767db6ccd96d0b4e1e" }
	};
	const double expected_snr = 120.0;

	test_case(zimg::resize::BilinearFilter{}, true, src_w, h, dst_w, h, format, expected_sha1[0], NAN);
	test_case(zimg::resize::Spline16Filter{}, true, src_w, h, dst_w, h, format, expected_sha1[1], NAN);
	test_case(zimg::resize::LanczosFilter{ 4 }, true, src_w, h, dst_w, h, format, expected_sha1[2], NAN);
	test_case(zimg::resize::LanczosFilter{ 4 }, true, dst_w, h, src_w, h, format, expected_sha1[3], NAN);
}


TEST(ResizeImplAVX2Test, test_resize_v_f16)
{
	const unsigned w = 640;
	const unsigned src_h = 480;
	const unsigned dst_h = 720;
	const zimg::PixelType type = zimg::PixelType::HALF;

	const char *expected_sha1[][3] = {
		{ "43bef3b996733efa9d2b25e9096edc06ceee99cd" },
		{ "ccf24249d20be7ffe8707a33c5996483c8fb4500" },
		{ "0831975c4802cd243d3978f0874c05eba590ab08" },
		{ "b4bb1a5a6654c9b9a45928c852575f79de6bf710" }
	};
	const double expected_snr = 120.0;

	test_case(zimg::resize::BilinearFilter{}, false, w, src_h, w, dst_h, type, expected_sha1[0], NAN);
	test_case(zimg::resize::Spline16Filter{}, false, w, src_h, w, dst_h, type, expected_sha1[1], NAN);
	test_case(zimg::resize::LanczosFilter{ 4 }, false, w, src_h, w, dst_h, type, expected_sha1[2], NAN);
	test_case(zimg::resize::LanczosFilter{ 4 }, false, w, dst_h, w, src_h, type, expected_sha1[3], NAN);
}

TEST(ResizeImplAVX2Test, test_resize_h_f32)
{
	const unsigned src_w = 640;
	const unsigned dst_w = 960;
	const unsigned h = 480;
	const zimg::PixelType format = zimg::PixelType::FLOAT;

	const char *expected_sha1[][3] = {
		{ "1b2e37a345d315b0fa4d11e3532c70cb57b1e569" },
		{ "2d0582a2f6af8a480e8f053fbd89eac0668b33f3" },
		{ "967f921dc3fd2b3d166a276fe671105c3fac0756" },
		{ "166dfd1881724fe546571c2d7ac959e6433623be" }
	};
	const double expected_snr = 120.0;

	test_case(zimg::resize::BilinearFilter{}, true, src_w, h, dst_w, h, format, expected_sha1[0], expected_snr);
	test_case(zimg::resize::Spline16Filter{}, true, src_w, h, dst_w, h, format, expected_sha1[1], expected_snr);
	test_case(zimg::resize::LanczosFilter{ 4 }, true, src_w, h, dst_w, h, format, expected_sha1[2], expected_snr);
	test_case(zimg::resize::LanczosFilter{ 4 }, true, dst_w, h, src_w, h, format, expected_sha1[3], expected_snr);
}


TEST(ResizeImplAVX2Test, test_resize_v_f32)
{
	const unsigned w = 640;
	const unsigned src_h = 480;
	const unsigned dst_h = 720;
	const zimg::PixelType type = zimg::PixelType::FLOAT;

	const char *expected_sha1[][3] = {
		{ "6b7507617dc89d5d3077f9cc4c832b261dea2be0" },
		{ "46283014e580fa47deacae5e0cec1ce952973f51" },
		{ "47946b5a3aba5e9ee6967659e8aeb26070ae80d6" },
		{ "bcedc16781dc7781557d744b75ccac510a98a3ac" }
	};
	const double expected_snr = 120.0;

	test_case(zimg::resize::BilinearFilter{}, false, w, src_h, w, dst_h, type, expected_sha1[0], expected_snr);
	test_case(zimg::resize::Spline16Filter{}, false, w, src_h, w, dst_h, type, expected_sha1[1], expected_snr);
	test_case(zimg::resize::LanczosFilter{ 4 }, false, w, src_h, w, dst_h, type, expected_sha1[2], expected_snr);
	test_case(zimg::resize::LanczosFilter{ 4 }, false, w, dst_h, w, src_h, type, expected_sha1[3], expected_snr);
}

#endif // ZIMG_X86
