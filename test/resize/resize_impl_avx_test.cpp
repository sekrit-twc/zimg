#ifdef ZIMG_X86

#include <typeinfo>
#include "common/cpuinfo.h"
#include "common/pixel.h"
#include "resize/filter.h"
#include "resize/resize_impl.h"

#include "gtest/gtest.h"
#include "graph/filter_validator.h"

namespace {;

void test_case(const zimg::resize::Filter &filter, bool horizontal, unsigned src_w, unsigned src_h, unsigned dst_w, unsigned dst_h,
               zimg::PixelType type, const char * const expected_sha1[3], double expected_snr)
{
	if (!zimg::query_x86_capabilities().avx) {
		SUCCEED() << "avx not available, skipping";
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

	auto filter_c = builder.set_cpu(zimg::CPUClass::CPU_NONE).create();
	auto filter_avx = builder.set_cpu(zimg::CPUClass::CPU_X86_AVX).create();

	ASSERT_NE(typeid(*filter_c), typeid(*filter_avx)) << typeid(*filter_c).name() << " " << typeid(*filter_avx).name();

	validate_filter(filter_avx.get(), src_w, src_h, type, expected_sha1);
	validate_filter_reference(filter_c.get(), filter_avx.get(), src_w, src_h, type, expected_snr);
}

} // namespace


TEST(ResizeImplAVXTest, test_resize_v_f32)
{
	const unsigned w = 640;
	const unsigned src_h = 480;
	const unsigned dst_h = 720;
	const zimg::PixelType type = zimg::PixelType::FLOAT;

	const char *expected_sha1[][3] = {
		{ "6b7507617dc89d5d3077f9cc4c832b261dea2be0" },
		{ "d07a8c6f3452ada7bd865a3283dc308176541db3" },
		{ "127be47bf5124d8ed61f8da2a397d9f5eb14da4a" },
		{ "3113e07cb62b071a6ec71e41914e8a2f965020b6" }
	};
	const double expected_snr = 120.0;

	test_case(zimg::resize::BilinearFilter{}, false, w, src_h, w, dst_h, type, expected_sha1[0], expected_snr);
	test_case(zimg::resize::Spline16Filter{}, false, w, src_h, w, dst_h, type, expected_sha1[1], expected_snr);
	test_case(zimg::resize::LanczosFilter{ 4 }, false, w, src_h, w, dst_h, type, expected_sha1[2], expected_snr);
	test_case(zimg::resize::LanczosFilter{ 4 }, false, w, dst_h, w, src_h, type, expected_sha1[3], expected_snr);
}

#endif // ZIMG_X86
