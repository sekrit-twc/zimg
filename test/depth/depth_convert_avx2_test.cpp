#ifdef ZIMG_X86

#include <cmath>
#include "common/cpuinfo.h"
#include "common/pixel.h"
#include "graph/image_filter.h"
#include "depth/depth_convert.h"

#include "gtest/gtest.h"
#include "graph/filter_validator.h"

namespace {

void test_case_depth_convert(const zimg::PixelFormat &pixel_in, const zimg::PixelFormat &pixel_out, const char * const expected_sha1[3], double expected_snr)
{
	const unsigned w = 640;
	const unsigned h = 480;

	if (!zimg::query_x86_capabilities().avx2) {
		SUCCEED() << "avx2 not available, skipping";
		return;
	}

	auto filter_c = zimg::depth::create_convert_to_float(w, h, pixel_in, pixel_out, zimg::CPUClass::NONE);
	auto filter_avx2 = zimg::depth::create_convert_to_float(w, h, pixel_in, pixel_out, zimg::CPUClass::X86_AVX2);

	validate_filter(filter_avx2.get(), w, h, pixel_in, expected_sha1);
	validate_filter_reference(filter_c.get(), filter_avx2.get(), w, h, pixel_in, expected_snr);
}

} // namespace


TEST(DepthConvertAVX2Test, test_depth_convert_b2h)
{
	zimg::PixelFormat pixel_in{ zimg::PixelType::BYTE, 8, true };
	zimg::PixelFormat pixel_out{ zimg::PixelType::HALF };

	const char *expected_sha1[3] = {
		"f0e4a68158eab0ab350c7161498a8eed3196c233"
	};

	test_case_depth_convert(pixel_in, pixel_out, expected_sha1, INFINITY);
}

TEST(DepthConvertAVX2Test, test_depth_convert_b2f)
{
	zimg::PixelFormat pixel_in{ zimg::PixelType::BYTE, 8, true };
	zimg::PixelFormat pixel_out{ zimg::PixelType::FLOAT };

	const char *expected_sha1[3] = {
		"20c77820ff7d4443a0de7991218e2f8eee551e8d"
	};

	test_case_depth_convert(pixel_in, pixel_out, expected_sha1, INFINITY);
}

TEST(DepthConvertAVX2Test, test_depth_convert_w2h)
{
	zimg::PixelFormat pixel_in{ zimg::PixelType::WORD, 16, true };
	zimg::PixelFormat pixel_out{ zimg::PixelType::HALF };

	const char *expected_sha1[3] = {
		"07b6aebbfe48004c8acb12a3c76137db57ba9a0b"
	};

	test_case_depth_convert(pixel_in, pixel_out, expected_sha1, INFINITY);
}

TEST(DepthConvertAVX2Test, test_depth_convert_w2f)
{
	zimg::PixelFormat pixel_in{ zimg::PixelType::WORD, 16, true };
	zimg::PixelFormat pixel_out{ zimg::PixelType::FLOAT };

	const char *expected_sha1[3] = {
		"7ad2bc4ba1be92699ec22f489ae93a8b0dc89821"
	};

	test_case_depth_convert(pixel_in, pixel_out, expected_sha1, INFINITY);
}

#endif // ZIMG_X86
