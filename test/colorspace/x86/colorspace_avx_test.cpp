#ifdef ZIMG_X86

#include <cmath>
#include "colorspace/colorspace.h"
#include "common/cpuinfo.h"
#include "common/pixel.h"
#include "common/x86/cpuinfo_x86.h"
#include "graphengine/filter.h"

#include "gtest/gtest.h"
#include "graphengine/filter_validation.h"

namespace {

void test_case(const zimg::colorspace::ColorspaceDefinition &csp_in, const zimg::colorspace::ColorspaceDefinition &csp_out,
               const char * const expected_sha1[3], double expected_snr)
{
	const unsigned w = 640;
	const unsigned h = 480;

	if (!zimg::query_x86_capabilities().avx) {
		SUCCEED() << "avx not available, skipping";
		return;
	}

	zimg::PixelFormat format = zimg::PixelType::FLOAT;
	auto builder = zimg::colorspace::ColorspaceConversion{ w, h }
		.set_csp_in(csp_in)
		.set_csp_out(csp_out);

	auto filter_c = builder.set_cpu(zimg::CPUClass::NONE).create_ge();
	auto filter_avx = builder.set_cpu(zimg::CPUClass::X86_AVX).create_ge();

	ASSERT_TRUE(filter_c);
	ASSERT_TRUE(filter_avx);

	graphengine::FilterValidation(filter_avx.get(), { w, h, zimg::pixel_size(zimg::PixelType::FLOAT) })
		.set_reference_filter(filter_c.get(), expected_snr)
		.set_input_pixel_format(0, { zimg::pixel_depth(zimg::PixelType::FLOAT), true, false })
		.set_input_pixel_format(1, { zimg::pixel_depth(zimg::PixelType::FLOAT), true, csp_in.matrix != zimg::colorspace::MatrixCoefficients::RGB })
		.set_input_pixel_format(2, { zimg::pixel_depth(zimg::PixelType::FLOAT), true, csp_in.matrix != zimg::colorspace::MatrixCoefficients::RGB })
		.set_output_pixel_format(0, { zimg::pixel_depth(zimg::PixelType::FLOAT), true, false })
		.set_output_pixel_format(1, { zimg::pixel_depth(zimg::PixelType::FLOAT), true, csp_out.matrix != zimg::colorspace::MatrixCoefficients::RGB })
		.set_output_pixel_format(2, { zimg::pixel_depth(zimg::PixelType::FLOAT), true, csp_out.matrix != zimg::colorspace::MatrixCoefficients::RGB })
		.set_sha1(0, expected_sha1[0])
		.set_sha1(1, expected_sha1[1])
		.set_sha1(2, expected_sha1[2])
		.run();
}

} // namespace


TEST(ColorspaceConversionAVXTest, test_matrix)
{
	using namespace zimg::colorspace;

	static const char *expected_sha1[3] = {
		"1d559e4b2812a5940839b064f5bd74bc4fe0a2f9",
		"b32a33c4bbbf3901f89458f914e6d03cc81f2c1d",
		"4aadd644fae30cfd2098bb8d2b9f98483c8821fd"
	};
	const double expected_snr = INFINITY;

	test_case({ MatrixCoefficients::RGB, TransferCharacteristics::UNSPECIFIED, ColorPrimaries::UNSPECIFIED },
	          { MatrixCoefficients::REC_709, TransferCharacteristics::UNSPECIFIED, ColorPrimaries::UNSPECIFIED },
	          expected_sha1, expected_snr);
}

#endif // ZIMG_X86
