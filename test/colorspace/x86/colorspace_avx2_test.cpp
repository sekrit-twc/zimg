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

	if (!zimg::query_x86_capabilities().avx2) {
		SUCCEED() << "avx2 not available, skipping";
		return;
	}

	zimg::PixelFormat format = zimg::PixelType::FLOAT;
	auto builder = zimg::colorspace::ColorspaceConversion{ w, h }
		.set_csp_in(csp_in)
		.set_csp_out(csp_out)
		.set_approximate_gamma(true);

	auto filter_c = builder.set_cpu(zimg::CPUClass::NONE).create();
	auto filter_avx2 = builder.set_cpu(zimg::CPUClass::X86_AVX2).create();

	ASSERT_TRUE(filter_c);
	ASSERT_TRUE(filter_avx2);

	graphengine::FilterValidation(filter_avx2.get(), { w, h, zimg::pixel_size(zimg::PixelType::FLOAT) })
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


TEST(ColorspaceConversionAVX2Test, test_transfer_lut)
{
	using namespace zimg::colorspace;

	static const char *expected_sha1[][3] = {
		{
			"23d012fcb280f601e2e3c349229d0108e3cd632a",
			"7ae186215d5fa45065f7aeac74ab2dc74b556696",
			"bad84d4e0de8572c81df6d9f91fef05b1576f9e5"
		},
		{
			"011ee645ad30bb6ad6d93d8980d89a3e3e073c19",
			"5ae0e075b3856d9f491954b477568b17daf7f147",
			"84b20f8fa27c23a668540566b9df26c4b42c9afa"
		},
		{
			"8206be2ae5e8a0fc003daeec4178189eecf82a13",
			"24843f17600dd7bf9870f5c778549bd96c333427",
			"26a6b00801b41da17d849e02217bf69add6324a6"
		},
		{
			"16f2274ffac90927de0438114f0ea22e650981a0",
			"b1c8b15b6159ab43e7bfc4e715fe3b621628d26e",
			"632ae07d6919533c87d2ed28560a60cf070498e2"
		},
	};
	const double expected_tolinear_snr = 80.0;
	const double expected_togamma_snr = 80.0;

	SCOPED_TRACE("tolinear 709");
	test_case({ MatrixCoefficients::RGB, TransferCharacteristics::REC_709, ColorPrimaries::UNSPECIFIED },
	          { MatrixCoefficients::RGB, TransferCharacteristics::LINEAR, ColorPrimaries::UNSPECIFIED },
	          expected_sha1[0], expected_tolinear_snr);
	SCOPED_TRACE("togamma 709");
	test_case({ MatrixCoefficients::RGB, TransferCharacteristics::LINEAR, ColorPrimaries::UNSPECIFIED },
	          { MatrixCoefficients::RGB, TransferCharacteristics::REC_709, ColorPrimaries::UNSPECIFIED },
	          expected_sha1[1], expected_togamma_snr);
	SCOPED_TRACE("tolinear st2084");
	test_case({ MatrixCoefficients::RGB, TransferCharacteristics::ST_2084, ColorPrimaries::UNSPECIFIED },
	          { MatrixCoefficients::RGB, TransferCharacteristics::LINEAR, ColorPrimaries::UNSPECIFIED },
	          expected_sha1[2], expected_tolinear_snr);
	SCOPED_TRACE("togamma st2084");
	test_case({ MatrixCoefficients::RGB, TransferCharacteristics::LINEAR, ColorPrimaries::UNSPECIFIED },
	          { MatrixCoefficients::RGB, TransferCharacteristics::ST_2084, ColorPrimaries::UNSPECIFIED },
	          expected_sha1[3], expected_togamma_snr);
}

#endif // ZIMG_X86
