#ifdef ZIMG_X86

#include <cmath>
#include "common/cpuinfo.h"
#include "common/pixel.h"
#include "common/x86/cpuinfo_x86.h"
#include "graph/image_filter.h"
#include "colorspace/colorspace.h"

#include "gtest/gtest.h"
#include "graph/filter_validator.h"

namespace {

void test_case(const zimg::colorspace::ColorspaceDefinition &csp_in, const zimg::colorspace::ColorspaceDefinition &csp_out,
               const char * const expected_sha1[3], double expected_snr)
{
	const unsigned w = 640;
	const unsigned h = 480;

	if (!zimg::query_x86_capabilities().sse2) {
		SUCCEED() << "sse2 not available, skipping";
		return;
	}

	zimg::PixelFormat format = zimg::PixelType::FLOAT;
	auto builder = zimg::colorspace::ColorspaceConversion{ w, h }
		.set_csp_in(csp_in)
		.set_csp_out(csp_out)
		.set_approximate_gamma(true);

	auto filter_c = builder.set_cpu(zimg::CPUClass::NONE).create();
	auto filter_sse2 = builder.set_cpu(zimg::CPUClass::X86_SSE2).create();

	FilterValidator validator{ filter_sse2.get(), w, h, format };
	validator.set_sha1(expected_sha1)
	         .set_ref_filter(filter_c.get(), expected_snr)
	         .set_yuv(csp_in.matrix != zimg::colorspace::MatrixCoefficients::RGB)
	         .validate();
}

} // namespace


TEST(ColorspaceConversionSSE2Test, test_transfer_lut)
{
	using namespace zimg::colorspace;

	const char *expected_sha1[][3] = {
		{
			"910ce7272eb8fac2e82424e9307eadffefbdb445",
			"fa7ab9deaee4790fa49a1d9d8249e5d323674ec2",
			"162687e701627cdc17283a32c36ea711d28a953e"
		},
		{
			"492587e7ed75b7e3ab868bead6ade7a4137c6ea1",
			"3b0694e9fbce61466cb5a575f300d784089b6cad",
			"b68f103f52ccafae867d664d7f27fe56ae9208af"
		},
		{
			"4c0b5ffe768a7812d1ef102b4d8d52614838bc8e",
			"58eb1dde0eb88fff043364836e1844aa766b64c5",
			"85a277a80dfca2e21789cedd76aaee307dbc4562"
		},
		{
			"df546ce0ad6f859499a96d2d697d896067e60e38",
			"f0041b8a008ab45f0ea1319090ac7e8be0990d92",
			"06880efb598e41f96fa79e04dbdfcccd50d6dc6f"
		},
	};
	const double expected_tolinear_snr = 80.0;
	const double expected_togamma_snr = 40.0;

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
