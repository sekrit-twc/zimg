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
			"95f2715bd0d417028bebd5c5377180fcd5b01119",
			"76f7c88b198f1ab08167f8162c1237b54f22007a",
			"1099c3ae187c0a9f79acb9445761b6056218c779"
		},
		{
			"4c0b5ffe768a7812d1ef102b4d8d52614838bc8e",
			"58eb1dde0eb88fff043364836e1844aa766b64c5",
			"85a277a80dfca2e21789cedd76aaee307dbc4562"
		},
		{
			"5e35786d313e936566d9873ba7a08a8d6005b2ee",
			"829fa88acfbbb26801871bf3cadf5cc2eb6830c9",
			"f82fcad18a19b548d419a1952b6a7a423a684b62"
		},
	};
	const double expected_tolinear_snr = 80.0;
	const double expected_togamma_snr = 60.0;

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
