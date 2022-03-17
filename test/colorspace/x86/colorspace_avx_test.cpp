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

	if (!zimg::query_x86_capabilities().avx) {
		SUCCEED() << "avx not available, skipping";
		return;
	}

	zimg::PixelFormat format = zimg::PixelType::FLOAT;
	auto builder = zimg::colorspace::ColorspaceConversion{ w, h }
		.set_csp_in(csp_in)
		.set_csp_out(csp_out);

	auto filter_c = builder.set_cpu(zimg::CPUClass::NONE).create();
	auto filter_avx = builder.set_cpu(zimg::CPUClass::X86_AVX).create();

	FilterValidator validator{ filter_avx.get(), w, h, format };
	validator.set_sha1(expected_sha1)
	         .set_ref_filter(filter_c.get(), expected_snr)
	         .set_yuv(csp_in.matrix != zimg::colorspace::MatrixCoefficients::RGB)
	         .validate();
}

} // namespace


TEST(ColorspaceConversionAVXTest, test_matrix)
{
	using namespace zimg::colorspace;

	const char *expected_sha1[3] = {
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
