#ifdef ZIMG_ARM

#include <cmath>
#include "common/cpuinfo.h"
#include "common/pixel.h"
#include "common/arm/cpuinfo_arm.h"
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

	if (!zimg::query_arm_capabilities().neon) {
		SUCCEED() << "neon not available, skipping";
		return;
	}

	zimg::PixelFormat format = zimg::PixelType::FLOAT;
	auto builder = zimg::colorspace::ColorspaceConversion{ w, h }
		.set_csp_in(csp_in)
		.set_csp_out(csp_out)
		.set_approximate_gamma(true);

	auto filter_c = builder.set_cpu(zimg::CPUClass::NONE).create();
	auto filter_neon = builder.set_cpu(zimg::CPUClass::ARM_NEON).create();

	FilterValidator validator{ filter_neon.get(), w, h, format };
	validator.set_sha1(expected_sha1)
	         .set_ref_filter(filter_c.get(), expected_snr)
	         .set_yuv(csp_in.matrix != zimg::colorspace::MatrixCoefficients::RGB)
	         .validate();
}

} // namespace


TEST(ColorspaceConversionNeonTest, test_matrix)
{
	using namespace zimg::colorspace;

	const char *expected_sha1[3] = {
#if defined(_M_ARM64) || defined(__aarch64__)
		"7b2a05426e2ef61dc6adc16573fca46ea3480256",
		"9c69bc8fa775a8e877e66e79812e9f4c39cec647",
		"6010983126eb3f5ca2dd5c01f4753c0e9f36d0bb"
#else
		"0495adab9c82d98e73841e229a9b2041838fc0f2",
		"ece7edb1118d4b3063ad80f5d8febb6db7e9633a",
		"73a9ee951c7bde9ae0ada9b90afd1f7ce8b604df"
#endif
	};
#if defined(_M_ARM64) || defined(__aarch64__)
	const double expected_snr = 120.0;
#else
	const double expected_snr = INFINITY;
#endif

	test_case({ MatrixCoefficients::RGB, TransferCharacteristics::UNSPECIFIED, ColorPrimaries::UNSPECIFIED },
	          { MatrixCoefficients::REC_709, TransferCharacteristics::UNSPECIFIED, ColorPrimaries::UNSPECIFIED },
	          expected_sha1, expected_snr);
}

#endif // ZIMG_ARM
