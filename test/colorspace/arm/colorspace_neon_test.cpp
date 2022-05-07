#ifdef ZIMG_ARM

#include <cmath>
#include "common/cpuinfo.h"
#include "common/pixel.h"
#include "common/arm/cpuinfo_arm.h"
#include "graph/image_filter.h"
#include "colorspace/colorspace.h"

#include "gtest/gtest.h"
#include "graph/filter_validator.h"

// ARMv7a vs ARMv8a numerics:
//     ARMv7a NEON does not support round-half-to-even mode (i.e. IEEE-754).
// For performance reasons, the ARM 32-bit kernels use floor(x + 0.5) instead.
//     ARMv8a implements fused-multiply-add, which produces the same result as
// the x86 AVX2 kernels, but slightly different output from C.

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
		"749f74428406c019b1b727fa30352fcd1f0141ed",
		"334cfa73375f8afef8423a163f3cff8f8a196762",
		"aa3aab12d52e67b4d6765b4e8c03205a5375d8d9"
#else
		"1d559e4b2812a5940839b064f5bd74bc4fe0a2f9",
		"b32a33c4bbbf3901f89458f914e6d03cc81f2c1d",
		"4aadd644fae30cfd2098bb8d2b9f98483c8821fd"
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

TEST(ColorspaceConversionNeonTest, test_transfer_lut)
{
	using namespace zimg::colorspace;

	const char *expected_sha1[][3] = {
		{
#if defined(_M_ARM64) || defined(__aarch64__)
			"23d012fcb280f601e2e3c349229d0108e3cd632a",
			"7ae186215d5fa45065f7aeac74ab2dc74b556696",
			"bad84d4e0de8572c81df6d9f91fef05b1576f9e5"
#else
			"52451877e62e9fc31eb10b1e37c5f95fdd8851db",
			"06bc0aff436bbbf4ba633b2255dd096e628a129c",
			"a20570af1c05291029ea7d6b4215c451f4a9187a"
#endif
		},
		{
			"011ee645ad30bb6ad6d93d8980d89a3e3e073c19",
			"5ae0e075b3856d9f491954b477568b17daf7f147",
			"84b20f8fa27c23a668540566b9df26c4b42c9afa"
		},
		{
#if defined(_M_ARM64) || defined(__aarch64__)
			"8206be2ae5e8a0fc003daeec4178189eecf82a13",
			"24843f17600dd7bf9870f5c778549bd96c333427",
			"26a6b00801b41da17d849e02217bf69add6324a6"
#else
			"905d4d54eeae6458e8e0975c9cea66b25edcc234",
			"d380f54820f1e269ea54a1d552b0cb704f83dd7b",
			"552579149674b5e37f0d443ad19a59593fdca057"
#endif
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

#endif // ZIMG_ARM
