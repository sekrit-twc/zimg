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

TEST(ColorspaceConversionNeonTest, test_transfer_lut)
{
	using namespace zimg::colorspace;

	const char *expected_sha1[][3] = {
		{
#if defined(_M_ARM64) || defined(__aarch64__)
			"23d012fcb280f601e2e3c349229d0108e3cd632a",
			"21f5071b0a817c28295d51ead5137cabc6e0d5c4",
			"158c4ff0c91c08f82e9fc35a500a1c8166f5ae6b"
#else
			"52451877e62e9fc31eb10b1e37c5f95fdd8851db",
			"3e2ff4f017c343edbe787692ce169123124337b1",
			"f719a90e6a6c859bfcfc136f3296e65044495da0"
#endif
		},
		{
			"011ee645ad30bb6ad6d93d8980d89a3e3e073c19",
			"d64814ca78cbf4e07606f92f1644f59762271ca5",
			"f871247697737f9f8b6a59a58306e22cce472ea6"
		},
		{
#if defined(_M_ARM64) || defined(__aarch64__)
			"8206be2ae5e8a0fc003daeec4178189eecf82a13",
			"6bc5833cbd22f04c1965d230aad2ef8969da24b7",
			"6538399afe0b9fd55a95608b25c8036e16d658b8"
#else
			"905d4d54eeae6458e8e0975c9cea66b25edcc234",
			"c2e7015447b40ebb2f4bfba48b7b091f964b22f1",
			"d222f960fe874ac88608666c4af8de180d91868e"
#endif
		},
		{
			"16f2274ffac90927de0438114f0ea22e650981a0",
			"2e01c95f89ea26b5a55bed895223381ac3f17e70",
			"c61d9c5369a00af5bb40b70fbb21956c00a4a1e9"
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
