#include "common/cpuinfo.h"
#include "common/pixel.h"
#include "colorspace/colorspace_param.h"
#include "colorspace/colorspace.h"

#include "gtest/gtest.h"
#include "graph/filter_validator.h"

namespace {;

void test_case(const zimg::colorspace::ColorspaceDefinition &csp_in, const zimg::colorspace::ColorspaceDefinition &csp_out, const char * const expected_sha1[3])
{
	const unsigned w = 640;
	const unsigned h = 480;

	zimg::PixelFormat format = zimg::default_pixel_format(zimg::PixelType::FLOAT);
	zimg::colorspace::ColorspaceConversion convert{ w, h, csp_in, csp_out, zimg::CPUClass::CPU_NONE };
	validate_filter(&convert, w, h, format, expected_sha1);
}

} // namespace


TEST(ColorspaceConversionTest, test_nop)
{
	using namespace zimg::colorspace;

	const char *expected_sha1[3] = {
		"483b6bdf608afbf1fba6bbca9657a8ca3822eef1",
		"4b4d64db9e73e41d7c1f612dea4dd08c7e8c07f7",
		"1e49fae79df6a5497ef8c58b28c1ccb300e8f523"
	};

	test_case({ MatrixCoefficients::MATRIX_RGB, TransferCharacteristics::TRANSFER_UNSPECIFIED, ColorPrimaries::PRIMARIES_UNSPECIFIED },
	          { MatrixCoefficients::MATRIX_RGB, TransferCharacteristics::TRANSFER_UNSPECIFIED, ColorPrimaries::PRIMARIES_UNSPECIFIED },
	          expected_sha1);
}

TEST(ColorspaceConversionTest, test_matrix_only)
{
	using namespace zimg::colorspace;

	const char *expected_sha1[][3] = {
		{
			"0495adab9c82d98e73841e229a9b2041838fc0f2",
			"ece7edb1118d4b3063ad80f5d8febb6db7e9633a",
			"73a9ee951c7bde9ae0ada9b90afd1f7ce8b604df"
		},
		{
			"86fa6f86eeeea51ece937dbd8ad8be881636c5ec",
			"a596bf669ab6d6f8e8ce5e862e00dd0babe51c48",
			"cf8fbed8b60ae7328d43d06523ab25eab1095316"
		},
		{
			"b189703d18508db5e1fa374e7403415728dfaf51",
			"b81c2d5cf5251aad96797ec0025fe56b2c676853",
			"03d20fd3a89fa681a461491a2558734636bd7e9b"
		},
	};

	test_case({ MatrixCoefficients::MATRIX_RGB, TransferCharacteristics::TRANSFER_UNSPECIFIED, ColorPrimaries::PRIMARIES_UNSPECIFIED },
	          { MatrixCoefficients::MATRIX_709, TransferCharacteristics::TRANSFER_UNSPECIFIED, ColorPrimaries::PRIMARIES_UNSPECIFIED },
	          expected_sha1[0]);
	test_case({ MatrixCoefficients::MATRIX_709, TransferCharacteristics::TRANSFER_UNSPECIFIED, ColorPrimaries::PRIMARIES_UNSPECIFIED },
	          { MatrixCoefficients::MATRIX_RGB, TransferCharacteristics::TRANSFER_UNSPECIFIED, ColorPrimaries::PRIMARIES_UNSPECIFIED },
	          expected_sha1[1]);
	test_case({ MatrixCoefficients::MATRIX_601, TransferCharacteristics::TRANSFER_UNSPECIFIED, ColorPrimaries::PRIMARIES_UNSPECIFIED },
	          { MatrixCoefficients::MATRIX_709, TransferCharacteristics::TRANSFER_UNSPECIFIED, ColorPrimaries::PRIMARIES_UNSPECIFIED },
	          expected_sha1[2]);
}

TEST(ColorspaceConversionTest, test_matrix_transfer)
{
	using namespace zimg::colorspace;

	const char *expected_sha1[3] = {
		"bc43116e500bb281fc17020b731813cddde8e057",
		"b7da9c5d86dff2004f82466f42da419e1f3bb02e",
		"10748bd4fa7c424b52c255b82d371a8534e124f8"
	};

	test_case({ MatrixCoefficients::MATRIX_709, TransferCharacteristics::TRANSFER_LINEAR, ColorPrimaries::PRIMARIES_UNSPECIFIED },
	          { MatrixCoefficients::MATRIX_601, TransferCharacteristics::TRANSFER_709, ColorPrimaries::PRIMARIES_UNSPECIFIED },
	          expected_sha1);
}

TEST(ColorspaceConversionTest, test_matrix_transfer_primaries)
{
	using namespace zimg::colorspace;

	const char *expected_sha1[][3] = {
		{
			"801cd3a68eec5bdac4b14e40de6f76724a0a48af",
			"f40cf973dc5245610a6cf4cbd7640d0369a1ff97",
			"b8deaf80e09777de4c3260850b0347a47f611040"
		},
		{
			"ffc1ff3c7be80355c40cf2b7b105ffce9d0a23a6",
			"7cb906b65f88d997c354bedfd80b37c1959d883d",
			"2a1330b248efa25fc1acfc25c1fba733910ebff9"
		}
	};

	SCOPED_TRACE("709->smpte_c");
	test_case({ MatrixCoefficients::MATRIX_709, TransferCharacteristics::TRANSFER_709, ColorPrimaries::PRIMARIES_709 },
	          { MatrixCoefficients::MATRIX_601, TransferCharacteristics::TRANSFER_709, ColorPrimaries::PRIMARIES_SMPTE_C },
	          expected_sha1[0]);

	SCOPED_TRACE("709->2020");
	test_case({ MatrixCoefficients::MATRIX_709, TransferCharacteristics::TRANSFER_709, ColorPrimaries::PRIMARIES_709 },
	          { MatrixCoefficients::MATRIX_2020_NCL, TransferCharacteristics::TRANSFER_709, ColorPrimaries::PRIMARIES_2020 },
	          expected_sha1[1]);
}

TEST(ColorspaceConversionTest, test_rec2020_cl)
{
	using namespace zimg::colorspace;

	ColorspaceDefinition csp_2020cl{ MatrixCoefficients::MATRIX_2020_CL, TransferCharacteristics::TRANSFER_709, ColorPrimaries::PRIMARIES_2020 };

	const char *expected_sha1[][3] = {
		{
			"f97ba4eff570e8f913d81df0c4586b4acdfe9a09",
			"1ee95d1ede59503c04766b0ae784494ceab55ed5",
			"bfacffcdeaaf2c84aac56f16b48b1397703a4d43"
		},
		{
			"371a25d860f9afb1577da1fcc20899f0d674c273",
			"6a04c69315f33e16a1c93f88a138902ac5b9437d",
			"087305d52cec79f609294963f66a529b07b72b0d"
		},
		{
			"15cf79998d301bf6192c75e916e14f524e697210",
			"8159f5c700c0eb9c7792a7b3431e7ca257c882b7",
			"6b06893a576c7271ea0fb15724c87dc8771c251c"
		},
		{
			"85a4798b25fbf178eb687dd7b80137e9ffcdb524",
			"3174c7fa05b21434ced0462828f94586518d27e7",
			"0f30c42546ed72792e08115292a15ae00e01eeff"
		},
	};

	SCOPED_TRACE("2020cl->rgb");
	test_case(csp_2020cl, csp_2020cl.toRGB(), expected_sha1[0]);

	SCOPED_TRACE("rgb->2020cl");
	test_case(csp_2020cl.toRGB(), csp_2020cl, expected_sha1[1]);

	SCOPED_TRACE("2020cl->2020ncl");
	test_case(csp_2020cl, csp_2020cl.to(MatrixCoefficients::MATRIX_2020_NCL), expected_sha1[2]);

	SCOPED_TRACE("2020ncl->2020cl");
	test_case(csp_2020cl.to(MatrixCoefficients::MATRIX_2020_NCL), csp_2020cl, expected_sha1[3]);
}
