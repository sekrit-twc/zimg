#include "common/pixel.h"
#include "colorspace/colorspace.h"
#include "graph/image_filter.h"

#include "gtest/gtest.h"
#include "graph/filter_validator.h"

namespace {

void test_case(const zimg::colorspace::ColorspaceDefinition &csp_in, const zimg::colorspace::ColorspaceDefinition &csp_out, const char * const expected_sha1[3])
{
	const unsigned w = 640;
	const unsigned h = 480;

	zimg::PixelFormat format = zimg::PixelType::FLOAT;
	auto convert = zimg::colorspace::ColorspaceConversion{ w, h }
		.set_csp_in(csp_in)
		.set_csp_out(csp_out)
		.create();

	validate_filter(convert.get(), w, h, format, expected_sha1);
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

	test_case({ MatrixCoefficients::RGB, TransferCharacteristics::UNSPECIFIED, ColorPrimaries::UNSPECIFIED },
	          { MatrixCoefficients::RGB, TransferCharacteristics::UNSPECIFIED, ColorPrimaries::UNSPECIFIED },
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

	test_case({ MatrixCoefficients::RGB, TransferCharacteristics::UNSPECIFIED, ColorPrimaries::UNSPECIFIED },
	          { MatrixCoefficients::REC_709, TransferCharacteristics::UNSPECIFIED, ColorPrimaries::UNSPECIFIED },
	          expected_sha1[0]);
	test_case({ MatrixCoefficients::REC_709, TransferCharacteristics::UNSPECIFIED, ColorPrimaries::UNSPECIFIED },
	          { MatrixCoefficients::RGB, TransferCharacteristics::UNSPECIFIED, ColorPrimaries::UNSPECIFIED },
	          expected_sha1[1]);
	test_case({ MatrixCoefficients::REC_601, TransferCharacteristics::UNSPECIFIED, ColorPrimaries::UNSPECIFIED },
	          { MatrixCoefficients::REC_709, TransferCharacteristics::UNSPECIFIED, ColorPrimaries::UNSPECIFIED },
	          expected_sha1[2]);
}

TEST(ColorspaceConversionTest, test_transfer_only)
{
	using namespace zimg::colorspace;

	ColorspaceDefinition csp_linear{ MatrixCoefficients::RGB, TransferCharacteristics::LINEAR, ColorPrimaries::UNSPECIFIED };
	ColorspaceDefinition csp_gamma{ MatrixCoefficients::RGB, TransferCharacteristics::REC_709, ColorPrimaries::UNSPECIFIED };
	ColorspaceDefinition csp_st2084{ MatrixCoefficients::RGB, TransferCharacteristics::ST_2084, ColorPrimaries::UNSPECIFIED };
	ColorspaceDefinition csp_arib_b67{ MatrixCoefficients::RGB, TransferCharacteristics::ARIB_B67, ColorPrimaries::UNSPECIFIED };

	const char *expected_sha1[][6] = {
		{
			"330a4d5fb0f3681689c8ea75f12afc53d2abacfa",
			"3b199c185b9a2ae677147694b424c4f1e1c2aa71",
			"85ecb896add9a2cdcf3a859e90ebaf5c3db24428"
		},
		{
			"72263d2c4e701fad7e19a98f7d4a6fd12c97f237",
			"9aad4d81bd3ac29a2d647218b91b9e3bb6b031b9",
			"a13696a7a2931b3ba549ef50ed061386f23ee354"
		},
		{
			"9344bde7a73ad908baa5dd8ebe15cfbcf9e40b81",
			"f4364af193955c13ec7fe6fc36cf94e1e5e692b1",
			"823ac60ee963298d89552f97416f04058c6d8c80"
		},
		{
			"4498bb3edb2391f990849a1ac5341ce7ac6bc6a6",
			"e85d2e8b26c7d979ff79784c0f62f796ad099f6d",
			"1b975ddb99008f2c72de2af2c864b52ffdea4c58"
		},
		{
			"4c62e5d775548495a170b6876a2e91b00d4b5f14",
			"90eae848b7050edf12ca22f57bda4eeccad8d7ef",
			"e2dc601f663ea61899f37a9db1b50b5e4110a38e"
		},
		{
			"446c897635131babdcbc60527ea53a677777ab72",
			"040517cd0eabb893d4bb79dc52c86dd983410b11",
			"ceed0c7ee2be19bd334cd33548e5162078007cf8"
		},
	};

	SCOPED_TRACE("gamma->linear");
	test_case(csp_gamma, csp_linear, expected_sha1[0]);

	SCOPED_TRACE("st2084->linear");
	test_case(csp_st2084, csp_linear, expected_sha1[1]);

	SCOPED_TRACE("b67->linear");
	test_case(csp_arib_b67, csp_linear, expected_sha1[2]);

	SCOPED_TRACE("linear->gamma");
	test_case(csp_linear, csp_gamma, expected_sha1[3]);

	SCOPED_TRACE("linear->st2084");
	test_case(csp_linear, csp_st2084, expected_sha1[4]);

	SCOPED_TRACE("linear->arib_b67");
	test_case(csp_linear, csp_arib_b67, expected_sha1[5]);
}

TEST(ColorspaceConversionTest, test_matrix_transfer)
{
	using namespace zimg::colorspace;

	const char *expected_sha1[3] = {
		"bc43116e500bb281fc17020b731813cddde8e057",
		"b7da9c5d86dff2004f82466f42da419e1f3bb02e",
		"10748bd4fa7c424b52c255b82d371a8534e124f8"
	};

	test_case({ MatrixCoefficients::REC_709, TransferCharacteristics::LINEAR, ColorPrimaries::UNSPECIFIED },
	          { MatrixCoefficients::REC_601, TransferCharacteristics::REC_709, ColorPrimaries::UNSPECIFIED },
	          expected_sha1);
}

TEST(ColorspaceConversionTest, test_matrix_transfer_primaries)
{
	using namespace zimg::colorspace;

	const char *expected_sha1[][3] = {
		{
			"4cd9db8fd2ded345aafffda2358cd6de9c763e00",
			"2dd41a82f55933f35988ca5f2d57c09f8157d43a",
			"2262468fb4840160c880dcaace12a44568e44246"
		},
		{
			"1e14ec2d359417837a6501c948ec110597535bb4",
			"c564d33ae070a54e5f10baa262f6da6c8450ed80",
			"e511d90a33b8e27ee96adfda987038cd4d540f8e"
		},
		{
			"6eecdcbf5cb11437ecab1700f0b0769c09a942e7",
			"37c2a61353e006672fe4169b3eef588f4f08b69a",
			"be47b698c9252af496b7d5b1bf7c678f1e39e615"
		}
	};

	SCOPED_TRACE("709->smpte_c");
	test_case({ MatrixCoefficients::REC_709, TransferCharacteristics::REC_709, ColorPrimaries::REC_709 },
	          { MatrixCoefficients::REC_601, TransferCharacteristics::REC_709, ColorPrimaries::SMPTE_C },
	          expected_sha1[0]);

	SCOPED_TRACE("709->2020");
	test_case({ MatrixCoefficients::REC_709, TransferCharacteristics::REC_709, ColorPrimaries::REC_709 },
	          { MatrixCoefficients::REC_2020_NCL, TransferCharacteristics::REC_709, ColorPrimaries::REC_2020 },
	          expected_sha1[1]);

	SCOPED_TRACE("709->p3d65");
	test_case({ MatrixCoefficients::REC_709, TransferCharacteristics::REC_709, ColorPrimaries::REC_709 },
	          { MatrixCoefficients::REC_709, TransferCharacteristics::REC_709, ColorPrimaries::DCI_P3_D65 },
	          expected_sha1[2]);
}

TEST(ColorspaceConversionTest, test_rec2020_cl)
{
	using namespace zimg::colorspace;

	ColorspaceDefinition csp_2020cl{ MatrixCoefficients::REC_2020_CL, TransferCharacteristics::REC_709, ColorPrimaries::REC_2020 };

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
	test_case(csp_2020cl, csp_2020cl.to_rgb(), expected_sha1[0]);

	SCOPED_TRACE("rgb->2020cl");
	test_case(csp_2020cl.to_rgb(), csp_2020cl, expected_sha1[1]);

	SCOPED_TRACE("2020cl->2020ncl");
	test_case(csp_2020cl, csp_2020cl.to(MatrixCoefficients::REC_2020_NCL), expected_sha1[2]);

	SCOPED_TRACE("2020ncl->2020cl");
	test_case(csp_2020cl.to(MatrixCoefficients::REC_2020_NCL), csp_2020cl, expected_sha1[3]);
}
