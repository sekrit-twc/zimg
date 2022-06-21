#include "common/pixel.h"
#include "colorspace/colorspace.h"
#include "graphengine/filter.h"

#include "gtest/gtest.h"
#include "graphengine/filter_validation.h"

namespace {

void test_case(const zimg::colorspace::ColorspaceDefinition &csp_in, const zimg::colorspace::ColorspaceDefinition &csp_out, const char * const expected_sha1[3])
{
	const unsigned w = 640;
	const unsigned h = 480;
	bool yuv = csp_in.matrix != zimg::colorspace::MatrixCoefficients::RGB;

	auto filter = zimg::colorspace::ColorspaceConversion{ w, h }
		.set_csp_in(csp_in)
		.set_csp_out(csp_out)
		.create();

	ASSERT_TRUE(filter);

	graphengine::FilterValidation(filter.get(), { w, h, zimg::pixel_size(zimg::PixelType::FLOAT) })
		.set_input_pixel_format(0, { zimg::pixel_depth(zimg::PixelType::FLOAT), true, false })
		.set_input_pixel_format(1, { zimg::pixel_depth(zimg::PixelType::FLOAT), true, yuv })
		.set_input_pixel_format(2, { zimg::pixel_depth(zimg::PixelType::FLOAT), true, yuv })
		.set_output_pixel_format(0, { zimg::pixel_depth(zimg::PixelType::FLOAT), true, false })
		.set_output_pixel_format(1, { zimg::pixel_depth(zimg::PixelType::FLOAT), true, yuv })
		.set_output_pixel_format(2, { zimg::pixel_depth(zimg::PixelType::FLOAT), true, yuv })
		.set_sha1(0, expected_sha1[0])
		.set_sha1(1, expected_sha1[1])
		.set_sha1(2, expected_sha1[2])
		.run();
}

} // namespace


TEST(ColorspaceConversionTest, test_nop)
{
	using namespace zimg::colorspace;

	auto filter = zimg::colorspace::ColorspaceConversion{ 640, 480 }
		.set_csp_in({ MatrixCoefficients::RGB, TransferCharacteristics::UNSPECIFIED, ColorPrimaries::UNSPECIFIED })
		.set_csp_out({ MatrixCoefficients::RGB, TransferCharacteristics::UNSPECIFIED, ColorPrimaries::UNSPECIFIED })
		.create();
	EXPECT_FALSE(filter);
}

TEST(ColorspaceConversionTest, test_matrix_only)
{
	using namespace zimg::colorspace;

	static const char *expected_sha1[][3] = {
		{
			"1d559e4b2812a5940839b064f5bd74bc4fe0a2f9",
			"b32a33c4bbbf3901f89458f914e6d03cc81f2c1d",
			"4aadd644fae30cfd2098bb8d2b9f98483c8821fd"
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
		{
			"3732b8f5fb4b5282ab1912a689f3d25cf5651bcb",
			"da2c32ffe713f33b0bf16632e21e41048d3dfa9c",
			"2961ec857c9308ee9398356f86fb3ab6e4e874b1"
		},
		{
			"fe2953868a65076d0b33dc681ebda4d07d01b819",
			"9762a35ca80d564f6594686463c65b0ffaedcfa4",
			"4331ca28abf6c18dc69e2a719564dfe4f149a359"
		},
	};

	{
		SCOPED_TRACE("rgb->709");
		test_case({ MatrixCoefficients::RGB, TransferCharacteristics::UNSPECIFIED, ColorPrimaries::UNSPECIFIED },
				  { MatrixCoefficients::REC_709, TransferCharacteristics::UNSPECIFIED, ColorPrimaries::UNSPECIFIED },
		          expected_sha1[0]);
	}
	{
		SCOPED_TRACE("rgb->709 (derived)");
		test_case({ MatrixCoefficients::RGB, TransferCharacteristics::REC_709, ColorPrimaries::REC_709 },
				  { MatrixCoefficients::CHROMATICITY_DERIVED_NCL, TransferCharacteristics::REC_709, ColorPrimaries::REC_709 },
		          expected_sha1[0]);
	}
	{
		SCOPED_TRACE("709->rgb");
		test_case({ MatrixCoefficients::REC_709, TransferCharacteristics::UNSPECIFIED, ColorPrimaries::UNSPECIFIED },
				  { MatrixCoefficients::RGB, TransferCharacteristics::UNSPECIFIED, ColorPrimaries::UNSPECIFIED },
		          expected_sha1[1]);
	}
	{
		SCOPED_TRACE("709->rgb (derived)");
		test_case({ MatrixCoefficients::CHROMATICITY_DERIVED_NCL, TransferCharacteristics::REC_709, ColorPrimaries::REC_709 },
				  { MatrixCoefficients::RGB, TransferCharacteristics::REC_709, ColorPrimaries::REC_709 },
		          expected_sha1[1]);
	}
	{
		SCOPED_TRACE("601->709");
		test_case({ MatrixCoefficients::REC_601, TransferCharacteristics::UNSPECIFIED, ColorPrimaries::UNSPECIFIED },
				  { MatrixCoefficients::REC_709, TransferCharacteristics::UNSPECIFIED, ColorPrimaries::UNSPECIFIED },
		          expected_sha1[2]);
	}
	{
		SCOPED_TRACE("smpte_c->rgb (derived)");
		test_case({ MatrixCoefficients::CHROMATICITY_DERIVED_NCL, TransferCharacteristics::REC_709, ColorPrimaries::SMPTE_C },
				  { MatrixCoefficients::RGB, TransferCharacteristics::REC_709, ColorPrimaries::SMPTE_C },
		          expected_sha1[3]);
	}
	{
		SCOPED_TRACE("rgb->smpte_c (derived)");
		test_case({ MatrixCoefficients::RGB, TransferCharacteristics::REC_709, ColorPrimaries::SMPTE_C },
				  { MatrixCoefficients::CHROMATICITY_DERIVED_NCL, TransferCharacteristics::REC_709, ColorPrimaries::SMPTE_C },
		          expected_sha1[4]);
	}
}

TEST(ColorspaceConversionTest, test_transfer_only)
{
	using namespace zimg::colorspace;

	ColorspaceDefinition csp_linear{ MatrixCoefficients::RGB, TransferCharacteristics::LINEAR, ColorPrimaries::UNSPECIFIED };
	ColorspaceDefinition csp_gamma{ MatrixCoefficients::RGB, TransferCharacteristics::REC_709, ColorPrimaries::UNSPECIFIED };
	ColorspaceDefinition csp_st2084{ MatrixCoefficients::RGB, TransferCharacteristics::ST_2084, ColorPrimaries::UNSPECIFIED };

	static const char *expected_sha1[][3] = {
		{
			"de9ee500de2e6b83642248d48b9bfef0785b6a79",
			"b04e69753e756e365d74f92d2fc625c8d286fc0f",
			"8d4a41d7d010327e542a6b7f66f3c44f85ea1392"
		},
		{
			"887e4db52b882cdb3e37333bb5c3dfce731724e0",
			"c3bf38e1f5e8f27b7cb562a8668df068e4d29d5c",
			"7543357d80e15f6258ea1de7101affffd0434c5a"
		},
		{
			"7c0ff6a1eb9f88d6fb434d859f355a3068a9d039",
			"4a04bc65d5fbbb76e49767c5c29a818a33ee929a",
			"80f10329386f198eca239b987f0af46cac4de42a"
		},
		{
			"d031d2e4e94201ffa613aa160270e2e58e420c5d",
			"e2cfee25339fd938cb90d9fa4d667677ce1e90ce",
			"9ae597e16b49fa7956ec9fd6b7d439a570bda631"
		},
	};

	{
		SCOPED_TRACE("gamma->linear");
		test_case(csp_gamma, csp_linear, expected_sha1[0]);
	}
	{
		SCOPED_TRACE("st2084->linear");
		test_case(csp_st2084, csp_linear, expected_sha1[1]);
	}
	{
		SCOPED_TRACE("linear->gamma");
		test_case(csp_linear, csp_gamma, expected_sha1[2]);
	}
	{
		SCOPED_TRACE("linear->st2084");
		test_case(csp_linear, csp_st2084, expected_sha1[3]);
	}
}

TEST(ColorspaceConversionTest, test_transfer_only_b67)
{
	using namespace zimg::colorspace;

	ColorspaceDefinition csp_linear{ MatrixCoefficients::RGB, TransferCharacteristics::LINEAR, ColorPrimaries::UNSPECIFIED };
	ColorspaceDefinition csp_arib_b67{ MatrixCoefficients::RGB, TransferCharacteristics::ARIB_B67, ColorPrimaries::UNSPECIFIED };

	ColorspaceDefinition csp_linear_2020 = csp_linear.to(ColorPrimaries::REC_2020);
	ColorspaceDefinition csp_arib_b67_2020 = csp_arib_b67.to(ColorPrimaries::REC_2020);

	static const char *expected_sha1[][3] = {
		{
			"6cd027d113acb5ea5574ae4314e079b384db3c79",
			"9f65866d8bfaf552318457c4c90361a0fd58cba8",
			"92029f0bfd0435d455ca0943fe45a51460e5d7c3"
		},
		{
			"5f27412b2c502c753a7169c05ef0fb72c6f4eb3f",
			"6081cdca417c06cb902085dd5ee2a3569cb34b6b",
			"aa0b149fb14637713bc2f9ebd043ae95eed98b80"
		},
		{
			"f99f91c4d63699b0e716dad5d97b9a907260d82d",
			"81a6e91352da1397e36c778d531672a6ffcd88a0",
			"ff75325792f4066b790acd15f6d9dbe4aca6da86"
		},
		{
			"f56f644759e84a7dee94cdee52ff3294cac3fd28",
			"eca0081b690e17d9c2ea90d84b6df45daefc3d4c",
			"2e0d25b27ecf4f847e1849a1ae75a5172d6f1c52"
		},
	};

	{
		SCOPED_TRACE("b67->linear");
		test_case(csp_arib_b67, csp_linear, expected_sha1[0]);
	}
	{
		SCOPED_TRACE("b67->linear (2020)");
		test_case(csp_arib_b67_2020, csp_linear_2020, expected_sha1[1]);
	}
	{
		SCOPED_TRACE("linear->arib_b67");
		test_case(csp_linear, csp_arib_b67, expected_sha1[2]);
	}
	{
		SCOPED_TRACE("linear->arib_b67 (2020)");
		test_case(csp_linear_2020, csp_arib_b67_2020, expected_sha1[3]);
	}
}

TEST(ColorspaceConversionTest, test_matrix_transfer)
{
	using namespace zimg::colorspace;

	static const char *expected_sha1[3] = {
		"c5d494e4c8fefcb2a7978887514782aca1d150df",
		"fb54712faf91d6f94d71cbbdc744d4a1f1d5eee5",
		"267c0f231918f23cacf3acb8b2ac92b301510cc8"
	};

	test_case({ MatrixCoefficients::REC_709, TransferCharacteristics::LINEAR, ColorPrimaries::UNSPECIFIED },
	          { MatrixCoefficients::REC_601, TransferCharacteristics::REC_709, ColorPrimaries::UNSPECIFIED },
	          expected_sha1);
}

TEST(ColorspaceConversionTest, test_matrix_transfer_primaries)
{
	using namespace zimg::colorspace;

	static const char *expected_sha1[][3] = {
		{
			"785701962fcdadbd03be31c2376af08ae2353c45",
			"443268fb0971cdff694265263efc8d8d821b2fd1",
			"ac689fc4cb452178d061fbcb66c04366ec80f0d7"
		},
		{
			"1cc7bc8ecd85b29c68d5c07a7ea3eef293027ce5",
			"1343bb8e12fd879c74c7277339cdc7045dcc843f",
			"280588a9c4482be507a7c06340cff993509b24ed"
		},
		{
			"4aef7e59f2f81391ad47a7db1a2ad63d99cb082e",
			"8674cadfe701667f35517cf57f8f6b785f51ec1c",
			"1933b6721e782b7517fe4f06f3ee80d5402dfbbb"
		},
		{
			"4971289327f7bc4272d517135bca80e84e9f0d5c",
			"24bf2884101730eab35134cd5d2b05daf578a3b5",
			"b05ed2186e3c810477bd12db5851b14a2a8205b8"
		},
	};

	{
		SCOPED_TRACE("709->smpte_c");
		test_case({ MatrixCoefficients::REC_709, TransferCharacteristics::REC_709, ColorPrimaries::REC_709 },
		          { MatrixCoefficients::REC_601, TransferCharacteristics::REC_709, ColorPrimaries::SMPTE_C },
		          expected_sha1[0]);
	}
	{
		SCOPED_TRACE("709->2020");
		test_case({ MatrixCoefficients::REC_709, TransferCharacteristics::REC_709, ColorPrimaries::REC_709 },
		          { MatrixCoefficients::REC_2020_NCL, TransferCharacteristics::REC_709, ColorPrimaries::REC_2020 },
		          expected_sha1[1]);
	}
	{
		SCOPED_TRACE("709->p3d65");
		test_case({ MatrixCoefficients::REC_709, TransferCharacteristics::REC_709, ColorPrimaries::REC_709 },
		          { MatrixCoefficients::REC_709, TransferCharacteristics::REC_709, ColorPrimaries::DCI_P3_D65 },
		          expected_sha1[2]);
	}
	{
		SCOPED_TRACE("p3->p3d65");
		test_case({ MatrixCoefficients::REC_709, TransferCharacteristics::REC_709, ColorPrimaries::DCI_P3 },
		          { MatrixCoefficients::REC_709, TransferCharacteristics::REC_709, ColorPrimaries::DCI_P3_D65 },
		          expected_sha1[3]);
	}
}

TEST(ColorspaceConversionTest, test_constant_luminance)
{
	using namespace zimg::colorspace;

	ColorspaceDefinition csp_2020cl{ MatrixCoefficients::REC_2020_CL, TransferCharacteristics::REC_709, ColorPrimaries::REC_2020 };

	static const char *expected_sha1[][3] = {
		{
			"7470a448b91c0f239f4d48e3064767bc91d5f537",
			"68867bddd3d5fd26234d48cd2fc27f7ad72ed2a6",
			"922731ac779576582285ec37220a9f18be0a9bc3"
		},
		{
			"5267c511acc7a8e27e0032c66399c0124d5b4583",
			"656c0b051cedb2669fb9d7b79ecd619db952770d",
			"fb421c54d78240d73add26ed8e61b5739fdbb7df"
		},
		{
			"24bf088f24c11ae113f48ef1e101116973e38b5b",
			"b611cc18391b5b2621541a68ab9e6a786f32073f",
			"24fb9d4a8d8fcd7543878846b2b19a306fbd520d"
		},
		{
			"a5e14c950bb6fd2e28ec709fa73f4b1b930ec3dd",
			"6dcaa2414dd4b786b5d997002b8b212a1e5ac0d6",
			"7084215715115154c799486443d2283676232d22"
		},
		{
			"2ef93e9b37320ba08929f3639de647301fd9c423",
			"75168e69370b2ac628f87afdc2c34c627ae19135",
			"a948d3f33a796b133b50701e280a241bcc3d0051"
		},
		{
			"3cd60ee07738ca3969ef0c07a9da003a5c3ced61",
			"df16146e6e7794529bb738d3b02b12f6627a660e",
			"271147942457d4f48837461070617d486e99f13f"
		},
	};

	{
		SCOPED_TRACE("2020cl->rgb");
		test_case(csp_2020cl, csp_2020cl.to_rgb(), expected_sha1[0]);
	}
	{
		SCOPED_TRACE("2020cl->rgb (derived)");
		test_case(csp_2020cl.to(MatrixCoefficients::CHROMATICITY_DERIVED_CL), csp_2020cl.to_rgb(), expected_sha1[0]);
	}
	{
		SCOPED_TRACE("rgb->2020cl");
		test_case(csp_2020cl.to_rgb(), csp_2020cl, expected_sha1[1]);
	}
	{
		SCOPED_TRACE("rgb->2020cl (derived)");
		test_case(csp_2020cl.to_rgb(), csp_2020cl.to(MatrixCoefficients::CHROMATICITY_DERIVED_CL), expected_sha1[1]);
	}
	{
		SCOPED_TRACE("2020cl->2020ncl");
		test_case(csp_2020cl, csp_2020cl.to(MatrixCoefficients::REC_2020_NCL), expected_sha1[2]);
	}
	{
		SCOPED_TRACE("2020ncl->2020cl");
		test_case(csp_2020cl.to(MatrixCoefficients::REC_2020_NCL), csp_2020cl, expected_sha1[3]);
	}
	{
		SCOPED_TRACE("709cl->linear_rgb (derived)");
		test_case({ MatrixCoefficients::CHROMATICITY_DERIVED_CL, TransferCharacteristics::REC_709, ColorPrimaries::REC_709 },
	              { MatrixCoefficients::RGB, TransferCharacteristics::LINEAR, ColorPrimaries::REC_709 },
	              expected_sha1[4]);
	}
	{
		SCOPED_TRACE("linear_rgb->709cl (derived)");
		test_case({ MatrixCoefficients::RGB, TransferCharacteristics::LINEAR, ColorPrimaries::REC_709 },
	              { MatrixCoefficients::CHROMATICITY_DERIVED_CL, TransferCharacteristics::REC_709, ColorPrimaries::REC_709 },
		          expected_sha1[5]);
	}
}

TEST(ColorspaceConversionTest, test_rec2100_ictcp)
{
	using namespace zimg::colorspace;

	static const char *expected_sha1[][3] = {
		{
			"aa110d2d0de2b2c462c2317ff204d7735cf4b4f8",
			"2a53f8c677e4f83c9318333991b5700798e5da2b",
			"e855768b63061b1ac3ab87298c4e344721450ccb"
		},
		{
			"8b45d71051b82a337242c56e0ce462e06603effa",
			"c2d8a905f037da5a97b0eef1778349e2ac4cd453",
			"1c5a5a64f5230d332799f725f7addf2ee713145f"
		},
		{
			"e0394f90fd58f793e669b97c460caac5f2645169",
			"8c77d241c4b88c267dd3167eb9235f1576a2acb9",
			"7b63e093003bfbc1951804c6acbcbfd99c3d05f6"
		},
		{
			"bbafa5f4660daa3e644865a648af6bd7a69a16a0",
			"8e04e05bfad949cf801af86ea97c093c59c83c19",
			"b82946433d407a5ceee0a06f0ef8d6c951509bbb"
		},
	};

	{
		SCOPED_TRACE("linear rgb->st2084 ictcp");
		test_case({ MatrixCoefficients::RGB, TransferCharacteristics::LINEAR, ColorPrimaries::REC_2020 },
		          { MatrixCoefficients::REC_2100_ICTCP, TransferCharacteristics::ST_2084, ColorPrimaries::REC_2020 },
	          expected_sha1[0]);
	}
	{
		SCOPED_TRACE("linear rgb->b67 ictcp");
		test_case({ MatrixCoefficients::RGB, TransferCharacteristics::LINEAR, ColorPrimaries::REC_2020 },
		          { MatrixCoefficients::REC_2100_ICTCP, TransferCharacteristics::ARIB_B67, ColorPrimaries::REC_2020 },
	          expected_sha1[1]);
	}
	{
		SCOPED_TRACE("st2084 ictcp->linear rgb");
		test_case({ MatrixCoefficients::REC_2100_ICTCP, TransferCharacteristics::ST_2084, ColorPrimaries::REC_2020 },
		          { MatrixCoefficients::RGB, TransferCharacteristics::LINEAR, ColorPrimaries::REC_2020 },
		          expected_sha1[2]);
	}
	{
		SCOPED_TRACE("b67 ictcp->linear rgb");
		test_case({ MatrixCoefficients::REC_2100_ICTCP, TransferCharacteristics::ARIB_B67, ColorPrimaries::REC_2020 },
		          { MatrixCoefficients::RGB, TransferCharacteristics::LINEAR, ColorPrimaries::REC_2020 },
		          expected_sha1[3]);
	}
}

