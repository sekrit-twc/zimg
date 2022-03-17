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

	FilterValidator validator{ convert.get(), w, h, format };
	validator.set_sha1(expected_sha1)
	         .set_yuv(csp_in.matrix != zimg::colorspace::MatrixCoefficients::RGB)
	         .validate();
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
			"7a4acfb71b940364d841d02db7625e2a4cfdd4f1",
			"50caf91cb19a7cb5a42dfa1df53ffb3210a03385",
			"99303773321c293b06ec49626c4753c6931f93b0"
		},
		{
			"e73f69c76f158fb4d461d2d5ecd946096116f2b5",
			"46aa18f98fa9ffed9bea83817a76cacaf28ed062",
			"510cff5b3f97e26874cec2b8a986d0a205742b2a"
		},
		{
			"83e39222105eab15a79601f196d491e993da2cd6",
			"a5d1b53d1833d59f2a63193005a55bffdebc449c",
			"96521fc8c82677ab0a74a9fb47e1259ef93fe758"
		},
		{
			"5d452fb5eeb54741e5dcf206295e43aa4f26b44a",
			"64e60f652a18a8bbbc99ade0f3105ba00a2d2f25",
			"7eeba07cef9d99bb1876454c0b812dc7df56bdec"
		},
	};

	SCOPED_TRACE("rgb->709");
	test_case({ MatrixCoefficients::RGB, TransferCharacteristics::UNSPECIFIED, ColorPrimaries::UNSPECIFIED },
	          { MatrixCoefficients::REC_709, TransferCharacteristics::UNSPECIFIED, ColorPrimaries::UNSPECIFIED },
	          expected_sha1[0]);
	SCOPED_TRACE("rgb->709 (derived)");
	test_case({ MatrixCoefficients::RGB, TransferCharacteristics::REC_709, ColorPrimaries::REC_709 },
	          { MatrixCoefficients::CHROMATICITY_DERIVED_NCL, TransferCharacteristics::REC_709, ColorPrimaries::REC_709 },
	          expected_sha1[0]);
	SCOPED_TRACE("709->rgb");
	test_case({ MatrixCoefficients::REC_709, TransferCharacteristics::UNSPECIFIED, ColorPrimaries::UNSPECIFIED },
	          { MatrixCoefficients::RGB, TransferCharacteristics::UNSPECIFIED, ColorPrimaries::UNSPECIFIED },
	          expected_sha1[1]);
	SCOPED_TRACE("709->rgb (derived)");
	test_case({ MatrixCoefficients::CHROMATICITY_DERIVED_NCL, TransferCharacteristics::REC_709, ColorPrimaries::REC_709 },
	          { MatrixCoefficients::RGB, TransferCharacteristics::REC_709, ColorPrimaries::REC_709 },
	          expected_sha1[1]);
	SCOPED_TRACE("601->709");
	test_case({ MatrixCoefficients::REC_601, TransferCharacteristics::UNSPECIFIED, ColorPrimaries::UNSPECIFIED },
	          { MatrixCoefficients::REC_709, TransferCharacteristics::UNSPECIFIED, ColorPrimaries::UNSPECIFIED },
	          expected_sha1[2]);
	SCOPED_TRACE("smpte_c->rgb (derived)");
	test_case({ MatrixCoefficients::CHROMATICITY_DERIVED_NCL, TransferCharacteristics::REC_709, ColorPrimaries::SMPTE_C },
	          { MatrixCoefficients::RGB, TransferCharacteristics::REC_709, ColorPrimaries::SMPTE_C },
	          expected_sha1[3]);
	SCOPED_TRACE("rgb->smpte_c (derived)");
	test_case({ MatrixCoefficients::RGB, TransferCharacteristics::REC_709, ColorPrimaries::SMPTE_C },
	          { MatrixCoefficients::CHROMATICITY_DERIVED_NCL, TransferCharacteristics::REC_709, ColorPrimaries::SMPTE_C },
	          expected_sha1[4]);
}

TEST(ColorspaceConversionTest, test_transfer_only)
{
	using namespace zimg::colorspace;

	ColorspaceDefinition csp_linear{ MatrixCoefficients::RGB, TransferCharacteristics::LINEAR, ColorPrimaries::UNSPECIFIED };
	ColorspaceDefinition csp_gamma{ MatrixCoefficients::RGB, TransferCharacteristics::REC_709, ColorPrimaries::UNSPECIFIED };
	ColorspaceDefinition csp_st2084{ MatrixCoefficients::RGB, TransferCharacteristics::ST_2084, ColorPrimaries::UNSPECIFIED };

	const char *expected_sha1[][3] = {
		{
			"de9ee500de2e6b83642248d48b9bfef0785b6a79",
			"95409014b8206d6d513dbadc22320afd8cc28c86",
			"b1044c1f9e8986c419dc6e239f93c5cff994e1f2"
		},
		{
			"887e4db52b882cdb3e37333bb5c3dfce731724e0",
			"ed3267eaaced310988a8e1b7f07bde2891a395e1",
			"5ccd19037d075ba5bba2b1046955aac4fe8e2b04"
		},
		{
			"7c0ff6a1eb9f88d6fb434d859f355a3068a9d039",
			"16ae1b7f522b4e5529832307476a3ab59125a70d",
			"458037dddaddabffcb6d451ebe27e86673935e0a"
		},
		{
			"d031d2e4e94201ffa613aa160270e2e58e420c5d",
			"2e0f732a848310d9c6fed7473e1056410d5a6e48",
			"180eabe55ef2c086ce0d744a8fb92326b3caa682"
		},
	};

	SCOPED_TRACE("gamma->linear");
	test_case(csp_gamma, csp_linear, expected_sha1[0]);

	SCOPED_TRACE("st2084->linear");
	test_case(csp_st2084, csp_linear, expected_sha1[1]);

	SCOPED_TRACE("linear->gamma");
	test_case(csp_linear, csp_gamma, expected_sha1[2]);

	SCOPED_TRACE("linear->st2084");
	test_case(csp_linear, csp_st2084, expected_sha1[3]);
}

TEST(ColorspaceConversionTest, test_transfer_only_b67)
{
	using namespace zimg::colorspace;

	ColorspaceDefinition csp_linear{ MatrixCoefficients::RGB, TransferCharacteristics::LINEAR, ColorPrimaries::UNSPECIFIED };
	ColorspaceDefinition csp_arib_b67{ MatrixCoefficients::RGB, TransferCharacteristics::ARIB_B67, ColorPrimaries::UNSPECIFIED };

	ColorspaceDefinition csp_linear_2020 = csp_linear.to(ColorPrimaries::REC_2020);
	ColorspaceDefinition csp_arib_b67_2020 = csp_arib_b67.to(ColorPrimaries::REC_2020);

	const char *expected_sha1[][3] = {
		{
			"6cd027d113acb5ea5574ae4314e079b384db3c79",
			"76f47b9bd1d9199799c36fff9a3ced6c629f8340",
			"8f4fe7c710068ad87a42347b8faa57005d04840f"
		},
		{
			"4a8e904520a84525c5c22b76d211d1fab11022c0",
			"1f8b365c300b43a6c17b5d0d6b16b70b622983ac",
			"51f14032ac6a48411d45173c0620a7c2c314a660"
		},
		{
			"f99f91c4d63699b0e716dad5d97b9a907260d82d",
			"55a1bc59a5f913b4bc304940290ccd0283a2296a",
			"3d73118ba8ba2451d6a5ba625e9ae70a79200070"
		},
		{
			"d3403c044d17191d5dadb387fcf2f1e0d633d5e6",
			"3b1694b5e2bea4c0cd3151bfe2d5a3d55f00b013",
			"31e0320e73da998c2fbf5bfb5f676b6b94b8a770"
		},
	};

	SCOPED_TRACE("b67->linear");
	test_case(csp_arib_b67, csp_linear, expected_sha1[0]);

	SCOPED_TRACE("b67->linear (2020)");
	test_case(csp_arib_b67_2020, csp_linear_2020, expected_sha1[1]);

	SCOPED_TRACE("linear->arib_b67");
	test_case(csp_linear, csp_arib_b67, expected_sha1[2]);

	SCOPED_TRACE("linear->arib_b67 (2020)");
	test_case(csp_linear_2020, csp_arib_b67_2020, expected_sha1[3]);
}

TEST(ColorspaceConversionTest, test_matrix_transfer)
{
	using namespace zimg::colorspace;

	const char *expected_sha1[3] = {
		"8ae350623a239cf75ab0eccdfde6d34b83b9d4e9",
		"ed627b2534efcb986434df8def1b11d78501d022",
		"b3c76e1eb49193987511a6c81765a1c416e19807"
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
			"634e7c12ca6f2b35098575031679dc9d4f6dc4d6",
			"dfff56afa7a10a50c61be104dcbe9e1388db25ec",
			"43dcab4ad5af0f002b38b76fad020f1bc89f7e27"
		},
		{
			"fd97f472f1b5fe4b8c286dc9c7aa7486bf4d57f3",
			"fbaa5b42057fecd15c32cc9f4f6aba75644f84b3",
			"1fe474ac70aa08480fc1aefd2ee34028b41be316"
		},
		{
			"276e70ed99be30f0d28a182ac5168b5aff1924f7",
			"9d8f38681e2e530a5a6c8f57b643f6bebaeb5767",
			"bcaba405eb4022ad0beb9ce4efe47fe3964f6519"
		},
		{
			"daa76f7089d1021ddf5ebb6491140b90dd28a000",
			"2e8161dcf02a79c054f26b2c4982811e0bec6e43",
			"b64198f1f920d5b53b8d0fdd4692c9dad437991d"
		},
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

	SCOPED_TRACE("p3->p3d65");
	test_case({ MatrixCoefficients::REC_709, TransferCharacteristics::REC_709, ColorPrimaries::DCI_P3 },
	          { MatrixCoefficients::REC_709, TransferCharacteristics::REC_709, ColorPrimaries::DCI_P3_D65 },
	          expected_sha1[3]);
}

TEST(ColorspaceConversionTest, test_constant_luminance)
{
	using namespace zimg::colorspace;

	ColorspaceDefinition csp_2020cl{ MatrixCoefficients::REC_2020_CL, TransferCharacteristics::REC_709, ColorPrimaries::REC_2020 };

	const char *expected_sha1[][3] = {
		{
			"bc16780aa8113623f84d94923147829076c2940d",
			"bce4c7158a86657a745fdbae54369cd29b2d9673",
			"a90c32990e9067c214da18ea59c8f4f8ee4d5044"
		},
		{
			"6d6ee771397e6c1a37222d3770fbe91cf413e298",
			"573a964fc1a05f5e8b89e6ef8baa72de36ee1be5",
			"8a81ef90c2f6fe1c2aa0014971e94f33b50b7933"
		},
		{
			"604dede63242b76005f49504e2b551d0400d12e4",
			"04ef8fd6cb80b7e27ef0a40ffac238652c9a32c9",
			"6f2f4bd611f5874387e6d420544ed4114f333703"
		},
		{
			"d0ad093619b6ab139374b65a176d7bbaf6671c6b",
			"b93460065a3287de0f8d3f1dcc4819ada661e258",
			"16c45e2d234cf0aee1fb180928e2f757d3ccb245"
		},
		{
			"f3168488c8de035f93cc1c80d02a8bd656f8131a",
			"52094ce2ef745d1a5adee337ce11e4329c8668c7",
			"5e3a0a16dfbe0587df7f07675e5899ce8be9f75d"
		},
		{
			"5f3ff22abf441d3276203390c4d25ff42200a4a2",
			"4023ac08b9631ba9882b18e30509ffee40bddf1e",
			"2e7af737cd42a5d38ced36288285d614f2d5d2ad"
		},
	};

	SCOPED_TRACE("2020cl->rgb");
	test_case(csp_2020cl, csp_2020cl.to_rgb(), expected_sha1[0]);

	SCOPED_TRACE("2020cl->rgb (derived)");
	test_case(csp_2020cl.to(MatrixCoefficients::CHROMATICITY_DERIVED_CL), csp_2020cl.to_rgb(), expected_sha1[0]);

	SCOPED_TRACE("rgb->2020cl");
	test_case(csp_2020cl.to_rgb(), csp_2020cl, expected_sha1[1]);

	SCOPED_TRACE("rgb->2020cl (derived)");
	test_case(csp_2020cl.to_rgb(), csp_2020cl.to(MatrixCoefficients::CHROMATICITY_DERIVED_CL), expected_sha1[1]);

	SCOPED_TRACE("2020cl->2020ncl");
	test_case(csp_2020cl, csp_2020cl.to(MatrixCoefficients::REC_2020_NCL), expected_sha1[2]);

	SCOPED_TRACE("2020ncl->2020cl");
	test_case(csp_2020cl.to(MatrixCoefficients::REC_2020_NCL), csp_2020cl, expected_sha1[3]);

	SCOPED_TRACE("709cl->linear_rgb (derived)");
	test_case({ MatrixCoefficients::CHROMATICITY_DERIVED_CL, TransferCharacteristics::REC_709, ColorPrimaries::REC_709 },
	          { MatrixCoefficients::RGB, TransferCharacteristics::LINEAR, ColorPrimaries::REC_709 },
	          expected_sha1[4]);

	SCOPED_TRACE("linear_rgb->709cl (derived)");
	test_case({ MatrixCoefficients::RGB, TransferCharacteristics::LINEAR, ColorPrimaries::REC_709 },
	          { MatrixCoefficients::CHROMATICITY_DERIVED_CL, TransferCharacteristics::REC_709, ColorPrimaries::REC_709 },
	          expected_sha1[5]);
}

TEST(ColorspaceConversionTest, test_rec2100_ictcp)
{
	using namespace zimg::colorspace;

	const char *expected_sha1[][3] = {
		{
			"6034b9f906877163406be7647013d9d62e86c18e",
			"8b77ba1058bc9b561af346a31e0c82c8f9a68e4a",
			"e4db705d18d07f8ca1f4f1c557b0ae6858af86d3"
		},
		{
			"e63f1eaca56682952511c1c1e028c2f3e1aaecd9",
			"aa6c76a7bdb801d1f75dbbf8412bd18f9782877f",
			"139711f578d484b26f99136169aa40419180b64a"
		},
		{
			"f85035f377c130b174adeb339ef0bb1b2e661e52",
			"1332c6bcf8fa47dc2216e1a6162d4401f92f8f02",
			"409008e215b9533e633a6127186f46d6180cfbe8"
		},
		{
			"5b2096d9c7e3432d9b6d2221aece753473ff97b7",
			"6be8fab629eab38d95b086798d53e8c1542a1d76",
			"e451a6ff61eefb459acc2426f03f7837a3d6ca73"
		},
	};

	SCOPED_TRACE("linear rgb->st2084 ictcp");
	test_case({ MatrixCoefficients::RGB, TransferCharacteristics::LINEAR, ColorPrimaries::REC_2020 },
	          { MatrixCoefficients::REC_2100_ICTCP, TransferCharacteristics::ST_2084, ColorPrimaries::REC_2020 },
	          expected_sha1[0]);
	SCOPED_TRACE("linear rgb->b67 ictcp");
	test_case({ MatrixCoefficients::RGB, TransferCharacteristics::LINEAR, ColorPrimaries::REC_2020 },
	          { MatrixCoefficients::REC_2100_ICTCP, TransferCharacteristics::ARIB_B67, ColorPrimaries::REC_2020 },
	          expected_sha1[1]);
	SCOPED_TRACE("st2084 ictcp->linear rgb");
	test_case({ MatrixCoefficients::REC_2100_ICTCP, TransferCharacteristics::ST_2084, ColorPrimaries::REC_2020 },
	          { MatrixCoefficients::RGB, TransferCharacteristics::LINEAR, ColorPrimaries::REC_2020 },
	          expected_sha1[2]);
	SCOPED_TRACE("b67 ictcp->linear rgb");
	test_case({ MatrixCoefficients::REC_2100_ICTCP, TransferCharacteristics::ARIB_B67, ColorPrimaries::REC_2020 },
	          { MatrixCoefficients::RGB, TransferCharacteristics::LINEAR, ColorPrimaries::REC_2020 },
	          expected_sha1[3]);
}
