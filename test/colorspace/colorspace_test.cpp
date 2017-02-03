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
			"b606c8645f868f1e763b3e6a88b16bd004db998a",
			"1020e7d1b081bde7ca05ba61656811aeead95f01",
			"4e74929009d94738f47c47001c54c2293d58f03d"
		},
		{
			"72263d2c4e701fad7e19a98f7d4a6fd12c97f237",
			"9aad4d81bd3ac29a2d647218b91b9e3bb6b031b9",
			"a13696a7a2931b3ba549ef50ed061386f23ee354"
		},
		{
			"4c190cb9ddfef9a6e220c61cb2e480c69a8f0b4d",
			"17e41c66ddd3eb07a4eee31b9bb6e6c717cbd92f",
			"19aa1aa5d54b0231bfe7ec03caede636f5dfc429"
		},
		{
			"d39fa08fda52893d294c2bf3c6563bc3035392a9",
			"e99ba9e53c3b43e5babb580c279b2d1558a6ffa0",
			"790eb9960fd57ff146029a0783b033e2bbdbd836"
		},
		{
			"4c62e5d775548495a170b6876a2e91b00d4b5f14",
			"90eae848b7050edf12ca22f57bda4eeccad8d7ef",
			"e2dc601f663ea61899f37a9db1b50b5e4110a38e"
		},
		{
			"e142f8335b67b30acaa562fa84b0fd79aeea86b8",
			"48459e4fc6b1c93df3cad0642a2dc60232eeac2a",
			"a61729320be4d2232ea08ca45d1b0cff0e3e5dda"
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
		"55cce8cf81b5cb0fd6c88744ee3f8bd76ea1928d",
		"b707cf6b81449aa34165a47aa4dc9e3bee784bdd",
		"508318c1837019b91880df83cb7f9d110548e30b"
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
			"211a5883cf7ce891c69d87420d6a3b827429d901",
			"a7f3794f55bf613915cd853b93950e514a619759",
			"08b8786d953c752b796a3e3222dbc587ebefd75e"
		},
		{
			"f171d249eb0cb409dd9b39fd547b556032616130",
			"ba3b6ad3789627a6c91e77c83d921714b5a989af",
			"f336a1b1fad140166fb366763e3104a6898043c7"
		},
		{
			"9196061dd36c8455a10b68767f8baad59280ef1b",
			"14f947d077d545930765b457e767da2852186180",
			"0894c577a74198718602d4fd7449d2f3ce5ecf28"
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
			"d70224e0034240051238069d5e6592ffd726f779",
			"2520e0d9dcce7be837159b92107c2e9f6a0c78dc",
			"2f0ac11e5be3fac4dbb0858d3de22f08deddbaf5"
		},
		{
			"ebc7919d4d491c5d98c27793974d6f5e37ecb906",
			"df01a3bd7bd43812c9f56e4be8abbcfa23c729e7",
			"41a894bc2bc0f40243d3b5397c9d99a522e97ffd"
		},
		{
			"3c278d7effe09b339cfc9da123a1bff7c662b174",
			"ea7d3047bc0b892eab63387af233e2bca6d7bdbf",
			"13acf6a2323436f653947d8a763c06784301554d"
		},
		{
			"e66509c42232798a6e4e885d03d5abda39d01669",
			"a1c19f71619ca5484645bd607ee19ab0a194268a",
			"66e5f36240c45413b1519efc02c64907ec5bc287"
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
