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
		"7e524dfcdf751f628d32c2f511a1c44247e96691",
		"40cc64287b861f0a55268ee0a4698689a48bb65c",
		"c3f0840878fe4d267bac85f21778ba293f494cf2"
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
			"23704c9a7da6e06cdc479ef2027c72b66bb7d96a",
			"0fe8a961ec40272e73b5a9fa7d4827c927aad8cc",
			"d7542d498a9942e767cd6c64c0a463d34fde8614"
		},
		{
			"9915126526bff4830a0060176f1ad0ae8241c2c6",
			"6e252e9ee7e287dcd5a31cc39b1fb6934a4c2f0a",
			"72f558ba789b112ff64136007218124b6f26c144"
		},
		{
			"54bcee0dbf129d8e8647b5686f61f6e5fd5a154e",
			"6743f2845f8025105cf5ce30c68cde578af5ddfb",
			"5a20b8900a6f388ea1aad469a8c1e77595b9732c"
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
			"5088f01f06b08983b7cb08b6a00532eb2084f505",
			"ad891e72a53caa3d1b597ea653d16dc08c1dd5f2",
			"a08f04519a7a11c6eeab86a9a5bd0c6b5989eda1"
		},
		{
			"ebc7919d4d491c5d98c27793974d6f5e37ecb906",
			"df01a3bd7bd43812c9f56e4be8abbcfa23c729e7",
			"41a894bc2bc0f40243d3b5397c9d99a522e97ffd"
		},
		{
			"8f8479117084312cb3e0dc74d0c0600b284615af",
			"71f9932b2e992836c7c16671dbf0648c0e847f79",
			"8d51f73a86f8b04b7f781db8932260d35e96349a"
		},
		{
			"7d6b3cdfad24dc8693dd30f8cc50740888c426e3",
			"dfb7b58f0757078c71b631ab73496d1f5e7593e0",
			"e8b545d4ff09ad9da6dcf8122d34de8391658362"
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
