#ifdef ZIMG_X86_AVX512

#include <cmath>
#include "colorspace/colorspace.h"
#include "common/cpuinfo.h"
#include "common/pixel.h"
#include "common/x86/cpuinfo_x86.h"
#include "graphengine/filter.h"

#include "gtest/gtest.h"
#include "graphengine/filter_validation.h"

namespace {

void test_case(const zimg::colorspace::ColorspaceDefinition &csp_in, const zimg::colorspace::ColorspaceDefinition &csp_out,
               const char * const expected_sha1[3], double expected_snr)
{
	const unsigned w = 640;
	const unsigned h = 480;

	if (!zimg::query_x86_capabilities().avx512f) {
		SUCCEED() << "avx512 not available, skipping";
		return;
	}

	zimg::PixelFormat format = zimg::PixelType::FLOAT;
	auto builder = zimg::colorspace::ColorspaceConversion{ w, h }
		.set_csp_in(csp_in)
		.set_csp_out(csp_out)
		.set_approximate_gamma(true);

	auto filter_c = builder.set_cpu(zimg::CPUClass::NONE).create_ge();
	auto filter_avx512 = builder.set_cpu(zimg::CPUClass::X86_AVX512).create_ge();

	ASSERT_TRUE(filter_c);
	ASSERT_TRUE(filter_avx512);

	graphengine::FilterValidation(filter_avx512.get(), { w, h, zimg::pixel_size(zimg::PixelType::FLOAT) })
		.set_reference_filter(filter_c.get(), expected_snr)
		.set_input_pixel_format(0, { zimg::pixel_depth(zimg::PixelType::FLOAT), true, false })
		.set_input_pixel_format(1, { zimg::pixel_depth(zimg::PixelType::FLOAT), true, csp_in.matrix != zimg::colorspace::MatrixCoefficients::RGB })
		.set_input_pixel_format(2, { zimg::pixel_depth(zimg::PixelType::FLOAT), true, csp_in.matrix != zimg::colorspace::MatrixCoefficients::RGB })
		.set_output_pixel_format(0, { zimg::pixel_depth(zimg::PixelType::FLOAT), true, false })
		.set_output_pixel_format(1, { zimg::pixel_depth(zimg::PixelType::FLOAT), true, csp_out.matrix != zimg::colorspace::MatrixCoefficients::RGB })
		.set_output_pixel_format(2, { zimg::pixel_depth(zimg::PixelType::FLOAT), true, csp_out.matrix != zimg::colorspace::MatrixCoefficients::RGB })
		.set_sha1(0, expected_sha1[0])
		.set_sha1(1, expected_sha1[1])
		.set_sha1(2, expected_sha1[2])
		.run();
}

} // namespace


TEST(ColorspaceConversionAVX512Test, test_matrix)
{
	using namespace zimg::colorspace;

	static const char *expected_sha1[3] = {
		"749f74428406c019b1b727fa30352fcd1f0141ed",
		"334cfa73375f8afef8423a163f3cff8f8a196762",
		"aa3aab12d52e67b4d6765b4e8c03205a5375d8d9"
	};
	const double expected_snr = 120.0;

	test_case({ MatrixCoefficients::RGB, TransferCharacteristics::UNSPECIFIED, ColorPrimaries::UNSPECIFIED },
	          { MatrixCoefficients::REC_709, TransferCharacteristics::UNSPECIFIED, ColorPrimaries::UNSPECIFIED },
	          expected_sha1, expected_snr);
}

TEST(ColorspaceConversionAVX512Test, test_transfer_rec_1886)
{
	using namespace zimg::colorspace;

	static const char *expected_sha1[][3] = {
		{
			"08d5b0c5299a03d6ca7a477dcf2853ece6f31a96",
			"1f1421c4a1c923f286314bcca9cb3b5d9a6ba4cc",
			"1f64edd1c042f14261b4b6a4d1f6a7eb3aeb32b6"
		},
		{
			"739bebe894129fad33141416ddfda4e3a1574b68",
			"bcff693b60e8285850bc3e534387509e00798666",
			"8e748683546247b1b715d9bff57f1e490d5a778a"
		},
	};
	const double expected_tolinear_snr = 120.0;
	const double expected_togamma_snr = 120.0;

	SCOPED_TRACE("tolinear");
	test_case({ MatrixCoefficients::RGB, TransferCharacteristics::REC_709, ColorPrimaries::UNSPECIFIED },
	          { MatrixCoefficients::RGB, TransferCharacteristics::LINEAR, ColorPrimaries::UNSPECIFIED },
	          expected_sha1[0], expected_tolinear_snr);
	SCOPED_TRACE("togamma");
	test_case({ MatrixCoefficients::RGB, TransferCharacteristics::LINEAR, ColorPrimaries::UNSPECIFIED },
	          { MatrixCoefficients::RGB, TransferCharacteristics::REC_709, ColorPrimaries::UNSPECIFIED },
	          expected_sha1[1], expected_togamma_snr);
}

TEST(ColorspaceConversionAVX512Test, test_transfer_srgb)
{
	using namespace zimg::colorspace;

	static const char *expected_sha1[][3] = {
		{
			"43c2d947ab229997b225ac5ba9d96010048fa895",
			"06c3c947a9b14727ea8f4344550e4a39f1407cc7",
			"37ff78604039771f13bc986031aa1e94bf87828f"
		},
		{
			"c77c34e6bef590d4998160c4d37a466eed692f06",
			"ed265e31b5efbb006cb4abbbbef35fb435800963",
			"b3ff9ddc265623a259c5bef3d365a4cea207484c"
		},
	};
	const double expected_tolinear_snr = 120.0;
	const double expected_togamma_snr = 120.0;

	SCOPED_TRACE("tolinear");
	test_case({ MatrixCoefficients::RGB, TransferCharacteristics::SRGB, ColorPrimaries::UNSPECIFIED },
	          { MatrixCoefficients::RGB, TransferCharacteristics::LINEAR, ColorPrimaries::UNSPECIFIED },
	          expected_sha1[0], expected_tolinear_snr);
	SCOPED_TRACE("togamma");
	test_case({ MatrixCoefficients::RGB, TransferCharacteristics::LINEAR, ColorPrimaries::UNSPECIFIED },
	          { MatrixCoefficients::RGB, TransferCharacteristics::SRGB, ColorPrimaries::UNSPECIFIED },
	          expected_sha1[1], expected_togamma_snr);
}

TEST(ColorspaceConversionAVX512Test, test_transfer_st_2084)
{
	using namespace zimg::colorspace;

	static const char *expected_sha1[][3] = {
		{
			"a0d7b4c6dc3381e8831aa84ab7ba05eb86cafd2b",
			"f40fd965e3938548da9a631099df7338bc3f2722",
			"d8e4067b1d3b0d14a751f3f5dc33412d99f2506a"
		},
		{
			"e64ec0811294da6869b6220ff00c6f7d3dda7399",
			"dd169dbbbe31f2f7041bdf9d2a93b925ad4a9c72",
			"5162af5c0e10a2e46407d2c78e7839dd265974aa"
		},
	};
	const double expected_tolinear_snr = 79.0; // :(
	const double expected_togamma_snr = 100.0;

	SCOPED_TRACE("tolinear");
	test_case({ MatrixCoefficients::RGB, TransferCharacteristics::ST_2084, ColorPrimaries::UNSPECIFIED },
	          { MatrixCoefficients::RGB, TransferCharacteristics::LINEAR, ColorPrimaries::UNSPECIFIED },
	          expected_sha1[0], expected_tolinear_snr);
	SCOPED_TRACE("togamma");
	test_case({ MatrixCoefficients::RGB, TransferCharacteristics::LINEAR, ColorPrimaries::UNSPECIFIED },
	          { MatrixCoefficients::RGB, TransferCharacteristics::ST_2084, ColorPrimaries::UNSPECIFIED },
	          expected_sha1[1], expected_togamma_snr);
}

#endif // ZIMG_X86_AVX512
