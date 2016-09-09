#ifdef ZIMG_X86

#include <cmath>
#include "common/cpuinfo.h"
#include "common/pixel.h"
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

	if (!zimg::query_x86_capabilities().avx) {
		SUCCEED() << "sse not available, skipping";
		return;
	}

	zimg::PixelFormat format = zimg::PixelType::FLOAT;
	auto builder = zimg::colorspace::ColorspaceConversion{ w, h }
		.set_csp_in(csp_in)
		.set_csp_out(csp_out);

	auto filter_c = builder.set_cpu(zimg::CPUClass::NONE).create();
	auto filter_avx = builder.set_cpu(zimg::CPUClass::X86_AVX).create();

	validate_filter(filter_avx.get(), w, h, format, expected_sha1);
	validate_filter_reference(filter_avx.get(), filter_c.get(), w, h, format, expected_snr);
}

} // namespace


TEST(ColorspaceConversionAVXTest, test_matrix)
{
	using namespace zimg::colorspace;

	const char *expected_sha1[3] = {
		"0495adab9c82d98e73841e229a9b2041838fc0f2",
		"ece7edb1118d4b3063ad80f5d8febb6db7e9633a",
		"73a9ee951c7bde9ae0ada9b90afd1f7ce8b604df"
	};
	const double expected_snr = INFINITY;

	test_case({ MatrixCoefficients::RGB, TransferCharacteristics::UNSPECIFIED, ColorPrimaries::UNSPECIFIED },
	          { MatrixCoefficients::REC_709, TransferCharacteristics::UNSPECIFIED, ColorPrimaries::UNSPECIFIED },
	          expected_sha1, expected_snr);
}

#endif // ZIMG_X86
