#include "common/x86/avx512_msvc_compat.h"

#ifdef ZIMG_X86_AVX512

#include <cmath>
#include "common/cpuinfo.h"
#include "common/pixel.h"
#include "common/x86/cpuinfo_x86.h"
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

	if (!zimg::query_x86_capabilities().avx512f) {
		SUCCEED() << "sse not available, skipping";
		return;
	}

	zimg::PixelFormat format = zimg::PixelType::FLOAT;
	auto builder = zimg::colorspace::ColorspaceConversion{ w, h }
		.set_csp_in(csp_in)
		.set_csp_out(csp_out);

	auto filter_c = builder.set_cpu(zimg::CPUClass::NONE).create();
	auto filter_avx512 = builder.set_cpu(zimg::CPUClass::X86_AVX512).create();

	FilterValidator validator{ filter_avx512.get(), w, h, format };
	validator.set_sha1(expected_sha1)
	         .set_ref_filter(filter_c.get(), expected_snr)
	         .set_yuv(csp_in.matrix != zimg::colorspace::MatrixCoefficients::RGB)
	         .validate();
}

} // namespace


TEST(ColorspaceConversionAVX512Test, test_matrix)
{
	using namespace zimg::colorspace;

	const char *expected_sha1[3] = {
		"7b2a05426e2ef61dc6adc16573fca46ea3480256",
		"9c69bc8fa775a8e877e66e79812e9f4c39cec647",
		"6010983126eb3f5ca2dd5c01f4753c0e9f36d0bb"
	};
	const double expected_snr = 120.0;

	test_case({ MatrixCoefficients::RGB, TransferCharacteristics::UNSPECIFIED, ColorPrimaries::UNSPECIFIED },
	          { MatrixCoefficients::REC_709, TransferCharacteristics::UNSPECIFIED, ColorPrimaries::UNSPECIFIED },
	          expected_sha1, expected_snr);
}

#endif // ZIMG_X86_AVX512
