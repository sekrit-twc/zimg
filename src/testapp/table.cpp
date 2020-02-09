#include <cmath>
#include "common/cpuinfo.h"
#include "common/pixel.h"
#include "colorspace/colorspace.h"
#include "depth/depth.h"
#include "resize/filter.h"

#include "table.h"

using zimg::CPUClass;
using zimg::PixelType;
using zimg::colorspace::MatrixCoefficients;
using zimg::colorspace::TransferCharacteristics;
using zimg::colorspace::ColorPrimaries;
using zimg::depth::DitherType;

namespace {

typedef std::unique_ptr<zimg::resize::Filter> filter_uptr;

template <class T>
filter_uptr make_filter(double, double)
{
	return filter_uptr{ new T{} };
}

filter_uptr make_bicubic_filter(double b, double c)
{
	b = std::isnan(b) ? zimg::resize::BicubicFilter::DEFAULT_B : b;
	c = std::isnan(c) ? zimg::resize::BicubicFilter::DEFAULT_C : c;

	return filter_uptr{ new zimg::resize::BicubicFilter{ b, c } };
}

filter_uptr make_lanczos_filter(double taps, double)
{
	taps = std::isnan(taps) ? zimg::resize::LanczosFilter::DEFAULT_TAPS : std::max(taps, 1.0);
	return filter_uptr{ new zimg::resize::LanczosFilter{ static_cast<unsigned>(taps) } };
}

filter_uptr make_null_filter(double, double) { return nullptr; }

} // namespace


const zimg::static_string_map<CPUClass, 9> g_cpu_table{
	{ "none", CPUClass::NONE },
	{ "auto", CPUClass::AUTO_64B },
#ifdef ZIMG_X86
	{ "sse",        CPUClass::X86_SSE },
	{ "sse2",       CPUClass::X86_SSE2 },
	{ "avx",        CPUClass::X86_AVX },
	{ "f16c",       CPUClass::X86_F16C },
	{ "avx2",       CPUClass::X86_AVX2 },
	{ "avx512",     CPUClass::X86_AVX512 },
	{ "avx512_clx", CPUClass::X86_AVX512_CLX },
#endif
};

const zimg::static_string_map<PixelType, 4> g_pixel_table{
	{ "byte",  PixelType::BYTE },
	{ "word",  PixelType::WORD },
	{ "half",  PixelType::HALF },
	{ "float", PixelType::FLOAT },
};

const zimg::static_string_map<MatrixCoefficients, 12> g_matrix_table{
	{ "unspec",     MatrixCoefficients::UNSPECIFIED },
	{ "rgb",        MatrixCoefficients::RGB },
	{ "601",        MatrixCoefficients::REC_601 },
	{ "709",        MatrixCoefficients::REC_709 },
	{ "fcc",        MatrixCoefficients::FCC },
	{ "240m",       MatrixCoefficients::SMPTE_240M },
	{ "ycgco",      MatrixCoefficients::YCGCO },
	{ "2020_ncl",   MatrixCoefficients::REC_2020_NCL },
	{ "2020_cl",    MatrixCoefficients::REC_2020_CL },
	{ "chroma_ncl", MatrixCoefficients::CHROMATICITY_DERIVED_NCL },
	{ "chroma_cl",  MatrixCoefficients::CHROMATICITY_DERIVED_CL },
	{ "ictcp",      MatrixCoefficients::REC_2100_ICTCP },
};

const zimg::static_string_map<TransferCharacteristics, 12> g_transfer_table{
	{ "unspec",   TransferCharacteristics::UNSPECIFIED },
	{ "linear",   TransferCharacteristics::LINEAR },
	{ "log100",   TransferCharacteristics::LOG_100 },
	{ "log316",   TransferCharacteristics::LOG_316 },
	{ "240m",     TransferCharacteristics::SMPTE_240M },
	{ "709",      TransferCharacteristics::REC_709 },
	{ "470m",     TransferCharacteristics::REC_470_M },
	{ "470bg",    TransferCharacteristics::REC_470_BG },
	{ "xvycc",    TransferCharacteristics::XVYCC },
	{ "srgb",     TransferCharacteristics::SRGB },
	{ "st_2084",  TransferCharacteristics::ST_2084 },
	{ "arib_b67", TransferCharacteristics::ARIB_B67 },
};

const zimg::static_string_map<ColorPrimaries, 12> g_primaries_table{
	{ "unspec",    ColorPrimaries::UNSPECIFIED },
	{ "470_m",     ColorPrimaries::REC_470_M },
	{ "470_bg",    ColorPrimaries::REC_470_BG },
	{ "smpte_c",   ColorPrimaries::SMPTE_C },
	{ "709",       ColorPrimaries::REC_709 },
	{ "film",      ColorPrimaries::FILM },
	{ "2020",      ColorPrimaries::REC_2020 },
	{ "xyz",       ColorPrimaries::XYZ },
	{ "dcip3",     ColorPrimaries::DCI_P3 },
	{ "dcip3_d65", ColorPrimaries::DCI_P3_D65 },
	{ "jedec_p22", ColorPrimaries::JEDEC_P22 },
};

const zimg::static_string_map<DitherType, 4> g_dither_table{
	{ "none",            DitherType::NONE },
	{ "ordered",         DitherType::ORDERED },
	{ "random",          DitherType::RANDOM },
	{ "error_diffusion", DitherType::ERROR_DIFFUSION },
};

const zimg::static_string_map<std::unique_ptr<zimg::resize::Filter>(*)(double, double), 8> g_resize_table{
	{ "point",    make_filter<zimg::resize::PointFilter> },
	{ "bilinear", make_filter<zimg::resize::BilinearFilter> },
	{ "bicubic",  make_bicubic_filter },
	{ "spline16", make_filter<zimg::resize::Spline16Filter> },
	{ "spline36", make_filter<zimg::resize::Spline36Filter> },
	{ "spline64", make_filter<zimg::resize::Spline64Filter> },
	{ "lanczos",  make_lanczos_filter },
	{ "unresize", make_null_filter },
};
