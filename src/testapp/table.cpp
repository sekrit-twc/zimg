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
	b = std::isnan(b) ? 1.0 / 3.0 : b;
	c = std::isnan(c) ? 1.0 / 3.0 : c;

	return filter_uptr{ new zimg::resize::BicubicFilter{ b, c } };
}

filter_uptr make_lanczos_filter(double taps, double)
{
	taps = std::isnan(taps) ? 4.0 : taps;
	return filter_uptr{ new zimg::resize::LanczosFilter{ static_cast<unsigned>(std::floor(taps)) } };
}

filter_uptr make_null_filter(double, double) { return nullptr; }

} // namespace


const zimg::static_string_map<CPUClass, 8> g_cpu_table{
	{ "none", CPUClass::NONE },
	{ "auto", CPUClass::AUTO_64B },
#ifdef ZIMG_X86
	{ "sse",    CPUClass::X86_SSE },
	{ "sse2",   CPUClass::X86_SSE2 },
	{ "avx",    CPUClass::X86_AVX },
	{ "f16c",   CPUClass::X86_F16C },
	{ "avx2",   CPUClass::X86_AVX2 },
	{ "avx512", CPUClass::X86_AVX512 },
#endif
};

const zimg::static_string_map<PixelType, 4> g_pixel_table{
	{ "byte",  PixelType::BYTE },
	{ "word",  PixelType::WORD },
	{ "half",  PixelType::HALF },
	{ "float", PixelType::FLOAT },
};

const zimg::static_string_map<MatrixCoefficients, 8> g_matrix_table{
	{ "unspec",   MatrixCoefficients::UNSPECIFIED },
	{ "rgb",      MatrixCoefficients::RGB },
	{ "601",      MatrixCoefficients::REC_601 },
	{ "709",      MatrixCoefficients::REC_709 },
	{ "ycgco",    MatrixCoefficients::YCGCO },
	{ "2020_ncl", MatrixCoefficients::REC_2020_NCL },
	{ "2020_cl",  MatrixCoefficients::REC_2020_CL },
	{ "ictcp",    MatrixCoefficients::REC_2100_ICTCP },
};

const zimg::static_string_map<TransferCharacteristics, 6> g_transfer_table{
	{ "unspec",   TransferCharacteristics::UNSPECIFIED },
	{ "linear",   TransferCharacteristics::LINEAR },
	{ "709",      TransferCharacteristics::REC_709 },
	{ "srgb",     TransferCharacteristics::SRGB },
	{ "st_2084",  TransferCharacteristics::ST_2084 },
	{ "arib_b67", TransferCharacteristics::ARIB_B67 },
};

const zimg::static_string_map<ColorPrimaries, 5> g_primaries_table{
	{ "unspec",    ColorPrimaries::UNSPECIFIED },
	{ "smpte_c",   ColorPrimaries::SMPTE_C },
	{ "709",       ColorPrimaries::REC_709 },
	{ "2020",      ColorPrimaries::REC_2020 },
	{ "dcip3_d65", ColorPrimaries::DCI_P3_D65 }
};

const zimg::static_string_map<DitherType, 4> g_dither_table{
	{ "none",            DitherType::NONE },
	{ "ordered",         DitherType::ORDERED },
	{ "random",          DitherType::RANDOM },
	{ "error_diffusion", DitherType::ERROR_DIFFUSION },
};

const zimg::static_string_map<std::unique_ptr<zimg::resize::Filter>(*)(double, double), 7> g_resize_table{
	{ "point",    make_filter<zimg::resize::PointFilter> },
	{ "bilinear", make_filter<zimg::resize::BilinearFilter> },
	{ "bicubic",  make_bicubic_filter },
	{ "spline16", make_filter<zimg::resize::Spline16Filter> },
	{ "spline36", make_filter<zimg::resize::Spline36Filter> },
	{ "lanczos",  make_lanczos_filter },
	{ "unresize", make_null_filter },
};
