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

namespace {;

typedef std::unique_ptr<zimg::resize::Filter> filter_uptr;

template <class T>
filter_uptr make_filter(double, double)
{
	return filter_uptr{ new T{} };
}

filter_uptr make_bicubic_filter(double b, double c)
{
	b = isnan(b) ? 1.0 / 3.0 : b;
	c = isnan(c) ? 1.0 / 3.0 : c;

	return filter_uptr{ new zimg::resize::BicubicFilter{ b, c } };
}

filter_uptr make_lanczos_filter(double taps, double)
{
	taps = isnan(taps) ? 4.0 : taps;
	return filter_uptr{ new zimg::resize::LanczosFilter{ static_cast<int>(std::floor(taps)) } };
}

} // namespace


const zimg::static_string_map<CPUClass, 4> g_cpu_table{
	{ "none", CPUClass::CPU_NONE },
	{ "auto", CPUClass::CPU_AUTO },
#ifdef ZIMG_X86
	{ "sse",  CPUClass::CPU_X86_SSE },
	{ "sse2", CPUClass::CPU_X86_SSE2 },
#endif
};

const zimg::static_string_map<PixelType, 4> g_pixel_table{
	{ "byte",  PixelType::BYTE },
	{ "word",  PixelType::WORD },
	{ "half",  PixelType::HALF },
	{ "float", PixelType::FLOAT },
};

const zimg::static_string_map<MatrixCoefficients, 7> g_matrix_table{
	{ "unspec",   MatrixCoefficients::MATRIX_UNSPECIFIED },
	{ "rgb",      MatrixCoefficients::MATRIX_RGB },
	{ "601",      MatrixCoefficients::MATRIX_601 },
	{ "709",      MatrixCoefficients::MATRIX_709 },
	{ "ycgco",    MatrixCoefficients::MATRIX_YCGCO },
	{ "2020_ncl", MatrixCoefficients::MATRIX_2020_NCL },
	{ "2020_cl",  MatrixCoefficients::MATRIX_2020_CL },
};

const zimg::static_string_map<TransferCharacteristics, 3> g_transfer_table{
	{ "unspec", TransferCharacteristics::TRANSFER_UNSPECIFIED },
	{ "linear", TransferCharacteristics::TRANSFER_LINEAR },
	{ "709",    TransferCharacteristics::TRANSFER_709 },
};

const zimg::static_string_map<ColorPrimaries, 4> g_primaries_table{
	{ "unspec",  ColorPrimaries::PRIMARIES_UNSPECIFIED },
	{ "smpte_c", ColorPrimaries::PRIMARIES_SMPTE_C },
	{ "709",     ColorPrimaries::PRIMARIES_709 },
	{ "2020",    ColorPrimaries::PRIMARIES_2020 },
};

const zimg::static_string_map<DitherType, 4> g_dither_table{
	{ "none",            DitherType::DITHER_NONE },
	{ "ordered",         DitherType::DITHER_ORDERED },
	{ "random",          DitherType::DITHER_RANDOM },
	{ "error_diffusion", DitherType::DITHER_ERROR_DIFFUSION },
};

const zimg::static_string_map<std::unique_ptr<zimg::resize::Filter>(*)(double, double), 6> g_resize_table{
	{ "point",    make_filter<zimg::resize::PointFilter> },
	{ "bilinear", make_filter<zimg::resize::BilinearFilter> },
	{ "bicubic",  make_bicubic_filter },
	{ "spline16", make_filter<zimg::resize::Spline16Filter> },
	{ "spline36", make_filter<zimg::resize::Spline36Filter> },
	{ "lanczos",  make_lanczos_filter },
};
