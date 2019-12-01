#include <cmath>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include "common/cpuinfo.h"
#include "common/except.h"
#include "common/make_unique.h"
#include "common/pixel.h"
#include "common/static_map.h"
#include "common/zassert.h"
#include "graph/filtergraph2.h"
#include "graph/graphbuilder2.h"
#include "graph/image_buffer.h"
#include "colorspace/colorspace.h"
#include "depth/depth.h"
#include "resize/filter.h"
#include "zimg.h"

#define ZIMG_RESIZE_UNRESIZE static_cast<zimg_resample_filter_e>(-1)

namespace {

constexpr unsigned API_VERSION_2_0 = ZIMG_MAKE_API_VERSION(2, 0);
constexpr unsigned API_VERSION_2_1 = ZIMG_MAKE_API_VERSION(2, 1);
constexpr unsigned API_VERSION_2_2 = ZIMG_MAKE_API_VERSION(2, 2);
constexpr unsigned API_VERSION_2_4 = ZIMG_MAKE_API_VERSION(2, 4);

#define API_VERSION_ASSERT(x) zassert_d((x) >= API_VERSION_2_0, "API version invalid")
#define POINTER_ALIGNMENT_ASSERT(x) zassert_d(!(x) || reinterpret_cast<uintptr_t>(x) % zimg::ALIGNMENT_RELAXED == 0, "pointer not aligned")
#define STRIDE_ALIGNMENT_ASSERT(x) zassert_d(!(x) || (x) % zimg::ALIGNMENT_RELAXED == 0, "buffer stride not aligned")

#define POINTER_ALIGNMENT64_ASSERT(x) zassert_d(!(x) || reinterpret_cast<uintptr_t>(x) % zimg::ALIGNMENT == 0, "pointer not aligned")
#define STRIDE_ALIGNMENT64_ASSERT(x) zassert_d(!(x) || (x) % zimg::ALIGNMENT == 0, "buffer stride not aligned")

thread_local zimg_error_code_e g_last_error = ZIMG_ERROR_SUCCESS;
thread_local std::string g_last_error_msg;

constexpr unsigned VERSION_INFO[] = { 2, 9, 2 };


template <class T, class U>
T *assert_dynamic_type(U *ptr) noexcept
{
	zassert_d(dynamic_cast<T *>(ptr), "bad dynamic type");
	return static_cast<T *>(ptr);
}

void clear_last_error_message() noexcept
{
	g_last_error_msg.clear();
	g_last_error_msg.shrink_to_fit();
}

void record_exception_message(const zimg::error::Exception &e) noexcept
{
	try {
		g_last_error_msg = e.what();
	} catch (const std::bad_alloc &) {
		clear_last_error_message();
	}
}

zimg_error_code_e handle_exception(std::exception_ptr eptr) noexcept
{
	using namespace zimg::error;

	zimg_error_code_e code = ZIMG_ERROR_UNKNOWN;

#define CATCH(type, error_code) catch (const type &e) { record_exception_message(e); code = (error_code); }
#define FATAL(type, error_code, msg) catch (const type &e) { record_exception_message(e); code = (error_code); zassert_dfatal(msg); }
	try {
		std::rethrow_exception(eptr);
	}
	CATCH(UnknownError,            ZIMG_ERROR_UNKNOWN)
	CATCH(OutOfMemory,             ZIMG_ERROR_OUT_OF_MEMORY)
	CATCH(UserCallbackFailed,      ZIMG_ERROR_USER_CALLBACK_FAILED)

	CATCH(GreyscaleSubsampling,    ZIMG_ERROR_GREYSCALE_SUBSAMPLING)
	CATCH(ColorFamilyMismatch,     ZIMG_ERROR_COLOR_FAMILY_MISMATCH)
	CATCH(ImageNotDivisible,       ZIMG_ERROR_IMAGE_NOT_DIVISIBLE)
	CATCH(BitDepthOverflow,        ZIMG_ERROR_BIT_DEPTH_OVERFLOW)
	CATCH(LogicError,              ZIMG_ERROR_LOGIC)

	CATCH(EnumOutOfRange,          ZIMG_ERROR_ENUM_OUT_OF_RANGE)
	CATCH(InvalidImageSize,        ZIMG_ERROR_INVALID_IMAGE_SIZE)
	CATCH(IllegalArgument,         ZIMG_ERROR_ILLEGAL_ARGUMENT)

	CATCH(UnsupportedSubsampling,  ZIMG_ERROR_UNSUPPORTED_SUBSAMPLING)
	CATCH(NoColorspaceConversion,  ZIMG_ERROR_NO_COLORSPACE_CONVERSION)
	CATCH(NoFieldParityConversion, ZIMG_ERROR_NO_FIELD_PARITY_CONVERSION)
	CATCH(ResamplingNotAvailable,  ZIMG_ERROR_RESAMPLING_NOT_AVAILABLE)
	CATCH(UnsupportedOperation,    ZIMG_ERROR_UNSUPPORTED_OPERATION)

	FATAL(InternalError,           ZIMG_ERROR_UNKNOWN, "internal error generated")
	FATAL(Exception,               ZIMG_ERROR_UNKNOWN, "unregistered error generated")
	catch (...) {
		g_last_error_msg[0] = '\0';
		zassert_dfatal("bad exception type");
	}
#undef CATCH
#undef FATAL
	g_last_error = code;
	return code;
}

template <class Map, class Key>
typename Map::mapped_type search_enum_map(const Map &map, const Key &key, const char *msg)
{
	auto it = map.find(key);
	if (it == map.end())
		zimg::error::throw_<zimg::error::EnumOutOfRange>(msg);
	return it->second;
}

template <class Map, class Key>
typename Map::mapped_type search_itu_enum_map(const Map &map, const Key &key, const char *msg)
{
	if (static_cast<int>(key) < 0 || static_cast<int>(key) > 255)
		zimg::error::throw_<zimg::error::EnumOutOfRange>(msg);

	auto it = map.find(key);
	if (it == map.end())
		zimg::error::throw_<zimg::error::NoColorspaceConversion>(msg);
	return it->second;
}

zimg::CPUClass translate_cpu(zimg_cpu_type_e cpu)
{
	using zimg::CPUClass;

	static SM_CONSTEXPR_14 const zimg::static_map<zimg_cpu_type_e, CPUClass, 19> map{
		{ ZIMG_CPU_NONE,           CPUClass::NONE },
		{ ZIMG_CPU_AUTO,           CPUClass::AUTO },
		{ ZIMG_CPU_AUTO_64B,       CPUClass::AUTO_64B },
#ifdef ZIMG_X86
		{ ZIMG_CPU_X86_MMX,        CPUClass::NONE },
		{ ZIMG_CPU_X86_SSE,        CPUClass::X86_SSE },
		{ ZIMG_CPU_X86_SSE2,       CPUClass::X86_SSE2 },
		{ ZIMG_CPU_X86_SSE3,       CPUClass::X86_SSE2 },
		{ ZIMG_CPU_X86_SSSE3,      CPUClass::X86_SSE2 },
		{ ZIMG_CPU_X86_SSE41,      CPUClass::X86_SSE2 },
		{ ZIMG_CPU_X86_SSE42,      CPUClass::X86_SSE2 },
		{ ZIMG_CPU_X86_AVX,        CPUClass::X86_AVX },
		{ ZIMG_CPU_X86_F16C,       CPUClass::X86_F16C },
		{ ZIMG_CPU_X86_AVX2,       CPUClass::X86_AVX2 },
		{ ZIMG_CPU_X86_AVX512F,    CPUClass::X86_AVX2 },
		{ ZIMG_CPU_X86_AVX512_SKX, CPUClass::X86_AVX512 },
		{ ZIMG_CPU_X86_AVX512_CLX, CPUClass::X86_AVX512_CLX },
		{ ZIMG_CPU_X86_AVX512_PMC, CPUClass::X86_AVX512 },
		{ ZIMG_CPU_X86_AVX512_SNC, CPUClass::X86_AVX512_CLX },
#endif
	};
	return search_enum_map(map, cpu, "unrecognized cpu type");
}

zimg::PixelType translate_pixel_type(zimg_pixel_type_e pixel_type)
{
	using zimg::PixelType;

	static SM_CONSTEXPR_14 const zimg::static_map<zimg_pixel_type_e, zimg::PixelType, 4> map{
		{ ZIMG_PIXEL_BYTE,  PixelType::BYTE },
		{ ZIMG_PIXEL_WORD,  PixelType::WORD },
		{ ZIMG_PIXEL_HALF,  PixelType::HALF },
		{ ZIMG_PIXEL_FLOAT, PixelType::FLOAT },
	};
	return search_enum_map(map, pixel_type, "unrecognized pixel type");
}

bool translate_pixel_range(zimg_pixel_range_e range)
{
	static SM_CONSTEXPR_14 const zimg::static_map<zimg_pixel_range_e, bool, 2> map{
		{ ZIMG_RANGE_LIMITED, false },
		{ ZIMG_RANGE_FULL,    true },
	};
	return search_enum_map(map, range, "unrecognized pixel range");
}

zimg::graph::GraphBuilder2::ColorFamily translate_color_family(zimg_color_family_e family)
{
	using zimg::graph::GraphBuilder2;

	static SM_CONSTEXPR_14 const zimg::static_map<zimg_color_family_e, GraphBuilder2::ColorFamily, 3> map{
		{ ZIMG_COLOR_GREY, GraphBuilder2::ColorFamily::GREY },
		{ ZIMG_COLOR_RGB,  GraphBuilder2::ColorFamily::RGB },
		{ ZIMG_COLOR_YUV,  GraphBuilder2::ColorFamily::YUV },
	};
	return search_enum_map(map, family, "unrecognized color family");
}

zimg::graph::GraphBuilder2::AlphaType translate_alpha(zimg_alpha_type_e alpha)
{
	using zimg::graph::GraphBuilder2;

	static SM_CONSTEXPR_14 const zimg::static_map<zimg_alpha_type_e, GraphBuilder2::AlphaType, 3> map{
		{ ZIMG_ALPHA_NONE, GraphBuilder2::AlphaType::NONE },
		{ ZIMG_ALPHA_STRAIGHT, GraphBuilder2::AlphaType::STRAIGHT },
		{ ZIMG_ALPHA_PREMULTIPLIED, GraphBuilder2::AlphaType::PREMULTIPLED },
	};
	return search_enum_map(map, alpha, "unrecognized alpha type");
}

zimg::graph::GraphBuilder2::FieldParity translate_field_parity(zimg_field_parity_e field)
{
	using zimg::graph::GraphBuilder2;

	static SM_CONSTEXPR_14 const zimg::static_map<zimg_field_parity_e, GraphBuilder2::FieldParity, 3> map{
		{ ZIMG_FIELD_PROGRESSIVE, GraphBuilder2::FieldParity::PROGRESSIVE },
		{ ZIMG_FIELD_TOP,         GraphBuilder2::FieldParity::TOP },
		{ ZIMG_FIELD_BOTTOM,      GraphBuilder2::FieldParity::BOTTOM },
	};
	return search_enum_map(map, field, "unrecognized field parity");
}

std::pair<zimg::graph::GraphBuilder2::ChromaLocationW, zimg::graph::GraphBuilder2::ChromaLocationH> translate_chroma_location(zimg_chroma_location_e chromaloc)
{
	typedef zimg::graph::GraphBuilder2::ChromaLocationW ChromaLocationW;
	typedef zimg::graph::GraphBuilder2::ChromaLocationH ChromaLocationH;

	// Workaround for std::pair assignment operator missing constexpr.
	struct chroma_pair {
		ChromaLocationW first;
		ChromaLocationH second;

		operator std::pair<ChromaLocationW, ChromaLocationH>() const { return{ first, second }; }
	};

	static SM_CONSTEXPR_14 const zimg::static_map<zimg_chroma_location_e, chroma_pair, 6> map{
		{ ZIMG_CHROMA_LEFT,        { ChromaLocationW::LEFT,   ChromaLocationH::CENTER } },
		{ ZIMG_CHROMA_CENTER,      { ChromaLocationW::CENTER, ChromaLocationH::CENTER } },
		{ ZIMG_CHROMA_TOP_LEFT,    { ChromaLocationW::LEFT,   ChromaLocationH::TOP } },
		{ ZIMG_CHROMA_TOP,         { ChromaLocationW::CENTER, ChromaLocationH::TOP } },
		{ ZIMG_CHROMA_BOTTOM_LEFT, { ChromaLocationW::LEFT,   ChromaLocationH::BOTTOM } },
		{ ZIMG_CHROMA_BOTTOM,      { ChromaLocationW::CENTER, ChromaLocationH::BOTTOM } },
	};
	return search_enum_map(map, chromaloc, "unrecognized chroma location");
}

zimg::colorspace::MatrixCoefficients translate_matrix(zimg_matrix_coefficients_e matrix)
{
	using zimg::colorspace::MatrixCoefficients;

	static SM_CONSTEXPR_14 const zimg::static_map<zimg_matrix_coefficients_e, zimg::colorspace::MatrixCoefficients, 13> map{
		{ ZIMG_MATRIX_RGB,                      MatrixCoefficients::RGB },
		{ ZIMG_MATRIX_BT709,                    MatrixCoefficients::REC_709 },
		{ ZIMG_MATRIX_UNSPECIFIED,              MatrixCoefficients::UNSPECIFIED },
		{ ZIMG_MATRIX_FCC,                      MatrixCoefficients::FCC },
		{ ZIMG_MATRIX_BT470_BG,                 MatrixCoefficients::REC_601 },
		{ ZIMG_MATRIX_ST170_M,                  MatrixCoefficients::REC_601 },
		{ ZIMG_MATRIX_ST240_M,                  MatrixCoefficients::SMPTE_240M },
		{ ZIMG_MATRIX_YCGCO,                    MatrixCoefficients::YCGCO },
		{ ZIMG_MATRIX_BT2020_NCL,               MatrixCoefficients::REC_2020_NCL },
		{ ZIMG_MATRIX_BT2020_CL,                MatrixCoefficients::REC_2020_CL },
		{ ZIMG_MATRIX_CHROMATICITY_DERIVED_NCL, MatrixCoefficients::CHROMATICITY_DERIVED_NCL },
		{ ZIMG_MATRIX_CHROMATICITY_DERIVED_CL,  MatrixCoefficients::CHROMATICITY_DERIVED_CL },
		{ ZIMG_MATRIX_ICTCP,                    MatrixCoefficients::REC_2100_ICTCP },
	};
	return search_itu_enum_map(map, matrix, "unrecognized matrix coefficients");
}

zimg::colorspace::TransferCharacteristics translate_transfer(zimg_transfer_characteristics_e transfer)
{
	using zimg::colorspace::TransferCharacteristics;

	static SM_CONSTEXPR_14 const zimg::static_map<zimg_transfer_characteristics_e, TransferCharacteristics, 15> map{
		{ ZIMG_TRANSFER_BT709,         TransferCharacteristics::REC_709 },
		{ ZIMG_TRANSFER_UNSPECIFIED,   TransferCharacteristics::UNSPECIFIED },
		{ ZIMG_TRANSFER_ST240_M,       TransferCharacteristics::SMPTE_240M },
		{ ZIMG_TRANSFER_BT601,         TransferCharacteristics::REC_709 },
		{ ZIMG_TRANSFER_BT470_M,       TransferCharacteristics::REC_470_M },
		{ ZIMG_TRANSFER_BT470_BG,      TransferCharacteristics::REC_470_BG },
		{ ZIMG_TRANSFER_IEC_61966_2_4, TransferCharacteristics::XVYCC },
		{ ZIMG_TRANSFER_IEC_61966_2_1, TransferCharacteristics::SRGB },
		{ ZIMG_TRANSFER_BT2020_10,     TransferCharacteristics::REC_709 },
		{ ZIMG_TRANSFER_BT2020_12,     TransferCharacteristics::REC_709 },
		{ ZIMG_TRANSFER_LINEAR,        TransferCharacteristics::LINEAR },
		{ ZIMG_TRANSFER_LOG_100,       TransferCharacteristics::LOG_100 },
		{ ZIMG_TRANSFER_LOG_316,       TransferCharacteristics::LOG_316 },
		{ ZIMG_TRANSFER_ST2084,        TransferCharacteristics::ST_2084 },
		{ ZIMG_TRANSFER_ARIB_B67,      TransferCharacteristics::ARIB_B67 },
	};
	return search_itu_enum_map(map, transfer, "unrecognized transfer characteristics");
}

zimg::colorspace::ColorPrimaries translate_primaries(zimg_color_primaries_e primaries)
{
	using zimg::colorspace::ColorPrimaries;

	static SM_CONSTEXPR_14 const zimg::static_map<zimg_color_primaries_e, ColorPrimaries, 12> map{
		{ ZIMG_PRIMARIES_BT470_M,     ColorPrimaries::REC_470_M },
		{ ZIMG_PRIMARIES_BT470_BG,    ColorPrimaries::REC_470_BG },
		{ ZIMG_PRIMARIES_BT709,       ColorPrimaries::REC_709 },
		{ ZIMG_PRIMARIES_UNSPECIFIED, ColorPrimaries::UNSPECIFIED },
		{ ZIMG_PRIMARIES_ST170_M,     ColorPrimaries::SMPTE_C },
		{ ZIMG_PRIMARIES_ST240_M,     ColorPrimaries::SMPTE_C },
		{ ZIMG_PRIMARIES_FILM,        ColorPrimaries::FILM },
		{ ZIMG_PRIMARIES_BT2020,      ColorPrimaries::REC_2020 },
		{ ZIMG_PRIMARIES_ST428,       ColorPrimaries::XYZ },
		{ ZIMG_PRIMARIES_ST431_2,     ColorPrimaries::DCI_P3 },
		{ ZIMG_PRIMARIES_ST432_1,     ColorPrimaries::DCI_P3_D65 },
		{ ZIMG_PRIMARIES_EBU3213_E,   ColorPrimaries::JEDEC_P22 },
	};
	return search_itu_enum_map(map, primaries, "unrecognized color primaries");
}

zimg::depth::DitherType translate_dither(zimg_dither_type_e dither)
{
	using zimg::depth::DitherType;

	static SM_CONSTEXPR_14 const zimg::static_map<zimg_dither_type_e, DitherType, 4> map{
		{ ZIMG_DITHER_NONE,            DitherType::NONE },
		{ ZIMG_DITHER_ORDERED,         DitherType::ORDERED },
		{ ZIMG_DITHER_RANDOM,          DitherType::RANDOM },
		{ ZIMG_DITHER_ERROR_DIFFUSION, DitherType::ERROR_DIFFUSION },
	};
	return search_enum_map(map, dither, "unrecognized dither type");
}

std::unique_ptr<zimg::resize::Filter> translate_resize_filter(zimg_resample_filter_e filter_type, double param_a, double param_b)
{
	if (filter_type == ZIMG_RESIZE_UNRESIZE)
		return nullptr;

	try {
		switch (filter_type) {
		case ZIMG_RESIZE_POINT:
			return ztd::make_unique<zimg::resize::PointFilter>();
		case ZIMG_RESIZE_BILINEAR:
			return ztd::make_unique<zimg::resize::BilinearFilter>();
		case ZIMG_RESIZE_BICUBIC:
			param_a = std::isnan(param_a) ? zimg::resize::BicubicFilter::DEFAULT_B : param_a;
			param_b = std::isnan(param_b) ? zimg::resize::BicubicFilter::DEFAULT_C : param_b;
			return ztd::make_unique<zimg::resize::BicubicFilter>(param_a, param_b);
		case ZIMG_RESIZE_SPLINE16:
			return ztd::make_unique<zimg::resize::Spline16Filter>();
		case ZIMG_RESIZE_SPLINE36:
			return ztd::make_unique<zimg::resize::Spline36Filter>();
		case ZIMG_RESIZE_LANCZOS:
			param_a = std::isnan(param_a) ? zimg::resize::LanczosFilter::DEFAULT_TAPS : std::max(param_a, 1.0);
			return ztd::make_unique<zimg::resize::LanczosFilter>(static_cast<unsigned>(param_a));
		default:
			zimg::error::throw_<zimg::error::EnumOutOfRange>("unrecognized resampling filter");
		}
	} catch (const std::bad_alloc &) {
		zimg::error::throw_<zimg::error::OutOfMemory>();
	}
}

zimg::graph::ColorImageBuffer<void> import_image_buffer(const zimg_image_buffer &src)
{
	zimg::graph::ColorImageBuffer<void> dst{};

	API_VERSION_ASSERT(src.version);

	if (src.version >= API_VERSION_2_0) {
		unsigned num_planes = src.version >= API_VERSION_2_4 ? 4 : 3;

		for (unsigned p = 0; p < num_planes; ++p) {
			dst[p] = zimg::graph::ImageBuffer<void>{ src.plane[p].data, src.plane[p].stride, src.plane[p].mask };
		}
	}
	return dst;
}

zimg::graph::ColorImageBuffer<const void> import_image_buffer(const zimg_image_buffer_const &src)
{
	zimg::graph::ColorImageBuffer<const void> dst{};

	API_VERSION_ASSERT(src.version);

	if (src.version >= API_VERSION_2_0) {
		unsigned num_planes = src.version >= API_VERSION_2_4 ? 4 : 3;

		for (unsigned p = 0; p < num_planes; ++p) {
			dst[p] = zimg::graph::ImageBuffer<const void>{ src.plane[p].data, src.plane[p].stride, src.plane[p].mask };
		}
	}
	return dst;
}

void import_graph_state_common(const zimg_image_format &src, zimg::graph::GraphBuilder2::state *out)
{
	if (src.version >= API_VERSION_2_0) {
		out->width = src.width;
		out->height = src.height;
		out->type = translate_pixel_type(src.pixel_type);
		out->subsample_w = src.subsample_w;
		out->subsample_h = src.subsample_h;

		// Handle colorspace constants separately.
		out->color = translate_color_family(src.color_family);

		out->depth = src.depth ? src.depth : zimg::pixel_depth(out->type);
		out->fullrange = translate_pixel_range(src.pixel_range);

		out->parity = translate_field_parity(src.field_parity);
		std::tie(out->chroma_location_w, out->chroma_location_h) = translate_chroma_location(src.chroma_location);
	}
	if (src.version >= API_VERSION_2_1) {
		out->active_left = std::isnan(src.active_region.left) ? 0 : src.active_region.left;
		out->active_top = std::isnan(src.active_region.top) ? 0 : src.active_region.top;
		out->active_width = std::isnan(src.active_region.width) ? src.width : src.active_region.width;
		out->active_height = std::isnan(src.active_region.height) ? src.height : src.active_region.height;
	} else {
		out->active_left = 0.0;
		out->active_top = 0.0;
		out->active_width = src.width;
		out->active_height = src.height;
	}
	if (src.version >= API_VERSION_2_4)
		out->alpha = translate_alpha(src.alpha);
}

std::pair<zimg::graph::GraphBuilder2::state, zimg::graph::GraphBuilder2::state> import_graph_state(const zimg_image_format &src, const zimg_image_format &dst)
{
	API_VERSION_ASSERT(src.version);
	API_VERSION_ASSERT(dst.version);
	zassert_d(src.version == dst.version, "image format versions do not match");

	zimg::graph::GraphBuilder2::state src_state{};
	zimg::graph::GraphBuilder2::state dst_state{};

	import_graph_state_common(src, &src_state);
	import_graph_state_common(dst, &dst_state);

	if (src.version >= API_VERSION_2_0) {
		// Accept unenumerated colorspaces if they form the basic no-op case.
		if (src.color_family == dst.color_family &&
		    src.matrix_coefficients == dst.matrix_coefficients &&
		    src.transfer_characteristics == dst.transfer_characteristics &&
		    src.color_primaries == dst.color_primaries)
		{
			src_state.colorspace = zimg::colorspace::ColorspaceDefinition{};
			dst_state.colorspace = zimg::colorspace::ColorspaceDefinition{};
		} else {
			src_state.colorspace.matrix = translate_matrix(src.matrix_coefficients);
			src_state.colorspace.transfer = translate_transfer(src.transfer_characteristics);
			src_state.colorspace.primaries = translate_primaries(src.color_primaries);

			dst_state.colorspace.matrix = translate_matrix(dst.matrix_coefficients);
			dst_state.colorspace.transfer = translate_transfer(dst.transfer_characteristics);
			dst_state.colorspace.primaries = translate_primaries(dst.color_primaries);
		}
	}

	return{ src_state, dst_state };
}

zimg::graph::GraphBuilder2::params import_graph_params(const zimg_graph_builder_params &src)
{
	API_VERSION_ASSERT(src.version);

	zimg::graph::GraphBuilder2::params params{};

	if (src.version >= API_VERSION_2_0) {
		params.filter = translate_resize_filter(src.resample_filter, src.filter_param_a, src.filter_param_b);
		params.filter_uv = translate_resize_filter(src.resample_filter_uv, src.filter_param_a_uv, src.filter_param_b_uv);
		params.unresize = src.resample_filter == ZIMG_RESIZE_UNRESIZE;
		params.dither_type = translate_dither(src.dither_type);
		params.cpu = translate_cpu(src.cpu_type);
	}
	if (src.version >= API_VERSION_2_2) {
		params.peak_luminance = src.nominal_peak_luminance;
		params.approximate_gamma = !!src.allow_approximate_gamma;
	}

	return params;
}

} // namespace


void zimg_get_version_info(unsigned *major, unsigned *minor, unsigned *micro)
{
	zassert_d(major, "null pointer");
	zassert_d(minor, "null pointer");
	zassert_d(micro, "null pointer");

	*major = VERSION_INFO[0];
	*minor = VERSION_INFO[1];
	*micro = VERSION_INFO[2];
}

unsigned zimg_get_api_version(unsigned *major, unsigned *minor)
{
	if (major)
		*major = ZIMG_API_VERSION_MAJOR;
	if (minor)
		*minor = ZIMG_API_VERSION_MINOR;

	return ZIMG_API_VERSION;
}

zimg_error_code_e zimg_get_last_error(char *err_msg, size_t n)
{
	if (err_msg && n) {
		std::strncpy(err_msg, g_last_error_msg.c_str(), n);
		err_msg[n - 1] = '\0';
	}

	return g_last_error;
}

void zimg_clear_last_error(void)
{
	g_last_error = ZIMG_ERROR_SUCCESS;
	clear_last_error_message();
}

unsigned zimg_select_buffer_mask(unsigned count)
{
	return zimg::graph::select_zimg_buffer_mask(count);
}

#define EX_BEGIN \
  zimg_error_code_e ret = ZIMG_ERROR_SUCCESS; \
  try {
#define EX_END \
  } catch (...) { \
    ret = handle_exception(std::current_exception()); \
  } \
  return ret;

void zimg_filter_graph_free(zimg_filter_graph *ptr)
{
	delete ptr;
}

zimg_error_code_e zimg_filter_graph_get_tmp_size(const zimg_filter_graph *ptr, size_t *out)
{
	zassert_d(ptr, "null pointer");
	zassert_d(out, "null pointer");

	EX_BEGIN
	*out = assert_dynamic_type<const zimg::graph::FilterGraph2>(ptr)->get_tmp_size();
	EX_END
}

zimg_error_code_e zimg_filter_graph_get_input_buffering(const zimg_filter_graph *ptr, unsigned *out)
{
	zassert_d(ptr, "null pointer");
	zassert_d(out, "null pointer");

	EX_BEGIN
	*out = assert_dynamic_type<const zimg::graph::FilterGraph2>(ptr)->get_input_buffering();
	EX_END
}

zimg_error_code_e zimg_filter_graph_get_output_buffering(const zimg_filter_graph *ptr, unsigned *out)
{
	zassert_d(ptr, "null pointer");
	zassert_d(out, "null pointer");

	EX_BEGIN
	*out = assert_dynamic_type<const zimg::graph::FilterGraph2>(ptr)->get_output_buffering();
	EX_END
}

zimg_error_code_e zimg_filter_graph_process(const zimg_filter_graph *ptr, const zimg_image_buffer_const *src, const zimg_image_buffer *dst, void *tmp,
                                             zimg_filter_graph_callback unpack_cb, void *unpack_user,
                                             zimg_filter_graph_callback pack_cb, void *pack_user)
{
	zassert_d(ptr, "null pointer");
	zassert_d(src, "null pointer");
	zassert_d(dst, "null pointer");

	EX_BEGIN
	const zimg::graph::FilterGraph2 *graph = assert_dynamic_type<const zimg::graph::FilterGraph2>(ptr);

	if (false /*graph->requires_64b_alignment()*/) {
		POINTER_ALIGNMENT64_ASSERT(src->plane[0].data);
		POINTER_ALIGNMENT64_ASSERT(src->plane[1].data);
		POINTER_ALIGNMENT64_ASSERT(src->plane[2].data);

		STRIDE_ALIGNMENT64_ASSERT(src->plane[0].stride);
		STRIDE_ALIGNMENT64_ASSERT(src->plane[1].stride);
		STRIDE_ALIGNMENT64_ASSERT(src->plane[2].stride);

		POINTER_ALIGNMENT64_ASSERT(dst->plane[0].data);
		POINTER_ALIGNMENT64_ASSERT(dst->plane[1].data);
		POINTER_ALIGNMENT64_ASSERT(dst->plane[2].data);

		STRIDE_ALIGNMENT64_ASSERT(dst->plane[0].stride);
		STRIDE_ALIGNMENT64_ASSERT(dst->plane[1].stride);
		STRIDE_ALIGNMENT64_ASSERT(dst->plane[2].stride);

		if (src->version >= API_VERSION_2_4) {
			POINTER_ALIGNMENT64_ASSERT(src->plane[3].data);
			STRIDE_ALIGNMENT64_ASSERT(src->plane[3].stride);
		}
		if (dst->version >= API_VERSION_2_4) {
			POINTER_ALIGNMENT64_ASSERT(dst->plane[3].data);
			STRIDE_ALIGNMENT64_ASSERT(dst->plane[3].stride);
		}

		POINTER_ALIGNMENT64_ASSERT(tmp);
	} else {
		POINTER_ALIGNMENT_ASSERT(src->plane[0].data);
		POINTER_ALIGNMENT_ASSERT(src->plane[1].data);
		POINTER_ALIGNMENT_ASSERT(src->plane[2].data);

		STRIDE_ALIGNMENT_ASSERT(src->plane[0].stride);
		STRIDE_ALIGNMENT_ASSERT(src->plane[1].stride);
		STRIDE_ALIGNMENT_ASSERT(src->plane[2].stride);

		POINTER_ALIGNMENT_ASSERT(dst->plane[0].data);
		POINTER_ALIGNMENT_ASSERT(dst->plane[1].data);
		POINTER_ALIGNMENT_ASSERT(dst->plane[2].data);

		STRIDE_ALIGNMENT_ASSERT(dst->plane[0].stride);
		STRIDE_ALIGNMENT_ASSERT(dst->plane[1].stride);
		STRIDE_ALIGNMENT_ASSERT(dst->plane[2].stride);

		if (src->version >= API_VERSION_2_4) {
			POINTER_ALIGNMENT_ASSERT(src->plane[3].data);
			STRIDE_ALIGNMENT_ASSERT(src->plane[3].stride);
		}
		if (dst->version >= API_VERSION_2_4) {
			POINTER_ALIGNMENT_ASSERT(dst->plane[3].data);
			STRIDE_ALIGNMENT_ASSERT(dst->plane[3].stride);
		}

		POINTER_ALIGNMENT_ASSERT(tmp);
	}

	auto src_buf = import_image_buffer(*src);
	auto dst_buf = import_image_buffer(*dst);
	graph->process(src_buf, dst_buf, tmp, { unpack_cb, unpack_user }, { pack_cb, pack_user });
	EX_END
}

#undef EX_BEGIN
#undef EX_END

void zimg_image_format_default(zimg_image_format *ptr, unsigned version)
{
	zassert_d(ptr, "null pointer");
	API_VERSION_ASSERT(version);

	ptr->version = version;

	if (version >= API_VERSION_2_0) {
		ptr->width = 0;
		ptr->height = 0;
		ptr->pixel_type = static_cast<zimg_pixel_type_e>(-1);

		ptr->subsample_w = 0;
		ptr->subsample_h = 0;

		ptr->color_family = ZIMG_COLOR_GREY;
		ptr->matrix_coefficients = ZIMG_MATRIX_UNSPECIFIED;
		ptr->transfer_characteristics = ZIMG_TRANSFER_UNSPECIFIED;
		ptr->color_primaries = ZIMG_PRIMARIES_UNSPECIFIED;

		ptr->depth = 0;
		ptr->pixel_range = ZIMG_RANGE_LIMITED;

		ptr->field_parity = ZIMG_FIELD_PROGRESSIVE;
		ptr->chroma_location = ZIMG_CHROMA_LEFT;
	}
	if (version >= API_VERSION_2_1) {
		ptr->active_region.left = NAN;
		ptr->active_region.top = NAN;
		ptr->active_region.width = NAN;
		ptr->active_region.height = NAN;
	}
	if (version >= API_VERSION_2_4) {
		ptr->alpha = ZIMG_ALPHA_NONE;
	}
}

void zimg_graph_builder_params_default(zimg_graph_builder_params *ptr, unsigned version)
{
	zassert_d(ptr, "null pointer");
	API_VERSION_ASSERT(version);

	ptr->version = version;

	if (version >= API_VERSION_2_0) {
		ptr->resample_filter = ZIMG_RESIZE_BICUBIC;
		ptr->filter_param_a = NAN;
		ptr->filter_param_b = NAN;

		ptr->resample_filter_uv = ZIMG_RESIZE_BILINEAR;
		ptr->filter_param_a_uv = NAN;
		ptr->filter_param_b_uv = NAN;

		ptr->dither_type = ZIMG_DITHER_NONE;

		ptr->cpu_type = ZIMG_CPU_AUTO;
	}
	if (version >= API_VERSION_2_2) {
		ptr->nominal_peak_luminance = NAN;
		ptr->allow_approximate_gamma = 0;
	}
}

zimg_filter_graph *zimg_filter_graph_build(const zimg_image_format *src_format, const zimg_image_format *dst_format, const zimg_graph_builder_params *params)
{
	zassert_d(src_format, "null pointer");
	zassert_d(dst_format, "null pointer");

	try {
		zimg::graph::GraphBuilder2::state src_state;
		zimg::graph::GraphBuilder2::state dst_state;
		zimg::graph::GraphBuilder2::params graph_params;

		std::tie(src_state, dst_state) = import_graph_state(*src_format, *dst_format);
		if (params)
			graph_params = import_graph_params(*params);

		return zimg::graph::GraphBuilder2{}.set_source(src_state)
		                                   .connect_graph(dst_state, params ? &graph_params : nullptr)
		                                   .complete_graph()
		                                   .release();
	} catch (...) {
		handle_exception(std::current_exception());
		return nullptr;
	}
}
