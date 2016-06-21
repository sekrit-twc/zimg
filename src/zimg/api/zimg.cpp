#include <cmath>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include "common/ccdep.h"
#include "common/cpuinfo.h"
#include "common/except.h"
#include "common/make_unique.h"
#include "common/pixel.h"
#include "common/static_map.h"
#include "common/zassert.h"
#include "graph/filtergraph.h"
#include "graph/graphbuilder.h"
#include "graph/image_buffer.h"
#include "colorspace/colorspace.h"
#include "depth/depth.h"
#include "resize/filter.h"
#include "zimg.h"

namespace {

const unsigned API_VERSION_2_0 = ZIMG_MAKE_API_VERSION(2, 0);

#define API_VERSION_ASSERT(x) _zassert_d((x) >= API_VERSION_2_0, "API version invalid")
#define POINTER_ALIGNMENT_ASSERT(x) _zassert_d(!(x) || reinterpret_cast<uintptr_t>(x) % zimg::ALIGNMENT == 0, "pointer not aligned")
#define STRIDE_ALIGNMENT_ASSERT(x) _zassert_d(!(x) || (x) % zimg::ALIGNMENT == 0, "buffer stride not aligned")

thread_local zimg_error_code_e g_last_error = ZIMG_ERROR_SUCCESS;
thread_local char g_last_error_msg[1024];

const unsigned VERSION_INFO[] = { 2, 1, 0 };


template <class T, class U>
T *assert_dynamic_cast(U *ptr)
{
	static_assert(std::is_base_of<U, T>::value, "type does not derive from base");
	T *pptr = dynamic_cast<T *>(ptr);
	_zassert_d(pptr, "bad dynamic type");
	return pptr;
}

template <class T>
void record_exception_message(const T &e)
{
	strncpy(g_last_error_msg, e.what(), sizeof(g_last_error_msg) - 1);
	g_last_error_msg[sizeof(g_last_error_msg) - 1] = '\0';
}

zimg_error_code_e handle_exception(std::exception_ptr eptr)
{
	using namespace zimg::error;

	zimg_error_code_e code = ZIMG_ERROR_UNKNOWN;

#define CATCH(type, error_code) catch (const type &e) { record_exception_message(e); code = (error_code); }
#define FATAL(type, error_code, msg) catch (const type &e) { _zassert_d(false, msg); record_exception_message(e); code = (error_code); }
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
	CATCH(ZeroImageSize,           ZIMG_ERROR_ZERO_IMAGE_SIZE)
	CATCH(IllegalArgument,         ZIMG_ERROR_ILLEGAL_ARGUMENT)

	CATCH(UnsupportedSubsampling,  ZIMG_ERROR_UNSUPPORTED_SUBSAMPLING)
	CATCH(NoColorspaceConversion,  ZIMG_ERROR_NO_COLORSPACE_CONVERSION)
	CATCH(NoFieldParityConversion, ZIMG_ERROR_NO_FIELD_PARITY_CONVERSION)
	CATCH(ResamplingNotAvailable,  ZIMG_ERROR_RESAMPLING_NOT_AVAILABLE)
	CATCH(UnsupportedOperation,    ZIMG_ERROR_UNSUPPORTED_OPERATION)

	FATAL(InternalError,           ZIMG_ERROR_UNKNOWN, "internal error generated")
	FATAL(Exception,               ZIMG_ERROR_UNKNOWN, "unregistered error generated")
#undef CATCH
#undef FATAL
	g_last_error = code;
	return code;
}

template <class Map, class Key>
typename Map::mapped_type search_enum_map(const Map &map, const Key &key, const char *msg)
{
	auto it = map.find(key);
	return it == map.end() ? throw zimg::error::EnumOutOfRange{ msg } : it->second;
}

template <class Map, class Key>
typename Map::mapped_type search_itu_enum_map(const Map &map, const Key &key, const char *msg)
{
	if (static_cast<int>(key) < 0 || static_cast<int>(key) > 255)
		throw zimg::error::EnumOutOfRange{ msg };

	auto it = map.find(key);
	return it == map.end() ? throw zimg::error::NoColorspaceConversion{ msg } : it->second;
}

zimg::CPUClass translate_cpu(zimg_cpu_type_e cpu)
{
	using zimg::CPUClass;

	static const zimg::static_map<zimg_cpu_type_e, CPUClass, 12> map{
		{ ZIMG_CPU_NONE,      CPUClass::CPU_NONE },
		{ ZIMG_CPU_AUTO,      CPUClass::CPU_AUTO },
#ifdef ZIMG_X86
		{ ZIMG_CPU_X86_MMX,   CPUClass::CPU_NONE },
		{ ZIMG_CPU_X86_SSE,   CPUClass::CPU_X86_SSE },
		{ ZIMG_CPU_X86_SSE2,  CPUClass::CPU_X86_SSE2 },
		{ ZIMG_CPU_X86_SSE3,  CPUClass::CPU_X86_SSE2 },
		{ ZIMG_CPU_X86_SSSE3, CPUClass::CPU_X86_SSE2 },
		{ ZIMG_CPU_X86_SSE41, CPUClass::CPU_X86_SSE2 },
		{ ZIMG_CPU_X86_SSE42, CPUClass::CPU_X86_SSE2 },
		{ ZIMG_CPU_X86_AVX,   CPUClass::CPU_X86_AVX },
		{ ZIMG_CPU_X86_F16C,  CPUClass::CPU_X86_F16C },
		{ ZIMG_CPU_X86_AVX2,  CPUClass::CPU_X86_AVX2 },
#endif
	};
	return search_enum_map(map, cpu, "unrecognized cpu type");
}

zimg::PixelType translate_pixel_type(zimg_pixel_type_e pixel_type)
{
	using zimg::PixelType;

	static const zimg::static_map<zimg_pixel_type_e, zimg::PixelType, 4> map{
		{ ZIMG_PIXEL_BYTE,  PixelType::BYTE },
		{ ZIMG_PIXEL_WORD,  PixelType::WORD },
		{ ZIMG_PIXEL_HALF,  PixelType::HALF },
		{ ZIMG_PIXEL_FLOAT, PixelType::FLOAT },
	};
	return search_enum_map(map, pixel_type, "unrecognized pixel type");
}

bool translate_pixel_range(zimg_pixel_range_e range)
{
	static const zimg::static_map<zimg_pixel_range_e, bool, 2> map{
		{ ZIMG_RANGE_LIMITED, false },
		{ ZIMG_RANGE_FULL,    true },
	};
	return search_enum_map(map, range, "unrecognized pixel range");
}

zimg::graph::GraphBuilder::ColorFamily translate_color_family(zimg_color_family_e family)
{
	using zimg::graph::GraphBuilder;

	static const zimg::static_map<zimg_color_family_e,GraphBuilder::ColorFamily, 3> map{
		{ ZIMG_COLOR_GREY, GraphBuilder::ColorFamily::COLOR_GREY },
		{ ZIMG_COLOR_RGB,  GraphBuilder::ColorFamily::COLOR_RGB },
		{ ZIMG_COLOR_YUV,  GraphBuilder::ColorFamily::COLOR_YUV},
	};
	return search_enum_map(map, family, "unrecognized color family");
}

zimg::graph::GraphBuilder::FieldParity translate_field_parity(zimg_field_parity_e field)
{
	using zimg::graph::GraphBuilder;

	static const zimg::static_map<zimg_field_parity_e, GraphBuilder::FieldParity, 3> map{
		{ ZIMG_FIELD_PROGRESSIVE, GraphBuilder::FieldParity::FIELD_PROGRESSIVE },
		{ ZIMG_FIELD_TOP,         GraphBuilder::FieldParity::FIELD_TOP },
		{ ZIMG_FIELD_BOTTOM,      GraphBuilder::FieldParity::FIELD_BOTTOM },
	};
	return search_enum_map(map, field, "unrecognized field parity");
}

std::pair<zimg::graph::GraphBuilder::ChromaLocationW, zimg::graph::GraphBuilder::ChromaLocationH> translate_chroma_location(zimg_chroma_location_e chromaloc)
{
	typedef zimg::graph::GraphBuilder::ChromaLocationW ChromaLocationW;
	typedef zimg::graph::GraphBuilder::ChromaLocationH ChromaLocationH;

	static const zimg::static_map<zimg_chroma_location_e, std::pair<ChromaLocationW, ChromaLocationH>, 6> map{
		{ ZIMG_CHROMA_LEFT,        { ChromaLocationW::CHROMA_W_LEFT,   ChromaLocationH::CHROMA_H_CENTER } },
		{ ZIMG_CHROMA_CENTER,      { ChromaLocationW::CHROMA_W_CENTER, ChromaLocationH::CHROMA_H_CENTER } },
		{ ZIMG_CHROMA_TOP_LEFT,    { ChromaLocationW::CHROMA_W_LEFT,   ChromaLocationH::CHROMA_H_TOP } },
		{ ZIMG_CHROMA_TOP,         { ChromaLocationW::CHROMA_W_CENTER, ChromaLocationH::CHROMA_H_TOP } },
		{ ZIMG_CHROMA_BOTTOM_LEFT, { ChromaLocationW::CHROMA_W_LEFT,   ChromaLocationH::CHROMA_H_BOTTOM } },
		{ ZIMG_CHROMA_BOTTOM,      { ChromaLocationW::CHROMA_W_CENTER, ChromaLocationH::CHROMA_H_BOTTOM } },
	};
	return search_enum_map(map, chromaloc, "unrecognized chroma location");
}

zimg::colorspace::MatrixCoefficients translate_matrix(zimg_matrix_coefficients_e matrix)
{
	using zimg::colorspace::MatrixCoefficients;

	static const zimg::static_map<zimg_matrix_coefficients_e, zimg::colorspace::MatrixCoefficients, 8> map{
		{ ZIMG_MATRIX_RGB,         MatrixCoefficients::MATRIX_RGB },
		{ ZIMG_MATRIX_709,         MatrixCoefficients::MATRIX_709 },
		{ ZIMG_MATRIX_UNSPECIFIED, MatrixCoefficients::MATRIX_UNSPECIFIED },
		{ ZIMG_MATRIX_470BG,       MatrixCoefficients::MATRIX_601 },
		{ ZIMG_MATRIX_170M,        MatrixCoefficients::MATRIX_601 },
		{ ZIMG_MATRIX_YCGCO,       MatrixCoefficients::MATRIX_YCGCO },
		{ ZIMG_MATRIX_2020_NCL,    MatrixCoefficients::MATRIX_2020_NCL },
		{ ZIMG_MATRIX_2020_CL,     MatrixCoefficients::MATRIX_2020_CL },
	};
	return search_itu_enum_map(map, matrix, "unrecognized matrix coefficients");
}

zimg::colorspace::TransferCharacteristics translate_transfer(zimg_transfer_characteristics_e transfer)
{
	using zimg::colorspace::TransferCharacteristics;

	static const zimg::static_map<zimg_transfer_characteristics_e, TransferCharacteristics, 7> map{
		{ ZIMG_TRANSFER_709,         TransferCharacteristics::TRANSFER_709 },
		{ ZIMG_TRANSFER_UNSPECIFIED, TransferCharacteristics::TRANSFER_UNSPECIFIED },
		{ ZIMG_TRANSFER_601,         TransferCharacteristics::TRANSFER_709 },
		{ ZIMG_TRANSFER_LINEAR,      TransferCharacteristics::TRANSFER_LINEAR },
		{ ZIMG_TRANSFER_2020_10,     TransferCharacteristics::TRANSFER_709 },
		{ ZIMG_TRANSFER_2020_12,     TransferCharacteristics::TRANSFER_709 },
		{ ZIMG_TRANSFER_2084,        TransferCharacteristics::TRANSFER_2084 },
	};
	return search_itu_enum_map(map, transfer, "unrecognized transfer characteristics");
}

zimg::colorspace::ColorPrimaries translate_primaries(zimg_color_primaries_e primaries)
{
	using zimg::colorspace::ColorPrimaries;

	static const zimg::static_map<zimg_color_primaries_e, ColorPrimaries, 5> map{
		{ ZIMG_PRIMARIES_709,         ColorPrimaries::PRIMARIES_709 },
		{ ZIMG_PRIMARIES_UNSPECIFIED, ColorPrimaries::PRIMARIES_UNSPECIFIED },
		{ ZIMG_PRIMARIES_170M,        ColorPrimaries::PRIMARIES_SMPTE_C },
		{ ZIMG_PRIMARIES_240M,        ColorPrimaries::PRIMARIES_SMPTE_C },
		{ ZIMG_PRIMARIES_2020,        ColorPrimaries::PRIMARIES_2020 },
	};
	return search_itu_enum_map(map, primaries, "unrecognized color primaries");
}

zimg::depth::DitherType translate_dither(zimg_dither_type_e dither)
{
	using zimg::depth::DitherType;

	static const zimg::static_map<zimg_dither_type_e, DitherType, 4> map{
		{ ZIMG_DITHER_NONE,            DitherType::DITHER_NONE },
		{ ZIMG_DITHER_ORDERED,         DitherType::DITHER_ORDERED },
		{ ZIMG_DITHER_RANDOM,          DitherType::DITHER_RANDOM },
		{ ZIMG_DITHER_ERROR_DIFFUSION, DitherType::DITHER_ERROR_DIFFUSION },
	};
	return search_enum_map(map, dither, "unrecognized dither type");
}

std::unique_ptr<zimg::resize::Filter> translate_resize_filter(zimg_resample_filter_e filter_type, double param_a, double param_b)
{
	try {
		switch (filter_type) {
		case ZIMG_RESIZE_POINT:
			return ztd::make_unique<zimg::resize::PointFilter>();
		case ZIMG_RESIZE_BILINEAR:
			return ztd::make_unique<zimg::resize::BilinearFilter>();
		case ZIMG_RESIZE_BICUBIC:
			param_a = std::isnan(param_a) ? 1.0 / 3.0 : param_a;
			param_b = std::isnan(param_b) ? 1.0 / 3.0 : param_b;
			return ztd::make_unique<zimg::resize::BicubicFilter>(param_a, param_b);
		case ZIMG_RESIZE_SPLINE16:
			return ztd::make_unique<zimg::resize::Spline16Filter>();
		case ZIMG_RESIZE_SPLINE36:
			return ztd::make_unique<zimg::resize::Spline36Filter>();
		case ZIMG_RESIZE_LANCZOS:
			param_a = std::isnan(param_a) ? 3.0 : std::floor(param_a);
			return ztd::make_unique<zimg::resize::LanczosFilter>((int)param_a);
		default:
			throw zimg::error::EnumOutOfRange{ "unrecognized resampling filter" };
		}
	} catch (const std::bad_alloc &) {
		throw zimg::error::OutOfMemory{};
	}
}

zimg::graph::ColorImageBuffer<void> import_image_buffer(const zimg_image_buffer &src)
{
	zimg::graph::ColorImageBuffer<void> dst{};

	API_VERSION_ASSERT(src.version);

	if (src.version >= API_VERSION_2_0) {
		for (unsigned p = 0; p < 3; ++p) {
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
		for (unsigned p = 0; p < 3; ++p) {
			dst[p] = zimg::graph::ImageBuffer<const void>{ src.plane[p].data, src.plane[p].stride, src.plane[p].mask };
		}
	}
	return dst;
}

void import_graph_state_common(const zimg_image_format &src, zimg::graph::GraphBuilder::state *out)
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
}

std::pair<zimg::graph::GraphBuilder::state, zimg::graph::GraphBuilder::state> import_graph_state(const zimg_image_format &src, const zimg_image_format &dst)
{
	API_VERSION_ASSERT(src.version);
	API_VERSION_ASSERT(dst.version);
	_zassert_d(src.version == dst.version, "image format versions do not match");

	zimg::graph::GraphBuilder::state src_state{};
	zimg::graph::GraphBuilder::state dst_state{};

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

zimg::graph::GraphBuilder::params import_graph_params(const zimg_graph_builder_params &src)
{
	API_VERSION_ASSERT(src.version);

	zimg::graph::GraphBuilder::params params{};

	if (src.version >= API_VERSION_2_0) {
		params.filter = translate_resize_filter(src.resample_filter, src.filter_param_a, src.filter_param_b);
		params.filter_uv = translate_resize_filter(src.resample_filter_uv, src.filter_param_a_uv, src.filter_param_b_uv);
		params.dither_type = translate_dither(src.dither_type);
		params.cpu = translate_cpu(src.cpu_type);
	}

	return params;
}

} // namespace


void zimg_get_version_info(unsigned *major, unsigned *minor, unsigned *micro)
{
	_zassert_d(major, "null pointer");
	_zassert_d(minor, "null pointer");
	_zassert_d(micro, "null pointer");

	*major = VERSION_INFO[0];
	*minor = VERSION_INFO[1];
	*micro = VERSION_INFO[2];
}

unsigned zimg_get_api_version(unsigned *major, unsigned *minor)
{
	if (major)
		*major = static_cast<unsigned>((ZIMG_API_VERSION >> 8) & 0xFF);
	if (minor)
		*minor = static_cast<unsigned>(ZIMG_API_VERSION & 0xFF);

	return ZIMG_API_VERSION;
}

zimg_error_code_e zimg_get_last_error(char *err_msg, size_t n)
{
	if (err_msg && n) {
		strncpy(err_msg, g_last_error_msg, n);
		err_msg[n - 1] = '\0';
	}

	return g_last_error;
}

void zimg_clear_last_error(void)
{
	g_last_error_msg[0] = '\0';
	g_last_error = ZIMG_ERROR_SUCCESS;
}

unsigned zimg_select_buffer_mask(unsigned count)
{
	return zimg::graph::select_zimg_buffer_mask(count);
}

#define EX_BEGIN \
  zimg_error_code_e ret = ZIMG_ERROR_SUCCESS; \
  try {
#define EX_END \
  } catch (const zimg::error::Exception &) { \
    ret = handle_exception(std::current_exception()); \
  } \
  return ret;

void zimg_filter_graph_free(zimg_filter_graph *ptr)
{
	delete ptr;
}

zimg_error_code_e zimg_filter_graph_get_tmp_size(const zimg_filter_graph *ptr, size_t *out)
{
	_zassert_d(ptr, "null pointer");
	_zassert_d(out, "null pointer");

	EX_BEGIN
	*out = assert_dynamic_cast<const zimg::graph::FilterGraph>(ptr)->get_tmp_size();
	EX_END
}

zimg_error_code_e zimg_filter_graph_get_input_buffering(const zimg_filter_graph *ptr, unsigned *out)
{
	_zassert_d(ptr, "null pointer");
	_zassert_d(out, "null pointer");

	EX_BEGIN
	*out = assert_dynamic_cast<const zimg::graph::FilterGraph>(ptr)->get_input_buffering();
	EX_END
}

zimg_error_code_e zimg_filter_graph_get_output_buffering(const zimg_filter_graph *ptr, unsigned *out)
{
	_zassert_d(ptr, "null pointer");
	_zassert_d(out, "null pointer");

	EX_BEGIN
	*out = assert_dynamic_cast<const zimg::graph::FilterGraph>(ptr)->get_output_buffering();
	EX_END
}

zimg_error_code_e zimg_filter_graph_process(const zimg_filter_graph *ptr, const zimg_image_buffer_const *src, const zimg_image_buffer *dst, void *tmp,
                                             zimg_filter_graph_callback unpack_cb, void *unpack_user,
                                             zimg_filter_graph_callback pack_cb, void *pack_user)
{
	_zassert_d(ptr, "null pointer");
	_zassert_d(src, "null pointer");
	_zassert_d(dst, "null pointer");

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

	POINTER_ALIGNMENT_ASSERT(tmp);

	EX_BEGIN
	const zimg::graph::FilterGraph *graph = assert_dynamic_cast<const zimg::graph::FilterGraph>(ptr);
	auto src_buf = import_image_buffer(*src);
	auto dst_buf = import_image_buffer(*dst);

	graph->process(src_buf, dst_buf, tmp, { unpack_cb, unpack_user }, { pack_cb, pack_user });
	EX_END
}

#undef EX_BEGIN
#undef EX_END

void zimg_image_format_default(zimg_image_format *ptr, unsigned version)
{
	_zassert_d(ptr, "null pointer");
	API_VERSION_ASSERT(version);

	ptr->version = version;

	if (version >= API_VERSION_2_0) {
		ptr->width = 0;
		ptr->height = 0;
		ptr->pixel_type = (zimg_pixel_type_e)-1;

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
}

void zimg_graph_builder_params_default(zimg_graph_builder_params *ptr, unsigned version)
{
	_zassert_d(ptr, "null pointer");
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
}

zimg_filter_graph *zimg_filter_graph_build(const zimg_image_format *src_format, const zimg_image_format *dst_format, const zimg_graph_builder_params *params)
{
	_zassert_d(src_format, "null pointer");
	_zassert_d(dst_format, "null pointer");

	try {
		zimg::graph::GraphBuilder::state src_state;
		zimg::graph::GraphBuilder::state dst_state;
		zimg::graph::GraphBuilder::params graph_params;
		zimg::graph::DefaultFilterFactory factory;

		std::tie(src_state, dst_state) = import_graph_state(*src_format, *dst_format);
		if (params)
			graph_params = import_graph_params(*params);

		return zimg::graph::GraphBuilder{}.set_factory(&factory).
		                                   set_source(src_state).
		                                   connect_graph(dst_state, params ? &graph_params : nullptr).
		                                   complete_graph().release();
	} catch (const zimg::error::Exception &) {
		handle_exception(std::current_exception());
		return nullptr;
	}
}
