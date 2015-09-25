#include <algorithm>
#include <cassert>
#include <cmath>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include "Common/cpuinfo.h"
#include "Common/except.h"
#include "Common/filtergraph.h"
#include "Common/mux_filter.h"
#include "Common/osdep.h"
#include "Common/pixel.h"
#include "Common/static_map.h"
#include "Common/zassert.h"
#include "Common/zfilter.h"
#include "Colorspace/colorspace2.h"
#include "Colorspace/colorspace_param.h"
#include "Depth/depth2.h"
#include "Resize/filter.h"
#include "Resize/resize2.h"
#include "zimg3.h"

#define API_VERSION_ASSERT(x) _zassert_d((x) >= 2 && (x) <= ZIMG_API_VERSION, "API version invalid")
#define POINTER_ALIGNMENT_ASSERT(x) _zassert_d(!(x) || reinterpret_cast<uintptr_t>(x) % zimg::ALIGNMENT == 0, "pointer not aligned")
#define STRIDE_ALIGNMENT_ASSERT(x) _zassert_d(!(x) || (x) % zimg::ALIGNMENT == 0, "buffer stride not aligned")

namespace {;

THREAD_LOCAL zimg_error_code_e g_last_error = ZIMG_ERROR_SUCCESS;
THREAD_LOCAL char g_last_error_msg[1024];

const unsigned VERSION_INFO[] = { 1, 91, 0 };


enum class ColorFamily {
	COLOR_GREY,
	COLOR_RGB,
	COLOR_YUV
};

enum class FieldParity {
	FIELD_PROGRESSIVE,
	FIELD_TOP,
	FIELD_BOTTOM
};

enum class ChromaLocationW {
	CHROMA_W_LEFT,
	CHROMA_W_CENTER
};

enum class ChromaLocationH {
	CHROMA_H_CENTER,
	CHROMA_H_TOP,
	CHROMA_H_BOTTOM
};


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
	zimg_error_code_e code = ZIMG_ERROR_UNKNOWN;

#define CATCH(type, error_code) catch (const type &e) { record_exception_message(e); code = (error_code); }
#define FATAL(type, error_code, msg) catch (const type &e) { _zassert_d(false, msg); record_exception_message(e); code = (error_code); }
	try {
		std::rethrow_exception(eptr);
	}

	CATCH(zimg::error::UnknownError,            ZIMG_ERROR_UNKNOWN)
	CATCH(zimg::error::OutOfMemory,             ZIMG_ERROR_OUT_OF_MEMORY)
	CATCH(zimg::error::UserCallbackFailed,      ZIMG_ERROR_USER_CALLBACK_FAILED)

	CATCH(zimg::error::GreyscaleSubsampling,    ZIMG_ERROR_GREYSCALE_SUBSAMPLING)
	CATCH(zimg::error::ColorFamilyMismatch,     ZIMG_ERROR_COLOR_FAMILY_MISMATCH)
	CATCH(zimg::error::ImageNotDivislbe,        ZIMG_ERROR_IMAGE_NOT_DIVISIBLE)
	CATCH(zimg::error::BitDepthOverflow,        ZIMG_ERROR_BIT_DEPTH_OVERFLOW)
	CATCH(zimg::error::LogicError,              ZIMG_ERROR_LOGIC)

	CATCH(zimg::error::EnumOutOfRange,          ZIMG_ERROR_ENUM_OUT_OF_RANGE)
	CATCH(zimg::error::ZeroImageSize,           ZIMG_ERROR_ZERO_IMAGE_SIZE)
	CATCH(zimg::error::IllegalArgument,         ZIMG_ERROR_ILLEGAL_ARGUMENT)

	CATCH(zimg::error::UnsupportedSubsampling,  ZIMG_ERROR_UNSUPPORTED_SUBSAMPLING)
	CATCH(zimg::error::NoColorspaceConversion,  ZIMG_ERROR_NO_COLORSPACE_CONVERSION)
	CATCH(zimg::error::NoFieldParityConversion, ZIMG_ERROR_NO_FIELD_PARITY_CONVERSION)
	CATCH(zimg::error::ResamplingNotAvailable,  ZIMG_ERROR_RESAMPLING_NOT_AVAILABLE)
	CATCH(zimg::error::UnsupportedOperation,    ZIMG_ERROR_UNSUPPORTED_OPERATION)

	FATAL(zimg::error::InternalError,           ZIMG_ERROR_UNKNOWN, "internal error generated")
	FATAL(zimg::error::Exception,               ZIMG_ERROR_UNKNOWN, "unregistered error generated")
#undef CATCH
#undef FATAL
	g_last_error = code;
	return code;
}

zimg_error_code_e handle_exception(const std::bad_alloc &e)
{
	zimg_error_code_e code = ZIMG_ERROR_OUT_OF_MEMORY;

	record_exception_message(e);
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
	static const zimg::static_enum_map<zimg_cpu_type_e, zimg::CPUClass, 12> map{
		{ ZIMG_CPU_NONE,      zimg::CPUClass::CPU_NONE },
		{ ZIMG_CPU_AUTO,      zimg::CPUClass::CPU_AUTO },
#ifdef ZIMG_X86
		{ ZIMG_CPU_X86_MMX,   zimg::CPUClass::CPU_NONE },
		{ ZIMG_CPU_X86_SSE,   zimg::CPUClass::CPU_X86_SSE },
		{ ZIMG_CPU_X86_SSE2,  zimg::CPUClass::CPU_X86_SSE2 },
		{ ZIMG_CPU_X86_SSE3,  zimg::CPUClass::CPU_X86_SSE2 },
		{ ZIMG_CPU_X86_SSSE3, zimg::CPUClass::CPU_X86_SSE2 },
		{ ZIMG_CPU_X86_SSE41, zimg::CPUClass::CPU_X86_SSE2 },
		{ ZIMG_CPU_X86_SSE42, zimg::CPUClass::CPU_X86_SSE2 },
		{ ZIMG_CPU_X86_AVX,   zimg::CPUClass::CPU_X86_SSE2 },
		{ ZIMG_CPU_X86_F16C,  zimg::CPUClass::CPU_X86_SSE2 },
		{ ZIMG_CPU_X86_AVX2,  zimg::CPUClass::CPU_X86_AVX2 },
#endif
	};
	return search_enum_map(map, cpu, "unrecognized cpu type");
}

zimg::PixelType translate_pixel_type(zimg_pixel_type_e pixel_type)
{
	static const zimg::static_enum_map<zimg_pixel_type_e, zimg::PixelType, 4> map{
		{ ZIMG_PIXEL_BYTE,  zimg::PixelType::BYTE },
		{ ZIMG_PIXEL_WORD,  zimg::PixelType::WORD },
		{ ZIMG_PIXEL_HALF,  zimg::PixelType::HALF },
		{ ZIMG_PIXEL_FLOAT, zimg::PixelType::FLOAT },
	};
	return search_enum_map(map, pixel_type, "unrecognized pixel type");
}

bool translate_pixel_range(zimg_pixel_range_e range)
{
	static const zimg::static_enum_map<zimg_pixel_range_e, bool, 2> map{
		{ ZIMG_RANGE_LIMITED, false },
		{ ZIMG_RANGE_FULL,    true },
	};
	return search_enum_map(map, range, "unrecognized pixel range");
}

ColorFamily translate_color_family(zimg_color_family_e family)
{
	static const zimg::static_enum_map<zimg_color_family_e, ColorFamily, 3> map{
		{ ZIMG_COLOR_GREY, ColorFamily::COLOR_GREY },
		{ ZIMG_COLOR_RGB,  ColorFamily::COLOR_RGB },
		{ ZIMG_COLOR_YUV,  ColorFamily::COLOR_YUV},
	};
	return search_enum_map(map, family, "unrecognized color family");
}

FieldParity translate_field_parity(zimg_field_parity_e field)
{
	static const zimg::static_enum_map<zimg_field_parity_e, FieldParity, 3> map{
		{ ZIMG_FIELD_PROGRESSIVE, FieldParity::FIELD_PROGRESSIVE },
		{ ZIMG_FIELD_TOP,         FieldParity::FIELD_TOP },
		{ ZIMG_FIELD_BOTTOM,      FieldParity::FIELD_BOTTOM },
	};
	return search_enum_map(map, field, "unrecognized field parity");
}

std::pair<ChromaLocationW, ChromaLocationH> translate_chroma_location(zimg_chroma_location_e chromaloc)
{
	static const zimg::static_enum_map<zimg_chroma_location_e, std::pair<ChromaLocationW, ChromaLocationH>, 6> map{
		{ ZIMG_CHROMA_LEFT,        { ChromaLocationW::CHROMA_W_LEFT,   ChromaLocationH::CHROMA_H_CENTER } },
		{ ZIMG_CHROMA_CENTER,      { ChromaLocationW::CHROMA_W_CENTER, ChromaLocationH::CHROMA_H_CENTER } },
		{ ZIMG_CHROMA_TOP_LEFT,    { ChromaLocationW::CHROMA_W_LEFT,   ChromaLocationH::CHROMA_H_TOP } },
		{ ZIMG_CHROMA_TOP,         { ChromaLocationW::CHROMA_W_CENTER, ChromaLocationH::CHROMA_H_TOP } },
		{ ZIMG_CHROMA_BOTTOM_LEFT, { ChromaLocationW::CHROMA_W_LEFT,   ChromaLocationH::CHROMA_H_BOTTOM } },
		{ ZIMG_CHROMA_BOTTOM,      { ChromaLocationW::CHROMA_W_CENTER, ChromaLocationH::CHROMA_H_BOTTOM } },
	};
	return search_enum_map(map, chromaloc, "unregonized chroma location");
}

zimg::colorspace::MatrixCoefficients translate_matrix(zimg_matrix_coefficients_e matrix)
{
	static const zimg::static_enum_map<zimg_matrix_coefficients_e, zimg::colorspace::MatrixCoefficients, 7> map{
		{ ZIMG_MATRIX_RGB,         zimg::colorspace::MatrixCoefficients::MATRIX_RGB },
		{ ZIMG_MATRIX_709,         zimg::colorspace::MatrixCoefficients::MATRIX_709 },
		{ ZIMG_MATRIX_UNSPECIFIED, zimg::colorspace::MatrixCoefficients::MATRIX_UNSPECIFIED },
		{ ZIMG_MATRIX_470BG,       zimg::colorspace::MatrixCoefficients::MATRIX_601 },
		{ ZIMG_MATRIX_170M,        zimg::colorspace::MatrixCoefficients::MATRIX_601 },
		{ ZIMG_MATRIX_2020_NCL,    zimg::colorspace::MatrixCoefficients::MATRIX_2020_NCL },
		{ ZIMG_MATRIX_2020_CL,     zimg::colorspace::MatrixCoefficients::MATRIX_2020_CL },
	};
	return search_itu_enum_map(map, matrix, "unrecognized matrix coefficients");
}

zimg::colorspace::TransferCharacteristics translate_transfer(zimg_transfer_characteristics_e transfer)
{
	static const zimg::static_enum_map<zimg_transfer_characteristics_e, zimg::colorspace::TransferCharacteristics, 6> map{
		{ ZIMG_TRANSFER_709,         zimg::colorspace::TransferCharacteristics::TRANSFER_709 },
		{ ZIMG_TRANSFER_UNSPECIFIED, zimg::colorspace::TransferCharacteristics::TRANSFER_UNSPECIFIED },
		{ ZIMG_TRANSFER_601,         zimg::colorspace::TransferCharacteristics::TRANSFER_709 },
		{ ZIMG_TRANSFER_2020_10,     zimg::colorspace::TransferCharacteristics::TRANSFER_709 },
		{ ZIMG_TRANSFER_2020_12,     zimg::colorspace::TransferCharacteristics::TRANSFER_709 },
		{ ZIMG_TRANSFER_LINEAR,      zimg::colorspace::TransferCharacteristics::TRANSFER_LINEAR },
	};
	return search_itu_enum_map(map, transfer, "unrecognized transfer characteristics");
}

zimg::colorspace::ColorPrimaries translate_primaries(zimg_color_primaries_e primaries)
{
	static const zimg::static_enum_map<zimg_color_primaries_e, zimg::colorspace::ColorPrimaries, 5> map{
		{ ZIMG_PRIMARIES_709,         zimg::colorspace::ColorPrimaries::PRIMARIES_709 },
		{ ZIMG_PRIMARIES_UNSPECIFIED, zimg::colorspace::ColorPrimaries::PRIMARIES_UNSPECIFIED },
		{ ZIMG_PRIMARIES_170M,        zimg::colorspace::ColorPrimaries::PRIMARIES_SMPTE_C },
		{ ZIMG_PRIMARIES_240M,        zimg::colorspace::ColorPrimaries::PRIMARIES_SMPTE_C },
		{ ZIMG_PRIMARIES_2020,        zimg::colorspace::ColorPrimaries::PRIMARIES_2020 },
	};
	return search_itu_enum_map(map, primaries, "unrecognized color primaries");
}

zimg::depth::DitherType translate_dither(zimg_dither_type_e dither)
{
	static const zimg::static_enum_map<zimg_dither_type_e, zimg::depth::DitherType, 4> map{
		{ ZIMG_DITHER_NONE,            zimg::depth::DitherType::DITHER_NONE },
		{ ZIMG_DITHER_ORDERED,         zimg::depth::DitherType::DITHER_ORDERED },
		{ ZIMG_DITHER_RANDOM,          zimg::depth::DitherType::DITHER_RANDOM },
		{ ZIMG_DITHER_ERROR_DIFFUSION, zimg::depth::DitherType::DITHER_ERROR_DIFFUSION },
	};
	return search_enum_map(map, dither, "urecognized dither type");
}

zimg::resize::Filter *translate_resize_filter(zimg_resample_filter_e filter_type, double param_a, double param_b)
{
	try {
		switch (filter_type) {
		case ZIMG_RESIZE_POINT:
			return new zimg::resize::PointFilter{};
		case ZIMG_RESIZE_BILINEAR:
			return new zimg::resize::BilinearFilter{};
		case ZIMG_RESIZE_BICUBIC:
			param_a = std::isnan(param_a) ? 1.0 / 3.0 : param_a;
			param_b = std::isnan(param_b) ? 1.0 / 3.0 : param_b;
			return new zimg::resize::BicubicFilter{ param_a, param_b };
		case ZIMG_RESIZE_SPLINE16:
			return new zimg::resize::Spline16Filter{};
		case ZIMG_RESIZE_SPLINE36:
			return new zimg::resize::Spline36Filter{};
		case ZIMG_RESIZE_LANCZOS:
			param_a = std::isnan(param_a) ? 3.0 : std::floor(param_a);
			return new zimg::resize::LanczosFilter{ (int)param_a };
		default:
			throw zimg::error::EnumOutOfRange{ "unrecognized resampling filter" };
		}
	} catch (const std::bad_alloc &) {
		throw zimg::ZimgOutOfMemory{};
	}
}

zimg::ZimgImageBuffer import_image_buffer(const zimg_image_buffer &src)
{
	zimg::ZimgImageBuffer dst{};

	API_VERSION_ASSERT(src.m.version);

#define COPY_ARRAY(x, y) std::copy(std::begin(x), std::end(x), y)
	if (src.m.version >= 2) {
		COPY_ARRAY(src.m.data, dst.data);
		COPY_ARRAY(src.m.stride, dst.stride);
		COPY_ARRAY(src.m.mask, dst.mask);
	}
#undef COPY_ARRAY
	return dst;
}

zimg::ZimgImageBufferConst import_image_buffer(const zimg_image_buffer_const &src)
{
	zimg::ZimgImageBufferConst dst{};

	API_VERSION_ASSERT(src.version);

#define COPY_ARRAY(x, y) std::copy(std::begin(x), std::end(x), y)
	if (src.version >= 2) {
		COPY_ARRAY(src.data, dst.data);
		COPY_ARRAY(src.stride, dst.stride);
		COPY_ARRAY(src.mask, dst.mask);
	}
#undef COPY_ARRAY
	return dst;
}


double chroma_shift_raw(ChromaLocationW loc, FieldParity)
{
	if (loc == ChromaLocationW::CHROMA_W_LEFT)
		return -0.5;
	else
		return 0.0;
}

double chroma_shift_raw(ChromaLocationH loc, FieldParity parity)
{
	double shift;

	if (loc == ChromaLocationH::CHROMA_H_TOP)
		shift = -0.5;
	else if (loc == ChromaLocationH::CHROMA_H_BOTTOM)
		shift = 0.5;
	else
		shift = 0;

	if (parity == FieldParity::FIELD_TOP)
		shift = (shift - 0.5) / 2.0;
	else if (parity == FieldParity::FIELD_BOTTOM)
		shift = (shift + 0.5) / 2.0;

	return shift;
}

template <class T>
double chroma_shift_factor(T loc_in, T loc_out, unsigned subsample_in, unsigned subsample_out, FieldParity parity, unsigned src_dim, unsigned dst_dim)
{
	double shift = 0.0;
	double sub_scale = 1.0 / (1 << subsample_in);

	if (subsample_in)
		shift -= sub_scale * chroma_shift_raw(loc_in, parity);
	if (subsample_out)
		shift += sub_scale * chroma_shift_raw(loc_out, parity) * (double)src_dim / dst_dim;

	return shift;
}

double luma_shift_factor(FieldParity parity, unsigned src_height, unsigned dst_height)
{
	double shift = 0.0;

	if (parity == FieldParity::FIELD_TOP)
		shift = -0.25;
	else if (parity == FieldParity::FIELD_BOTTOM)
		shift = 0.25;

	return shift * (double)src_height / dst_height - shift;
}

class GraphBuilder {
public:
	struct params {
		std::unique_ptr<zimg::resize::Filter> filter;
		std::unique_ptr<zimg::resize::Filter> filter_uv;
		zimg::depth::DitherType dither_type;
		zimg::CPUClass cpu;
	};

	struct state {
		unsigned width;
		unsigned height;
		zimg::PixelType type;
		unsigned subsample_w;
		unsigned subsample_h;

		ColorFamily color;
		zimg::colorspace::ColorspaceDefinition colorspace;

		unsigned depth;
		bool fullrange;

		FieldParity parity;
		ChromaLocationW chroma_location_w;
		ChromaLocationH chroma_location_h;

		void validate() const
		{
			if (!width || !height)
				throw zimg::error::ZeroImageSize{ "image dimensions must be non-zero" };

			if (color == ColorFamily::COLOR_GREY) {
				if (subsample_w || subsample_h)
					throw zimg::error::GreyscaleSubsampling{ "cannot subsample greyscale image" };
				if (colorspace.matrix != zimg::colorspace::MatrixCoefficients::MATRIX_UNSPECIFIED ||
					colorspace.transfer != zimg::colorspace::TransferCharacteristics::TRANSFER_UNSPECIFIED ||
					colorspace.primaries != zimg::colorspace::ColorPrimaries::PRIMARIES_UNSPECIFIED)
					throw zimg::error::NoColorspaceConversion{ "cannot specify colorspace of greyscale image" };
			}

			if (color == ColorFamily::COLOR_RGB) {
				if (subsample_w || subsample_h)
					throw zimg::error::UnsupportedSubsampling{ "subsampled RGB image not supported" };
				if (colorspace.matrix != zimg::colorspace::MatrixCoefficients::MATRIX_UNSPECIFIED &&
					colorspace.matrix != zimg::colorspace::MatrixCoefficients::MATRIX_RGB)
					throw zimg::error::ColorFamilyMismatch{ "RGB color family cannot be YUV" };
			}

			if (color == ColorFamily::COLOR_YUV) {
				if (colorspace.matrix == zimg::colorspace::MatrixCoefficients::MATRIX_RGB)
					throw zimg::error::ColorFamilyMismatch{ "YUV color family cannot be RGB" };
			}

			if (subsample_h > 1 && parity != FieldParity::FIELD_PROGRESSIVE)
				throw zimg::error::UnsupportedSubsampling{ "vertical subsampling greater than 2x is not supported" };
			if (subsample_w > 2 || subsample_h > 2)
				throw zimg::error::UnsupportedSubsampling{ "subsampling greater than 4x is not supported" };

			if (width % (1 << subsample_w) || height % (1 << subsample_h))
				throw zimg::error::ImageNotDivislbe{ "image dimensions must be divisible by subsampling factor" };

			if (depth > (unsigned)zimg::default_pixel_format(type).depth)
				throw zimg::error::BitDepthOverflow{ "bit depth exceeds limits of type" };
		}

		bool is_greyscale() const
		{
			return color == ColorFamily::COLOR_GREY;
		}

		bool is_rgb() const
		{
			return color == ColorFamily::COLOR_RGB;
		}

		bool is_yuv() const
		{
			return color == ColorFamily::COLOR_YUV;
		}
	};
private:
	std::unique_ptr<zimg::FilterGraph> m_graph;
	state m_state;
	bool m_dirty;

	zimg::PixelType select_working_type(const state &target) const
	{
		if (needs_colorspace(target)) {
			return zimg::PixelType::FLOAT;
		} else if (needs_resize(target)) {
			if (m_state.type == zimg::PixelType::BYTE)
				return zimg::PixelType::WORD;
			else if (m_state.type == zimg::PixelType::HALF)
				return zimg::PixelType::FLOAT;
			else
				return m_state.type;
		} else {
			return m_state.type;
		}
	}

	bool needs_colorspace(const state &target) const
	{
		return m_state.colorspace != target.colorspace;
	}

	bool needs_depth(const state &target) const
	{
		return m_state.type != target.type || m_state.depth != target.depth || m_state.fullrange != target.fullrange;
	}

	bool needs_resize(const state &target) const
	{
		return m_state.width != target.width ||
		       m_state.height != target.height ||
		       m_state.subsample_w != target.subsample_w ||
		       m_state.subsample_h != target.subsample_h ||
		       (m_state.subsample_w && m_state.chroma_location_w != target.chroma_location_w) ||
		       (m_state.subsample_h && m_state.chroma_location_h != target.chroma_location_h);
	}

	void attach_filter(std::unique_ptr<zimg::IZimgFilter> &&filter)
	{
		m_graph->attach_filter(filter.get());
		filter.release();
		m_dirty = true;
	}

	void attach_filter_uv(std::unique_ptr<zimg::IZimgFilter> &&filter)
	{
		m_graph->attach_filter_uv(filter.get());
		filter.release();
		m_dirty = true;
	}

	void convert_colorspace(const zimg::colorspace::ColorspaceDefinition &colorspace, const params *params)
	{
		if (m_state.is_greyscale())
			throw zimg::error::NoColorspaceConversion{ "cannot apply colorspace conversion to greyscale image" };

		std::unique_ptr<zimg::IZimgFilter> filter;
		zimg::CPUClass cpu = params ? params->cpu : zimg::CPUClass::CPU_AUTO;

		if (m_state.colorspace == colorspace)
			return;

		filter.reset(new zimg::colorspace::ColorspaceConversion2{ m_state.width, m_state.height, m_state.colorspace, colorspace, cpu });
		attach_filter(std::move(filter));

		m_state.color = colorspace.matrix == zimg::colorspace::MatrixCoefficients::MATRIX_RGB ? ColorFamily::COLOR_RGB : ColorFamily::COLOR_YUV;
		m_state.colorspace = colorspace;
	}

	void convert_depth(const zimg::PixelFormat &format, const params *params)
	{
		zimg::depth::DitherType dither_type = params ? params->dither_type : zimg::depth::DitherType::DITHER_NONE;
		zimg::PixelFormat src_format = zimg::default_pixel_format(m_state.type);

		std::unique_ptr<zimg::IZimgFilter> filter;
		std::unique_ptr<zimg::IZimgFilter> filter_uv;
		zimg::CPUClass cpu = params ? params->cpu : zimg::CPUClass::CPU_AUTO;

		if (src_format == format)
			return;

		src_format.depth = m_state.depth;
		src_format.fullrange = m_state.fullrange;

		filter.reset(zimg::depth::create_depth2(dither_type, m_state.width, m_state.height, src_format, format, cpu));

		if (m_state.is_yuv()) {
			zimg::PixelFormat src_format_uv = src_format;
			zimg::PixelFormat format_uv = format;

			src_format_uv.chroma = true;
			format_uv.chroma = true;

			filter_uv.reset(
				zimg::depth::create_depth2(dither_type, m_state.width >> m_state.subsample_w, m_state.height >> m_state.subsample_h,
				                           src_format_uv, format_uv, cpu));
		} else if (m_state.is_rgb()) {
			std::unique_ptr<zimg::IZimgFilter> mux{ new zimg::MuxFilter{ filter.get(), filter_uv.get() } };
			filter.release();
			filter_uv.release();

			filter = std::move(mux);
		}

		attach_filter(std::move(filter));
		if (filter_uv)
			attach_filter_uv(std::move(filter_uv));

		m_state.type = format.type;
		m_state.depth = format.depth;
		m_state.fullrange = format.fullrange;
	}

	void convert_resize(unsigned width, unsigned height, unsigned subsample_w, unsigned subsample_h,
	                    ChromaLocationW chroma_location_w, ChromaLocationH chroma_location_h, const params *params)
	{
		zimg::resize::BicubicFilter bicubic_filter{ 1.0 / 3.0, 1.0 / 3.0 };
		zimg::resize::BilinearFilter bilinear_filter;

		if (m_state.is_greyscale()) {
			subsample_w = 0;
			subsample_h = 0;
		}
		if (!subsample_w)
			chroma_location_w = ChromaLocationW::CHROMA_W_CENTER;
		if (!subsample_h)
			chroma_location_h = ChromaLocationH::CHROMA_H_CENTER;

		if (m_state.width == width &&
		    m_state.height == height &&
		    m_state.subsample_w == subsample_w &&
		    m_state.subsample_h == subsample_h &&
		    m_state.chroma_location_w == chroma_location_w &&
		    m_state.chroma_location_h == chroma_location_h)
			return;

		const zimg::resize::Filter *resample_filter = params ? params->filter.get() : &bicubic_filter;
		const zimg::resize::Filter *resample_filter_uv = params ? params->filter_uv.get() : &bilinear_filter;
		zimg::CPUClass cpu = params ? params->cpu : zimg::CPUClass::CPU_AUTO;

		bool do_resize_luma = m_state.width != width || m_state.height != height;
		bool do_resize_chroma = (m_state.width >> m_state.subsample_w != width >> subsample_w) ||
		                        (m_state.height >> m_state.subsample_h != height >> subsample_h) ||
		                        ((m_state.subsample_w || subsample_w) && m_state.chroma_location_w != chroma_location_w) ||
		                        ((m_state.subsample_h || subsample_h) && m_state.chroma_location_h != chroma_location_h);

		std::unique_ptr<zimg::IZimgFilter> filter1;
		std::unique_ptr<zimg::IZimgFilter> filter2;
		std::unique_ptr<zimg::IZimgFilter> filter1_uv;
		std::unique_ptr<zimg::IZimgFilter> filter2_uv;

		if (do_resize_luma) {
			double shift_h = luma_shift_factor(m_state.parity, m_state.height, height);
			auto filter_pair = zimg::resize::create_resize2(*resample_filter, m_state.type, m_state.depth, m_state.width, m_state.height, width, height,
			                                                0.0, shift_h, m_state.width, m_state.height, cpu);
			filter1.reset(filter_pair.first);
			filter2.reset(filter_pair.second);

			if (m_state.is_rgb()) {
				std::unique_ptr<zimg::IZimgFilter> mux;

				mux.reset(new zimg::MuxFilter{ filter1.get(), nullptr });
				filter1.release();
				filter1 = std::move(mux);

				if (filter2) {
					mux.reset(new zimg::MuxFilter{ filter2.get(), nullptr });
					filter2.release();
					filter2 = std::move(mux);
				}
			}
		}
		if (m_state.is_yuv() && do_resize_chroma) {
			double shift_w = chroma_shift_factor(m_state.chroma_location_w, chroma_location_w, m_state.subsample_w, subsample_w, m_state.parity, m_state.width, width);
			double shift_h = chroma_shift_factor(m_state.chroma_location_h, chroma_location_h, m_state.subsample_h, subsample_h, m_state.parity, m_state.height, height);

			unsigned chroma_width_in = m_state.width >> m_state.subsample_w;
			unsigned chroma_height_in = m_state.height >> m_state.subsample_h;
			unsigned chroma_width_out = width >> subsample_w;
			unsigned chroma_height_out = height >> subsample_h;

			auto filter_pair = zimg::resize::create_resize2(*resample_filter_uv, m_state.type, m_state.depth, chroma_width_in, chroma_height_in, chroma_width_out, chroma_height_out,
			                                                shift_w, shift_h, chroma_width_in, chroma_height_in, cpu);
			filter1_uv.reset(filter_pair.first);
			filter2_uv.reset(filter_pair.second);
		}

		if (filter1)
			attach_filter(std::move(filter1));
		if (filter2)
			attach_filter(std::move(filter2));
		if (filter1_uv)
			attach_filter_uv(std::move(filter1_uv));
		if (filter2_uv)
			attach_filter_uv(std::move(filter2_uv));

		m_state.width = width;
		m_state.height = height;
		m_state.subsample_w = subsample_w;
		m_state.subsample_h = subsample_h;
		m_state.chroma_location_w = chroma_location_w;
		m_state.chroma_location_h = chroma_location_h;
	}
public:
	GraphBuilder() : m_graph{}, m_state{}, m_dirty{}
	{
	}

	void set_source(const state &source)
	{
		if (m_dirty || m_graph)
			throw zimg::error::InternalError{ "source already set" };

		source.validate();

		m_graph.reset(new zimg::FilterGraph{ source.width, source.height, source.type, source.subsample_w, source.subsample_h, source.color != ColorFamily::COLOR_GREY });
		m_state = source;
	}

	zimg::FilterGraph *build(const state &target, const params *params)
	{
		if (m_dirty || !m_graph)
			throw zimg::error::InternalError{ "graph already built" };
		if (m_state.parity != target.parity)
			throw zimg::error::NoFieldParityConversion{ "conversion between field parity not supported" };
		if (m_state.is_greyscale() && !target.is_greyscale())
			throw zimg::error::NoColorspaceConversion{ "conversion between greyscale and color image not supported" };

		target.validate();
		convert_depth(zimg::default_pixel_format(select_working_type(target)), params);

		while (true) {
			if (needs_colorspace(target)) {
				unsigned width_444 = std::min(m_state.width, target.width);
				unsigned height_444 = std::min(m_state.height, target.height);

				convert_resize(width_444, height_444, 0, 0, ChromaLocationW::CHROMA_W_CENTER, ChromaLocationH::CHROMA_H_CENTER, params);
				convert_colorspace(target.colorspace, params);
			} else if (needs_resize(target)) {
				convert_resize(target.width, target.height, target.subsample_w, target.subsample_h, target.chroma_location_w, target.chroma_location_h, params);
			} else if (needs_depth(target)) {
				zimg::PixelFormat format = zimg::default_pixel_format(target.type);
				format.depth = target.depth;
				format.fullrange = target.fullrange;

				convert_depth(format, params);
			} else {
				break;
			}
		}

		m_graph->complete();
		return m_graph.release();
	}
};


void import_graph_state_common(const zimg_image_format &src, GraphBuilder::state *out)
{
	API_VERSION_ASSERT(src.version);

	if (src.version >= 2) {
		out->width = src.width;
		out->height = src.height;
		out->type = translate_pixel_type(src.pixel_type);
		out->subsample_w = src.subsample_w;
		out->subsample_h = src.subsample_h;
		out->color = translate_color_family(src.color_family);

		out->colorspace.matrix = translate_matrix(src.matrix_coefficients);
		out->colorspace.transfer = translate_transfer(src.transfer_characteristics);
		out->colorspace.primaries = translate_primaries(src.color_primaries);

		out->depth = src.depth ? src.depth : zimg::default_pixel_format(out->type).depth;
		out->fullrange = translate_pixel_range(src.pixel_range);

		out->parity = translate_field_parity(src.field_parity);
		std::tie(out->chroma_location_w, out->chroma_location_h) = translate_chroma_location(src.chroma_location);
	}
}

std::pair<GraphBuilder::state, GraphBuilder::state> import_graph_state(const zimg_image_format &src, const zimg_image_format &dst)
{
	API_VERSION_ASSERT(src.version);
	API_VERSION_ASSERT(dst.version);
	_zassert_d(src.version == dst.version, "image format versions do not match");

	GraphBuilder::state src_state{};
	GraphBuilder::state dst_state{};

	import_graph_state_common(src, &src_state);
	import_graph_state_common(dst, &dst_state);

	if (src.version >= 2) {
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

GraphBuilder::params import_graph_params(const zimg_filter_graph_params &src)
{
	API_VERSION_ASSERT(src.version);

	GraphBuilder::params params{};

	if (src.version >= 2) {
		params.filter.reset(translate_resize_filter(src.resample_filter, src.filter_param_a, src.filter_param_b));
		params.filter_uv.reset(translate_resize_filter(src.resample_filter_uv, src.filter_param_a_uv, src.filter_param_b_uv));

		params.dither_type = translate_dither(src.dither_type);

		params.cpu = translate_cpu(src.cpu_type);
	}

	return params;
}

} // namespace


void zimg2_get_version_info(unsigned *major, unsigned *minor, unsigned *micro)
{
	_zassert_d(major, "null pointer");
	_zassert_d(minor, "null pointer");
	_zassert_d(micro, "null pointer");

	*major = VERSION_INFO[0];
	*minor = VERSION_INFO[1];
	*micro = VERSION_INFO[2];
}

unsigned zimg2_get_api_version(void)
{
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

unsigned zimg2_select_buffer_mask(unsigned count)
{
	return zimg::select_zimg_buffer_mask(count);
}

#define EX_BEGIN \
  zimg_error_code_e ret = ZIMG_ERROR_SUCCESS; \
  try {
#define EX_END \
  } catch (const zimg::error::Exception &) { \
    ret = handle_exception(std::current_exception()); \
  } \
  return ret;

void zimg2_filter_graph_free(zimg_filter_graph *ptr)
{
	delete ptr;
}

zimg_error_code_e zimg2_filter_graph_get_tmp_size(const zimg_filter_graph *ptr, size_t *out)
{
	_zassert_d(ptr, "null pointer");
	_zassert_d(out, "null pointer");

	EX_BEGIN
	*out = assert_dynamic_cast<const zimg::FilterGraph>(ptr)->get_tmp_size();
	EX_END
}

zimg_error_code_e zimg2_filter_graph_get_input_buffering(const zimg_filter_graph *ptr, unsigned *out)
{
	_zassert_d(ptr, "null pointer");
	_zassert_d(out, "null pointer");

	EX_BEGIN
	*out = assert_dynamic_cast<const zimg::FilterGraph>(ptr)->get_input_buffering();
	EX_END
}

zimg_error_code_e zimg2_filter_graph_get_output_buffering(const zimg_filter_graph *ptr, unsigned *out)
{
	_zassert_d(ptr, "null pointer");
	_zassert_d(out, "null pointer");

	EX_BEGIN
	*out = assert_dynamic_cast<const zimg::FilterGraph>(ptr)->get_output_buffering();
	EX_END
}

zimg_error_code_e zimg2_filter_graph_process(const zimg_filter_graph *ptr, const zimg_image_buffer_const *src, const zimg_image_buffer *dst, void *tmp,
                                             zimg_filter_graph_callback unpack_cb, void *unpack_user,
                                             zimg_filter_graph_callback pack_cb, void *pack_user)
{
	_zassert_d(ptr, "null pointer");
	_zassert_d(src, "null pointer");
	_zassert_d(dst, "null pointer");

	POINTER_ALIGNMENT_ASSERT(src->data[0]);
	POINTER_ALIGNMENT_ASSERT(src->data[1]);
	POINTER_ALIGNMENT_ASSERT(src->data[2]);

	STRIDE_ALIGNMENT_ASSERT(src->stride[0]);
	STRIDE_ALIGNMENT_ASSERT(src->stride[1]);
	STRIDE_ALIGNMENT_ASSERT(src->stride[2]);

	POINTER_ALIGNMENT_ASSERT(dst->m.data[0]);
	POINTER_ALIGNMENT_ASSERT(dst->m.data[1]);
	POINTER_ALIGNMENT_ASSERT(dst->m.data[2]);

	STRIDE_ALIGNMENT_ASSERT(dst->m.stride[0]);
	STRIDE_ALIGNMENT_ASSERT(dst->m.stride[1]);
	STRIDE_ALIGNMENT_ASSERT(dst->m.stride[2]);

	POINTER_ALIGNMENT_ASSERT(tmp);

	EX_BEGIN
	const zimg::FilterGraph *graph = assert_dynamic_cast<const zimg::FilterGraph>(ptr);
	zimg::ZimgImageBufferConst src_buf = import_image_buffer(*src);
	zimg::ZimgImageBuffer dst_buf = import_image_buffer(*dst);

	graph->process(src_buf, dst_buf, tmp, { unpack_cb, unpack_user }, { pack_cb, pack_user });
	EX_END
}

#undef EX_BEGIN
#undef EX_END

void zimg2_image_format_default(zimg_image_format *ptr, unsigned version)
{
	_zassert_d(ptr, "null pointer");
	API_VERSION_ASSERT(version);

	ptr->version = version;

	if (version >= 2) {
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

void zimg2_filter_graph_params_default(zimg_filter_graph_params *ptr, unsigned version)
{
	_zassert_d(ptr, "null pointer");
	API_VERSION_ASSERT(version);

	ptr->version = version;

	if (version >= 2) {
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

zimg_filter_graph *zimg2_filter_graph_build(const zimg_image_format *src_format, const zimg_image_format *dst_format, const zimg_filter_graph_params *params)
{
	_zassert_d(src_format, "null pointer");
	_zassert_d(dst_format, "null pointer");

	try {
		GraphBuilder builder;
		GraphBuilder::state src_state;
		GraphBuilder::state dst_state;
		GraphBuilder::params graph_params;

		std::tie(src_state, dst_state) = import_graph_state(*src_format, *dst_format);

		if (params)
			graph_params = import_graph_params(*params);

		builder.set_source(src_state);
		return builder.build(dst_state, params ? &graph_params : nullptr);
	} catch (const zimg::error::Exception &) {
		handle_exception(std::current_exception());
		return nullptr;
	} catch (const std::bad_alloc &e) {
		handle_exception(e);
		return nullptr;
	}
}
