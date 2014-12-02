#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstring>
#include "Common/cpuinfo.h"
#include "Common/except.h"
#include "Common/osdep.h"
#include "Common/pixel.h"
#include "Common/plane.h"
#include "Colorspace/colorspace.h"
#include "Colorspace/colorspace_param.h"
#include "Depth/depth.h"
#include "Resize/filter.h"
#include "Resize/resize.h"
#include "zimg.h"

using namespace zimg;

namespace {;

std::atomic<CPUClass> g_cpu_type{ CPUClass::CPU_NONE };
THREAD_LOCAL int g_last_error = 0;
THREAD_LOCAL char g_last_error_msg[1024];

CPUClass get_cpu_class(int cpu)
{
	switch (cpu) {
	case ZIMG_CPU_NONE:
		return CPUClass::CPU_NONE;
#ifdef ZIMG_X86
	case ZIMG_CPU_AUTO:
		return CPUClass::CPU_X86_AUTO;
	case ZIMG_CPU_X86_SSE2:
	case ZIMG_CPU_X86_SSE3:
	case ZIMG_CPU_X86_SSSE3:
	case ZIMG_CPU_X86_SSE41:
	case ZIMG_CPU_X86_SSE42:
	case ZIMG_CPU_X86_AVX:
	case ZIMG_CPU_X86_F16C:
		return CPUClass::CPU_X86_SSE2;
	case ZIMG_CPU_X86_AVX2:
		return CPUClass::CPU_X86_AVX2;
#endif
	default:
		return CPUClass::CPU_NONE;
	}
}

PixelType get_pixel_type(int pixel_type)
{
	switch (pixel_type) {
	case ZIMG_PIXEL_BYTE:
		return PixelType::BYTE;
	case ZIMG_PIXEL_WORD:
		return PixelType::WORD;
	case ZIMG_PIXEL_HALF:
		return PixelType::HALF;
	case ZIMG_PIXEL_FLOAT:
		return PixelType::FLOAT;
	default:
		throw ZimgIllegalArgument{ "unknown pixel type" };
	}
}

colorspace::MatrixCoefficients get_matrix_coeffs(int matrix)
{
	switch (matrix) {
	case ZIMG_MATRIX_RGB:
		return colorspace::MatrixCoefficients::MATRIX_RGB;
	case ZIMG_MATRIX_709:
		return colorspace::MatrixCoefficients::MATRIX_709;
	case ZIMG_MATRIX_470BG:
	case ZIMG_MATRIX_170M:
		return colorspace::MatrixCoefficients::MATRIX_601;
	case ZIMG_MATRIX_2020_NCL:
		return colorspace::MatrixCoefficients::MATRIX_2020_NCL;
	case ZIMG_MATRIX_2020_CL:
		return colorspace::MatrixCoefficients::MATRIX_2020_CL;
	default:
		throw ZimgIllegalArgument{ "unknown matrix coefficients" };
	}
}

colorspace::TransferCharacteristics get_transfer_characteristics(int transfer)
{
	switch (transfer) {
	case ZIMG_TRANSFER_709:
	case ZIMG_TRANSFER_601:
	case ZIMG_TRANSFER_2020_10:
	case ZIMG_TRANSFER_2020_12:
		return colorspace::TransferCharacteristics::TRANSFER_709;
	case ZIMG_TRANSFER_LINEAR:
		return colorspace::TransferCharacteristics::TRANSFER_LINEAR;
	default:
		throw ZimgIllegalArgument{ "unknown transfer characteristics" };
	}
}

colorspace::ColorPrimaries get_color_primaries(int primaries)
{
	switch (primaries) {
	case ZIMG_PRIMARIES_709:
		return colorspace::ColorPrimaries::PRIMARIES_709;
	case ZIMG_PRIMARIES_170M:
	case ZIMG_PRIMARIES_240M:
		return colorspace::ColorPrimaries::PRIMARIES_SMPTE_C;
	case ZIMG_PRIMARIES_2020:
		return colorspace::ColorPrimaries::PRIMARIES_2020;
	default:
		throw ZimgIllegalArgument{ "unknown color primaries" };
	}
}

depth::DitherType get_dither_type(int dither)
{
	switch (dither) {
	case ZIMG_DITHER_NONE:
		return depth::DitherType::DITHER_NONE;
	case ZIMG_DITHER_ORDERED:
		return depth::DitherType::DITHER_ORDERED;
	case ZIMG_DITHER_RANDOM:
		return depth::DitherType::DITHER_RANDOM;
	case ZIMG_DITHER_ERROR_DIFFUSION:
		return depth::DitherType::DITHER_ERROR_DIFFUSION;
	default:
		throw ZimgIllegalArgument{ "unknown dither type" };
	}
}

resize::Filter *create_filter(int filter_type, double filter_param_a, double filter_param_b)
{
	switch (filter_type) {
	case ZIMG_RESIZE_POINT:
		return new resize::PointFilter{};
	case ZIMG_RESIZE_BILINEAR:
		return new resize::BilinearFilter{};
	case ZIMG_RESIZE_BICUBIC:
		filter_param_a = std::isfinite(filter_param_a) ? filter_param_a : 1.0 / 3.0;
		filter_param_b = std::isfinite(filter_param_b) ? filter_param_b : 1.0 / 3.0;
		return new resize::BicubicFilter{ filter_param_a, filter_param_b };
	case ZIMG_RESIZE_SPLINE16:
		return new resize::Spline16Filter{};
	case ZIMG_RESIZE_SPLINE36:
		return new resize::Spline36Filter{};
	case ZIMG_RESIZE_LANCZOS:
		filter_param_a = std::isfinite(filter_param_a) ? std::floor(filter_param_a) : 3.0;
		return new resize::LanczosFilter{ (int)filter_param_a };
	default:
		throw ZimgIllegalArgument{ "unknown resampling filter" };
	}
}

void handle_exception(const ZimgException &e)
{
	try {
		zimg_clear_last_error();

		std::strncpy(g_last_error_msg, e.what(), sizeof(g_last_error_msg));
		g_last_error_msg[sizeof(g_last_error_msg) - 1] = '\0';

		throw e;
	} catch (const ZimgUnknownError &) {
		g_last_error = ZIMG_ERROR_UNKNOWN;
	} catch (const ZimgLogicError &) {
		g_last_error = ZIMG_ERROR_LOGIC;
	} catch (const ZimgOutOfMemory &) {
		g_last_error = ZIMG_ERROR_OUT_OF_MEMORY;
	} catch (const ZimgIllegalArgument &) {
		g_last_error = ZIMG_ERROR_ILLEGAL_ARGUMENT;
	} catch (const ZimgUnsupportedError &) {
		g_last_error = ZIMG_ERROR_UNSUPPORTED;
	} catch (...) {
		g_last_error = ZIMG_ERROR_UNKNOWN;
	}
}

void handle_bad_alloc()
{
	zimg_clear_last_error();
	g_last_error = ZIMG_ERROR_OUT_OF_MEMORY;
}

} // namespace


struct zimg_colorspace_context {
	colorspace::ColorspaceConversion p;
};

struct zimg_depth_context {
	depth::Depth p;
};

struct zimg_resize_context {
	resize::Resize p;
};


int zimg_check_api_version(int ver)
{
	return ZIMG_API_VERSION >= ver;
}

int zimg_get_last_error(char *err_msg, size_t n)
{
	size_t sz;
	size_t to_copy;

	if (err_msg && n) {
		sz = std::strlen(g_last_error_msg) + 1;
		to_copy = sz > n ? n : sz;

		std::memcpy(err_msg, g_last_error_msg, to_copy);
		err_msg[n - 1] = '\0';
	}

	return g_last_error;
}

void zimg_clear_last_error(void)
{
	std::memset(g_last_error_msg, 0, sizeof(g_last_error_msg));
	g_last_error = 0;
}

void zimg_set_cpu(int cpu)
{
	g_cpu_type = get_cpu_class(cpu);
}


zimg_colorspace_context *zimg_colorspace_create(int matrix_in, int transfer_in, int primaries_in,
                                                int matrix_out, int transfer_out, int primaries_out)
{
	zimg_colorspace_context *ret = nullptr;

	try {
		colorspace::ColorspaceDefinition csp_in;
		colorspace::ColorspaceDefinition csp_out;

		csp_in.matrix = get_matrix_coeffs(matrix_in);
		csp_in.transfer = get_transfer_characteristics(transfer_in);
		csp_in.primaries = get_color_primaries(primaries_in);

		csp_out.matrix = get_matrix_coeffs(matrix_out);
		csp_out.transfer = get_transfer_characteristics(transfer_out);
		csp_out.primaries = get_color_primaries(primaries_out);

		ret = new zimg_colorspace_context{ { csp_in, csp_out, g_cpu_type } };
	} catch (const ZimgException &e) {
		handle_exception(e);
	} catch (const std::bad_alloc &) {
		handle_bad_alloc();
	}

	return ret;
}

size_t zimg_colorspace_tmp_size(zimg_colorspace_context *ctx, int width)
{
	return ctx->p.tmp_size(width) * pixel_size(PixelType::FLOAT);
}

int zimg_colorspace_process(zimg_colorspace_context *ctx, const void * const src[3], void * const dst[3], void *tmp,
                            int width, int height, const int src_stride[3], const int dst_stride[3], int pixel_type)
{
	zimg_clear_last_error();

	try {
		PixelType type = get_pixel_type(pixel_type);
		int pxsize = pixel_size(type);

		ImagePlane<const void> src_planes[3];
		ImagePlane<void> dst_planes[3];

		for (int p = 0; p < 3; ++p) {
			src_planes[p] = ImagePlane<const void>{ src[p], width, height, src_stride[p] / pxsize, type };
			dst_planes[p] = ImagePlane<void>{ dst[p], width, height, dst_stride[p] / pxsize, type };
		}

		ctx->p.process(src_planes, dst_planes, tmp);
	} catch (const ZimgException &e) {
		handle_exception(e);
	}

	return g_last_error;
}

void zimg_colorspace_delete(zimg_colorspace_context *ctx)
{
	delete ctx;
}


zimg_depth_context *zimg_depth_create(int dither_type)
{
	zimg_depth_context *ret = nullptr;

	try {
		depth::DitherType dither = get_dither_type(dither_type);
		ret = new zimg_depth_context{ { dither, g_cpu_type } };
	} catch (const ZimgException &e) {
		handle_exception(e);
	} catch (const std::bad_alloc &) {
		handle_bad_alloc();
	}

	return ret;
}

size_t zimg_depth_tmp_size(zimg_depth_context *ctx, int width)
{
	return ctx->p.tmp_size(width) * pixel_size(PixelType::FLOAT);
}

int zimg_depth_process(zimg_depth_context *ctx, const void *src, void *dst, void *tmp,
                       int width, int height, int src_stride, int dst_stride,
					   int pixel_in, int pixel_out, int depth_in, int depth_out, int fullrange_in, int fullrange_out, int chroma)
{
	zimg_clear_last_error();

	try {
		PixelFormat src_format;
		PixelFormat dst_format;

		int src_pxsize;
		int dst_pxsize;

		src_format.type = get_pixel_type(pixel_in);
		src_format.depth = depth_in;
		src_format.fullrange = !!fullrange_in;
		src_format.chroma = !!chroma;

		dst_format.type = get_pixel_type(pixel_out);
		dst_format.depth = depth_out;
		dst_format.fullrange = !!fullrange_out;
		dst_format.chroma = !!chroma;

		src_pxsize = pixel_size(src_format.type);
		dst_pxsize = pixel_size(dst_format.type);

		ImagePlane<const void> src_plane{ src, width, height, src_stride / src_pxsize, src_format };
		ImagePlane<void> dst_plane{ dst, width, height, dst_stride / dst_pxsize, dst_format };

		ctx->p.process(src_plane, dst_plane, tmp);
	} catch (const ZimgException &e) {
		handle_exception(e);
	}

	return g_last_error;
}

void zimg_depth_delete(zimg_depth_context *ctx)
{
	delete ctx;
}


zimg_resize_context *zimg_resize_create(int filter_type, int src_width, int src_height, int dst_width, int dst_height,
                                        double shift_w, double shift_h, double subwidth, double subheight,
                                        double filter_param_a, double filter_param_b)
{
	zimg_resize_context *ret = nullptr;
	resize::Filter *filter = nullptr;

	try {
		filter = create_filter(filter_type, filter_param_a, filter_param_b);
		ret = new zimg_resize_context{ { *filter, src_width, src_height, dst_width, dst_height, shift_w, shift_h, subwidth, subheight, g_cpu_type } };
	} catch (const ZimgException &e) {
		handle_exception(e);
	} catch (const std::bad_alloc &) {
		handle_bad_alloc();
	}

	delete filter;
	return ret;
}

size_t zimg_resize_tmp_size(zimg_resize_context *ctx, int pixel_type)
{
	size_t ret = 0;

	try {
		PixelType type = get_pixel_type(pixel_type);
		ret = ctx->p.tmp_size(type) * pixel_size(type);
	} catch (const ZimgException &e) {
		handle_exception(e);
	}

	return ret;
}

int zimg_resize_process(zimg_resize_context *ctx, const void *src, void *dst, void *tmp,
                        int src_width, int src_height, int dst_width, int dst_height,
                        int src_stride, int dst_stride, int pixel_type)
{
	zimg_clear_last_error();

	try {
		PixelType type = get_pixel_type(pixel_type);
		int pxsize = pixel_size(type);

		ImagePlane<const void> src_plane{ src, src_width, src_height, src_stride / pxsize, type };
		ImagePlane<void> dst_plane{ dst, dst_width, dst_height, dst_stride / pxsize, type };

		ctx->p.process(src_plane, dst_plane, tmp);
	} catch (const ZimgException &e) {
		handle_exception(e);
	}

	return g_last_error;
}

void zimg_resize_delete(zimg_resize_context *ctx)
{
	delete ctx;
}
