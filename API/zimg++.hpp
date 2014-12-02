#ifndef ZIMGPLUSPLUS_H_
#define ZIMGPLUSPLUS_H_

#include "zimg.h"

struct ZimgError {
	int code;
	char msg[1024];

	ZimgError()
	{
		code = zimg_get_last_error(msg, sizeof(msg));
	}
};

class ZimgColorspaceContext {
	zimg_colorspace_context *m_ctx;
public:
	ZimgColorspaceContext(int matrix_in, int transfer_in, int primaries_in,
	                      int matrix_out, int transfer_out, int primaries_out)
	{
		if (!(m_ctx = zimg_colorspace_create(matrix_in, transfer_in, primaries_in, matrix_out, transfer_out, primaries_out)))
			throw ZimgError{};
	}

	ZimgColorspaceContext(const ZimgColorspaceContext &) = delete;

	ZimgColorspaceContext &operator=(const ZimgColorspaceContext &) = delete;

	~ZimgColorspaceContext()
	{
		zimg_colorspace_delete(m_ctx);
	}

	size_t tmp_size(int width)
	{
		return zimg_colorspace_tmp_size(m_ctx, width);
	}

	void process(const void * const src[3], void * const dst[3], void *tmp, int width, int height, const int src_stride[3], const int dst_stride[3], int pixel_type)
	{
		if (zimg_colorspace_process(m_ctx, src, dst, tmp, width, height, src_stride, dst_stride, pixel_type))
			throw ZimgError{};
	}
};

class ZimgDepthContext {
	zimg_depth_context *m_ctx;
public:
	ZimgDepthContext(int dither_type)
	{
		if (!(m_ctx = zimg_depth_create(dither_type)))
			throw ZimgError{};
	}

	ZimgDepthContext(const ZimgDepthContext &) = delete;

	ZimgDepthContext &operator=(const ZimgDepthContext &) = delete;

	~ZimgDepthContext()
	{
		zimg_depth_delete(m_ctx);
	}

	size_t tmp_size(int width)
	{
		return zimg_depth_tmp_size(m_ctx, width);
	}

	void process(const void *src, void *dst, void *tmp, int width, int height, int src_stride, int dst_stride,
	             int pixel_in, int pixel_out, int depth_in, int depth_out, int fullrange_in, int fullrange_out, int chroma)
	{
		if (zimg_depth_process(m_ctx, src, dst, tmp, width, height, src_stride, dst_stride, pixel_in, pixel_out, depth_in, depth_out, fullrange_in, fullrange_out, chroma))
			throw ZimgError{};
	}
};

class ZimgResizeContext {
	zimg_resize_context *m_ctx;
public:
	ZimgResizeContext(int filter_type, int src_width, int src_height, int dst_width, int dst_height,
	                  double shift_w, double shift_h, double subwidth, double subheight,
	                  double filter_param_a, double filter_param_b)
	{
		if (!(m_ctx = zimg_resize_create(filter_type, src_width, src_height, dst_width, dst_height,
		                                 shift_w, shift_h, subwidth, subheight, filter_param_a, filter_param_b)))
			throw ZimgError{};
	}

	ZimgResizeContext(const ZimgResizeContext &) = delete;

	ZimgResizeContext &operator=(const ZimgResizeContext &) = delete;

	~ZimgResizeContext()
	{
		zimg_resize_delete(m_ctx);
	}

	size_t tmp_size(int pixel_type)
	{
		return zimg_resize_tmp_size(m_ctx, pixel_type);
	}

	void process(const void *src, void *dst, void *tmp, int src_width, int src_height, int dst_width, int dst_height, int src_stride, int dst_stride, int pixel_type)
	{
		if (zimg_resize_process(m_ctx, src, dst, tmp, src_width, src_height, dst_width, dst_height, src_stride, dst_stride, pixel_type))
			throw ZimgError{};
	}
};

#endif // ZIMGPLUSPLUS_H_
