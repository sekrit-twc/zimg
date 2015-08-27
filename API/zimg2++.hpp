#ifndef ZIMG2PLUSPLUS_HPP_
#define ZIMG2PLUSPLUS_HPP_

#include "zimg2.h"

namespace zimgapi {;

struct zerror {
	int code;
	char msg[1024];

	zerror()
	{
		code = zimg_get_last_error(msg, sizeof(msg));
	}
};


struct zimage_buffer {
	zimg_image_buffer m_buffer;

	zimage_buffer()
	{
		m_buffer.m.version = ZIMG_API_VERSION;
	}

	operator zimg_image_buffer &()
	{
		return m_buffer;
	}

	operator const zimg_image_buffer &() const
	{
		return m_buffer;
	}

	operator zimg_image_buffer_const &()
	{
		return m_buffer.c;
	}

	operator const zimg_image_buffer_const &() const
	{
		return m_buffer.c;
	}
};

class zfilter {
public:
	struct pair_unsigned {
		unsigned first;
		unsigned second;
	};
private:
	zimg_filter *m_filter;

	zfilter(const zfilter &);
	zfilter &operator=(const zfilter &);

	void check_throw(unsigned x) const
	{
		if (x)
			throw zerror();
	}
public:
	zfilter(zimg_filter *filter) : m_filter(filter)
	{
	}

	~zfilter()
	{
		zimg2_filter_free(m_filter);
	}

	zimg_filter *release()
	{
		zimg_filter *ret = m_filter;
		m_filter = 0;
		return ret;
	}

	void get_flags(zimg_filter_flags *flags) const
	{
		check_throw(zimg2_filter_get_flags(m_filter, flags, ZIMG_API_VERSION));
	}

	pair_unsigned get_required_row_range(unsigned i) const
	{
		pair_unsigned ret;

		check_throw(zimg2_filter_get_required_row_range(m_filter, i, &ret.first, &ret.second));
		return ret;
	}

	pair_unsigned get_required_col_range(unsigned left, unsigned right) const
	{
		pair_unsigned ret;

		check_throw(zimg2_filter_get_required_col_range(m_filter, left, right, &ret.first, &ret.second));
		return ret;
	}

	unsigned get_max_buffering() const
	{
		unsigned ret;

		check_throw(zimg2_filter_get_max_buffering(m_filter, &ret));
		return ret;
	}

	size_t get_context_size() const
	{
		size_t ret;

		check_throw(zimg2_filter_get_context_size(m_filter, &ret));
		return ret;
	}

	size_t get_tmp_size(unsigned left, unsigned right) const
	{
		size_t ret;

		check_throw(zimg2_filter_get_tmp_size(m_filter, left, right, &ret));
		return ret;
	}

	void init_context(void *ctx) const
	{
		check_throw(zimg2_filter_init_context(m_filter, ctx));
	}

	void process(void *ctx, const zimg_image_buffer_const &src, const zimg_image_buffer &dst, void *tmp, unsigned i, unsigned left, unsigned right)
	{
		check_throw(zimg2_filter_process(m_filter, ctx, &src, &dst, tmp, i, left, right));
	}

	size_t get_tmp_size_plane(int width, int height)
	{
		size_t ret;

		check_throw(zimg2_plane_filter_get_tmp_size(m_filter, width, height, &ret));
		return ret;
	}

	void process_plane(void *tmp_pool, const void * const src[3], void * const dst[3], const ptrdiff_t src_stride[3], const ptrdiff_t dst_stride[3], unsigned width, unsigned height)
	{
		check_throw(zimg2_plane_filter_process(m_filter, tmp_pool, src, dst, src_stride, dst_stride, width, height));
	}
};


struct zcolorspace_params : public zimg_colorspace_params {
	zcolorspace_params()
	{
		zimg2_colorspace_params_default(this, ZIMG_API_VERSION);
	}
};

struct zdepth_params : public zimg_depth_params {
	zdepth_params()
	{
		zimg2_depth_params_default(this, ZIMG_API_VERSION);
	}
};

struct zresize_params : public zimg_resize_params {
	zresize_params()
	{
		zimg2_resize_params_default(this, ZIMG_API_VERSION);
	}
};


zfilter zfilter_create(const zimg_colorspace_params &params)
{
	zimg_filter *filter;

	if (!(zimg2_colorspace_create(&params)))
		throw zerror();

	return filter;
}

zfilter zfilter_create(const zimg_depth_params &params)
{
	zimg_filter *filter;

	if (!(zimg2_depth_create(&params)))
		throw zerror();

	return filter;
}

zfilter zfilter_create(const zimg_resize_params &params)
{
	zimg_filter *filter;

	if (!(zimg2_resize_create(&params)))
		throw zerror();

	return filter;
}

} // namespace zimgapi


#ifdef ZIMG_API_V1
typedef zimgapi::zerror ZimgError;

class ZimgColorspaceContext {
	zimg_colorspace_context *m_ctx;

	ZimgColorspaceContext(const ZimgColorspaceContext &);
	ZimgColorspaceContext &operator=(const ZimgColorspaceContext &);
public:
	ZimgColorspaceContext(int matrix_in, int transfer_in, int primaries_in,
						  int matrix_out, int transfer_out, int primaries_out)
	{
		if (!(m_ctx = zimg_colorspace_create(matrix_in, transfer_in, primaries_in, matrix_out, transfer_out, primaries_out)))
			throw ZimgError();
	}

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
			throw ZimgError();
	}
};

class ZimgDepthContext {
	zimg_depth_context *m_ctx;

	ZimgDepthContext(const ZimgDepthContext &);

	ZimgDepthContext &operator=(const ZimgDepthContext &);
public:
	ZimgDepthContext(int dither_type)
	{
		if (!(m_ctx = zimg_depth_create(dither_type)))
			throw ZimgError();
	}

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
			throw ZimgError();
	}
};

class ZimgResizeContext {
	zimg_resize_context *m_ctx;

	ZimgResizeContext(const ZimgResizeContext &);

	ZimgResizeContext &operator=(const ZimgResizeContext &);
public:
	ZimgResizeContext(int filter_type, int src_width, int src_height, int dst_width, int dst_height,
					  double shift_w, double shift_h, double subwidth, double subheight,
					  double filter_param_a, double filter_param_b)
	{
		if (!(m_ctx = zimg_resize_create(filter_type, src_width, src_height, dst_width, dst_height,
										 shift_w, shift_h, subwidth, subheight, filter_param_a, filter_param_b)))
			throw ZimgError();
	}

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
			throw ZimgError();
	}
};
#endif // ZIMG_API_V1

#endif // ZIMG2PLUSPLUS_HPP_
