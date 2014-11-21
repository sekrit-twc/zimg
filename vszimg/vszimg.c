#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "zimg.h"
#include "VapourSynth.h"
#include "VSHelper.h"

typedef enum chroma_location {
	CHROMA_LOC_MPEG1,
	CHROMA_LOC_MPEG2
} chroma_location;

static int translate_dither(const char *dither)
{
	if (!strcmp(dither, "none"))
		return ZIMG_DITHER_NONE;
	else if (!strcmp(dither, "ordered"))
		return ZIMG_DITHER_ORDERED;
	else if (!strcmp(dither, "random"))
		return ZIMG_DITHER_RANDOM;
	else if (!strcmp(dither, "error_diffusion"))
		return ZIMG_DITHER_ERROR_DIFFUSION;
	else
		return ZIMG_DITHER_NONE;
}

static int translate_pixel(const VSFormat *format)
{
	if (format->sampleType == stInteger && format->bytesPerSample == 1)
		return ZIMG_PIXEL_BYTE;
	else if (format->sampleType == stInteger && format->bytesPerSample == 2)
		return ZIMG_PIXEL_WORD;
	else if (format->sampleType == stFloat && format->bitsPerSample == 16)
		return ZIMG_PIXEL_HALF;
	else if (format->sampleType == stFloat && format->bitsPerSample == 32)
		return ZIMG_PIXEL_FLOAT;
	else
		return -1;
}

static int translate_filter(const char *filter)
{
	if (!strcmp(filter, "point"))
		return ZIMG_RESIZE_POINT;
	else if (!strcmp(filter, "bilinear"))
		return ZIMG_RESIZE_BILINEAR;
	else if (!strcmp(filter, "bicubic"))
		return ZIMG_RESIZE_BICUBIC;
	else if (!strcmp(filter, "spline16"))
		return ZIMG_RESIZE_SPLINE16;
	else if (!strcmp(filter, "spline36"))
		return ZIMG_RESIZE_SPLINE36;
	else if (!strcmp(filter, "lanczos"))
		return ZIMG_RESIZE_LANCZOS;
	else
		return ZIMG_RESIZE_POINT;
}

/* Offset needed to go from 4:4:4 to chroma location at given subsampling, relative to 4:4:4 grid. */
static double chroma_h_mpeg1_distance(const char *chroma_loc, int subsample)
{
	return (!strcmp(chroma_loc, "mpeg2") && subsample == 1) ? -0.5 : 0.0;
}

/* Adjustment to shift needed to convert between chroma locations. */
static double chroma_adjust_h(const char *loc_in, const char *loc_out, int subsample_in, int subsample_out)
{
	double scale = 1.0f / (double)(1 << subsample_in);
	double to_444_offset = -chroma_h_mpeg1_distance(loc_in, subsample_in) * scale;
	double from_444_offset = chroma_h_mpeg1_distance(loc_out, subsample_out) * scale;

	return to_444_offset + from_444_offset;
}

static double chroma_adjust_v(const char *loc_in, const char *loc_out, int subsample_in, int subsample_out)
{
	return 0.0;
}


typedef struct vs_colorspace_data {
	zimg_colorspace_context *colorspace_ctx;
	VSNodeRef *node;
	VSVideoInfo vi;
} vs_colorspace_data;

static void VS_CC vs_colorspace_init(VSMap *in, VSMap *out, void **instanceData, VSNode *node, VSCore *core, const VSAPI *vsapi)
{
	const vs_colorspace_data *data = *instanceData;
	vsapi->setVideoInfo(&data->vi, 1, node);
}

static const VSFrameRef * VS_CC vs_colorspace_get_frame(int n, int activationReason, void **instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi)
{
	vs_colorspace_data *data = *instanceData;
	VSFrameRef *ret = 0;
	char fail_str[1024] = { 0 };
	int err = 0;
	int p;

	zimg_clear_last_error();

	if (activationReason == arInitial) {
		vsapi->requestFrameFilter(n, data->node, frameCtx);
	} else if (activationReason == arAllFramesReady) {
		const VSFrameRef *src_frame = vsapi->getFrameFilter(n, data->node, frameCtx);
		VSFrameRef *dst_frame = 0;

		int width = vsapi->getFrameWidth(src_frame, 0);
		int height = vsapi->getFrameHeight(src_frame, 0);
		int pixel_type = translate_pixel(vsapi->getFrameFormat(src_frame));

		const VSFrameRef *frame_order[3] = { src_frame, src_frame, src_frame };
		const int plane_order[3] = { 0, 1, 2 };

		const void *src_plane[3];
		void *dst_plane[3];
		int src_stride[3];
		int dst_stride[3];

		size_t tmp_size;
		void *tmp = 0;

		dst_frame = vsapi->newVideoFrame2(data->vi.format, width, height, frame_order, plane_order, src_frame, core);
		
		for (p = 0; p < 3; ++p) {
			src_plane[p] = vsapi->getReadPtr(src_frame, p);
			dst_plane[p] = vsapi->getWritePtr(dst_frame, p);
			src_stride[p] = vsapi->getStride(src_frame, p);
			dst_stride[p] = vsapi->getStride(dst_frame, p);
		}

		tmp_size = zimg_colorspace_tmp_size(data->colorspace_ctx, width);
		VS_ALIGNED_MALLOC(&tmp, tmp_size, 32);
		if (!tmp) {
			strcpy(fail_str, "error allocating temporary buffer");
			err = 1;
			goto fail;
		}

		err = zimg_colorspace_process(data->colorspace_ctx, src_plane, dst_plane, tmp, width, height, src_stride, dst_stride, pixel_type);
		if (err) {
			zimg_get_last_error(fail_str, sizeof(fail_str));
			goto fail;
		}
		ret = dst_frame;
		dst_frame = 0;
	fail:
		vsapi->freeFrame(src_frame);
		vsapi->freeFrame(dst_frame);
		VS_ALIGNED_FREE(tmp);
	}

	if (err)
		vsapi->setFilterError(fail_str, frameCtx);
	return ret;
}

static void VS_CC vs_colorspace_free(void *instanceData, VSCore *core, const VSAPI *vsapi)
{
	vs_colorspace_data *data = instanceData;
	zimg_colorspace_delete(data->colorspace_ctx);
	vsapi->freeNode(data->node);
	free(data);
}

static void VS_CC vs_colorspace_create(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi)
{
	vs_colorspace_data *data = 0;
	zimg_colorspace_context *colorspace_ctx = 0;
	char fail_str[1024] = { 0 };
	int err;

	VSNodeRef *node = 0;
	const VSVideoInfo *node_vi;
	const VSFormat *node_fmt;
	VSVideoInfo vi;

	int matrix_in;
	int transfer_in;
	int primaries_in;
	int matrix_out;
	int transfer_out;
	int primaries_out;

	zimg_clear_last_error();

	node = vsapi->propGetNode(in, "clip", 0, 0);
	node_vi = vsapi->getVideoInfo(node);
	node_fmt = node_vi->format;

	if (!node_fmt) {
		strcpy(fail_str, "clip must have a defined format");
		goto fail;
	}

	matrix_in = (int)vsapi->propGetInt(in, "matrix_in", 0, 0);
	transfer_in = (int)vsapi->propGetInt(in, "transfer_in", 0, 0);
	primaries_in = (int)vsapi->propGetInt(in, "primaries_in", 0, 0);

	matrix_out = (int)vsapi->propGetInt(in, "matrix_out", 0, &err);
	if (err)
		matrix_out = matrix_in;

	transfer_out = (int)vsapi->propGetInt(in, "transfer_out", 0, &err);
	if (err)
		transfer_out = transfer_in;

	primaries_out = (int)vsapi->propGetInt(in, "primaries_out", 0, &err);
	if (err)
		primaries_out = primaries_in;

	if (node_fmt->numPlanes < 3 || node_fmt->subSamplingW || node_fmt->subSamplingH) {
		strcpy(fail_str, "colorspace conversion can only be performed on 4:4:4 clips");
		goto fail;
	}

	vi = *node_vi;
	vi.format = vsapi->registerFormat(matrix_out == ZIMG_MATRIX_RGB ? cmRGB : cmYUV,
	                                  node_fmt->sampleType, node_fmt->bitsPerSample, node_fmt->subSamplingW, node_fmt->subSamplingH, core);

	colorspace_ctx = zimg_colorspace_create(matrix_in, transfer_in, primaries_in, matrix_out, transfer_out, primaries_out);
	if (!colorspace_ctx) {
		zimg_get_last_error(fail_str, sizeof(fail_str));
		goto fail;
	}

	data = malloc(sizeof(vs_colorspace_data));
	if (!data) {
		strcpy(fail_str, "error allocating vs_colorspace_data");
		goto fail;
	}

	data->colorspace_ctx = colorspace_ctx;
	data->node = node;
	data->vi = vi;

	vsapi->createFilter(in, out, "colorspace", vs_colorspace_init, vs_colorspace_get_frame, vs_colorspace_free, fmParallel, 0, data, core);
	return;
fail:
	vsapi->setError(out, fail_str);
	vsapi->freeNode(node);
	zimg_colorspace_delete(colorspace_ctx);
	free(data);
	return;
}


typedef struct vs_depth_data {
	zimg_depth_context *depth_ctx;
	VSNodeRef *node;
	VSVideoInfo vi;
	int tv_in;
	int tv_out;
} vs_depth_data;

static void VS_CC vs_depth_init(VSMap *in, VSMap *out, void **instanceData, VSNode *node, VSCore *core, const VSAPI *vsapi)
{
	const vs_depth_data *data = *instanceData;
	vsapi->setVideoInfo(&data->vi, 1, node);
}

static const VSFrameRef * VS_CC vs_depth_get_frame(int n, int activationReason, void **instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi)
{
	vs_depth_data *data = *instanceData;
	VSFrameRef *ret = 0;
	char fail_str[1024] = { 0 };
	int err = 0;
	int p;

	zimg_clear_last_error();

	if (activationReason == arInitial) {
		vsapi->requestFrameFilter(n, data->node, frameCtx);
	} else if (activationReason == arAllFramesReady) {
		const VSFrameRef *src_frame = vsapi->getFrameFilter(n, data->node, frameCtx);
		VSFrameRef *dst_frame = vsapi->newVideoFrame(data->vi.format, data->vi.width, data->vi.height, src_frame, core);

		const VSFormat *src_format = vsapi->getFrameFormat(src_frame);
		const VSFormat *dst_format = data->vi.format;

		int src_pixel = translate_pixel(src_format);
		int dst_pixel = translate_pixel(dst_format);
		int yuv = src_format->colorFamily == cmYUV || src_format->colorFamily == cmYCoCg;

		void *tmp = 0;
		size_t tmp_size = zimg_depth_tmp_size(data->depth_ctx, vsapi->getFrameWidth(src_frame, 0));

		VS_ALIGNED_MALLOC(&tmp, tmp_size, 32);
		if (!tmp) {
			strcpy(fail_str, "error allocating temporary buffer");
			err = 1;
			goto fail;
		}

		for (p = 0; p < data->vi.format->numPlanes; ++p) {
			err = zimg_depth_process(data->depth_ctx,
			                         vsapi->getReadPtr(src_frame, p),
			                         vsapi->getWritePtr(dst_frame, p),
			                         tmp,
			                         vsapi->getFrameWidth(src_frame, p),
			                         vsapi->getFrameHeight(src_frame, p),
			                         vsapi->getStride(src_frame, p),
			                         vsapi->getStride(dst_frame, p),
			                         src_pixel,
			                         dst_pixel,
			                         src_format->bitsPerSample,
			                         dst_format->bitsPerSample,
			                         data->tv_in,
			                         data->tv_out,
			                         p > 0 && yuv);
			if (err) {
				zimg_get_last_error(fail_str, sizeof(fail_str));
				goto fail;
			}
		}
		ret = dst_frame;
		dst_frame = 0;
	fail:
		vsapi->freeFrame(src_frame);
		vsapi->freeFrame(dst_frame);
		VS_ALIGNED_FREE(tmp);
	}

	if (err)
		vsapi->setFilterError(fail_str, frameCtx);
	return ret;
}

static void VS_CC vs_depth_free(void *instanceData, VSCore *core, const VSAPI *vsapi)
{
	vs_depth_data *data = instanceData;
	zimg_depth_delete(data->depth_ctx);
	vsapi->freeNode(data->node);
	free(data);
}

static void VS_CC vs_depth_create(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi)
{
	vs_depth_data *data = 0;
	zimg_depth_context *depth_ctx = 0;
	char fail_str[1024] = { 0 };
	int err;

	VSNodeRef *node = 0;
	const VSVideoInfo *node_vi;
	const VSFormat *node_fmt;
	const VSFormat *preset_fmt;

	VSVideoInfo out_vi;
	const VSFormat *out_fmt;

	const char *dither;
	int format;
	int sample;
	int depth;
	int tv_in;
	int tv_out;

	zimg_clear_last_error();

	node = vsapi->propGetNode(in, "clip", 0, 0);
	node_vi = vsapi->getVideoInfo(node);
	node_fmt = node_vi->format;

	if (!node_fmt) {
		strcpy(fail_str, "clip must have a defined format");
		goto fail;
	}

	format = (int)vsapi->propGetInt(in, "format", 0, &err);
	if (err)
		format = node_fmt->id;

	preset_fmt = vsapi->getFormatPreset(format, core);
	if (!preset_fmt) {
		strcpy(fail_str, "invalid format preset provided");
		goto fail;
	}

	dither = vsapi->propGetData(in, "dither", 0, &err);
	if (err)
		dither = "none";

	sample = (int)vsapi->propGetInt(in, "sample", 0, &err);
	if (err)
		sample = preset_fmt->sampleType;

	depth = (int)vsapi->propGetInt(in, "depth", 0, &err);
	if (err)
		depth = preset_fmt->bitsPerSample;

	tv_in = !!vsapi->propGetInt(in, "fullrange_in", 0, &err);
	if (err)
		tv_in = preset_fmt->colorFamily == cmRGB;

	tv_out = !!vsapi->propGetInt(in, "fullrange_out", 0, &err);
	if (err)
		tv_out = preset_fmt->colorFamily == cmRGB;

	if (sample != stInteger && sample != stFloat) {
		strcpy(fail_str, "invalid sample type: must be stInteger or stFloat");
		goto fail;
	}
	if (sample == stFloat && depth != 16 && depth != 32) {
		strcpy(fail_str, "only half and single-precision supported for floats");
		goto fail;
	}
	if (sample == stInteger && (depth <= 0 || depth > 16)) {
		strcpy(fail_str, "only bit depths 1-16 are supported for int");
		goto fail;
	}

	if (preset_fmt->colorFamily != node_fmt->colorFamily) {
		strcpy(fail_str, "cannot change color family with depth filter");
		goto fail;
	}
	if (preset_fmt->subSamplingW != node_fmt->subSamplingW ||
	    preset_fmt->subSamplingH != node_fmt->subSamplingH)
	{
		strcpy(fail_str, "cannot change chroma subsampling with depth filter");
		goto fail;
	}

	out_fmt = vsapi->registerFormat(preset_fmt->colorFamily, sample, depth, preset_fmt->subSamplingW, preset_fmt->subSamplingH, core);
	out_vi.format = out_fmt;
	out_vi.fpsNum = node_vi->fpsNum;
	out_vi.fpsDen = node_vi->fpsDen;
	out_vi.width = node_vi->width;
	out_vi.height = node_vi->height;
	out_vi.numFrames = node_vi->numFrames;
	out_vi.flags = 0;

	depth_ctx = zimg_depth_create(translate_dither(dither));
	if (!depth_ctx) {
		zimg_get_last_error(fail_str, sizeof(fail_str));
		goto fail;
	}

	data = malloc(sizeof(vs_depth_data));
	if (!data) {
		strcpy(fail_str, "error allocating vs_depth_data");
		goto fail;
	}

	data->depth_ctx = depth_ctx;
	data->node = node;
	data->vi = out_vi;
	data->tv_in = tv_in;
	data->tv_out = tv_out;

	vsapi->createFilter(in, out, "depth", vs_depth_init, vs_depth_get_frame, vs_depth_free, fmParallel, 0, data, core);
	return;
fail:
	vsapi->setError(out, fail_str);
	vsapi->freeNode(node);
	zimg_depth_delete(depth_ctx);
	free(data);
	return;
}


typedef struct vs_resize_data {
	zimg_resize_context *resize_ctx_y;
	zimg_resize_context *resize_ctx_uv;
	VSNodeRef *node;
	VSVideoInfo vi;
} vs_resize_data;

static void VS_CC vs_resize_init(VSMap *in, VSMap *out, void **instanceData, VSNode *node, VSCore *core, const VSAPI *vsapi)
{
	const vs_resize_data *data = *instanceData;
	vsapi->setVideoInfo(&data->vi, 1, node);
}

static const VSFrameRef * VS_CC vs_resize_get_frame(int n, int activationReason, void **instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi)
{
	vs_resize_data *data = *instanceData;
	VSFrameRef *ret = 0;
	char fail_str[1024] = { 0 };
	int err = 0;
	int p;

	zimg_clear_last_error();

	if (activationReason == arInitial) {
		vsapi->requestFrameFilter(n, data->node, frameCtx);
	} else if (activationReason == arAllFramesReady) {
		const VSFrameRef *src_frame = vsapi->getFrameFilter(n, data->node, frameCtx);
		VSFrameRef *dst_frame = vsapi->newVideoFrame(data->vi.format, data->vi.width, data->vi.height, src_frame, core);

		const VSFormat *format = data->vi.format;
		int pixel_type = translate_pixel(format);

		void *tmp = 0;
		size_t tmp_size = zimg_resize_tmp_size(data->resize_ctx_y, pixel_type);
		size_t tmp_size_uv = data->resize_ctx_uv ? zimg_resize_tmp_size(data->resize_ctx_uv, pixel_type) : 0;
		tmp_size = tmp_size_uv > tmp_size ? tmp_size_uv : tmp_size;

		VS_ALIGNED_MALLOC(&tmp, tmp_size, 32);
		if (!tmp) {
			strcpy(fail_str, "error allocating temporary buffer");
			err = 1;
			goto fail;
		}

		for (p = 0; p < format->numPlanes; ++p) {
			zimg_resize_context *resize_ctx = ((p == 1 || p == 2) && data->resize_ctx_uv) ? data->resize_ctx_uv : data->resize_ctx_y;

			err = zimg_resize_process(resize_ctx,
			                          vsapi->getReadPtr(src_frame, p),
			                          vsapi->getWritePtr(dst_frame, p),
			                          tmp,
			                          vsapi->getFrameWidth(src_frame, p),
			                          vsapi->getFrameHeight(src_frame, p),
			                          data->vi.width,
			                          data->vi.height,
			                          vsapi->getStride(src_frame, p),
			                          vsapi->getStride(dst_frame, p),
			                          pixel_type);
			if (err) {
				zimg_get_last_error(fail_str, sizeof(fail_str));
				goto fail;
			}
		}
		ret = dst_frame;
		dst_frame = 0;
	fail:
		vsapi->freeFrame(src_frame);
		vsapi->freeFrame(dst_frame);
		VS_ALIGNED_FREE(tmp);
	}

	if (err)
		vsapi->setFilterError(fail_str, frameCtx);
	return ret;
}

static void VS_CC vs_resize_free(void *instanceData, VSCore *core, const VSAPI *vsapi)
{
	vs_resize_data *data = instanceData;
	vsapi->freeNode(data->node);
	zimg_resize_delete(data->resize_ctx_y);
	zimg_resize_delete(data->resize_ctx_uv);
	free(data);
}

static void VS_CC vs_resize_create(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi)
{
	vs_resize_data *data = 0;
	zimg_resize_context *resize_ctx_y = 0;
	zimg_resize_context *resize_ctx_uv = 0;
	char fail_str[1024] = { 0 };
	int err;

	VSNodeRef *node = 0;
	const VSVideoInfo *node_vi;
	const VSFormat *node_fmt;
	VSVideoInfo out_vi;
	const VSFormat *out_fmt;

	int width;
	int height;

	const char *filter;
	double filter_param_a;
	double filter_param_b;

	double shift_w;
	double shift_h;
	double subwidth;
	double subheight;

	const char *filter_uv;
	double filter_param_a_uv;
	double filter_param_b_uv;

	int subsample_w;
	int subsample_h;

	const char *chroma_loc_in;
	const char *chroma_loc_out;

	node = vsapi->propGetNode(in, "clip", 0, 0);
	node_vi = vsapi->getVideoInfo(node);
	node_fmt = node_vi->format;

	if (!isConstantFormat(node_vi)) {
		strcpy(fail_str, "clip must have constant format");
		goto fail;
	}

	width = (int)vsapi->propGetInt(in, "width", 0, 0);
	height = (int)vsapi->propGetInt(in, "height", 0, 0);

	filter = vsapi->propGetData(in, "filter", 0, &err);
	if (err)
		filter = "point";

	filter_param_a = vsapi->propGetFloat(in, "filter_param_a", 0, &err);
	if (err)
		filter_param_a = NAN;

	filter_param_b = vsapi->propGetFloat(in, "filter_param_b", 0, &err);
	if (err)
		filter_param_b = NAN;

	shift_w = vsapi->propGetFloat(in, "shift_w", 0, &err);
	if (err)
		shift_w = 0.0;

	shift_h = vsapi->propGetFloat(in, "shift_h", 0, &err);
	if (err)
		shift_h = 0.0;

	subwidth = vsapi->propGetFloat(in, "subwidth", 0, &err);
	if (err)
		subwidth = node_vi->width;

	subheight = vsapi->propGetFloat(in, "subheight", 0, &err);
	if (err)
		subheight = node_vi->height;

	filter_uv = vsapi->propGetData(in, "filter_uv", 0, &err);
	if (err)
		filter_uv = filter;

	filter_param_a_uv = vsapi->propGetFloat(in, "filter_param_a_uv", 0, &err);
	if (err)
		filter_param_a_uv = !strcmp(filter, filter_uv) ? filter_param_a : NAN;

	filter_param_b_uv = vsapi->propGetFloat(in, "filter_param_b_uv", 0, &err);
	if (err)
		filter_param_b_uv = !strcmp(filter, filter_uv) ? filter_param_b : NAN;

	subsample_w = (int)vsapi->propGetInt(in, "subsample_w", 0, &err);
	if (err)
		subsample_w = node_fmt->subSamplingW;

	subsample_h = (int)vsapi->propGetInt(in, "subsample_h", 0, &err);
	if (err)
		subsample_h = node_fmt->subSamplingH;

	chroma_loc_in = vsapi->propGetData(in, "chroma_loc_in", 0, &err);
	if (err)
		chroma_loc_in = "mpeg2";

	chroma_loc_out = vsapi->propGetData(in, "chroma_loc_out", 0, &err);
	if (err)
		chroma_loc_out = "mpeg2";

	if (width <= 0 || height <= 0 || subwidth <= 0.0 || subheight <= 0.0) {
		strcpy(fail_str, "width and height must be positive");
		goto fail;
	}
	if ((node_fmt->colorFamily != cmYUV && node_fmt->colorFamily != cmYCoCg) && (subsample_w || subsample_h)) {
		strcpy(fail_str, "subsampling is only allowed for YUV");
		goto fail;
	}

	out_fmt = vsapi->registerFormat(node_fmt->colorFamily, node_fmt->sampleType, node_fmt->bitsPerSample, subsample_w, subsample_h, core);
	out_vi.format = out_fmt;
	out_vi.fpsNum = node_vi->fpsNum;
	out_vi.fpsDen = node_vi->fpsDen;
	out_vi.width = width;
	out_vi.height = height;
	out_vi.numFrames = node_vi->numFrames;
	out_vi.flags = 0;

	resize_ctx_y = zimg_resize_create(translate_filter(filter), node_vi->width, node_vi->height, 
	                                  width, height, shift_w, shift_h, subwidth, subheight, filter_param_a, filter_param_b);
	if (!resize_ctx_y) {
		zimg_get_last_error(fail_str, sizeof(fail_str));
		goto fail;
	}

	if (node_fmt->subSamplingW || node_fmt->subSamplingH || subsample_w || subsample_h) {
		int src_width_uv = node_vi->width >> node_fmt->subSamplingW;
		int src_height_uv = node_vi->height >> node_fmt->subSamplingH;
		int width_uv = width >> subsample_w;
		int height_uv = height >> subsample_h;

		double shift_w_uv = shift_w / (double)(1 << node_fmt->subSamplingW);
		double shift_h_uv = shift_h / (double)(1 << node_fmt->subSamplingH);
		double subwidth_uv = subwidth / (double)(1 << node_fmt->subSamplingW);
		double subheight_uv = subheight / (double)(1 << node_fmt->subSamplingH);

		shift_w_uv += chroma_adjust_h(chroma_loc_in, chroma_loc_out, node_fmt->subSamplingW, subsample_w);
		shift_h_uv += chroma_adjust_v(chroma_loc_in, chroma_loc_out, node_fmt->subSamplingH, subsample_h);

		resize_ctx_uv = zimg_resize_create(translate_filter(filter_uv), src_width_uv, src_height_uv, width_uv, height_uv,
		                                   shift_w_uv, shift_h_uv, subwidth_uv, subheight_uv, filter_param_a_uv, filter_param_b_uv);
		if (!resize_ctx_uv) {
			zimg_get_last_error(fail_str, sizeof(fail_str));
			goto fail;
		}
	}

	data = malloc(sizeof(vs_resize_data));
	if (!data) {
		strcpy(fail_str, "error allocaing vs_resize_data");
		goto fail;
	}

	data->node = node;
	data->resize_ctx_y = resize_ctx_y;
	data->resize_ctx_uv = resize_ctx_uv;
	data->vi = out_vi;

	vsapi->createFilter(in, out, "resize", vs_resize_init, vs_resize_get_frame, vs_resize_free, fmParallel, 0, data, core);
	return;
fail:
	vsapi->freeNode(node);
	zimg_resize_delete(resize_ctx_y);
	zimg_resize_delete(resize_ctx_uv);
	free(data);
	return;
}

static void VS_CC vs_set_cpu(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi)
{
	const char *cpu = vsapi->propGetData(in, "cpu", 0, 0);

	if (!strcmp(cpu, "none"))
		zimg_set_cpu(ZIMG_CPU_NONE);
	else if (!strcmp(cpu, "auto"))
		zimg_set_cpu(ZIMG_CPU_AUTO);
#if defined(__i386) || defined(_M_IX86) || defined(_M_X64) || defined(__x86_64__)
	else if (!strcmp(cpu, "mmx"))
		zimg_set_cpu(ZIMG_CPU_X86_MMX);
	else if (!strcmp(cpu, "sse"))
		zimg_set_cpu(ZIMG_CPU_X86_SSE);
	else if (!strcmp(cpu, "sse2"))
		zimg_set_cpu(ZIMG_CPU_X86_SSE2);
	else if (!strcmp(cpu, "sse3"))
		zimg_set_cpu(ZIMG_CPU_X86_SSE3);
	else if (!strcmp(cpu, "ssse3"))
		zimg_set_cpu(ZIMG_CPU_X86_SSSE3);
	else if (!strcmp(cpu, "sse41"))
		zimg_set_cpu(ZIMG_CPU_X86_SSE41);
	else if (!strcmp(cpu, "sse42"))
		zimg_set_cpu(ZIMG_CPU_X86_SSE42);
	else if (!strcmp(cpu, "avx"))
		zimg_set_cpu(ZIMG_CPU_X86_AVX);
	else if (!strcmp(cpu, "f16c"))
		zimg_set_cpu(ZIMG_CPU_X86_F16C);
	else if (!strcmp(cpu, "avx2"))
		zimg_set_cpu(ZIMG_CPU_X86_AVX2);
#endif
	else
		zimg_set_cpu(ZIMG_CPU_NONE);
}

VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin)
{
	if (!zimg_check_api_version(ZIMG_API_VERSION))
		return;

	configFunc("the.weather.channel", "z", "batman", VAPOURSYNTH_API_VERSION, 1, plugin);
	
	registerFunc("Colorspace", "clip:clip;"
	                           "matrix_in:int;"
	                           "transfer_in:int;"
	                           "primaries_in:int;"
	                           "matrix_out:int:opt;"
	                           "transfer_out:int:opt;"
	                           "primaries_out:int:opt", vs_colorspace_create, 0, plugin);
	registerFunc("Depth", "clip:clip;"
	                      "dither:data:opt;"
	                      "format:int:opt;"
	                      "sample:int:opt;"
	                      "depth:int:opt;"
	                      "fullrange_in:int:opt;"
	                      "fullrange_out:int:opt", vs_depth_create, 0, plugin);
	registerFunc("Resize", "clip:clip;"
	                       "width:int;"
	                       "height:int;"
	                       "filter:data:opt;"
	                       "filter_param_a:float:opt;"
	                       "filter_param_b:float:opt;"
	                       "shift_w:float:opt;"
	                       "shift_h:float:opt;"
	                       "subwidth:float:opt;"
	                       "subheight:float:opt;"
	                       "filter_uv:data:opt;"
	                       "filter_param_a_uv:float:opt;"
	                       "filter_param_b_uv:float:opt;"
	                       "subsample_w:int:opt;"
	                       "subsample_h:int:opt;"
	                       "chroma_loc_in:data:opt;"
	                       "chroma_loc_out:data:opt;", vs_resize_create, 0, plugin);

	registerFunc("SetCPU", "cpu:data", vs_set_cpu, 0, plugin);

	zimg_set_cpu(ZIMG_CPU_AUTO);
}
