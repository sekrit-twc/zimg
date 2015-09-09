#if 0

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "zimg2.h"
#include "VapourSynth.h"
#include "VSHelper.h"

#if ZIMG_API_VERSION < 2
  #error zAPI v2 or greater required
#endif


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

static int64_t propGetIntDefault(const VSAPI *vsapi, const VSMap *map, const char *key, int index, int64_t default_value)
{
	int64_t x;
	int err;

	x = vsapi->propGetInt(map, key, index, &err);
	return err ? default_value : x;
}

static double propGetFloatDefault(const VSAPI *vsapi, const VSMap *map, const char *key, int index, double default_value)
{
	double x;
	int err;

	x = vsapi->propGetFloat(map, key, index, &err);
	return err ? default_value : x;
}

static const char *propGetDataDefault(const VSAPI *vsapi, const VSMap *map, const char *key, int index, const char *default_value)
{
	const char *x;
	int err;

	x = vsapi->propGetData(map, key, index, &err);
	return err ? default_value : x;
}


typedef struct vs_colorspace_data {
	zimg_filter *filter;
	VSNodeRef *node;
	VSVideoInfo vi;
	int matrix_out;
	int transfer_out;
	int primaries_out;
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
	char fail_str[1024];
	int err = 0;

	if (activationReason == arInitial) {
		vsapi->requestFrameFilter(n, data->node, frameCtx);
	} else if (activationReason == arAllFramesReady) {
		const VSFrameRef *src_frame = 0;
		VSFrameRef *dst_frame = 0;
		void *tmp = 0;

		const void *src_plane[3];
		void *dst_plane[3];
		ptrdiff_t src_stride[3];
		ptrdiff_t dst_stride[3];
		int width;
		int height;
		size_t tmp_size;
		VSMap *props;
		unsigned p;

		width = data->vi.width;
		height = data->vi.height;

		src_frame = vsapi->getFrameFilter(n, data->node, frameCtx);
		dst_frame = vsapi->newVideoFrame(data->vi.format, width, height, src_frame, core);

		if ((err = zimg2_plane_filter_get_tmp_size(data->filter, &tmp_size))) {
			zimg_get_last_error(fail_str, sizeof(fail_str));
			goto fail;
		}

		VS_ALIGNED_MALLOC(&tmp, tmp_size, 32);
		if (!tmp) {
			strcpy(fail_str, "error allocating temporary buffer");
			err = 1;
			goto fail;
		}

		for (p = 0; p < 3; ++p) {
			src_plane[p] = vsapi->getReadPtr(src_frame, p);
			src_stride[p] = vsapi->getStride(src_frame, p);

			dst_plane[p] = vsapi->getWritePtr(dst_frame, p);
			dst_stride[p] = vsapi->getStride(dst_frame, p);
		}

		if ((err = zimg2_plane_filter_process(data->filter, tmp, src_plane, dst_plane, src_stride, dst_stride))) {
			zimg_get_last_error(fail_str, sizeof(fail_str));
			goto fail;
		}

		props = vsapi->getFramePropsRW(dst_frame);
		vsapi->propSetInt(props, "_Matrix", data->matrix_out, paReplace);
		vsapi->propSetInt(props, "_Transfer", data->transfer_out, paReplace);
		vsapi->propSetInt(props, "_Primaries", data->primaries_out, paReplace);

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
	zimg2_filter_free(data->filter);
	vsapi->freeNode(data->node);
	free(data);
}

static void VS_CC vs_colorspace_create(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi)
{
	vs_colorspace_data *data = 0;
	zimg_colorspace_params params;
	zimg_filter *filter = 0;
	char fail_str[1024];

	VSNodeRef *node = 0;
	const VSVideoInfo *node_vi;
	const VSFormat *node_fmt;
	VSVideoInfo vi;

	node = vsapi->propGetNode(in, "clip", 0, 0);
	node_vi = vsapi->getVideoInfo(node);
	node_fmt = node_vi->format;

	if (!isConstantFormat(node_vi)) {
		strcpy(fail_str, "clip must have a defined format");
		goto fail;
	}
	if (node_fmt->numPlanes < 3 || node_fmt->subSamplingW || node_fmt->subSamplingH) {
		strcpy(fail_str, "colorspace conversion can only be performed on 4:4:4 clips");
		goto fail;
	}

	zimg2_colorspace_params_default(&params, ZIMG_API_VERSION);

	params.width = node_vi->width;
	params.height = node_vi->height;

	params.matrix_in = (int)vsapi->propGetInt(in, "matrix_in", 0, 0);
	params.transfer_in = (int)vsapi->propGetInt(in, "transfer_in", 0, 0);
	params.primaries_in = (int)vsapi->propGetInt(in, "primaries_in", 0, 0);

	params.matrix_out = (int)propGetIntDefault(vsapi, in, "matrix_out", 0, params.matrix_in);
	params.transfer_out = (int)propGetIntDefault(vsapi, in, "transfer_out", 0, params.transfer_in);
	params.primaries_out = (int)propGetIntDefault(vsapi, in, "primaries_out", 0, params.primaries_in);

	params.pixel_type = translate_pixel(node_fmt);
	params.depth = node_fmt->bitsPerSample;
	params.range_in = (int)!!propGetIntDefault(vsapi, in, "fullrange_in", 0, params.matrix_in == ZIMG_MATRIX_RGB ? ZIMG_RANGE_FULL : ZIMG_RANGE_LIMITED);
	params.range_out = (int)!!propGetIntDefault(vsapi, in, "fullrange_out", 0, params.matrix_out == ZIMG_MATRIX_RGB ? ZIMG_RANGE_FULL : ZIMG_RANGE_LIMITED);

	vi = *node_vi;
	vi.format = vsapi->registerFormat(params.matrix_out == ZIMG_MATRIX_RGB ? cmRGB : cmYUV,
	                                  node_fmt->sampleType, node_fmt->bitsPerSample, node_fmt->subSamplingW, node_fmt->subSamplingH, core);

	if (!(filter = zimg2_colorspace_create(&params))) {
		zimg_get_last_error(fail_str, sizeof(fail_str));
		goto fail;
	}
	if (!(data = malloc(sizeof(*data)))) {
		strcpy(fail_str, "error allocating vs_colorspace_data");
		goto fail;
	}

	data->filter = filter;
	data->node = node;
	data->vi = vi;
	data->matrix_out = params.matrix_out;
	data->transfer_out = params.transfer_out;
	data->primaries_out = params.primaries_out;

	vsapi->createFilter(in, out, "colorspace", vs_colorspace_init, vs_colorspace_get_frame, vs_colorspace_free, fmParallel, 0, data, core);
	return;
fail:
	vsapi->setError(out, fail_str);
	vsapi->freeNode(node);
	zimg2_filter_free(filter);
	free(data);
}


typedef struct vs_depth_data {
	zimg_filter *filter;
	zimg_filter *filter_uv;
	VSNodeRef *node;
	VSVideoInfo vi;
	int fullrange;
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
	char fail_str[1024];
	int err = 0;

	if (activationReason == arInitial) {
		vsapi->requestFrameFilter(n, data->node, frameCtx);
	} else {
		const VSFrameRef *src_frame = 0;
		VSFrameRef *dst_frame = 0;
		void *tmp = 0;

		const void *src_plane[3] = { 0 };
		void *dst_plane[3] = { 0 };
		ptrdiff_t src_stride[3] = { 0 };
		ptrdiff_t dst_stride[3] = { 0 };
		int width;
		int height;
		unsigned num_planes;
		size_t tmp_size;
		size_t tmp_size_uv;
		VSMap *props;
		unsigned p;

		width = data->vi.width;
		height = data->vi.height;
		num_planes = data->vi.format->numPlanes;

		src_frame = vsapi->getFrameFilter(n, data->node, frameCtx);
		dst_frame = vsapi->newVideoFrame(data->vi.format, width, height, src_frame, core);

		if ((err = zimg2_plane_filter_get_tmp_size(data->filter, &tmp_size))) {
			zimg_get_last_error(fail_str, sizeof(fail_str));
			goto fail;
		}
		if (data->filter_uv) {
			if ((err = zimg2_plane_filter_get_tmp_size(data->filter_uv, &tmp_size_uv))) {
				zimg_get_last_error(fail_str, sizeof(fail_str));
				goto fail;
			}
		} else {
			tmp_size_uv = 0;
		}

		VS_ALIGNED_MALLOC(&tmp, VSMAX(tmp_size, tmp_size_uv), 32);
		if (!tmp) {
			strcpy(fail_str, "error allocating temporary buffer");
			err = 1;
			goto fail;
		}

		for (p = 0; p < num_planes; ++p) {
			const zimg_filter *filter;

			if ((p == 1 || p == 2) && data->filter_uv)
				filter = data->filter_uv;
			else
				filter = data->filter;

			src_plane[0] = vsapi->getReadPtr(src_frame, p);
			src_stride[0] = vsapi->getStride(src_frame, p);

			dst_plane[0] = vsapi->getWritePtr(dst_frame, p);
			dst_stride[0] = vsapi->getStride(dst_frame, p);

			if ((err = zimg2_plane_filter_process(filter, tmp, src_plane, dst_plane, src_stride, dst_stride))) {
				zimg_get_last_error(fail_str, sizeof(fail_str));
				goto fail;
			}
		}

		props = vsapi->getFramePropsRW(dst_frame);
		vsapi->propSetInt(props, "_ColorRange", !data->fullrange, paReplace);

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
	zimg2_filter_free(data->filter);
	zimg2_filter_free(data->filter_uv);
	vsapi->freeNode(data->node);
	free(data);
}

static void VS_CC vs_depth_create(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi)
{
	vs_depth_data *data = 0;
	zimg_depth_params params;
	zimg_filter *filter = 0;
	zimg_filter *filter_uv = 0;
	char fail_str[1024];
	int err;

	VSNodeRef *node = 0;
	const VSVideoInfo *node_vi;
	const VSFormat *node_fmt;
	VSVideoInfo vi;

	const char *dither_str;
	int sample_type;
	int depth;

	node = vsapi->propGetNode(in, "clip", 0, 0);
	node_vi = vsapi->getVideoInfo(node);
	node_fmt = node_vi->format;

	if (!isConstantFormat(node_vi)) {
		strcpy(fail_str, "clip must have a defined format");
		goto fail;
	}

	sample_type = (int)propGetIntDefault(vsapi, in, "sample", 0, node_fmt->sampleType);
	depth = (int)propGetIntDefault(vsapi, in, "depth", 0, node_fmt->bitsPerSample);

	if (sample_type != stInteger && sample_type != stFloat) {
		strcpy(fail_str, "invalid sample type: must be stInteger or stFloat");
		goto fail;
	}
	if (sample_type == stFloat && depth != 16 && depth != 32) {
		strcpy(fail_str, "invalid depth: must be 16 or 32 for stFloat");
		goto fail;
	}
	if (sample_type == stInteger && (depth <= 0 || depth > 16)) {
		strcpy(fail_str, "invalid depth: must be between 1-16 for stInteger");
		goto fail;
	}

	vi = *node_vi;
	vi.format = vsapi->registerFormat(node_fmt->colorFamily, sample_type, depth < 8 ? 8 : depth, node_fmt->subSamplingW, node_fmt->subSamplingH, core);

	if (!vi.format) {
		strcpy(fail_str, "unable to register output VSFormat");
		goto fail;
	}

	zimg2_depth_params_default(&params, ZIMG_API_VERSION);

	params.width = node_vi->width;
	params.height = node_vi->height;

	dither_str = vsapi->propGetData(in, "dither", 0, &err);
	if (!err)
		params.dither_type = translate_dither(dither_str);

	params.chroma = 0;

	params.pixel_in = translate_pixel(node_fmt);
	params.depth_in = node_fmt->bitsPerSample;
	params.range_in = (int)propGetIntDefault(vsapi, in, "range_in", 0, node_fmt->colorFamily == cmRGB ? ZIMG_RANGE_FULL : ZIMG_RANGE_LIMITED);

	params.pixel_out = translate_pixel(vi.format);
	params.depth_out = depth;
	params.range_out = (int)propGetIntDefault(vsapi, in, "range_out", 0, params.range_in);

	if (!(filter = zimg2_depth_create(&params))) {
		zimg_get_last_error(fail_str, sizeof(fail_str));
		goto fail;
	}

	if (node_fmt->colorFamily == cmYUV || node_fmt->colorFamily == cmYCoCg) {
		params.width = params.width >> node_fmt->subSamplingW;
		params.height = params.height >> node_fmt->subSamplingH;
		params.chroma = 1;

		if (!(filter_uv = zimg2_depth_create(&params))) {
			zimg_get_last_error(fail_str, sizeof(fail_str));
			goto fail;
		}
	}

	if (!(data = malloc(sizeof(*data)))) {
		strcpy(fail_str, "error allocating vs_depth_data");
		goto fail;
	}

	data->filter = filter;
	data->filter_uv = filter_uv;
	data->node = node;
	data->vi = vi;
	data->fullrange = params.range_out == ZIMG_RANGE_FULL;

	vsapi->createFilter(in, out, "depth", vs_depth_init, vs_depth_get_frame, vs_depth_free, fmParallel, 0, data, core);
	return;
fail:
	vsapi->setError(out, fail_str);
	vsapi->freeNode(node);
	zimg2_filter_free(filter);
	zimg2_filter_free(filter_uv);
	free(data);
}


typedef struct vs_resize_data {
	zimg_filter *filter;
	zimg_filter *filter_uv;
	VSNodeRef *node;
	VSVideoInfo vi;
	int mpeg2;
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
	char fail_str[1024];
	int err = 0;

	if (activationReason == arInitial) {
		vsapi->requestFrameFilter(n, data->node, frameCtx);
	} else {
		const VSFrameRef *src_frame = 0;
		VSFrameRef *dst_frame = 0;
		void *tmp = 0;

		const void *src_plane[3] = { 0 };
		void *dst_plane[3] = { 0 };
		ptrdiff_t src_stride[3] = { 0 };
		ptrdiff_t dst_stride[3] = { 0 };
		int width;
		int height;
		unsigned num_planes;
		size_t tmp_size;
		size_t tmp_size_uv;
		VSMap *props;
		unsigned p;

		width = data->vi.width;
		height = data->vi.height;
		num_planes = data->vi.format->numPlanes;

		src_frame = vsapi->getFrameFilter(n, data->node, frameCtx);
		dst_frame = vsapi->newVideoFrame(data->vi.format, width, height, src_frame, core);

		if ((err = zimg2_plane_filter_get_tmp_size(data->filter, &tmp_size))) {
			zimg_get_last_error(fail_str, sizeof(fail_str));
			goto fail;
		}
		if (data->filter_uv) {
			if ((err = zimg2_plane_filter_get_tmp_size(data->filter_uv, &tmp_size_uv))) {
				zimg_get_last_error(fail_str, sizeof(fail_str));
				goto fail;
			}
		} else {
			tmp_size_uv = 0;
		}

		VS_ALIGNED_MALLOC(&tmp, VSMAX(tmp_size, tmp_size_uv), 32);
		if (!tmp) {
			strcpy(fail_str, "error allocating temporary buffer");
			err = 1;
			goto fail;
		}

		for (p = 0; p < num_planes; ++p) {
			const zimg_filter *filter;

			if ((p == 1 || p == 2) && data->filter_uv)
				filter = data->filter_uv;
			else
				filter = data->filter;

			src_plane[0] = vsapi->getReadPtr(src_frame, p);
			src_stride[0] = vsapi->getStride(src_frame, p);

			dst_plane[0] = vsapi->getWritePtr(dst_frame, p);
			dst_stride[0] = vsapi->getStride(dst_frame, p);

			if ((err = zimg2_plane_filter_process(filter, tmp, src_plane, dst_plane, src_stride, dst_stride))) {
				zimg_get_last_error(fail_str, sizeof(fail_str));
				goto fail;
			}
		}

		props = vsapi->getFramePropsRW(dst_frame);
		vsapi->propSetInt(props, "_ChromaLocation", data->mpeg2 ? 0 : 1, paReplace);

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
	zimg2_filter_free(data->filter);
	zimg2_filter_free(data->filter_uv);
	free(data);
}

static void VS_CC vs_resize_create(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi)
{
	vs_resize_data *data = 0;
	zimg_resize_params params;
	zimg_filter *filter = 0;
	zimg_filter *filter_uv = 0;
	char fail_str[1024];
	int err = 0;

	VSNodeRef *node = 0;
	const VSVideoInfo *node_vi;
	const VSFormat *node_fmt;
	VSVideoInfo vi;

	const char *filter_str;
	unsigned subsampling_w;
	unsigned subsampling_h;
	int mpeg2 = 0;

	node = vsapi->propGetNode(in, "clip", 0, 0);
	node_vi = vsapi->getVideoInfo(node);
	node_fmt = node_vi->format;

	if (!isConstantFormat(node_vi)) {
		strcpy(fail_str, "clip must have constant format");
		goto fail;
	}

	zimg2_resize_params_default(&params, ZIMG_API_VERSION);

	params.src_width = node_vi->width;
	params.src_height = node_vi->height;
	params.dst_width = (unsigned)vsapi->propGetInt(in, "width", 0, 0);
	params.dst_height = (unsigned)vsapi->propGetInt(in, "height", 0, 0);

	params.pixel_type = translate_pixel(node_fmt);
	params.depth = node_fmt->bitsPerSample;

	params.shift_w = propGetFloatDefault(vsapi, in, "shift_w", 0, params.shift_w);
	params.shift_h = propGetFloatDefault(vsapi, in, "shift_h", 0, params.shift_h);
	params.subwidth = propGetFloatDefault(vsapi, in, "subwidth", 0, params.subwidth);
	params.subheight = propGetFloatDefault(vsapi, in, "subheight", 0, params.subheight);

	filter_str = vsapi->propGetData(in, "filter", 0, &err);
	if (!err)
		params.filter_type = translate_filter(filter_str);

	params.filter_param_a = propGetFloatDefault(vsapi, in, "filter_param_a", 0, params.filter_param_a);
	params.filter_param_b = propGetFloatDefault(vsapi, in, "filter_param_b", 0, params.filter_param_b);

	subsampling_w = (int)propGetIntDefault(vsapi, in, "subsample_w", 0, node_fmt->subSamplingW);
	subsampling_h = (int)propGetIntDefault(vsapi, in, "subsample_h", 0, node_fmt->subSamplingH);

	if ((node_fmt->colorFamily != cmYUV && node_fmt->colorFamily != cmYCoCg) && (subsampling_w || subsampling_h)) {
		strcpy(fail_str, "subsampling not allowed for colorfamily");
		goto fail;
	}

	vi = *node_vi;
	vi.width = params.dst_width;
	vi.height = params.dst_height;
	vi.format = vsapi->registerFormat(node_fmt->colorFamily, node_fmt->sampleType, node_fmt->bitsPerSample, subsampling_w, subsampling_h, core);

	if (!(filter = zimg2_resize_create(&params))) {
		zimg_get_last_error(fail_str, sizeof(fail_str));
		goto fail;
	}

	if (node_fmt->subSamplingW || node_fmt->subSamplingH || subsampling_w || subsampling_h) {
		zimg_resize_params params_uv;

		double scale_w = 1.0 / (double)(1 << node_fmt->subSamplingW);
		double scale_h = 1.0 / (double)(1 << node_fmt->subSamplingH);

		const char *chroma_loc_in = propGetDataDefault(vsapi, in, "chroma_loc_in", 0, "mpeg2");
		const char *chroma_loc_out = propGetDataDefault(vsapi, in, "chroma_loc_out", 0, chroma_loc_in);

		zimg2_resize_params_default(&params_uv, ZIMG_API_VERSION);

		params_uv.src_width = params.src_width >> node_fmt->subSamplingW;
		params_uv.src_height = params.src_height >> node_fmt->subSamplingH;
		params_uv.dst_width = params.dst_width >> subsampling_w;
		params_uv.dst_height = params.dst_height >> subsampling_h;

		params_uv.pixel_type = params.pixel_type;
		params_uv.depth = params.depth;

		params_uv.shift_w = params.shift_w * scale_w;
		params_uv.shift_h = params.shift_h * scale_h;
		params_uv.subwidth = params.subwidth * scale_w;
		params_uv.subheight = params.subheight * scale_h;

		params_uv.shift_w += chroma_adjust_h(chroma_loc_in, chroma_loc_out, node_fmt->subSamplingW, subsampling_w);
		params_uv.shift_h += chroma_adjust_v(chroma_loc_in, chroma_loc_out, node_fmt->subSamplingH, subsampling_h);

		filter_str = vsapi->propGetData(in, "filter_uv", 0, &err);
		if (!err) {
			params_uv.filter_type = translate_filter(filter_str);
		} else {
			params_uv.filter_type = params.filter_type;
			params_uv.filter_param_a = params.filter_param_a;
			params_uv.filter_param_b = params.filter_param_b;
		}

		params_uv.filter_param_a = propGetFloatDefault(vsapi, in, "filter_param_a_uv", 0, params_uv.filter_param_a);
		params_uv.filter_param_b = propGetFloatDefault(vsapi, in, "filter_param_b_uv", 0, params_uv.filter_param_b);

		if (!(filter_uv = zimg2_resize_create(&params_uv))) {
			zimg_get_last_error(fail_str, sizeof(fail_str));
			goto fail;
		}

		mpeg2 = !strcmp(chroma_loc_out, "mpeg2") && (subsampling_w || subsampling_h);
	}

	if (!(data = malloc(sizeof(*data)))) {
		strcpy(fail_str, "error allocating vs_resize_data");
		goto fail;
	}

	data->filter = filter;
	data->filter_uv = filter_uv;
	data->node = node;
	data->vi = vi;
	data->mpeg2 = mpeg2;

	vsapi->createFilter(in, out, "resize", vs_resize_init, vs_resize_get_frame, vs_resize_free, fmParallel, 0, data, core);
	return;
fail:
	vsapi->setError(out, fail_str);
	vsapi->freeNode(node);
	zimg2_filter_free(filter);
	zimg2_filter_free(filter_uv);
	free(data);
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
	configFunc("the.weather.channel", "z", "batman", VAPOURSYNTH_API_VERSION, 1, plugin);

	registerFunc("Colorspace",
	             "clip:clip;"
	             "matrix_in:int;"
	             "transfer_in:int;"
	             "primaries_in:int;"
	             "matrix_out:int:opt;"
	             "transfer_out:int:opt;"
	             "primaries_out:int:opt;"
	             "range_in:int:opt;"
	             "range_out:int:opt;",
	             vs_colorspace_create, 0, plugin);
	registerFunc("Depth",
	             "clip:clip;"
	             "dither:data:opt;"
	             "sample:int:opt;"
	             "depth:int:opt;"
	             "range_in:int:opt;"
	             "range_out:int:opt;",
	             vs_depth_create, 0, plugin);
	registerFunc("Resize",
	             "clip:clip;"
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
	             "chroma_loc_out:data:opt;",
	             vs_resize_create, 0, plugin);

	registerFunc("SetCPU", "cpu:data;", vs_set_cpu, 0, plugin);

	zimg_set_cpu(ZIMG_CPU_AUTO);
}

#endif
