/* z.lib example code for C API.
 *
 * Example code demonstrates the use of z.lib to scale a YV12/I420 image.
 */

#ifndef _WIN32
  #define _POSIX_C_SOURCE 200112L
#endif

#include <stddef.h>
#include <stdio.h>

#include <zimg.h>

#include "aligned_malloc.h"
#include "argparse.h"

struct Arguments {
	const char *inpath;
	const char *outpath;
	unsigned in_w;
	unsigned in_h;
	unsigned out_w;
	unsigned out_h;
};

static const ArgparseOption program_positional[] = {
	{ OPTION_STRING, 0, "inpath",     offsetof(struct Arguments, inpath),  0, "input path" },
	{ OPTION_STRING, 0, "outpath",    offsetof(struct Arguments, outpath), 0, "output path" },
	{ OPTION_UINT,   0, "in_width",   offsetof(struct Arguments, in_w),    0, "input width" },
	{ OPTION_UINT,   0, "in_height",  offsetof(struct Arguments, in_h),    0, "input height" },
	{ OPTION_UINT,   0, "out_width",  offsetof(struct Arguments, out_w),   0, "output width" },
	{ OPTION_UINT,   0, "out_height", offsetof(struct Arguments, out_h),   0, "output height" },
	{ OPTION_NULL }
};

static const ArgparseCommandLine program_def = { 0, program_positional, "api_example_c", "resize 4:2:0 images" };

static ptrdiff_t width_to_stride(unsigned w)
{
	return ((size_t)w + 31) & ~31;
}

/* Allocate an image buffer and initialize plane pointers and strides. */
static void *init_image(unsigned w, unsigned h, void *data[3], ptrdiff_t stride[3])
{
	size_t rowsize = width_to_stride(w);
	size_t rowsize_uv = width_to_stride(w / 2);
	size_t size = 0;
	void *ptr;
	char *pptr;
	unsigned p;

	size += rowsize * h;
	size += rowsize * (h / 2);
	size += rowsize * (h / 2);

	if (!(ptr = aligned_malloc(size, 32)))
		return 0;

	pptr = ptr;
	for (p = 0; p < 3; ++p) {
		size_t rowsize_p = p ? rowsize_uv : rowsize;
		size_t plane_sz = rowsize_p * (p ? h / 2 : h);

		data[p] = pptr;
		stride[p] = rowsize_p;

		pptr += plane_sz;
	}
	return ptr;
}

static int read_image_from_file(const char *path, unsigned w, unsigned h, void * const data[3], const ptrdiff_t stride[3])
{
	FILE *file;
	unsigned p, i;
	int ret = 1;

	if (!(file = fopen(path, "rb")))
		goto fail;

	for (p = 0; p < 3; ++p) {
		unsigned width = p ? w / 2 : w;
		unsigned height = p ? h / 2 : h;
		char *ptr = data[p];

		for (i = 0; i < height; ++i) {
			char *row_ptr = ptr;
			size_t to_read = width;

			while (to_read) {
				size_t n = fread(row_ptr, 1, to_read, file);

				if (n != to_read) {
					if (ferror(file)) {
						perror("error reading file");
						goto fail;
					}
					if (feof(file)) {
						fprintf(stderr, "unexpected end of file at: p=%u, i=%u\n", p, i);
						goto fail;
					}
				}

				row_ptr += n;
				to_read -= n;
			}
			ptr += stride[p];
		}
	}

	ret = 0;
fail:
	if (file)
		fclose(file);
	return ret;
}

static int write_image_to_file(const char *path, unsigned w, unsigned h, const void * const data[3], const ptrdiff_t stride[3])
{
	FILE *file;
	unsigned p, i;
	int ret = 1;

	if (!(file = fopen(path, "wb")))
		goto fail;

	for (p = 0; p < 3; ++p) {
		unsigned width = p ? w / 2 : w;
		unsigned height = p ? h / 2 : h;
		const char *ptr = data[p];

		for (i = 0; i < height; ++i) {
			const char *row_ptr = ptr;
			size_t to_write = width;

			while (to_write) {
				size_t n = fwrite(row_ptr, 1, to_write, file);

				if (n != to_write && ferror(file)) {
					perror("error writing file");
					goto fail;
				}

				row_ptr += n;
				to_write -= n;
			}
			ptr += stride[p];
		}
	}

	ret = 0;
fail:
	if (file)
		fclose(file);
	return ret;
}

static void print_zimg_error(void)
{
	char err_msg[1024];
	int err_code = zimg_get_last_error(err_msg, sizeof(err_msg));

	fprintf(stderr, "zimg error %d: %s\n", err_code, err_msg);
}

static int process(const struct Arguments *args, const void * const src_p[3], void * const dst_p[3], const ptrdiff_t src_stride[3], const ptrdiff_t dst_stride[3])
{
	zimg_filter_graph *graph = 0;
	zimg_image_buffer_const src_buf = { ZIMG_API_VERSION };
	zimg_image_buffer dst_buf = { ZIMG_API_VERSION };
	zimg_image_format src_format;
	zimg_image_format dst_format;
	size_t tmp_size;
	void *tmp = 0;
	unsigned p;
	int ret = 1;

	/* (1) Initialize structures with default values. */
	zimg_image_format_default(&src_format, ZIMG_API_VERSION);
	zimg_image_format_default(&dst_format, ZIMG_API_VERSION);

	/* (2) Fill the format descriptors for the input and output images. */
	src_format.width = args->in_w;
	src_format.height = args->in_h;
	src_format.pixel_type = ZIMG_PIXEL_BYTE;

	src_format.subsample_w = 1;
	src_format.subsample_h = 1;

	src_format.color_family = ZIMG_COLOR_YUV;

	dst_format.width = args->out_w;
	dst_format.height = args->out_h;
	dst_format.pixel_type = ZIMG_PIXEL_BYTE;

	dst_format.subsample_w = 1;
	dst_format.subsample_h = 1;

	dst_format.color_family = ZIMG_COLOR_YUV;

	/* (3) Build the processing context. */
	if (!(graph = zimg_filter_graph_build(&src_format, &dst_format, 0))) {
		print_zimg_error();
		goto fail;
	}
	if ((ret = zimg_filter_graph_get_tmp_size(graph, &tmp_size))) {
		print_zimg_error();
		goto fail;
	}

	printf("heap usage: %lu\n", (unsigned long)tmp_size);

	/* (4) Allocate a temporary buffer for use during processing. If additional
	 * images need to be processed, the same temporary buffer can be used in
	 * subsequent calls.
	 */
	if (!(tmp = aligned_malloc(tmp_size, 32)))
		goto fail;

	/* (5) Set the buffer pointers and strides. In this example, the input and
	 * output images are already in planar format, so the images are used
	 * directly as the scanline buffers. To indicate this, the buffer mask is
	 * set to ZIMG_BUFFER_MAX.
	 */
	for (p = 0; p < 3; ++p) {
		src_buf.plane[p].data = src_p[p];
		src_buf.plane[p].stride = src_stride[p];
		src_buf.plane[p].mask = ZIMG_BUFFER_MAX;

		dst_buf.plane[p].data = dst_p[p];
		dst_buf.plane[p].stride = dst_stride[p];
		dst_buf.plane[p].mask = ZIMG_BUFFER_MAX;
	}

	/* (6) Perform the image scaling operation. */
	if ((ret = zimg_filter_graph_process(graph, &src_buf, &dst_buf, tmp, 0, 0, 0, 0))) {
		print_zimg_error();
		goto fail;
	}

	ret = 0;
fail:
	zimg_filter_graph_free(graph);
	aligned_free(tmp);
	return ret;
}

static int execute(const struct Arguments *args)
{
	void *src_handle = 0;
	void *dst_handle = 0;
	void *src_planes[3];
	void *dst_planes[3];
	const void *src_planes_const[3];
	const void *dst_planes_const[3];
	ptrdiff_t src_stride[3];
	ptrdiff_t dst_stride[3];
	int ret = 1;

	if (!(src_handle = init_image(args->in_w, args->in_h, src_planes, src_stride)))
		goto fail;
	if (!(dst_handle = init_image(args->out_w, args->out_h, dst_planes, dst_stride)))
		goto fail;

	src_planes_const[0] = src_planes[0];
	src_planes_const[1] = src_planes[1];
	src_planes_const[2] = src_planes[2];

	dst_planes_const[0] = dst_planes[0];
	dst_planes_const[1] = dst_planes[1];
	dst_planes_const[2] = dst_planes[2];

	if ((ret = read_image_from_file(args->inpath, args->in_w, args->in_h, src_planes, src_stride)))
		goto fail;
	if ((ret = process(args, src_planes_const, dst_planes, src_stride, dst_stride)))
		goto fail;
	if ((ret = write_image_to_file(args->outpath, args->out_w, args->out_h, dst_planes_const, dst_stride)))
		goto fail;

	ret = 0;
fail:
	aligned_free(src_handle);
	aligned_free(dst_handle);
	return ret;
}

int main(int argc, char **argv)
{
	struct Arguments args = { 0 };
	int ret;

	if ((ret = argparse_parse(&program_def, &args, argc, argv)) < 0)
		return ret == ARGPARSE_HELP_MESSAGE ? 0 : ret;

	if ((ret = execute(&args)))
		fprintf(stderr, "error: %d\n", ret);

	return ret;
}
