// z.lib example code for interlacing API.
//
// Example code demonstrates the use of z.lib to scale an interlaced YV12/I420
// image. Emphasis is placed on illustrating how this operation can be performed
// without copying the individual fields into separate buffers.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <system_error>

#include <zimg++.hpp>

#include "aligned_malloc.h"
#include "argparse.h"
#include "mmap.h"

namespace {

struct Arguments {
	const char *inpath;
	const char *outpath;
	unsigned in_w;
	unsigned in_h;
	unsigned out_w;
	unsigned out_h;
};

const ArgparseOption program_positional[] = {
	{ OPTION_STRING, nullptr, "inpath",     offsetof(struct Arguments, inpath),  nullptr, "input path" },
	{ OPTION_STRING, nullptr, "outpath",    offsetof(struct Arguments, outpath), nullptr, "output path" },
	{ OPTION_UINT,   nullptr, "in_width",   offsetof(struct Arguments, in_w),    nullptr, "input width" },
	{ OPTION_UINT,   nullptr, "in_height",  offsetof(struct Arguments, in_h),    nullptr, "input height" },
	{ OPTION_UINT,   nullptr, "out_width",  offsetof(struct Arguments, out_w),   nullptr, "output width" },
	{ OPTION_UINT,   nullptr, "out_height", offsetof(struct Arguments, out_h),   nullptr, "output height" },
	{ OPTION_NULL }
};

const ArgparseCommandLine program_def = { nullptr, program_positional, "interlace_example", "resize interlaced 4:2:0 images" };


struct YV12Image {
	std::shared_ptr<void> handle;
	void *image_base[3];
	unsigned width;
	unsigned height;
};

struct Callback {
	const zimgxx::zimage_buffer *buffer;
	const YV12Image *file;
	bool from_buffer;
	bool top_field;
};

YV12Image open_yv12_file(const char *path, unsigned w, unsigned h, bool write)
{
	YV12Image file;

	std::shared_ptr<MemoryMappedFile> mmap;
	size_t size = static_cast<size_t>(w) * h + static_cast<size_t>(w / 2) * (h / 2) * 2;
	uint8_t *file_base;

	if (write) {
		mmap = std::make_shared<MemoryMappedFile>(path, size, MemoryMappedFile::CREATE_TAG);
		file_base = static_cast<uint8_t *>(mmap->write_ptr());
	} else {
		mmap = std::make_shared<MemoryMappedFile>(path, MemoryMappedFile::READ_TAG);

		if (mmap->size() != size)
			throw std::runtime_error{ "bad YV12 file size" };

		file_base = static_cast<uint8_t *>(const_cast<void *>(mmap->read_ptr()));
	}

	file.handle = std::move(mmap);

	file.image_base[0] = file_base;
	file.image_base[1] = file_base + static_cast<ptrdiff_t>(w) * h;
	file.image_base[2] = file_base + static_cast<ptrdiff_t>(w) * h + static_cast<ptrdiff_t>(w / 2) * (h / 2);

	file.width = w;
	file.height = h;

	return file;
}

zimgxx::zimage_format get_image_format(const YV12Image &file, bool top)
{
	zimgxx::zimage_format format;

	format.width = file.width;
	format.height = file.height / 2;
	format.pixel_type = ZIMG_PIXEL_BYTE;

	format.subsample_w = 1;
	format.subsample_h = 1;

	format.color_family = ZIMG_COLOR_YUV;
	format.field_parity = top ? ZIMG_FIELD_TOP : ZIMG_FIELD_BOTTOM;

	return format;
}

std::pair<zimgxx::zimage_buffer, std::shared_ptr<void>> allocate_buffer(const zimgxx::zimage_format &format, unsigned count)
{
	zimgxx::zimage_buffer buffer;
	std::shared_ptr<void> handle;
	unsigned char *ptr;

	unsigned mask = zimg_select_buffer_mask(count);
	size_t channel_size[3] = { 0 };
	size_t pixel_size;

	count = (mask == ZIMG_BUFFER_MAX) ? format.height : mask + 1;

	if (format.pixel_type == ZIMG_PIXEL_FLOAT)
		pixel_size = sizeof(float);
	else if (format.pixel_type == ZIMG_PIXEL_WORD || format.pixel_type == ZIMG_PIXEL_HALF)
		pixel_size = sizeof(uint16_t);
	else
		pixel_size = sizeof(uint8_t);

	for (unsigned p = 0; p < (format.color_family == ZIMG_COLOR_GREY ? 1U : 3U); ++p) {
		unsigned count_plane = p ? count >> format.subsample_h : count;
		unsigned mask_plane = (mask == ZIMG_BUFFER_MAX) ? mask : mask >> format.subsample_h;
		size_t row_size = format.width * pixel_size;
		ptrdiff_t stride = (row_size + 31) & ~31;

		buffer.mask(p) = mask_plane;
		buffer.stride(p) = stride;
		channel_size[p] = static_cast<size_t>(stride) * count_plane;
	}

	handle.reset(aligned_malloc(channel_size[0] + channel_size[1] + channel_size[2], 32), &aligned_free);
	ptr = static_cast<unsigned char *>(handle.get());

	for (unsigned p = 0; p < (format.color_family == ZIMG_COLOR_GREY ? 1U : 3U); ++p) {
		buffer.data(p) = ptr;
		ptr += channel_size[p];
	}

	return{ buffer, handle };
}

std::shared_ptr<void> allocate_buffer(size_t size)
{
	return{ aligned_malloc(size, 32), &aligned_free };
}

// I/O callback that copies the requested lines between buffers.
//
// In this example, the image format is 4:2:0, so the callback must copy data
// from two scanlines in each call.
int yv12_bitblt_callback(void *user, unsigned i, unsigned left, unsigned right)
{
	const Callback *cb = static_cast<Callback *>(user);
	const zimgxx::zimage_buffer &buf = *cb->buffer;
	unsigned file_phase = cb->top_field ? 0 : 1;

	uint8_t *src_p[4];
	uint8_t *dst_p[4];
	uint8_t **buf_pp;
	uint8_t **img_pp;

	if (cb->from_buffer) {
		buf_pp = src_p;
		img_pp = dst_p;
	} else {
		buf_pp = dst_p;
		img_pp = src_p;
	}

	left = left % 2 ? left - 1 : left;
	right = right % 2 ? right + 1 : right;

	buf_pp[0] = static_cast<uint8_t *>(buf.line_at(i + 0, 0));
	buf_pp[1] = static_cast<uint8_t *>(buf.line_at(i + 1, 0));
	buf_pp[2] = static_cast<uint8_t *>(buf.line_at(i / 2, 1));
	buf_pp[3] = static_cast<uint8_t *>(buf.line_at(i / 2, 2));

	// Since the fields are being processed individually, double the line numbers.
	img_pp[0] = static_cast<uint8_t *>(cb->file->image_base[0]) + ((i + 0) * 2 + file_phase) * cb->file->width + left;
	img_pp[1] = static_cast<uint8_t *>(cb->file->image_base[0]) + ((i + 1) * 2 + file_phase) * cb->file->width + left;
	img_pp[2] = static_cast<uint8_t *>(cb->file->image_base[1]) + (i + file_phase) * (cb->file->width / 2) + left / 2;
	img_pp[3] = static_cast<uint8_t *>(cb->file->image_base[2]) + (i + file_phase) * (cb->file->width / 2) + left / 2;

	memcpy(dst_p[0], src_p[0], right - left);
	memcpy(dst_p[1], src_p[1], right - left);
	memcpy(dst_p[2], src_p[2], (right - left) / 2);
	memcpy(dst_p[3], src_p[3], (right - left) / 2);

	return 0;
}

void process(const YV12Image &in_data, const YV12Image &out_data)
{
	// (1) Fill the format descriptors for the top and bottom fields. The same
	// context can not be used for both fields, as they are located at opposite
	// offsets from the image center. If the fields were to be scaled as
	// progressive-scan images of half height, spatial misalignment of the
	// output would occur.
	zimgxx::zimage_format in_format_t = get_image_format(in_data, true);
	zimgxx::zimage_format in_format_b = get_image_format(in_data, false);
	zimgxx::zimage_format out_format_t = get_image_format(out_data, true);
	zimgxx::zimage_format out_format_b = get_image_format(out_data, false);

	// (2) Build the processing contexts.
	zimgxx::FilterGraph graph_t{ zimgxx::FilterGraph::build(in_format_t, out_format_t) };
	zimgxx::FilterGraph graph_b{ zimgxx::FilterGraph::build(in_format_b, out_format_b) };

	// (3) Allocate scanline and temporary buffers for input and output data. In
	// this case, the same buffers can be used for both fields.
	unsigned input_buffering_t = graph_t.get_input_buffering();
	unsigned input_buffering_b = graph_b.get_input_buffering();
	unsigned output_buffering_t = graph_t.get_input_buffering();
	unsigned output_buffering_b = graph_b.get_input_buffering();
	size_t tmp_size_t = graph_t.get_tmp_size();
	size_t tmp_size_b = graph_b.get_tmp_size();

	std::cout << "input buffering:  " << std::max(input_buffering_t, input_buffering_b) << '\n';
	std::cout << "output buffering: " << std::max(output_buffering_t, output_buffering_b) << '\n';
	std::cout << "heap usage: " << std::max(tmp_size_t, tmp_size_b) << '\n';

	auto in_buf = allocate_buffer(in_format_t, std::max(input_buffering_t, input_buffering_b));
	auto out_buf = allocate_buffer(out_format_t, std::max(output_buffering_t, output_buffering_b));
	auto tmp_buf = allocate_buffer(std::max(tmp_size_t, tmp_size_b));

	// (4) Store context information required by the I/O callbacks. The
	// callbacks convert between the on-disk and z.lib alignment requirements.
	Callback unpack_data = { &in_buf.first, &in_data, false, true };
	Callback pack_data = { &out_buf.first, &out_data, true, true };

	// (5) Process the top field.
	graph_t.process(in_buf.first.as_const(), out_buf.first, tmp_buf.get(),
	                yv12_bitblt_callback, &unpack_data, yv12_bitblt_callback, &pack_data);

	// (6) Process the bottom field.
	unpack_data.top_field = false;
	pack_data.top_field = false;

	graph_b.process(in_buf.first.as_const(), out_buf.first, tmp_buf.get(),
	                yv12_bitblt_callback, &unpack_data, yv12_bitblt_callback, &pack_data);
}

void execute(const Arguments &args)
{
	YV12Image in_image = open_yv12_file(args.inpath, args.in_w, args.in_h, false);
	YV12Image out_image = open_yv12_file(args.outpath, args.out_w, args.out_h, true);

	process(in_image, out_image);
}

} // namespace


int main(int argc, char **argv)
{
	Arguments args{};
	int ret;

	if ((ret = argparse_parse(&program_def, &args, argc, argv)) < 0)
		return ret == ARGPARSE_HELP_MESSAGE ? 0 : ret;

	try {
		execute(args);
	} catch (const std::system_error &e) {
		std::cerr << "system_error " << e.code() << ": " << e.what() << '\n';
		return 2;
	} catch (const zimgxx::zerror &e) {
		std::cerr << "zimg error " << e.code << ": " << e.msg << '\n';
		return 2;
	} catch (const std::runtime_error &e) {
		std::cerr << "runtime_error: " << e.what() << '\n';
		return 2;
	} catch (const std::logic_error &e) {
		std::cerr << "logic_error: " << e.what() << '\n';
		throw;
	}

	return 0;
}
