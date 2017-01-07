#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <system_error>
#include <utility>

#include <zimg++.hpp>

#include "aligned_malloc.h"
#include "argparse.h"
#include "mmap.h"
#include "win32_bitmap.h"

namespace {

struct Arguments {
	const char *inpath;
	const char *outpath;
	unsigned out_w;
	unsigned out_h;
	unsigned in_w;
	unsigned in_h;
	double shift_w = NAN;
	double shift_h = NAN;
	double subwidth = NAN;
	double subheight = NAN;
};

const ArgparseOption program_switches[] = {
	{ OPTION_UINT,  nullptr, "in-width",   offsetof(Arguments, in_w),      nullptr, "input width" },
	{ OPTION_UINT,  nullptr, "in-height",  offsetof(Arguments, in_h),      nullptr, "input height" },
	{ OPTION_FLOAT, nullptr, "shift-w",    offsetof(Arguments, shift_w),   nullptr, "shift image to the left by x subpixels" },
	{ OPTION_FLOAT, nullptr, "shift-h",    offsetof(Arguments, shift_h),   nullptr, "shift image to the top by x subpixels" },
	{ OPTION_FLOAT, nullptr, "sub-width",  offsetof(Arguments, subwidth),  nullptr, "treat image width differently from actual width" },
	{ OPTION_FLOAT, nullptr, "sub-height", offsetof(Arguments, subheight), nullptr, "treat image height differently from actual height" },
	{ OPTION_NULL }
};

const ArgparseOption program_positional[] = {
	{ OPTION_STRING, nullptr, "inpath",  offsetof(Arguments, inpath),  nullptr, "input path specifier" },
	{ OPTION_STRING, nullptr, "outpath", offsetof(Arguments, outpath), nullptr, "output path specifier" },
	{ OPTION_UINT,   "w",     "width",   offsetof(Arguments, out_w),   nullptr, "output width" },
	{ OPTION_UINT,   "h",     "height",  offsetof(Arguments, out_h),   nullptr, "output height" },
	{ OPTION_NULL }
};

const ArgparseCommandLine program_def = {
	program_switches,
	program_positional,
	"api_example",
	"convert images between BMP and YUY2",
	"Path specifier format: (bmp|yuy2):path"
};


enum class FileFormat {
	FILE_BMP,
	FILE_YUY2
};

struct ImageFile {
	FileFormat fmt;
	std::shared_ptr<void> handle;
	void *image_base;
	unsigned width;
	unsigned height;
	unsigned bmp_bit_count;
	ptrdiff_t stride;
};

struct Callback {
	const zimgxx::zimage_buffer *buffer;
	const ImageFile *file;
};

std::pair<FileFormat, const char *> parse_path_specifier(const char *pathspec)
{
	const char *sep = strchr(pathspec, ':');
	FileFormat fmt;

	if (!sep)
		throw std::runtime_error{ "bad path spec" };

	if (!strncmp(pathspec, "bmp", sep - pathspec))
		fmt = FileFormat::FILE_BMP;
	else if (!strncmp(pathspec, "yuy2", sep - pathspec))
		fmt = FileFormat::FILE_YUY2;
	else
		throw std::runtime_error{ "bad path spec: invalid file type" };

	return{ fmt, sep + 1 };
}

ImageFile open_file(FileFormat fmt, const char *path, unsigned w, unsigned h, bool write)
{
	ImageFile file = { fmt };

	if (fmt == FileFormat::FILE_BMP) {
		std::shared_ptr<WindowsBitmap> bmp;

		if (write) {
			bmp = std::make_shared<WindowsBitmap>(path, w, h, 24);
			file.image_base = bmp->write_ptr();
		} else {
			bmp = std::make_shared<WindowsBitmap>(path, WindowsBitmap::READ_TAG);
			file.image_base = const_cast<unsigned char *>(bmp->read_ptr());
		}

		file.width = bmp->width();
		file.height = bmp->height();
		file.bmp_bit_count = bmp->bit_count();
		file.stride = bmp->stride();

		file.handle = std::move(bmp);
	} else if (fmt == FileFormat::FILE_YUY2) {
		std::shared_ptr<MemoryMappedFile> mmap;
		size_t size = static_cast<size_t>(w) * h * 2;

		if (write) {
			mmap = std::make_shared<MemoryMappedFile>(path, size, MemoryMappedFile::CREATE_TAG);
			file.image_base = mmap->write_ptr();
		} else {
			mmap = std::make_shared<MemoryMappedFile>(path, MemoryMappedFile::READ_TAG);

			if (mmap->size() != size)
				throw std::runtime_error{ "bad YUY2 file size" };

			file.image_base = const_cast<void *>(mmap->read_ptr());
		}

		file.width = w;
		file.height = h;
		file.stride = w * 2;

		file.handle = std::move(mmap);
	} else {
		throw std::logic_error{ "bad file format" };
	}

	return file;
}

zimgxx::zimage_format get_image_format(const ImageFile &file)
{
	zimgxx::zimage_format format;

	format.width = file.width;
	format.height = file.height;

	if (file.fmt == FileFormat::FILE_BMP) {
		format.subsample_w = 0;
		format.subsample_h = 0;

		format.pixel_type = ZIMG_PIXEL_BYTE;
		format.color_family = ZIMG_COLOR_RGB;
		format.matrix_coefficients = ZIMG_MATRIX_RGB;
		format.pixel_range = ZIMG_RANGE_FULL;
	} else if (file.fmt == FileFormat::FILE_YUY2) {
		format.subsample_w = 1;
		format.subsample_h = 0;

		format.pixel_type = ZIMG_PIXEL_BYTE;
		format.color_family = ZIMG_COLOR_YUV;
		format.matrix_coefficients = ZIMG_MATRIX_709;
		format.pixel_range = ZIMG_RANGE_LIMITED;
	} else {
		throw std::logic_error{ "bad file format" };
	}

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
		unsigned count_plane = p ? count : count >> format.subsample_h;
		unsigned mask_plane = (mask == ZIMG_BUFFER_MAX) ? mask : mask >> format.subsample_h;
		size_t row_size = format.width * pixel_size;
		ptrdiff_t stride = row_size % 64 ? row_size - row_size % 64 + 64 : row_size;

		buffer.mask(p) = mask_plane;
		buffer.stride(p) = stride;
		channel_size[p] = static_cast<size_t>(stride) * count_plane;
	}

	handle.reset(aligned_malloc(channel_size[0] + channel_size[1] + channel_size[2], 64), &aligned_free);
	ptr = static_cast<unsigned char *>(handle.get());

	for (unsigned p = 0; p < (format.color_family == ZIMG_COLOR_GREY ? 1U : 3U); ++p) {
		buffer.data(p) = ptr;
		ptr += channel_size[p];
	}

	return{ buffer, handle };
}

std::shared_ptr<void> allocate_buffer(size_t size)
{
	return{ aligned_malloc(size, 64), &aligned_free };
}

void unpack_bgr(const void *bgr, void * const planar[3], unsigned bit_depth, unsigned left, unsigned right)
{
	const uint8_t *packed_bgr = static_cast<const uint8_t *>(bgr);
	uint8_t *planar_r = static_cast<uint8_t *>(planar[0]);
	uint8_t *planar_g = static_cast<uint8_t *>(planar[1]);
	uint8_t *planar_b = static_cast<uint8_t *>(planar[2]);
	unsigned step = bit_depth / 8;

	for (unsigned j = left; j < right; ++j) {
		uint8_t r, g, b;

		b = packed_bgr[j * step + 0];
		g = packed_bgr[j * step + 1];
		r = packed_bgr[j * step + 2];

		planar_r[j] = r;
		planar_g[j] = g;
		planar_b[j] = b;
	}
}

void unpack_yuy2(const void *yuy2, void * const planar[3], unsigned left, unsigned right)
{
	const uint8_t *packed_yuy2 = static_cast<const uint8_t *>(yuy2);
	uint8_t *planar_y = static_cast<uint8_t *>(planar[0]);
	uint8_t *planar_u = static_cast<uint8_t *>(planar[1]);
	uint8_t *planar_v = static_cast<uint8_t *>(planar[2]);

	left = left % 2 ? left - 1 : left;
	right = right % 2 ? right + 1 : right;

	for (unsigned j = left; j < right; j += 2) {
		uint8_t y0, y1, u0, v0;

		y0 = packed_yuy2[j * 2 + 0];
		u0 = packed_yuy2[j * 2 + 1];
		y1 = packed_yuy2[j * 2 + 2];
		v0 = packed_yuy2[j * 2 + 3];

		planar_y[j + 0] = y0;
		planar_y[j + 1] = y1;
		planar_u[j / 2] = u0;
		planar_v[j / 2] = v0;
	}
}

void pack_bgr(const void * const planar[3], void *bgr, unsigned bit_depth, unsigned left, unsigned right)
{
	const uint8_t *planar_r = static_cast<const uint8_t *>(planar[0]);
	const uint8_t *planar_g = static_cast<const uint8_t *>(planar[1]);
	const uint8_t *planar_b = static_cast<const uint8_t *>(planar[2]);
	uint8_t *packed_bgr = static_cast<uint8_t *>(bgr);
	unsigned step = bit_depth / 8;

	for (unsigned j = left; j < right; ++j) {
		uint8_t r, g, b;

		r = planar_r[j];
		g = planar_g[j];
		b = planar_b[j];

		packed_bgr[j * step + 0] = b;
		packed_bgr[j * step + 1] = g;
		packed_bgr[j * step + 2] = r;
	}
}

void pack_yuy2(const void * const planar[3], void *yuy2, unsigned left, unsigned right)
{
	const uint8_t *planar_y = static_cast<const uint8_t *>(planar[0]);
	const uint8_t *planar_u = static_cast<const uint8_t *>(planar[1]);
	const uint8_t *planar_v = static_cast<const uint8_t *>(planar[2]);
	uint8_t *packed_yuy2 = static_cast<uint8_t *>(yuy2);

	left = left % 2 ? left - 1 : left;
	right = right % 2 ? right + 1 : right;

	for (unsigned j = left; j < right; j += 2) {
		uint8_t y0, y1, u0, v0;

		y0 = planar_y[j];
		y1 = planar_y[j + 1];
		u0 = planar_u[j / 2];
		v0 = planar_v[j / 2];

		packed_yuy2[j * 2 + 0] = y0;
		packed_yuy2[j * 2 + 1] = u0;
		packed_yuy2[j * 2 + 2] = y1;
		packed_yuy2[j * 2 + 3] = v0;
	}
}

int unpack_image(void *user, unsigned i, unsigned left, unsigned right)
{
	const Callback *cb = static_cast<Callback *>(user);
	const void *img = static_cast<uint8_t *>(cb->file->image_base) + i * cb->file->stride;
	const zimgxx::zimage_buffer &buf = *cb->buffer;
	FileFormat fmt = cb->file->fmt;
	void *buf_data[3];

	for (unsigned p = 0; p < 3; ++p) {
		buf_data[p] = static_cast<char *>(buf.line_at(i, p));
	}

	if (fmt == FileFormat::FILE_BMP) {
		unpack_bgr(img, buf_data, cb->file->bmp_bit_count, left, right);
		return 0;
	} else if (fmt == FileFormat::FILE_YUY2) {
		unpack_yuy2(img, buf_data, left, right);
		return 0;
	} else {
		return 1;
	}
}

int pack_image(void *user, unsigned i, unsigned left, unsigned right)
{
	const Callback *cb = static_cast<Callback *>(user);
	void *img = static_cast<uint8_t *>(cb->file->image_base) + i * cb->file->stride;
	const zimgxx::zimage_buffer &buf = *cb->buffer;
	FileFormat fmt = cb->file->fmt;
	const void *buf_data[3];

	for (unsigned p = 0; p < 3; ++p) {
		buf_data[p] = static_cast<const char *>(buf.line_at(i, p));
	}

	if (fmt == FileFormat::FILE_BMP) {
		pack_bgr(buf_data, img, cb->file->bmp_bit_count, left, right);
		return 0;
	} else if (fmt == FileFormat::FILE_YUY2) {
		pack_yuy2(buf_data, img, left, right);
		return 0;
	} else {
		return 1;
	}
}

void process(const Arguments &args, const ImageFile &in_data, const ImageFile &out_data)
{
	zimgxx::zimage_format in_format = get_image_format(in_data);
	zimgxx::zimage_format out_format = get_image_format(out_data);

	// Additional fields in API structures do not break binary compatibility.
	// If relying on the specific semantics of fields not present in earlier versions,
	// the application should also check the API version at runtime.
#if ZIMG_API_VERSION >= ZIMG_MAKE_API_VERSION(2, 1)
	if (!std::isnan(args.shift_w) || !std::isnan(args.shift_h) || !std::isnan(args.subheight) || !std::isnan(args.subheight)) {
		if (zimg_get_api_version(nullptr, nullptr) < ZIMG_MAKE_API_VERSION(2, 1))
			std::cerr << "warning: subpixel operation requires API 2.1\n";

		in_format.active_region.left = args.shift_w;
		in_format.active_region.top = args.shift_h;
		in_format.active_region.width = args.subwidth;
		in_format.active_region.height = args.subheight;
	}
#endif

	zimgxx::FilterGraph graph{ zimgxx::FilterGraph::build(in_format, out_format) };
	unsigned input_buffering = graph.get_input_buffering();
	unsigned output_buffering = graph.get_output_buffering();
	size_t tmp_size = graph.get_tmp_size();

	std::cout << "input buffering:  " << input_buffering << '\n';
	std::cout << "output buffering: " << output_buffering << '\n';
	std::cout << "heap usage: " << tmp_size << '\n';

	auto in_buf = allocate_buffer(in_format, input_buffering);
	auto out_buf = allocate_buffer(out_format, output_buffering);
	auto tmp_buf = allocate_buffer(tmp_size);

	Callback unpack_cb_data = { &in_buf.first, &in_data };
	Callback pack_cb_data = { &out_buf.first, &out_data };

	graph.process(in_buf.first.as_const(), out_buf.first, tmp_buf.get(),
	              unpack_image, &unpack_cb_data, pack_image, &pack_cb_data);
}

void execute(const Arguments &args)
{
	auto in_spec = parse_path_specifier(args.inpath);
	auto out_spec = parse_path_specifier(args.outpath);

	ImageFile in_image = open_file(in_spec.first, in_spec.second, args.in_w, args.in_h, false);
	ImageFile out_image = open_file(out_spec.first, out_spec.second, args.out_w, args.out_h, true);

	process(args, in_image, out_image);
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
		return 2;
	}

	return 0;
}
