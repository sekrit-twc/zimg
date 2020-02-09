// z.lib example code for HDR API.
//
// Example code demonstrates the use of z.lib to operate on HDR images. In the
// example, an HDR10 image is decomposed into SDR and HDR components.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>

#include <zimg++.hpp>
#if ZIMG_API_VERSION < ZIMG_MAKE_API_VERSION(2, 2)
  #error API 2.2 required for HDR
#endif

#include "aligned_malloc.h"
#include "argparse.h"
#include "mmap.h"
#include "win32_bitmap.h"

namespace {

int decode_mask_key(const struct ArgparseOption *, void *out, const char *param, int)
{
	const char HEX_DIGITS[] = "0123456789abcdefABCDEF";

	uint8_t *mask_key = static_cast<uint8_t *>(out);

	try {
		std::string s{ param };
		if (s.size() != 6 || s.find_first_not_of(HEX_DIGITS) != std::string::npos)
			throw std::runtime_error{ "bad hex string" };

		mask_key[0] = static_cast<uint8_t>(std::stoi(s.substr(0, 2), nullptr, 16));
		mask_key[1] = static_cast<uint8_t>(std::stoi(s.substr(2, 2), nullptr, 16));
		mask_key[2] = static_cast<uint8_t>(std::stoi(s.substr(4, 2), nullptr, 16));
	} catch (const std::exception &e) {
		std::cerr << e.what() << '\n';
		return -1;
	}

	return 0;
}


struct Arguments {
	const char *inpath;
	const char *sdrpath;
	const char *hdrpath;
	unsigned width;
	unsigned height;
	double luminance;
	uint8_t mask_key[3];
	char fast;
};

const ArgparseOption program_switches[] = {
	{ OPTION_FLAG,   "f", "fast",      offsetof(Arguments, fast),      nullptr, "use fast gamma functions" },
	{ OPTION_FLOAT,  "l", "luminance", offsetof(Arguments, luminance), nullptr, "legacy peak brightness (cd/m^2)" },
	{ OPTION_USER1,  "k", "key",       offsetof(Arguments, mask_key),  decode_mask_key, "HDR color key (RRGGBB hex string)" },
	{ OPTION_STRING, "m", "mask",      offsetof(Arguments, hdrpath),   nullptr, "HDR difference mask" },
	{ OPTION_NULL }
};

const ArgparseOption program_positional[] = {
	{ OPTION_STRING, nullptr, "inpath",  offsetof(Arguments, inpath),  nullptr, "input path" },
	{ OPTION_STRING, nullptr, "outpath", offsetof(Arguments, sdrpath), nullptr, "output path" },
	{ OPTION_UINT,   "w",     "width",   offsetof(Arguments, width),   nullptr, "image width" },
	{ OPTION_UINT,   "h",     "height",  offsetof(Arguments, height),  nullptr, "image height" },
	{ OPTION_NULL }
};

const ArgparseCommandLine program_def = {
	program_switches,
	program_positional,
	"hdr_example",
	"show legacy and HDR portion of HDR10 images",
	"Input must be HDR10 (YUV 4:2:0, 10 bpc), SDR output is BMP, HDR output is planar HDR10 RGB"
};


struct ImageBuffer {
	std::shared_ptr<void> handle;
	zimgxx::zimage_buffer buffer;
};

size_t align(size_t n)
{
	return (n + 31) & ~31;
}

ImageBuffer allocate_buffer(unsigned width, unsigned height, unsigned subsample_w, unsigned subsample_h, size_t bytes_per_pel)
{
	ImageBuffer buf;

	buf.buffer.stride(0) = align(width * bytes_per_pel);
	buf.buffer.stride(1) = align((width >> subsample_h) * bytes_per_pel);
	buf.buffer.stride(2) = align((width >> subsample_w) * bytes_per_pel);
	buf.buffer.mask(0) = ZIMG_BUFFER_MAX;
	buf.buffer.mask(1) = ZIMG_BUFFER_MAX;
	buf.buffer.mask(2) = ZIMG_BUFFER_MAX;

	size_t buffer_size = buf.buffer.stride(0) * height + 2 * buf.buffer.stride(1) * (height >> subsample_h);
	buf.handle = std::shared_ptr<void>(aligned_malloc(buffer_size, 32), aligned_free);

	uint8_t *ptr = static_cast<uint8_t *>(buf.handle.get());
	buf.buffer.data(0) = ptr;
	buf.buffer.data(1) = ptr + buf.buffer.stride(0) * height;
	buf.buffer.data(2) = ptr + buf.buffer.stride(0) * height + buf.buffer.stride(1) * (height >> subsample_h);

	return buf;
}

ImageBuffer read_from_file(const char *path, unsigned width, unsigned height, unsigned subsample_w, unsigned subsample_h, size_t bytes_per_pel)
{
	ImageBuffer buf = allocate_buffer(width, height, subsample_w, subsample_h, bytes_per_pel);
	MemoryMappedFile mmap{ path, MemoryMappedFile::READ_TAG };

	size_t file_size = bytes_per_pel * width * height + 2 * bytes_per_pel * (width >> subsample_w) * (height >> subsample_h);
	if (mmap.size() != file_size)
		throw std::runtime_error{ "bad file size" };

	const uint8_t *src_p = static_cast<const uint8_t *>(mmap.read_ptr());

	for (unsigned p = 0; p < 3; ++p) {
		size_t rowsize = (width >> (p ? subsample_w : 0)) * bytes_per_pel;

		for (unsigned i = 0; i < height >> (p ? subsample_h : 0); ++i) {
			memcpy(buf.buffer.line_at(i, p), src_p, rowsize);
			src_p += rowsize;
		}
	}

	return buf;
}

void write_to_file(const ImageBuffer &buf, const char *path, unsigned width, unsigned height, unsigned subsample_w, unsigned subsample_h, size_t bytes_per_pel)
{
	size_t file_size = bytes_per_pel * width * height + 2 * bytes_per_pel * (width >> subsample_w) * (height >> subsample_h);
	MemoryMappedFile mmap{ path, file_size, MemoryMappedFile::CREATE_TAG };

	uint8_t *dst_p = static_cast<uint8_t *>(mmap.write_ptr());

	for (unsigned p = 0; p < 3; ++p) {
		size_t rowsize = (width >> (p ? subsample_w : 0)) * bytes_per_pel;

		for (unsigned i = 0; i < height >> (p ? subsample_h : 0); ++i) {
			memcpy(dst_p, buf.buffer.line_at(i, p), rowsize);
			dst_p += rowsize;
		}
	}
}

void write_to_bmp(const ImageBuffer &buf, const char *path, unsigned width, unsigned height)
{
	WindowsBitmap bmp{ path, static_cast<int>(width), static_cast<int>(height), 24 };
	unsigned char *dst_p = bmp.write_ptr();

	for (unsigned i = 0; i < height; ++i) {
		const uint8_t *src_r = static_cast<const uint8_t *>(buf.buffer.line_at(i, 0));
		const uint8_t *src_g = static_cast<const uint8_t *>(buf.buffer.line_at(i, 1));
		const uint8_t *src_b = static_cast<const uint8_t *>(buf.buffer.line_at(i, 2));

		for (unsigned j = 0; j < width; ++j) {
			dst_p[j * 3 + 0] = src_b[j];
			dst_p[j * 3 + 1] = src_g[j];
			dst_p[j * 3 + 2] = src_r[j];
		}

		dst_p += bmp.stride();
	}
}

float undo_gamma(float x)
{
	if (x < 4.5f * 0.018053968510807f)
		x = x / 4.5f;
	else
		x = std::pow((x + (1.09929682680944f - 1.0f)) / 1.09929682680944f, 1.0f / 0.45f);

	return x;
}

void mask_pixels(const zimgxx::zimage_buffer& src_buf, const zimgxx::zimage_buffer& mask_buf, unsigned width, unsigned height, const uint8_t *mask_val)
{
	float r_mask = undo_gamma(mask_val[0] / 255.0f);
	float g_mask = undo_gamma(mask_val[1] / 255.0f);
	float b_mask = undo_gamma(mask_val[2] / 255.0f);

	for (unsigned i = 0; i < height; ++i) {
		float *src_r = static_cast<float *>(src_buf.line_at(i, 0));
		float *src_g = static_cast<float *>(src_buf.line_at(i, 1));
		float *src_b = static_cast<float *>(src_buf.line_at(i, 2));

		float *dst_r = static_cast<float *>(mask_buf.line_at(i, 0));
		float *dst_g = static_cast<float *>(mask_buf.line_at(i, 1));
		float *dst_b = static_cast<float *>(mask_buf.line_at(i, 2));

		for (unsigned j = 0; j < width; ++j) {
			float r = src_r[j];
			float g = src_g[j];
			float b = src_b[j];

			if (r < 0.0f || g < 0.0f || b < 0.0f || r > 1.0f || g > 1.0f || b > 1.0f) {
				src_r[j] = r_mask;
				src_g[j] = g_mask;
				src_b[j] = b_mask;

				dst_r[j] = r;
				dst_g[j] = g;
				dst_b[j] = b;
			} else {
				dst_r[j] = 0.0f;
				dst_g[j] = 0.0f;
				dst_b[j] = 0.0f;
			}
		}
	}
}

void execute(const Arguments &args)
{
	ImageBuffer in = read_from_file(args.inpath, args.width, args.height, 1, 1, sizeof(uint16_t));
	ImageBuffer linear = allocate_buffer(args.width, args.height, 0, 0, sizeof(float));
	ImageBuffer linear_mask = allocate_buffer(args.width, args.height, 0, 0, sizeof(float));
	ImageBuffer sdr = allocate_buffer(args.width, args.height, 0, 0, sizeof(uint8_t));
	ImageBuffer mask = allocate_buffer(args.width, args.height, 0, 0, sizeof(uint16_t));

	// If allow_approximate_gamma is set, out-of-range pixels may be clipped,
	// which could interfere with further processing of image highlights.
	zimgxx::zfilter_graph_builder_params params;
	params.nominal_peak_luminance = args.luminance;
	params.allow_approximate_gamma = !!args.fast;

	// HDR10 specification.
	zimgxx::zimage_format src_format;
	src_format.width = args.width;
	src_format.height = args.height;
	src_format.pixel_type = ZIMG_PIXEL_WORD;
	src_format.subsample_w = 1;
	src_format.subsample_h = 1;
	src_format.color_family = ZIMG_COLOR_YUV;
	src_format.matrix_coefficients = ZIMG_MATRIX_BT2020_NCL;
	src_format.transfer_characteristics = ZIMG_TRANSFER_ST2084;
	src_format.color_primaries = ZIMG_PRIMARIES_BT2020;
	src_format.depth = 10;
	src_format.pixel_range = ZIMG_RANGE_LIMITED;

	// Linear Rec.709 RGB corresponding to above.
	zimgxx::zimage_format linear_format;
	linear_format.width = args.width;
	linear_format.height = args.height;
	linear_format.pixel_type = ZIMG_PIXEL_FLOAT;
	linear_format.color_family = ZIMG_COLOR_RGB;
	linear_format.matrix_coefficients = ZIMG_MATRIX_RGB;
	linear_format.transfer_characteristics = ZIMG_TRANSFER_LINEAR;
	linear_format.color_primaries = ZIMG_PRIMARIES_BT709;

	// HDR10 RGB corresponding to above.
	zimgxx::zimage_format rgb_format;
	rgb_format.width = args.width;
	rgb_format.height = args.height;
	rgb_format.pixel_type = ZIMG_PIXEL_WORD;
	rgb_format.color_family = ZIMG_COLOR_RGB;
	rgb_format.matrix_coefficients = ZIMG_MATRIX_RGB;
	rgb_format.transfer_characteristics = ZIMG_TRANSFER_ST2084;
	rgb_format.color_primaries = ZIMG_PRIMARIES_BT2020;
	rgb_format.depth = 10;
	rgb_format.pixel_range = ZIMG_RANGE_FULL;

	// Rec.709 RGB corresponding to above.
	zimgxx::zimage_format sdr_format;
	sdr_format.width = args.width;
	sdr_format.height = args.height;
	sdr_format.pixel_type = ZIMG_PIXEL_BYTE;
	sdr_format.color_family = ZIMG_COLOR_RGB;
	sdr_format.matrix_coefficients = ZIMG_MATRIX_RGB;
	sdr_format.transfer_characteristics = ZIMG_TRANSFER_BT709;
	sdr_format.color_primaries = ZIMG_PRIMARIES_BT709;
	sdr_format.depth = 8;
	sdr_format.pixel_range = ZIMG_RANGE_FULL;

	zimgxx::FilterGraph tolinear_graph{ zimgxx::FilterGraph::build(src_format, linear_format, &params) };
	zimgxx::FilterGraph tohdr_graph{ zimgxx::FilterGraph::build(linear_format, rgb_format, &params) };
	zimgxx::FilterGraph tosdr_graph{ zimgxx::FilterGraph::build(linear_format, sdr_format, &params) };

	size_t tmp_size = std::max({ tolinear_graph.get_tmp_size(), tohdr_graph.get_tmp_size(), tosdr_graph.get_tmp_size() });
	std::shared_ptr<void> tmp_buf{ aligned_malloc(tmp_size, 32), aligned_free };

	// Convert from HDR10 to linear Rec.709.
	tolinear_graph.process(in.buffer.as_const(), linear.buffer, tmp_buf.get());

	// Search for out of range pixels and replace with color key.
	mask_pixels(linear.buffer, linear_mask.buffer, args.width, args.height, args.mask_key);

	// Convert linear image to Rec.709 for export.
	tosdr_graph.process(linear.buffer.as_const(), sdr.buffer, tmp_buf.get());

	// Convert mask image to HDR10 RGB for export.
	tohdr_graph.process(linear_mask.buffer.as_const(), mask.buffer, tmp_buf.get());

	write_to_bmp(sdr, args.sdrpath, args.width, args.height);
	if (args.hdrpath)
		write_to_file(mask, args.hdrpath, args.width, args.height, 0, 0, sizeof(uint16_t));
}

} // namespace


int main(int argc, char **argv)
{
	Arguments args{};
	int ret;

	args.luminance = NAN;

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
