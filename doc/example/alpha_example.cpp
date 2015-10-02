#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <system_error>
#include <utility>

#include <zimg++.hpp>

#include "aligned_malloc.h"
#include "argparse.h"
#include "win32_bitmap.h"

namespace {;

struct Arguments {
	const char *inpath;
	const char *outpath;
	unsigned out_w;
	unsigned out_h;
};

const ArgparseOption program_positional[] = {
	{ OPTION_STRING,   nullptr, "inpath",     offsetof(struct Arguments, inpath),  nullptr, "input path specifier" },
	{ OPTION_STRING,   nullptr, "outpath",    offsetof(struct Arguments, outpath), nullptr, "output path specifier" },
	{ OPTION_UINTEGER, nullptr, "out_width",  offsetof(struct Arguments, out_w),   nullptr, "output width" },
	{ OPTION_UINTEGER, nullptr, "out_height", offsetof(struct Arguments, out_h),   nullptr, "output height" }
};

const ArgparseCommandLine program_def = {
	nullptr,
	0,
	program_positional,
	sizeof(program_positional) / sizeof(program_positional[0]),
	"alpha_example",
	"resize BGRA bitmaps",
	nullptr
};


struct Callback {
	const zimgxx::zimage_buffer *buffer;
	const zimgxx::zimage_buffer *alpha_buffer;
	WindowsBitmap *bmp;
};


std::pair<zimgxx::zimage_buffer, std::shared_ptr<void>> allocate_buffer(unsigned width, unsigned height, unsigned count)
{
	zimgxx::zimage_buffer buffer;
	std::shared_ptr<void> handle;

	unsigned mask = zimg2_select_buffer_mask(count);
	ptrdiff_t stride = width % 64 ? width - width % 64 + 64 : width;
	size_t plane_size = (size_t)stride * ((mask == (unsigned)-1) ? height : mask + 1);

	handle.reset(aligned_malloc(3 * plane_size, 64), aligned_free);

	for (unsigned p = 0; p < 3; ++p) {
		buffer._.m.data[p] = (char *)handle.get() + p * plane_size;
		buffer._.m.stride[p] = stride;
		buffer._.m.mask[p] = mask;
	}

	return{ buffer, handle };
}

std::shared_ptr<void> allocate_buffer(size_t size)
{
	return{ aligned_malloc(size, 64), &aligned_free };
}

std::pair<zimgxx::zimage_buffer, std::shared_ptr<void>> allocate_plane(unsigned width, unsigned height)
{
	ptrdiff_t stride = width % 64 ? width - width % 64 + 64 : width;

	std::shared_ptr<void> handle{ aligned_malloc((size_t)stride * height, 64), aligned_free };
	zimgxx::zimage_buffer buffer{};

	buffer._.m.data[0] = handle.get();
	buffer._.m.stride[0] = stride;
	buffer._.m.mask[0] = -1;

	return{ buffer, handle };
}

void *buffer_at_line(const zimgxx::zimage_buffer &buffer, unsigned p, unsigned i)
{
	void *base = buffer._.m.data[p];
	ptrdiff_t stride = buffer._.m.stride[p];
	unsigned mask = buffer._.m.mask[p];

	return reinterpret_cast<char *>(base) + (ptrdiff_t)(i & mask) * stride;
}

int unpack_bgra(void *user, unsigned i, unsigned left, unsigned right)
{
	const Callback *callback = reinterpret_cast<Callback *>(user);
	const uint8_t *bgra = reinterpret_cast<const uint8_t *>(callback->bmp->read_ptr() + (ptrdiff_t)i * callback->bmp->stride());
	uint8_t *planes[4];

	for (unsigned p = 0; p < 3; ++p) {
		planes[p] = reinterpret_cast<uint8_t *>(buffer_at_line(*callback->buffer, p, i));
	}
	planes[3] = reinterpret_cast<uint8_t *>(buffer_at_line(*callback->alpha_buffer, 0, i));

	for (unsigned j = left; j < right; ++j) {
		uint8_t r, g, b, a;

		r = bgra[j * 4 + 2];
		g = bgra[j * 4 + 1];
		b = bgra[j * 4 + 0];
		a = bgra[j * 4 + 3];

		planes[0][j] = r;
		planes[1][j] = g;
		planes[2][j] = b;
		planes[3][j] = a;
	}

	return 0;
}

int pack_bgr(void *user, unsigned i, unsigned left, unsigned right)
{
	const Callback *callback = reinterpret_cast<Callback *>(user);
	uint8_t *bgra = reinterpret_cast<uint8_t *>(callback->bmp->write_ptr() + (ptrdiff_t)i * callback->bmp->stride());
	const uint8_t *planes[3];

	for (unsigned p = 0; p < 3; ++p) {
		planes[p] = reinterpret_cast<const uint8_t *>(buffer_at_line(*callback->buffer, p, i));
	}

	for (unsigned j = left; j < right; ++j) {
		uint8_t r, g, b;

		r = planes[0][j];
		g = planes[1][j];
		b = planes[2][j];

		bgra[j * 4 + 0] = b;
		bgra[j * 4 + 1] = g;
		bgra[j * 4 + 2] = r;
	}

	return 0;
}

int pack_alpha(void *user, unsigned i, unsigned left, unsigned right)
{
	const Callback *callback = reinterpret_cast<Callback *>(user);
	uint8_t *bgra = reinterpret_cast<uint8_t *>(callback->bmp->write_ptr() + (ptrdiff_t)i * callback->bmp->stride());
	const uint8_t *alpha = reinterpret_cast<const uint8_t *>(buffer_at_line(*callback->buffer, 0, i));

	for (unsigned j = left; j < right; ++j) {
		bgra[j * 4 + 3] = alpha[j];
	}

	return 0;
}

void process(const WindowsBitmap *in_bmp, WindowsBitmap *out_bmp)
{
	zimgxx::zimage_format in_format{};
	zimgxx::zimage_format out_format{};
	zimgxx::zimage_format in_format_alpha{};
	zimgxx::zimage_format out_format_alpha{};

	in_format.width = in_bmp->width();
	in_format.height = in_bmp->height();
	in_format.pixel_type = ZIMG_PIXEL_BYTE;
	in_format.color_family = ZIMG_COLOR_RGB;

	out_format.width = out_bmp->width();
	out_format.height = out_bmp->height();
	out_format.pixel_type = ZIMG_PIXEL_BYTE;
	out_format.color_family = ZIMG_COLOR_RGB;

	in_format_alpha.width = in_bmp->width();
	in_format_alpha.height = in_bmp->height();
	in_format_alpha.pixel_type = ZIMG_PIXEL_BYTE;
	in_format_alpha.color_family = ZIMG_COLOR_GREY;

	out_format_alpha.width = out_bmp->width();
	out_format_alpha.height = out_bmp->height();
	out_format_alpha.pixel_type = ZIMG_PIXEL_BYTE;
	out_format_alpha.color_family = ZIMG_COLOR_GREY;

	zimgxx::FilterGraph graph{ zimgxx::FilterGraph::build(&in_format, &out_format, nullptr) };
	zimgxx::FilterGraph graph_alpha{ zimgxx::FilterGraph::build(&in_format_alpha, &out_format_alpha, nullptr) };

	unsigned input_buffering = std::max(graph.get_input_buffering(), graph_alpha.get_input_buffering());
	unsigned output_buffering = std::max(graph.get_output_buffering(), graph_alpha.get_output_buffering());
	size_t tmp_size = std::max(graph.get_tmp_size(), graph_alpha.get_tmp_size());

	auto alpha = allocate_plane(in_bmp->width(), in_bmp->height());
	auto in_buf = allocate_buffer(in_bmp->width(), in_bmp->height(), input_buffering);
	auto out_buf = allocate_buffer(out_bmp->width(), out_bmp->height(), output_buffering);
	auto tmp_buf = allocate_buffer(tmp_size);

	Callback unpack_cb_data = { &in_buf.first, &alpha.first, const_cast<WindowsBitmap *>(in_bmp) };
	Callback pack_cb_data = { &out_buf.first, nullptr, out_bmp };

	graph.process(&in_buf.first.as_const(), &out_buf.first._, tmp_buf.get(),
	              unpack_bgra, &unpack_cb_data, pack_bgr, &pack_cb_data);
	graph_alpha.process(&alpha.first.as_const(), &out_buf.first._, tmp_buf.get(),
	                    nullptr, nullptr, pack_alpha, &pack_cb_data);

	alpha.second.reset();
	in_buf.second.reset();
	out_buf.second.reset();
	tmp_buf.reset();
}

void execute(const Arguments &args)
{
	std::shared_ptr<WindowsBitmap> in_bmp;
	std::shared_ptr<WindowsBitmap> out_bmp;

	in_bmp = std::make_shared<WindowsBitmap>(args.inpath, WindowsBitmap::READ_TAG);

	if (in_bmp->bit_count() != 32)
		throw std::runtime_error{ "no alpha component in bitmap" };

	out_bmp = std::make_shared<WindowsBitmap>(args.outpath, args.out_w, args.out_h, 32);
	process(in_bmp.get(), out_bmp.get());
}

} // namespace


int main(int argc, char **argv)
{
	Arguments args{};
	int ret;

	if ((ret = argparse_parse(&program_def, &args, argc, argv)))
		return ret == ARGPARSE_HELP ? 0 : ret;

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
