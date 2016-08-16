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

namespace {

struct Arguments {
	const char *inpath;
	const char *outpath;
	unsigned out_w;
	unsigned out_h;
	int premultiply;
};

const ArgparseOption program_switches[] = {
	{ OPTION_BOOL, nullptr, "premultiply", offsetof(Arguments, premultiply), nullptr, "color channels premultiplied by alpha" },
};

const ArgparseOption program_positional[] = {
	{ OPTION_STRING,   nullptr, "inpath",     offsetof(Arguments, inpath),  nullptr, "input path specifier" },
	{ OPTION_STRING,   nullptr, "outpath",    offsetof(Arguments, outpath), nullptr, "output path specifier" },
	{ OPTION_UINTEGER, nullptr, "out_width",  offsetof(Arguments, out_w),   nullptr, "output width" },
	{ OPTION_UINTEGER, nullptr, "out_height", offsetof(Arguments, out_h),   nullptr, "output height" }
};

const ArgparseCommandLine program_def = {
	program_switches,
	sizeof(program_switches) / sizeof(program_switches[0]),
	program_positional,
	sizeof(program_positional) / sizeof(program_positional[0]),
	"alpha_example",
	"resize BGRA bitmaps",
	nullptr
};


struct Callback {
	WindowsBitmap *bmp;
	zimgxx::zimage_buffer *rgb_buf;
	zimgxx::zimage_buffer *alpha_buf;
	bool premultiply;
};

zimgxx::zimage_format get_image_format(const WindowsBitmap *bmp, bool premultiply)
{
	zimgxx::zimage_format format;

	format.width = bmp->width();
	format.height = bmp->height();
	format.pixel_type = premultiply ? ZIMG_PIXEL_BYTE : ZIMG_PIXEL_WORD;

	format.color_family = ZIMG_COLOR_RGB;
	format.pixel_range = ZIMG_RANGE_FULL;

	return format;
}

zimgxx::zimage_format get_alpha_format(const WindowsBitmap *bmp, bool premultiply)
{
	zimgxx::zimage_format format;

	format.width = bmp->width();
	format.height = bmp->height();
	format.pixel_type = ZIMG_PIXEL_BYTE;

	format.color_family = ZIMG_COLOR_GREY;
	format.pixel_range = ZIMG_RANGE_FULL;

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

void unpack_bgra_straight(const void *bgra, void * const planar[4], unsigned left, unsigned right)
{
	const uint8_t *packed_bgra = static_cast<const uint8_t *>(bgra);
	uint16_t *planar_r = static_cast<uint16_t *>(planar[0]);
	uint16_t *planar_g = static_cast<uint16_t *>(planar[1]);
	uint16_t *planar_b = static_cast<uint16_t *>(planar[2]);
	uint8_t *planar_a = static_cast<uint8_t *>(planar[3]);

	for (unsigned j = left; j < right; ++j) {
		uint16_t r, g, b;
		uint8_t a;

		r = packed_bgra[j * 4 + 2];
		g = packed_bgra[j * 4 + 1];
		b = packed_bgra[j * 4 + 0];
		a = packed_bgra[j * 4 + 3];

		r = static_cast<uint16_t>(static_cast<uint32_t>(r) * a * 65535 / (255 * 255));
		g = static_cast<uint16_t>(static_cast<uint32_t>(g) * a * 65535 / (255 * 255));
		b = static_cast<uint16_t>(static_cast<uint32_t>(b) * a * 65535 / (255 * 255));

		planar_r[j] = r;
		planar_g[j] = g;
		planar_b[j] = b;
		planar_a[j] = a;
	}
}

void unpack_bgra_premul(const void *bgra, void * const planar[4], unsigned left, unsigned right)
{
	const uint8_t *packed_bgra = static_cast<const uint8_t *>(bgra);
	uint8_t *planar_r = static_cast<uint8_t *>(planar[0]);
	uint8_t *planar_g = static_cast<uint8_t *>(planar[1]);
	uint8_t *planar_b = static_cast<uint8_t *>(planar[2]);
	uint8_t *planar_a = static_cast<uint8_t *>(planar[3]);

	for (unsigned j = left; j < right; ++j) {
		uint8_t r, g, b, a;

		r = packed_bgra[j * 4 + 2];
		g = packed_bgra[j * 4 + 1];
		b = packed_bgra[j * 4 + 0];
		a = packed_bgra[j * 4 + 3];

		planar_r[j] = r;
		planar_g[j] = g;
		planar_b[j] = b;
		planar_a[j] = a;
	}
}

void pack_bgra_straight(const void * const planar[4], void *bgra, unsigned left, unsigned right)
{
	const uint16_t *planar_r = static_cast<const uint16_t *>(planar[0]);
	const uint16_t *planar_g = static_cast<const uint16_t *>(planar[1]);
	const uint16_t *planar_b = static_cast<const uint16_t *>(planar[2]);
	const uint8_t *planar_a = static_cast<const uint8_t *>(planar[3]);
	uint8_t *packed_bgra = static_cast<uint8_t *>(bgra);

	for (unsigned j = left; j < right; ++j) {
		uint16_t r, g, b;
		uint8_t a, a_eff;

		r = planar_r[j];
		g = planar_g[j];
		b = planar_b[j];
		a = planar_a[j];

		a_eff = std::max(a, static_cast<uint8_t>(1));

		r = static_cast<uint16_t>(static_cast<uint32_t>(r) * 255 * 255 / (static_cast<uint32_t>(65535) * a_eff));
		g = static_cast<uint16_t>(static_cast<uint32_t>(g) * 255 * 255 / (static_cast<uint32_t>(65535) * a_eff));
		b = static_cast<uint16_t>(static_cast<uint32_t>(b) * 255 * 255 / (static_cast<uint32_t>(65535) * a_eff));

		packed_bgra[j * 4 + 0] = static_cast<uint8_t>(b);
		packed_bgra[j * 4 + 1] = static_cast<uint8_t>(g);
		packed_bgra[j * 4 + 2] = static_cast<uint8_t>(r);
		packed_bgra[j * 4 + 3] = a;
	}
}

void pack_bgra_premul(const void * const planar[4], void *bgra, unsigned left, unsigned right)
{
	const uint8_t *planar_r = static_cast<const uint8_t *>(planar[0]);
	const uint8_t *planar_g = static_cast<const uint8_t *>(planar[1]);
	const uint8_t *planar_b = static_cast<const uint8_t *>(planar[2]);
	const uint8_t *planar_a = static_cast<const uint8_t *>(planar[3]);
	uint8_t *packed_bgra = static_cast<uint8_t *>(bgra);

	for (unsigned j = left; j < right; ++j) {
		uint8_t r, g, b, a;

		r = planar_r[j];
		g = planar_g[j];
		b = planar_b[j];
		a = planar_a[j];

		packed_bgra[j * 4 + 0] = b;
		packed_bgra[j * 4 + 1] = g;
		packed_bgra[j * 4 + 2] = r;
		packed_bgra[j * 4 + 3] = a;
	}
}

int unpack_bgra(void *user, unsigned i, unsigned left, unsigned right)
{
	const Callback *cb = static_cast<Callback *>(user);
	const zimgxx::zimage_buffer &rgb_buf = *cb->rgb_buf;
	const zimgxx::zimage_buffer &alpha_buf = *cb->alpha_buf;
	const void *packed_data = cb->bmp->read_ptr() + i * cb->bmp->stride();
	void *planar_data[4];

	for (unsigned p = 0; p < 3; ++p) {
		planar_data[p] = static_cast<char *>(rgb_buf.line_at(i, p));
	}
	planar_data[3] = static_cast<char *>(alpha_buf.line_at(i));

	if (cb->premultiply)
		unpack_bgra_premul(packed_data, planar_data, left, right);
	else
		unpack_bgra_straight(packed_data, planar_data, left, right);

	return 0;
}

int pack_bgra(void *user, unsigned i, unsigned left, unsigned right)
{
	const Callback * cb = static_cast<Callback *>(user);
	const zimgxx::zimage_buffer &rgb_buf = *cb->rgb_buf;
	const zimgxx::zimage_buffer &alpha_buf = *cb->alpha_buf;
	void *packed_data = cb->bmp->write_ptr() + i * cb->bmp->stride();
	const void *planar_data[4];

	for (unsigned p = 0; p < 3; ++p) {
		planar_data[p] = static_cast<const char *>(rgb_buf.line_at(i, p));
	}
	planar_data[3] = static_cast<const char *>(alpha_buf.line_at(i));

	if (cb->premultiply)
		pack_bgra_premul(planar_data, packed_data, left, right);
	else
		pack_bgra_straight(planar_data, packed_data, left, right);

	return 0;
}

void process(const WindowsBitmap *in_bmp, WindowsBitmap *out_bmp, bool premultiply)
{
	zimgxx::zimage_format in_format = get_image_format(in_bmp, premultiply);
	zimgxx::zimage_format out_format = get_image_format(out_bmp, premultiply);
	zimgxx::zimage_format in_format_alpha = get_alpha_format(in_bmp, premultiply);
	zimgxx::zimage_format out_format_alpha = get_alpha_format(out_bmp, premultiply);

	zimgxx::FilterGraph graph{ zimgxx::FilterGraph::build(in_format, out_format) };
	zimgxx::FilterGraph graph_alpha{ zimgxx::FilterGraph::build(in_format_alpha, out_format_alpha) };

	unsigned input_buffering = std::max(graph.get_input_buffering(), graph_alpha.get_input_buffering());
	unsigned output_buffering = std::max(graph.get_output_buffering(), graph_alpha.get_output_buffering());
	size_t tmp_size = std::max(graph.get_tmp_size(), graph_alpha.get_tmp_size());

	auto in_rgb_buf = allocate_buffer(in_format, input_buffering);
	auto in_alpha_plane_buf = allocate_buffer(in_format_alpha, ZIMG_BUFFER_MAX);

	auto out_rgb_plane_buf = allocate_buffer(out_format, ZIMG_BUFFER_MAX);
	auto out_alpha_buf = allocate_buffer(out_format_alpha, output_buffering);

	auto tmp = allocate_buffer(tmp_size);

	Callback unpack_cb_data{ const_cast<WindowsBitmap *>(in_bmp), &in_rgb_buf.first, &in_alpha_plane_buf.first, premultiply };
	Callback pack_cb_data{ out_bmp, &out_rgb_plane_buf.first, &out_alpha_buf.first, premultiply };

	graph.process(in_rgb_buf.first.as_const(), out_rgb_plane_buf.first, tmp.get(), unpack_bgra, &unpack_cb_data, nullptr, nullptr);
	graph_alpha.process(in_alpha_plane_buf.first.as_const(), out_alpha_buf.first, tmp.get(), nullptr, nullptr, pack_bgra, &pack_cb_data);
}

void execute(const Arguments &args)
{
	std::shared_ptr<WindowsBitmap> in_bmp;
	std::shared_ptr<WindowsBitmap> out_bmp;

	in_bmp = std::make_shared<WindowsBitmap>(args.inpath, WindowsBitmap::READ_TAG);

	if (in_bmp->bit_count() != 32)
		throw std::runtime_error{ "no alpha component in bitmap" };

	out_bmp = std::make_shared<WindowsBitmap>(args.outpath, args.out_w, args.out_h, 32);
	process(in_bmp.get(), out_bmp.get(), !!args.premultiply);
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
