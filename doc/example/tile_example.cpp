// z.lib example code for tile-based threading.
//
// Example code demonstrates the use of z.lib to scale a single image by
// dividing the output into tiles. For processing multiple images, it is
// recommended to use frame-based threading for higher efficiency.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <vector>

#include <zimg++.hpp>

#include "aligned_malloc.h"
#include "argparse.h"
#include "win32_bitmap.h"

#if ZIMG_API_VERSION < ZIMG_MAKE_API_VERSION(2, 1)
  #error API 2.1 required
#endif

namespace {

struct Arguments {
	const char *inpath;
	const char *outpath;
	unsigned out_w;
	unsigned out_h;
	unsigned tile_width;
	unsigned tile_height;
	unsigned threads;
	char interactive;
	char opt;
};

const ArgparseOption program_switches[] = {
	{ OPTION_UINT, nullptr, "tile-width",  offsetof(Arguments, tile_width),  nullptr, "tile width" },
	{ OPTION_UINT, nullptr, "tile-height", offsetof(Arguments, tile_height), nullptr, "tile height" },
	{ OPTION_UINT, nullptr, "threads",     offsetof(Arguments, threads),     nullptr, "number of threads" },
	{ OPTION_FLAG, "i",     "interactive", offsetof(Arguments, interactive), nullptr, "interactive mode" },
	{ OPTION_NULL },
};

const ArgparseOption program_positional[] = {
	{ OPTION_STRING, nullptr, "inpath",  offsetof(Arguments, inpath),  nullptr, "input path" },
	{ OPTION_STRING, nullptr, "outpath", offsetof(Arguments, outpath), nullptr, "output path" },
	{ OPTION_UINT,   "w",     "width",   offsetof(Arguments, out_w),   nullptr, "width" },
	{ OPTION_UINT,   "h",     "height",  offsetof(Arguments, out_h),   nullptr, "height" },
	{ OPTION_NULL },
};

const ArgparseCommandLine program_def = { program_switches, program_positional, "tile_example", "resize BMP images with tile-based threading" };


struct TileTask {
	zimgxx::zimage_format src_format;
	zimgxx::zimage_format dst_format;
	unsigned tile_left;
	unsigned tile_top;
};

struct Callback {
	const WindowsBitmap *in_bmp;
	WindowsBitmap *out_bmp;
	const TileTask *task;
	const zimgxx::zimage_buffer *src_buf;
	const zimgxx::zimage_buffer *dst_buf;
};

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

int unpack_bmp(void *user, unsigned i, unsigned left, unsigned right)
{
	Callback *cb = static_cast<Callback *>(user);

	// Pixel indices in the input image are relative to the whole image.
	const uint8_t *packed_bgr = cb->in_bmp->read_ptr() + static_cast<ptrdiff_t>(i) * cb->in_bmp->stride();
	uint8_t *planar_r = static_cast<uint8_t *>(cb->src_buf->line_at(i, 0));
	uint8_t *planar_g = static_cast<uint8_t *>(cb->src_buf->line_at(i, 1));
	uint8_t *planar_b = static_cast<uint8_t *>(cb->src_buf->line_at(i, 2));
	unsigned step = cb->in_bmp->bit_count() / 8;

	for (unsigned j = left; j < right; ++j) {
		uint8_t r, g, b;

		b = packed_bgr[j * step + 0];
		g = packed_bgr[j * step + 1];
		r = packed_bgr[j * step + 2];

		planar_r[j] = r;
		planar_g[j] = g;
		planar_b[j] = b;
	}

	return 0;
}

int pack_bmp(void *user, unsigned i, unsigned left, unsigned right)
{
	Callback *cb = static_cast<Callback *>(user);

	// Pixel indices in the output image are relative to the tile.
	const uint8_t *planar_r = static_cast<const uint8_t *>(cb->dst_buf->line_at(i, 0));
	const uint8_t *planar_g = static_cast<const uint8_t *>(cb->dst_buf->line_at(i, 1));
	const uint8_t *planar_b = static_cast<const uint8_t *>(cb->dst_buf->line_at(i, 2));
	uint8_t *packed_bgr = cb->out_bmp->write_ptr() + static_cast<ptrdiff_t>(i + cb->task->tile_top) * cb->out_bmp->stride() + (cb->task->tile_left * cb->out_bmp->bit_count() / 8);
	unsigned step = cb->out_bmp->bit_count() / 8;

	for (unsigned j = left; j < right; ++j) {
		uint8_t r, g, b;

		r = planar_r[j];
		g = planar_g[j];
		b = planar_b[j];

		packed_bgr[j * step + 0] = b;
		packed_bgr[j * step + 1] = g;
		packed_bgr[j * step + 2] = r;
	}

	return 0;
}

void thread_func(const WindowsBitmap *in_bmp, WindowsBitmap *out_bmp, std::vector<TileTask> *tasks, std::mutex *mutex, std::exception_ptr *eptr, bool interactive);

void execute(const Arguments &args)
{
	WindowsBitmap in_bmp{ args.inpath, WindowsBitmap::READ_TAG };
	WindowsBitmap out_bmp{ args.outpath, static_cast<int>(args.out_w), static_cast<int>(args.out_h), 24 };

	// (1) Fill the common fields in the format descriptors for the input and output files.
	zimgxx::zimage_format in_format;
	zimgxx::zimage_format out_format;

	in_format.width = in_bmp.width();
	in_format.height = in_bmp.height();
	in_format.pixel_type = ZIMG_PIXEL_BYTE;
	in_format.color_family = ZIMG_COLOR_RGB;
	in_format.pixel_range = ZIMG_RANGE_FULL;

	out_format.pixel_type = ZIMG_PIXEL_BYTE;
	out_format.color_family = ZIMG_COLOR_RGB;
	out_format.pixel_range = ZIMG_RANGE_FULL;

	std::vector<TileTask> task_queue;

	// (2) Calculate the bounds of the input regions from the output regions. For
	// each tile, a graph creates an image of tile_width x tile_height from a
	// subset of the input image. Note that the input tile is specified through
	// the active_region field, unlike the output tile.
	double scale_w = static_cast<double>(in_bmp.width()) / args.out_w;
	double scale_h = static_cast<double>(in_bmp.height()) / args.out_h;

	// The destination buffer passed to zimg_filter_graph_process will point to
	// the upper-left corner of the tile. As a result, when not using a pack
	// callback, incrementing the output image by tile_width pixels must
	// maintain alignment.
	if (args.tile_width % 32)
		std::cout << "warning: tile width results in unaligned image\n";

	for (unsigned i = 0; i < args.out_h; i += args.tile_height) {
		for (unsigned j = 0; j < args.out_w; j += args.tile_width) {
			zimgxx::zimage_format tile_in_format = in_format;
			zimgxx::zimage_format tile_out_format = out_format;

			unsigned tile_right = out_bmp.width() - j >= args.tile_width ? j + args.tile_width : out_bmp.width();
			unsigned tile_bottom = out_bmp.height() - i >= args.tile_height ? i + args.tile_height : out_bmp.height();

			tile_in_format.active_region.left = j * scale_w;
			tile_in_format.active_region.top = i * scale_h;
			tile_in_format.active_region.width = (tile_right - j) * scale_w;
			tile_in_format.active_region.height = (tile_bottom - i) * scale_h;

			tile_out_format.width = tile_right - j;
			tile_out_format.height = tile_bottom - i;

			task_queue.push_back({ tile_in_format, tile_out_format, j, i });
		}
	}

	// (3) Distribute the tiles across threads. Note that the calls to
	// zimg_filter_graph_create must also be parallelized for maximum effect.
	std::vector<std::thread> threads;
	unsigned num_threads = args.interactive ? 1 : (args.threads ? args.threads : std::thread::hardware_concurrency());
	std::exception_ptr eptr;
	std::mutex mutex;

	// Process tiles in raster order.
	std::reverse(task_queue.begin(), task_queue.end());

	threads.reserve(num_threads);
	for (unsigned i = 0; i < num_threads; ++i) {
		threads.emplace_back(thread_func, &in_bmp, &out_bmp, &task_queue, &mutex, &eptr, !!args.interactive);
	}

	for (std::thread &th : threads) {
		th.join();
	}

	if (eptr)
		std::rethrow_exception(eptr);
}

void thread_func(const WindowsBitmap *in_bmp, WindowsBitmap *out_bmp, std::vector<TileTask> *tasks, std::mutex *mutex, std::exception_ptr *eptr, bool interactive)
{
	try {
		while (true) {
			std::unique_lock<std::mutex> lock{ *mutex };
			if (tasks->empty())
				break;

			TileTask task = tasks->back();
			tasks->pop_back();
			lock.unlock();

			// (4) Build the processing context for the tile.
			zimgxx::FilterGraph graph{ zimgxx::FilterGraph::build(task.src_format, task.dst_format) };
			unsigned input_buffering = graph.get_input_buffering();
			unsigned output_buffering = graph.get_output_buffering();
			size_t tmp_size = graph.get_tmp_size();

			if (input_buffering == ZIMG_BUFFER_MAX || output_buffering == ZIMG_BUFFER_MAX)
				throw std::logic_error{ "graph can not be processed with tiles" };

			// (5) Allocate scanline buffers for the input and output data.
			auto src_buf = allocate_buffer(task.src_format, input_buffering);
			auto dst_buf = allocate_buffer(task.dst_format, output_buffering);
			auto tmp = allocate_buffer(tmp_size);

			// (6) Process the tile.
			Callback cb{ in_bmp, out_bmp, &task, &src_buf.first, &dst_buf.first };
			graph.process(src_buf.first.as_const(), dst_buf.first, tmp.get(), unpack_bmp, &cb, pack_bmp, &cb);

			if (interactive) {
				out_bmp->flush();
				std::cout << "Press enter to continue...";
				std::cin.get();
			}
		}
	} catch (...) {
		std::lock_guard<std::mutex> lock{ *mutex };
		*eptr = std::current_exception();
	}
}

} // namespace


int main(int argc, char **argv)
{
	Arguments args{};
	int ret;

	args.tile_width = 512;
	args.tile_height = 512;

	if ((ret = argparse_parse(&program_def, &args, argc, argv)) < 0)
		return ret == ARGPARSE_HELP_MESSAGE ? 0 : ret;

	if (zimg_get_api_version(nullptr, nullptr) < ZIMG_MAKE_API_VERSION(2, 1)) {
		std::cerr << "error: subpixel operation requires API 2.1\n";
		return 2;
	}

	// Prior to z.lib 2.9, using horizontal tiling results in redundant loading
	// of scanlines above the first row in the tile.
	if (args.tile_height < args.out_h) {
		unsigned version[3];
		zimg_get_version_info(version, version + 1, version + 2);

		if (version[0] < 2 || (version[0] == 2 && version[1] < 9))
			std::cerr << "warning: horizontal tiling may be slow in z.lib versions prior to 2.9\n";
	}

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
