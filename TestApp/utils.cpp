#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include "Common/align.h"
#include "Common/alloc.h"
#include "Common/cpuinfo.h"
#include "Common/linebuffer.h"
#include "Common/pixel.h"
#include "Common/plane.h"
#include "Common/zfilter.h"
#include "Depth/depth.h"
#include "frame.h"
#include "utils.h"
#include "timer.h"

using namespace zimg;

namespace {;

int required_args(OptionType type)
{
	if (type == OptionType::OPTION_SPECIAL || type == OptionType::OPTION_TRUE || type == OptionType::OPTION_FALSE)
		return 1;
	else
		return 2;
}

} // namespace


zimg::CPUClass select_cpu(const char *cpu)
{
#ifdef ZIMG_X86
	if (!strcmp(cpu, "auto"))
		return CPUClass::CPU_X86_AUTO;
	else if (!strcmp(cpu, "sse2"))
		return CPUClass::CPU_X86_SSE2;
	else if (!strcmp(cpu, "avx2"))
		return CPUClass::CPU_X86_AVX2;
	else
		return CPUClass::CPU_NONE;
#else
	return CPUClass::CPU_NONE;
#endif // ZIMG_X86
}

zimg::PixelType select_pixel_type(const char *pixtype)
{
	if (!strcmp(pixtype, "u8"))
		return PixelType::BYTE;
	else if (!strcmp(pixtype, "u16"))
		return PixelType::WORD;
	else if (!strcmp(pixtype, "f16"))
		return PixelType::HALF;
	else if (!strcmp(pixtype, "f32"))
		return PixelType::FLOAT;
	else
		throw std::invalid_argument{ "unknown pixel type" };
}

zimg::AlignedVector<char> allocate_buffer(size_t count, zimg::PixelType type)
{
	return AlignedVector<char>(count * pixel_size(type));
}

void convert_frame(const Frame &in, Frame &out, zimg::PixelType pxl_in, zimg::PixelType pxl_out, bool fullrange, bool yuv)
{
	std::unique_ptr<depth::Depth> convert{ new depth::Depth{ depth::DitherType::DITHER_NONE, CPUClass::CPU_NONE } };
	int width = in.width();
	int height = in.height();
	int planes = in.planes();

	for (int p = 0; p < planes; ++p) {
		bool plane_fullrange = fullrange || p == 3; // Always treat alpha as fullrange.
		bool plane_chroma = yuv && (p == 1 || p == 2); // Chroma planes.

		PixelFormat src_fmt = default_pixel_format(pxl_in);
		PixelFormat dst_fmt = default_pixel_format(pxl_out);

		src_fmt.fullrange = plane_fullrange;
		src_fmt.chroma = plane_chroma;
		dst_fmt.fullrange = plane_fullrange;
		dst_fmt.chroma = plane_chroma;

		ImagePlane<const void> src_plane{ in.data(p), width, height, in.stride(), src_fmt };
		ImagePlane<void> dst_plane{ out.data(p), width, height, out.stride(), dst_fmt };

		convert->process(src_plane, dst_plane, nullptr);
	}
}

zimg::AlignedVector<char> alloc_filter_tmp(const zimg::IZimgFilter &filter, const Frame &in, Frame &out)
{
	zimg_filter_flags flags = filter.get_flags();
	FakeAllocator alloc;

	unsigned input_lines = filter.get_max_buffering();
	unsigned output_lines = filter.get_simultaneous_lines();
	unsigned input_mask = select_zimg_buffer_mask(input_lines);
	unsigned output_mask = select_zimg_buffer_mask(output_lines);
	unsigned num_planes = flags.color ? 3 : 1;

	size_t input_buf_stride = align(in.width() * in.pxsize(), ALIGNMENT);
	size_t output_buf_stride = align(out.width() * out.pxsize(), ALIGNMENT);

	// Allocate a duplicate buffer to handle the image borders.
	if (!flags.entire_plane) {
		alloc.allocate(input_buf_stride * ((size_t)input_mask + 1) * num_planes);
		alloc.allocate(output_buf_stride * ((size_t)output_mask + 1) * num_planes);
	}

	alloc.allocate(filter.get_context_size());
	alloc.allocate(filter.get_tmp_size(0, out.width()));

	return allocate_buffer(alloc.count(), PixelType::BYTE);
}

void apply_filter(const zimg::IZimgFilter &filter, const Frame &in, Frame &out, void *alloc_pool, int plane)
{
	zimg_filter_flags flags = filter.get_flags();
	LinearAllocator alloc{ alloc_pool };

	unsigned input_lines = filter.get_max_buffering();
	unsigned output_lines = filter.get_simultaneous_lines();
	unsigned input_mask = select_zimg_buffer_mask(input_lines);
	unsigned output_mask = select_zimg_buffer_mask(output_lines);

	size_t input_buf_stride = align(in.width() * in.pxsize(), ALIGNMENT);
	size_t output_buf_stride = align(out.width() * out.pxsize(), ALIGNMENT);

	zimg_image_buffer in_buf1{};
	zimg_image_buffer in_buf2{};
	zimg_image_buffer out_buf1{};
	zimg_image_buffer out_buf2{};

	void *ctx = alloc.allocate(filter.get_context_size());
	void *tmp = alloc.allocate(filter.get_tmp_size(0, out.width()));

	for (int p = 0; p < (flags.color ? 3 : 1); ++p) {
		in_buf1.data[p] = const_cast<unsigned char *>(in.data(flags.color ? p : plane));
		in_buf1.stride[p] = in.stride() * in.pxsize();
		in_buf1.mask[p] = -1;

		if (!flags.entire_plane) {
			in_buf2.data[p] = alloc.allocate(input_buf_stride * ((size_t)input_mask + 1));
			in_buf2.stride[p] = input_buf_stride;
			in_buf2.mask[p] = input_mask;
		}
	}
	for (int p = 0; p < (flags.color ? 3 : 1); ++p) {
		out_buf1.data[p] = out.data(flags.color ? p : plane);
		out_buf1.stride[p] = out.stride() * out.pxsize();
		out_buf1.mask[p] = -1;

		if (!flags.entire_plane) {
			out_buf2.data[p] = alloc.allocate(output_buf_stride * ((size_t)output_mask + 1));
			out_buf2.stride[p] = output_buf_stride;
			out_buf2.mask[p] = output_mask;
		}
	}

	filter.init_context(ctx);

	for (unsigned i = 0; i < (unsigned)out.height(); i += output_lines) {
		const zimg_image_buffer *in_buf_p = &in_buf1;
		const zimg_image_buffer *out_buf_p = &out_buf1;

		auto input_bounds = filter.get_required_row_range(i);
		unsigned top = input_bounds.first;
		unsigned bot = input_bounds.second;

		if (top == 0 && bot - top < input_lines && !flags.entire_plane) {
			for (int p = 0; p < (flags.color ? 3 : 1); ++p) {
				LineBuffer<void> src_buf{ in_buf1.data[p], (unsigned)in.width(), (unsigned)in_buf1.stride[p], in_buf1.mask[p] };
				LineBuffer<void> dst_buf{ in_buf2.data[p], (unsigned)in.width(), (unsigned)in_buf2.stride[p], in_buf2.mask[p] };

				copy_buffer_lines(src_buf, dst_buf, in.width() * in.pxsize(), top, bot);
			}

			in_buf_p = &in_buf2;
		}

		if (i + output_lines > (unsigned)out.height() && !flags.entire_plane)
			out_buf_p = &out_buf2;

		filter.process(ctx, in_buf_p, out_buf_p, tmp, i, 0, out.width());

		if (i + output_lines > (unsigned)out.height() && !flags.entire_plane) {
			for (int p = 0; p < (flags.color ? 3 : 1); ++p) {
				LineBuffer<void> src_buf{ out_buf2.data[p], (unsigned)out.width(), (unsigned)out_buf2.stride[p], out_buf2.mask[p] };
				LineBuffer<void> dst_buf{ out_buf1.data[p], (unsigned)out.width(), (unsigned)out_buf1.stride[p], out_buf1.mask[p] };

				copy_buffer_lines(src_buf, dst_buf, out.width() * out.pxsize(), i, out.height());
			}
		}
	}
}

void measure_time(int times, std::function<void(void)> f)
{
	Timer timer;
	double min_time = INFINITY;
	double avg_time = 0.0;

	for (int n = 0; n < times; ++n) {
		double elapsed;

		timer.start();
		f();
		timer.stop();

		elapsed = timer.elapsed();
		std::cout << '#' << n << ": " << elapsed << '\n';

		avg_time += elapsed / times;
		min_time = min_time < elapsed ? min_time : elapsed;
	}
	std::cout << "average: " << avg_time << '\n';
	std::cout << "min: " << min_time << '\n';
}

void parse_opts(const char **first, const char **last, const AppOption *options_first, const AppOption *options_last, void *dst, void *user)
{
	std::unordered_map<std::string, const AppOption *> option_map;
	char *dst_byteptr = reinterpret_cast<char *>(dst);

	while (options_first != options_last) {
		option_map[options_first->name] = options_first;
		++options_first;
	}

	while (first < last) {
		std::string o{ *first };

		if (o.find_first_of("--") == 0) {
			o = o.substr(2);
		} else if (o.find_first_of("-") == 0) {
			o = o.substr(1);
		} else {
			std::cerr << "not an option: " << o << '\n';
			throw std::invalid_argument{ "not an option" };
		}

		auto it = option_map.find(o);
		if (it == option_map.end()) {
			std::cerr << "unknown option: " << o << '\n';
			throw std::invalid_argument{ "unknown option" };
		}

		const AppOption *cur = it->second;
		int nargs = required_args(cur->type);

		if (last - first < nargs) {
			std::cerr << "insufficient arguments to option: " << o << '\n';
			throw std::invalid_argument{ "insufficient arguments" };
		}

		char *dst_out = dst_byteptr + cur->offset;

		switch (cur->type) {
		case OptionType::OPTION_INTEGER:
			*(int *)dst_out = std::stoi(first[1]);
			break;
		case OptionType::OPTION_FLOAT:
			*(double *)dst_out = std::stod(first[1]);
			break;
		case OptionType::OPTION_STRING:
			*(const char **)dst_out = first[1];
			break;
		case OptionType::OPTION_FALSE:
			*(int *)dst_out = 0;
			break;
		case OptionType::OPTION_TRUE:
			*(int *)dst_out = 1;
			break;
		case OptionType::OPTION_CPUCLASS:
			*(CPUClass *)dst_out = select_cpu(first[1]);
			break;
		case OptionType::OPTION_PIXELTYPE:
			*(PixelType *)dst_out = select_pixel_type(first[1]);
			break;
		case OptionType::OPTION_SPECIAL:
			nargs = cur->func(first, last, dst, user);
			break;
		default:
			throw std::invalid_argument{ "unknown option type" };
		}

		first += nargs;
	}
}
