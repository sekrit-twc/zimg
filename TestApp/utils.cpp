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
#include "Common/static_map.h"
#include "Common/zfilter.h"
#include "Depth/depth2.h"
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
	static const static_string_map<CPUClass, 4> map{
		{ "none", CPUClass::CPU_NONE },
		{ "auto", CPUClass::CPU_AUTO },
#ifdef ZIMG_X86
		{ "sse2", CPUClass::CPU_X86_SSE2 },
		{ "avx2", CPUClass::CPU_X86_AVX2 },
#endif
	};
	auto it = map.find(cpu);
	return it == map.end() ? CPUClass::CPU_NONE : it->second;
}

zimg::PixelType select_pixel_type(const char *pixtype)
{
	static const static_string_map<PixelType, 4> map{
		{ "u8",  PixelType::BYTE },
		{ "u16", PixelType::WORD },
		{ "f16", PixelType::HALF },
		{ "f32", PixelType::FLOAT }
	};
	auto it = map.find(pixtype);
	return it == map.end() ? throw std::invalid_argument{ "unknown pixel type" } : it->second;
}

zimg::AlignedVector<char> allocate_buffer(size_t count, zimg::PixelType type)
{
	return AlignedVector<char>(count * pixel_size(type));
}

void convert_frame(const Frame &in, Frame &out, zimg::PixelType pxl_in, zimg::PixelType pxl_out, bool fullrange, bool yuv)
{
	PixelFormat src_fmt = default_pixel_format(pxl_in);
	PixelFormat dst_fmt = default_pixel_format(pxl_out);

	src_fmt.fullrange = fullrange;
	dst_fmt.fullrange = fullrange;

	src_fmt.chroma = false;
	dst_fmt.chroma = false;
	std::unique_ptr<IZimgFilter> convert{ depth::create_depth2(depth::DitherType::DITHER_NONE, (unsigned)in.width(), (unsigned)in.height(), src_fmt, dst_fmt, CPUClass::CPU_NONE) };

	src_fmt.chroma = yuv;
	dst_fmt.chroma = yuv;
	std::unique_ptr<IZimgFilter> convert_uv{ depth::create_depth2(depth::DitherType::DITHER_NONE, (unsigned)in.width(), (unsigned)in.height(), src_fmt, dst_fmt, CPUClass::CPU_NONE) };

	for (int p = 0; p < in.planes(); ++p) {
		bool plane_chroma = yuv && (p == 1 || p == 2); // Chroma planes.

		apply_filter(*(plane_chroma ? convert_uv : convert), in, out, nullptr, p);
	}
}

zimg::AlignedVector<char> alloc_filter_tmp(const zimg::IZimgFilter &filter, const Frame &in, Frame &out)
{
	FakeAllocator alloc;

	alloc.allocate(filter.get_context_size());
	alloc.allocate(filter.get_tmp_size(0, out.width()));

	return allocate_buffer(alloc.count(), PixelType::BYTE);
}

void apply_filter(const zimg::IZimgFilter &filter, const Frame &in, Frame &out, void *alloc_pool, int plane)
{
	ZimgFilterFlags flags = filter.get_flags();
	LinearAllocator alloc{ alloc_pool };

	unsigned output_lines = filter.get_simultaneous_lines();

	void *ctx = alloc.allocate(filter.get_context_size());
	void *tmp = alloc.allocate(filter.get_tmp_size(0, out.width()));

	ZimgImageBufferConst in_buf{};
	ZimgImageBuffer out_buf{};

	for (int p = 0; p < (flags.color ? 3 : 1); ++p) {
		in_buf.data[p] = in.data(flags.color ? p : plane);
		in_buf.stride[p] = in.stride() * in.pxsize();
		in_buf.mask[p] = -1;

		out_buf.data[p] = out.data(flags.color ? p : plane);
		out_buf.stride[p] = out.stride() * out.pxsize();
		out_buf.mask[p] = -1;
	}

	filter.init_context(ctx);

	for (unsigned i = 0; i < (unsigned)out.height(); i += output_lines) {
		filter.process(ctx, in_buf, out_buf, tmp, i, 0, out.width());
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
