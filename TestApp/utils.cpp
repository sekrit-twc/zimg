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
#include "Common/cpuinfo.h"
#include "Common/pixel.h"
#include "Common/plane.h"
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
	else if (!strcmp(cpu, "f16c"))
		return CPUClass::CPU_X86_F16C;
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

void convert_frame(const Frame &in, Frame &out, zimg::PixelType pxl_in, zimg::PixelType pxl_out, bool tv, bool yuv)
{
	std::unique_ptr<depth::Depth> convert{ new depth::Depth{ depth::DitherType::DITHER_NONE, CPUClass::CPU_NONE } };
	int width = in.width();
	int height = in.height();
	int planes = in.planes();

	for (int p = 0; p < planes; ++p) {
		bool plane_tv = tv && p != 3; // Always treat alpha as fullrange.
		bool plane_chroma = yuv && (p == 1 || p == 2); // Chroma planes.

		PixelFormat src_fmt = default_pixel_format(pxl_in);
		PixelFormat dst_fmt = default_pixel_format(pxl_out);

		src_fmt.tv = plane_tv;
		src_fmt.chroma = plane_chroma;
		dst_fmt.tv = plane_tv;
		dst_fmt.chroma = plane_chroma;

		ImagePlane<const void> src_plane{ in.data(p), width, height, in.stride(), src_fmt };
		ImagePlane<void> dst_plane{ out.data(p), width, height, out.stride(), dst_fmt };

		convert->process(src_plane, dst_plane, nullptr);
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
