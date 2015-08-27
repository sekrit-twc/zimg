#include <cstddef>
#include <cstring>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include "Common/cpuinfo.h"
#include "Common/pixel.h"
#include "Common/plane.h"
#include "Common/static_map.h"
#include "Resize/filter.h"
#include "Resize/resize2.h"
#include "apps.h"
#include "frame.h"
#include "utils.h"

using namespace zimg;

namespace {;

struct AppContext {
	const char *infile;
	const char *outfile;
	int width;
	int height;
	std::unique_ptr<resize::Filter> filter;
	double shift_w;
	double shift_h;
	double sub_w;
	double sub_h;
	int times;
	CPUClass cpu;
	PixelType pixtype;
};

int select_filter(const char **opt, const char **lastopt, void *p, void *)
{
	static resize::Filter *const _f = nullptr;
	static const static_string_map<resize::Filter *(*)(void), 6> map{
		{ "point",    []() { return 0 ? _f : new resize::PointFilter{}; } },
		{ "bilinear", []() { return 0 ? _f : new resize::BilinearFilter{}; } },
		{ "bicubc",   []() { return 0 ? _f : new resize::BicubicFilter{ 1.0 / 3.0, 1.0 / 3.0 }; } },
		{ "spline16", []() { return 0 ? _f : new resize::Spline16Filter{}; } },
		{ "spline36", []() { return 0 ? _f : new resize::Spline36Filter{}; } },
		{ "lanczos",  []() { return 0 ? _f : new resize::LanczosFilter{ 4 }; } },
	};

	AppContext *c = (AppContext *)p;

	if (lastopt - opt < 2)
		throw std::invalid_argument{ "insufficient arguments for option filter" };

	auto it = map.find(opt[1]);
	c->filter.reset(it == map.end() ? throw std::invalid_argument{ "unsupported filter type" } : it->second());

	return 2;
}

const AppOption OPTIONS[] = {
		{ "filter",   OptionType::OPTION_SPECIAL,   0, select_filter },
		{ "shift-w",  OptionType::OPTION_FLOAT,     offsetof(AppContext, shift_w) },
		{ "shift-h",  OptionType::OPTION_FLOAT,     offsetof(AppContext, shift_h) },
		{ "sub-w",    OptionType::OPTION_FLOAT,     offsetof(AppContext, sub_w) },
		{ "sub-h",    OptionType::OPTION_FLOAT,     offsetof(AppContext, sub_h) },
		{ "times",    OptionType::OPTION_INTEGER,   offsetof(AppContext, times) },
		{ "cpu",      OptionType::OPTION_CPUCLASS,  offsetof(AppContext, cpu) },
		{ "pixtype",  OptionType::OPTION_PIXELTYPE, offsetof(AppContext, pixtype) }
};

void usage()
{
	std::cout << "resize infile outfile w h [--filter filter] [--shift-w shift] [--shift-h shift] [--sub-w w] [--sub-h h] [--times n] [--cpu cpu] [--pixtype type]\n";
	std::cout << "    infile              input BMP file\n";
	std::cout << "    outfile             output BMP file\n";
	std::cout << "    w                   output width\n";
	std::cout << "    h                   output height\n";
	std::cout << "    --filter            resampling filter\n";
	std::cout << "    --shift-w           horizontal shift\n";
	std::cout << "    --shift-h           vertical shift\n";
	std::cout << "    --sub-w             subwindow width\n";
	std::cout << "    --sub-h             subwindow height\n";
	std::cout << "    --times             number of cycles\n";
	std::cout << "    --cpu               select CPU type\n";
	std::cout << "    --pixtype           select pixel format\n";
}

void execute(const IZimgFilter *resize, const Frame &in, Frame &out, int times, PixelType type)
{
	int pxsize = pixel_size(type);
	int planes = in.planes();

	Frame src{ in.width(), in.height(), pxsize, planes };
	Frame dst{ out.width(), out.height(), pxsize, planes };

	auto tmp = alloc_filter_tmp(*resize, src, dst);

	convert_frame(in, src, PixelType::BYTE, type, true, false);

	measure_time(times, [&]()
	{
		for (int p = 0; p < src.planes(); ++p) {
			apply_filter(*resize, src, dst, tmp.data(), p);
		}
	});

	convert_frame(dst, out, type, PixelType::BYTE, true, false);
}

} // namespace


int resize_main(int argc, const char **argv)
{
	if (argc < 5) {
		usage();
		return -1;
	}

	AppContext c{};

	c.infile  = argv[1];
	c.outfile = argv[2];
	c.width   = std::stoi(argv[3]);
	c.height  = std::stoi(argv[4]);
	c.shift_h = 0.0;
	c.shift_w = 0.0;
	c.sub_w   = -1.0;
	c.sub_h   = -1.0;
	c.times   = 1;
	c.pixtype = PixelType::FLOAT;
	c.cpu     = CPUClass::CPU_NONE;

	parse_opts(argv + 5, argv + argc, std::begin(OPTIONS), std::end(OPTIONS), &c, nullptr);

	Frame in{ read_frame_bmp(c.infile) };
	Frame out{ c.width, c.height, 1, in.planes() };

	if (c.sub_w < 0.0)
		c.sub_w = in.width();
	if (c.sub_h < 0.0)
		c.sub_h = in.height();
	if (!c.filter)
		c.filter.reset(new resize::BilinearFilter{});

	std::unique_ptr<IZimgFilter> resize{ resize::create_resize2(*c.filter, c.pixtype, pixel_size(c.pixtype) * 8, in.width(), in.height(), c.width, c.height,
	                                                            c.shift_w, c.shift_h, c.sub_w, c.sub_h, c.cpu) };

	execute(resize.get(), in, out, c.times, c.pixtype);
	write_frame_bmp(out, c.outfile);

	return 0;
}
