#include <cstddef>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include "Resize/filter.h"
#include "Resize/resize.h"
#include "apps.h"
#include "bitmap.h"

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
	AppContext *c = (AppContext *)p;
	const char *filter = opt[1];

	if (lastopt - opt < 2)
		throw std::invalid_argument{ "insufficient arguments for option filter" };

	if (!strcmp(filter, "point"))
		c->filter.reset(new resize::PointFilter{});
	else if (!strcmp(filter, "bilinear"))
		c->filter.reset(new resize::BilinearFilter{});
	else if (!strcmp(filter, "bicubic"))
		c->filter.reset(new resize::BicubicFilter{ 1.0 / 3.0, 1.0 / 3.0 });
	else if (!strcmp(filter, "lanczos"))
		c->filter.reset(new resize::LanczosFilter{ 4 });
	else if (!strcmp(filter, "spline16"))
		c->filter.reset(new resize::Spline16Filter{});
	else if (!strcmp(filter, "spline36"))
		c->filter.reset(new resize::Spline36Filter{});
	else
		throw std::invalid_argument{ "unsupported filter type" };

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
	std::cout << "resize infile outfile w h [--filter filter] [--shift-w shift] [--shift-h shift] [--sub-w w] [--sub-h h] [--times n] [--cpu cpu] [--u16 | --f16 | --f32]\n";
	std::cout << "    infile              input BMP file\n";
	std::cout << "    outfile             output BMP file\n";
	std::cout << "    width               output width\n";
	std::cout << "    height              output height\n";
	std::cout << "    --filter            resampling filter\n";
	std::cout << "    --shift-w           horizontal shift\n";
	std::cout << "    --shift-h           vertical shift\n";
	std::cout << "    --sub-w             subwindow width\n";
	std::cout << "    --sub-h             subwindow height\n";
	std::cout << "    --times             number of cycles\n";
	std::cout << "    --cpu               select CPU type\n";
	std::cout << "    --pixtype           select pixel format\n";
}

void execute(const resize::Resize &resize, const Bitmap &in, Bitmap &out, int times, PixelType type)
{
	int pxsize = pixel_size(type);

	int src_width = in.width();
	int src_height = in.height();
	int src_stride = width_to_stride(in.width(), type);
	int planes = in.planes();

	int dst_width = out.width();
	int dst_height = out.height();
	int dst_stride = width_to_stride(out.width(), type);

	size_t src_plane_size = image_plane_size(src_stride, src_height, type);
	size_t dst_plane_size = image_plane_size(dst_stride, dst_height, type);

	auto in_planes = allocate_frame(src_stride, src_height, planes, type);
	auto out_planes = allocate_frame(dst_stride, dst_height, planes, type);
	auto tmp = allocate_buffer(resize.tmp_size(type), type);

	for (int p = 0; p < planes; ++p) {
		convert_from_byte(type, in.data(p), in_planes.data() + p * src_plane_size, src_width, src_height, in.stride(), src_stride);
	}

	measure_time(times, [&]()
	{
		for (int p = 0; p < planes; ++p) {
			const void *src = in_planes.data() + p * src_plane_size;
			void *dst = out_planes.data() + p * dst_plane_size;

			resize.process(type, src, dst, tmp.data(), src_stride, dst_stride);
		}
	});

	for (int p = 0; p < planes; ++p) {
		convert_to_byte(type, out_planes.data() + p * dst_plane_size, out.data(p), dst_width, dst_height, dst_stride, out.stride());
	}
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

	Bitmap in = read_bitmap(c.infile);
	Bitmap out{ c.width, c.height, in.planes() == 4 };

	if (c.sub_w < 0.0)
		c.sub_w = in.width();
	if (c.sub_h < 0.0)
		c.sub_h = in.height();
	if (!c.filter)
		c.filter.reset(new resize::BilinearFilter{});

	resize::Resize resize{ *c.filter, in.width(), in.height(), c.width, c.height, c.shift_w, c.shift_h, c.sub_w, c.sub_h, c.cpu };

	execute(resize, in, out, c.times, c.pixtype);
	write_bitmap(out, c.outfile);

	return 0;
}
