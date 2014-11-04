#include <cstddef>
#include <iostream>
#include <iterator>
#include <string>
#include "Common/cpuinfo.h"
#include "Common/pixel.h"
#include "Common/plane.h"
#include "Unresize/unresize.h"
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
	double shift_w;
	double shift_h;
	int times;
	CPUClass cpu;
	PixelType pixtype;
};

const AppOption OPTIONS[] = {
	{ "shift-w", OptionType::OPTION_FLOAT,     offsetof(AppContext, shift_w) },
	{ "shift-h", OptionType::OPTION_FLOAT,     offsetof(AppContext, shift_h) },
	{ "times",   OptionType::OPTION_INTEGER,   offsetof(AppContext, times) },
	{ "cpu",     OptionType::OPTION_CPUCLASS,  offsetof(AppContext, cpu) },
	{ "pixtype", OptionType::OPTION_PIXELTYPE, offsetof(AppContext, pixtype) }
};

void usage()
{
	std::cout << "unresize infile outfile width height [--shift-w shift] [--shift-h shift] [--times n] [--cpu cpu] [--pixtype type]\n";
	std::cout << "    infile              input BMP file\n";
	std::cout << "    outfile             output BMP file\n";
	std::cout << "    w                   output width\n";
	std::cout << "    h                   output height\n";
	std::cout << "    --shift-w           horizontal shift\n";
	std::cout << "    --shift-h           vertical shift\n";
	std::cout << "    --times             number of cycles\n";
	std::cout << "    --cpu               select CPU type\n";
	std::cout << "    --pixtype           select pixel format\n";
}

void execute(const unresize::Unresize &unresize, const Frame &in, Frame &out, int times, PixelType type)
{
	int pxsize = pixel_size(type);
	int planes = in.planes();

	Frame src{ in.width(), in.height(), pxsize, planes };
	Frame dst{ out.width(), out.height(), pxsize, planes };
	auto tmp = allocate_buffer(unresize.tmp_size(type), type);

	convert_frame(in, src, PixelType::BYTE, type, true, false);

	measure_time(times, [&]()
	{
		for (int p = 0; p < src.planes(); ++p) {
			ImagePlane<const void> src_p{ src.data(p), src.width(), src.height(), src.stride(), type };
			ImagePlane<void> dst_p{ dst.data(p), dst.width(), dst.height(), dst.stride(), type };

			unresize.process(src_p, dst_p, tmp.data());
		}
	});

	convert_frame(dst, out, type, PixelType::BYTE, true, false);
}

} // namespace


int unresize_main(int argc, const char **argv)
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
	c.shift_w = 0.0;
	c.shift_h = 0.0;
	c.times   = 1;
	c.cpu     = CPUClass::CPU_NONE;
	c.pixtype = PixelType::FLOAT;

	parse_opts(argv + 5, argv + argc, std::begin(OPTIONS), std::end(OPTIONS), &c, nullptr);

	Frame in{ read_frame_bmp(c.infile) };
	Frame out{ c.width, c.height, 1, in.planes() };

	unresize::Unresize unresize{ in.width(), in.height(), out.width(), out.height(), (float)c.shift_w, (float)c.shift_h, c.cpu };
	execute(unresize, in, out, c.times, c.pixtype);
	write_frame_bmp(out, c.outfile);

	return 0;
}
