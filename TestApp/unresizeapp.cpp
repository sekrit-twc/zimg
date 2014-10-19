#include <cstddef>
#include <iostream>
#include <iterator>
#include <string>
#include "Unresize/unresize.h"
#include "apps.h"
#include "bitmap.h"

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
};

const AppOption OPTIONS[] = {
	{ "shift-w", OptionType::OPTION_FLOAT,    offsetof(AppContext, shift_w) },
	{ "shift-h", OptionType::OPTION_FLOAT,    offsetof(AppContext, shift_h) },
	{ "times",   OptionType::OPTION_INTEGER,  offsetof(AppContext, times) },
	{ "cpu",     OptionType::OPTION_CPUCLASS, offsetof(AppContext, cpu) }
};

void usage()
{
	std::cout << "unresize infile outfile width height [--shift-w shift] [--shift-h shift] [--times n] [--cpu cpu]\n";
	std::cout << "    infile              input BMP file\n";
	std::cout << "    outfile             output BMP file\n";
	std::cout << "    width               output width\n";
	std::cout << "    height              output height\n";
	std::cout << "    --shift-w           horizontal shift\n";
	std::cout << "    --shift-h           vertical shift\n";
	std::cout << "    --times             number of cycles\n";
	std::cout << "    --cpu               select CPU type\n";
}

void execute(const Bitmap &in, Bitmap &out, float shift_w, float shift_h, int times, CPUClass cpu)
{
	unresize::Unresize u(in.width(), in.height(), out.width(), out.height(), shift_w, shift_h, cpu);

	int src_stride = width_to_stride(in.width(), PixelType::FLOAT);
	int dst_stride = width_to_stride(out.width(), PixelType::FLOAT);
	size_t src_plane_size = image_plane_size(src_stride, in.height(), PixelType::FLOAT);
	size_t dst_plane_size = image_plane_size(dst_stride, out.height(), PixelType::FLOAT);

	auto src_planes = allocate_frame(src_stride, in.height(), 3, PixelType::FLOAT);
	auto dst_planes = allocate_frame(dst_stride, out.height(), 3, PixelType::FLOAT);
	auto tmp = allocate_buffer(u.tmp_size(PixelType::FLOAT), PixelType::FLOAT);

	for (int p = 0; p < 3; ++p) {
		convert_from_byte(PixelType::FLOAT, in.data(p), src_planes.data() + src_plane_size * p, in.width(), in.height(), in.stride(), src_stride);
	}

	measure_time(times, [&]()
	{
		for (int p = 0; p < 3; ++p) {
			const void *src_p = src_planes.data() + p * src_plane_size;
			void *dst_p = dst_planes.data() + p * dst_plane_size;

			u.process(PixelType::FLOAT, src_p, dst_p, tmp.data(), src_stride, dst_stride);
		}
	});

	for (int p = 0; p < 3; ++p) {
		convert_to_byte(PixelType::FLOAT, dst_planes.data() + dst_plane_size * p, out.data(p), out.width(), out.height(), dst_stride, out.stride());
	}
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

	parse_opts(argv + 5, argv + argc, std::begin(OPTIONS), std::end(OPTIONS), &c, nullptr);

	Bitmap in = read_bitmap(c.infile);
	Bitmap out(c.width, c.height, false);

	execute(in, out, (float)c.shift_w, (float)c.shift_h, c.times, c.cpu);
	write_bitmap(out, c.outfile);

	return 0;
}
