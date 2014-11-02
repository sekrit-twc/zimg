#include <cstring>
#include <iostream>
#include <iterator>
#include <string>
#include "Common/cpuinfo.h"
#include "Common/pixel.h"
#include "Common/plane.h"
#include "Depth/depth.h"
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
	PixelType pixtype_in;
	PixelType pixtype_out;
	depth::DitherType dither;
	int bits_in;
	int bits_out;
	bool fullrange_in;
	bool fullrange_out;
	bool yuv;
	const char *visualise;
	int times;
	CPUClass cpu;
};

int select_dither(const char **opt, const char **lastopt, void *p, void *user)
{
	AppContext *c = (AppContext *)p;

	if (lastopt - opt < 2)
		throw std::invalid_argument{ "insufficient arguments for option dither" };

	const char *dither = opt[1];

	if (!strcmp(dither, "none"))
		c->dither = depth::DitherType::DITHER_NONE;
	else if (!strcmp(dither, "ordered"))
		c->dither = depth::DitherType::DITHER_ORDERED;
	else if (!strcmp(dither, "random"))
		c->dither = depth::DitherType::DITHER_RANDOM;
	else if (!strcmp(dither, "error_diffusion"))
		c->dither = depth::DitherType::DITHER_ERROR_DIFFUSION;
	else
		throw std::invalid_argument{ "unsupported dither type" };

	return 2;
}

const AppOption OPTIONS[] = {
	{ "dither",    OptionType::OPTION_SPECIAL,  0, select_dither },
	{ "bits-in",   OptionType::OPTION_INTEGER,  offsetof(AppContext, bits_in) },
	{ "bits-out",  OptionType::OPTION_INTEGER,  offsetof(AppContext, bits_out) },
	{ "tv-in",     OptionType::OPTION_FALSE,    offsetof(AppContext, fullrange_in) },
	{ "pc-in",     OptionType::OPTION_TRUE,     offsetof(AppContext, fullrange_in) },
	{ "tv-out",    OptionType::OPTION_FALSE,    offsetof(AppContext, fullrange_out) },
	{ "pc-out",    OptionType::OPTION_TRUE,     offsetof(AppContext, fullrange_out) },
	{ "yuv",       OptionType::OPTION_TRUE,     offsetof(AppContext, yuv) },
	{ "rgb",       OptionType::OPTION_FALSE,    offsetof(AppContext, yuv) },
	{ "visualise", OptionType::OPTION_STRING,   offsetof(AppContext, visualise) },
	{ "times",     OptionType::OPTION_INTEGER,  offsetof(AppContext, times) },
	{ "cpu",       OptionType::OPTION_CPUCLASS, offsetof(AppContext, cpu) },
};

void usage()
{
	std::cout << "depth infile outfile w h pxl_in pxl_out [--dither dither] [--bits-in bits] [--bits-out bits] [--tv-in | pc-in] [--tv-out | --pc-out] [--yuv | --rgb] [--times n] [--cpu cpu]\n";
	std::cout << "    infile               input file\n";
	std::cout << "    outfile              output file\n";
	std::cout << "    w                    image width\n";
	std::cout << "    h                    image height\n";
	std::cout << "    pxl_in               input pixel type\n";
	std::cout << "    pxl_out              output pixel type\n";
	std::cout << "    --dither             select dithering type\n";
	std::cout << "    --bits-in            input bit depth (integer only)\n";
	std::cout << "    --bits-out           output bit depth (integer only)\n";
	std::cout << "    --tv-in | --pc-in    toggle TV vs PC range for input\n";
	std::cout << "    --tv-out | --pc-out  toggle TV vs PC range for output\n";
	std::cout << "    --yuv | --rgb        toggle YUV vs RGB\n";
	std::cout << "    --visualise          path to BMP file for visualisation\n";
	std::cout << "    --times              number of cycles\n";
	std::cout << "    --cpu                select CPU type\n";
}

void execute(const depth::Depth &depth, const Frame &in, Frame &out, int times,
             PixelType pxl_in, PixelType pxl_out, int bits_in, int bits_out, bool fullrange_in, bool fullrange_out, bool yuv)
{
	int width = in.width();
	int height = in.height();
	int src_stride = in.stride();
	int dst_stride = out.stride();

	auto tmp = allocate_buffer(depth.tmp_size(width), PixelType::FLOAT);

	measure_time(times, [&]()
	{
		for (int p = 0; p < 3; ++p) {
			bool chroma = yuv && (p == 1 || p == 2);

			PixelFormat src_format{ pxl_in, bits_in, fullrange_in, chroma };
			PixelFormat dst_format{ pxl_out, bits_out, fullrange_out, chroma };

			ImagePlane<const void> src_plane{ in.data(p), width, height, src_stride, src_format };
			ImagePlane<void> dst_plane{ out.data(p), width, height, dst_stride, dst_format };

			depth.process(src_plane, dst_plane, tmp.data());
		}
	});
}

void export_for_bmp(const Frame &in, Frame &out, PixelType type, int bits, bool fullrange, bool yuv)
{
	depth::Depth depth{ depth::DitherType::DITHER_NONE, CPUClass::CPU_NONE };

	int width = in.width();
	int height = in.height();
	int src_stride = in.stride();
	int dst_stride = out.stride();

	auto tmp = allocate_buffer(depth.tmp_size(width), PixelType::FLOAT);

	for (int p = 0; p < 3; ++p) {
		bool chroma = yuv && (p == 1 || p == 2);

		PixelFormat src_format{ type, bits, fullrange, chroma };
		PixelFormat dst_format{ PixelType::BYTE, 8, fullrange, chroma };

		ImagePlane<const void> src_plane{ in.data(p), width, height, src_stride, src_format };
		ImagePlane<void> dst_plane{ out.data(p), width, height, dst_stride, dst_format };

		depth.process(src_plane, dst_plane, tmp.data());
	}
}

} // namespace


int depth_main(int argc, const char **argv)
{
	if (argc < 7) {
		usage();
		return -1;
	}

	AppContext c{};

	c.infile        = argv[1];
	c.outfile       = argv[2];
	c.width         = std::stoi(argv[3]);
	c.height        = std::stoi(argv[4]);
	c.pixtype_in    = select_pixel_type(argv[5]);
	c.pixtype_out   = select_pixel_type(argv[6]);
	c.bits_in       = -1;
	c.bits_out      = -1;
	c.dither        = depth::DitherType::DITHER_NONE;
	c.fullrange_in  = false;
	c.fullrange_out = false;
	c.yuv           = false;
	c.visualise     = nullptr;
	c.times         = 1;
	c.cpu           = CPUClass::CPU_NONE;

	parse_opts(argv + 7, argv + argc, std::begin(OPTIONS), std::end(OPTIONS), &c, nullptr);

	c.bits_in  = c.bits_in < 0 ? pixel_size(c.pixtype_in) * 8 : c.bits_in;
	c.bits_out = c.bits_out < 0 ? pixel_size(c.pixtype_out) * 8 : c.bits_out;

	int width = c.width;
	int height = c.height;

	Frame in{ width, height, pixel_size(c.pixtype_in), 3 };
	Frame out{ width, height, pixel_size(c.pixtype_out), 3 };

	read_frame_raw(in, c.infile);

	depth::Depth depth{ c.dither, c.cpu };
	execute(depth, in, out, c.times, c.pixtype_in, c.pixtype_out, c.bits_in, c.bits_out, c.fullrange_in, c.fullrange_out, c.yuv);

	write_frame_raw(out, c.outfile);

	if (c.visualise) {
		Frame bmp{ width, height, 1, 3 };

		export_for_bmp(out, bmp, c.pixtype_out, c.bits_out, c.fullrange_out, c.yuv);
		write_frame_bmp(bmp, c.visualise);
	}

	return 0;
}
