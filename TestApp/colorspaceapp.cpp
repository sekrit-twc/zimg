#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include "Common/align.h"
#include "Common/cpuinfo.h"
#include "Common/pixel.h"
#include "Colorspace/colorspace_param.h"
#include "Colorspace/colorspace.h"
#include "apps.h"
#include "bitmap.h"

using namespace zimg;

namespace {;

struct AppContext {
	const char *infile;
	const char *outfile;
	int width;
	int height;
	colorspace::ColorspaceDefinition csp_in;
	colorspace::ColorspaceDefinition csp_out;
	const char *visualise;
	int tv_in;
	int tv_out;
	int times;
	CPUClass cpu;
	PixelType pixtype;
};

const AppOption OPTIONS[] = {
	{ "pc-in",     OptionType::OPTION_FALSE,     offsetof(AppContext, tv_in) },
	{ "tv-in",     OptionType::OPTION_TRUE,      offsetof(AppContext, tv_in) },
	{ "pc-out",    OptionType::OPTION_FALSE,     offsetof(AppContext, tv_out) },
	{ "tv-out",    OptionType::OPTION_TRUE,      offsetof(AppContext, tv_out) }, 
	{ "visualise", OptionType::OPTION_STRING,    offsetof(AppContext, visualise) },
	{ "times",     OptionType::OPTION_INTEGER,   offsetof(AppContext, times) },
	{ "cpu",       OptionType::OPTION_CPUCLASS,  offsetof(AppContext, cpu) },
	{ "pixtype",   OptionType::OPTION_PIXELTYPE, offsetof(AppContext, pixtype) }
};

void usage()
{
	std::cout << "colorspace infile outfile w h csp_in csp_out [--pc-in | --tv-in] [--pc-out | --tv-out] [--visualise path] [--times n] [--cpu cpu] [--pixtype type] \n";
	std::cout << "    infile               input file\n";
	std::cout << "    outfile              output file\n";
	std::cout << "    width                image width\n";
	std::cout << "    height               image height\n";
	std::cout << "    csp_in               input colorspace\n";
	std::cout << "    csp_out              output colorspace\n";
	std::cout << "    --pc-in | --tv-in    specify PC or TV range input (integer only)\n";
	std::cout << "    --pc-out | --tv-out  specify PC or TV range output (integer only)\n";
	std::cout << "    --visualise          path to BMP file for visualisation\n";
	std::cout << "    --times              number of cycles\n";
	std::cout << "    --cpu                select CPU type\n";
	std::cout << "    --pixtype            select pixel format\n";
}

colorspace::MatrixCoefficients parse_matrix(const char *matrix)
{
	if (!strcmp(matrix, "rgb"))
		return colorspace::MatrixCoefficients::MATRIX_RGB;
	else if (!strcmp(matrix, "601"))
		return colorspace::MatrixCoefficients::MATRIX_601;
	else if (!strcmp(matrix, "709"))
		return colorspace::MatrixCoefficients::MATRIX_709;
	else if (!strcmp(matrix, "2020_ncl"))
		return colorspace::MatrixCoefficients::MATRIX_2020_NCL;
	else if (!strcmp(matrix, "2020_cl"))
		return colorspace::MatrixCoefficients::MATRIX_2020_CL;
	else
		throw std::runtime_error{ "bad matrix coefficients" };
}

colorspace::TransferCharacteristics parse_transfer(const char *transfer)
{
	if (!strcmp(transfer, "linear"))
		return colorspace::TransferCharacteristics::TRANSFER_LINEAR;
	else if (!strcmp(transfer, "709"))
		return colorspace::TransferCharacteristics::TRANSFER_709;
	else
		throw std::runtime_error{ "bad transfer characteristics" };
}

colorspace::ColorPrimaries parse_primaries(const char *primaries)
{
	if (!strcmp(primaries, "smpte_c"))
		return colorspace::ColorPrimaries::PRIMARIES_SMPTE_C;
	else if (!strcmp(primaries, "709"))
		return colorspace::ColorPrimaries::PRIMARIES_709;
	else if (!strcmp(primaries, "2020"))
		return colorspace::ColorPrimaries::PRIMARIES_2020;
	else
		throw std::runtime_error{ "bad primaries" };
}

colorspace::ColorspaceDefinition parse_csp(const char *str)
{
	colorspace::ColorspaceDefinition csp;
	std::string s{ str };
	std::string sub;
	size_t prev;
	size_t curr;

	prev = 0;
	curr = s.find(':');
	if (curr == std::string::npos || curr == s.size() - 1)
		throw std::runtime_error{ "bad colorspace string" };

	sub = s.substr(prev, curr - prev);
	csp.matrix = parse_matrix(sub.c_str());

	prev = curr + 1;
	curr = s.find(':', prev);
	if (curr == std::string::npos || curr == s.size() - 1)
		throw std::runtime_error{ "bad colorspace string" };

	sub = s.substr(prev, curr - prev);
	csp.transfer = parse_transfer(sub.c_str());

	prev = curr + 1;
	curr = s.size();
	sub = s.substr(prev, curr - prev);
	csp.primaries = parse_primaries(sub.c_str());

	return csp;
}

void read_frame(const char *path, void *dst, int width, int height, PixelType type)
{
	int stride = width_to_stride(width, type);
	uint8_t *byteptr = (uint8_t *)dst;

	std::unique_ptr<FILE, decltype(&fclose)> file{ fopen(path, "rb"), fclose };

	if (!file)
		throw std::runtime_error{ "error opening file" };

	for (int p = 0; p < 3; ++p) {
		for (int i = 0; i < height; ++i) {
			if (fread(byteptr, pixel_size(type), width, file.get()) != width)
				throw std::runtime_error{ "read error" };

			byteptr += stride * pixel_size(type);
		}
	}
}

void write_frame(const char *path, const void *src, int width, int height, PixelType type)
{
	int stride = width_to_stride(width, type);
	const uint8_t *byteptr = (const uint8_t *)src;

	std::unique_ptr<FILE, decltype(&fclose)> file{ fopen(path, "wb"), fclose };

	if (!file)
		throw std::runtime_error{ "error opening file" };

	for (int p = 0; p < 3; ++p) {
		for (int i = 0; i < height; ++i) {
			if (fwrite(byteptr, pixel_size(type), width, file.get()) != width)
				throw std::runtime_error{ "write error" };

			byteptr += stride * pixel_size(type);
		}
	}
}

void execute(const colorspace::ColorspaceConversion &conv, const void *in, void *out, Bitmap *bmp, int width, int height, bool tv_in, bool tv_out, int times, PixelType type)
{
	int stride = width_to_stride(width, type);
	size_t plane_size = image_plane_size(stride, height, type);
	auto tmp = allocate_buffer(conv.tmp_size(width), PixelType::FLOAT);

	int stride_array[3] = { stride, stride, stride };
	const void *in_p[3];
	void *out_p[3];

	for (int p = 0; p < 3; ++p) {
		in_p[p] = (const char *)in + p * plane_size;
		out_p[p] = (char *)out + p * plane_size;
	}

	measure_time(times, [&]()
	{
		conv.process(type, in_p, out_p, tmp.data(), width, height, stride_array, stride_array, tv_in, tv_out);
	});

	if (bmp) {
		for (int p = 0; p < 3; ++p) {
			// Handle inverted RGB order.
			convert_to_byte(type, out_p[p], bmp->data(2 - p), width, height, stride, bmp->stride());
		}
	}
}

} // namespace


int colorspace_main(int argc, const char **argv)
{
	if (argc < 7) {
		usage();
		return -1;
	}

	AppContext c{};

	c.infile    = argv[1];
	c.outfile   = argv[2];
	c.width     = std::stoi(argv[3]);
	c.height    = std::stoi(argv[4]);
	c.csp_in    = parse_csp(argv[5]);
	c.csp_out   = parse_csp(argv[6]);
	c.tv_in     = 0;
	c.tv_out    = 0;
	c.visualise = nullptr;
	c.times     = 1;
	c.pixtype   = PixelType::FLOAT;
	c.cpu       = CPUClass::CPU_NONE;

	parse_opts(argv + 7, argv + argc, std::begin(OPTIONS), std::end(OPTIONS), &c, nullptr);

	auto in = allocate_frame(c.width, c.height, 3, c.pixtype);
	auto out = allocate_frame(c.width, c.height, 3, c.pixtype);
	Bitmap bmp = c.visualise ? Bitmap{ c.width, c.height, false } : Bitmap{ 0, 0, false };

	read_frame(c.infile, in.data(), c.width, c.height, c.pixtype);

	colorspace::ColorspaceConversion conv{ c.csp_in, c.csp_out, c.cpu };

	execute(conv, in.data(), out.data(), c.visualise ? &bmp : nullptr, c.width, c.height, !!c.tv_in, !!c.tv_out, c.times, c.pixtype);
	write_frame(c.outfile, out.data(), c.width, c.height, c.pixtype);

	if (c.visualise)
		write_bitmap(bmp, c.visualise);

	return 0;
}
