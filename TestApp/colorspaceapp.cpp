#include <cstddef>
#include <cstdio>
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
#include "Colorspace/colorspace2.h"
#include "Colorspace/colorspace_param.h"
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
	colorspace::ColorspaceDefinition csp_in;
	colorspace::ColorspaceDefinition csp_out;
	int fullrange_in;
	int fullrange_out;
	const char *visualise;
	int times;
	CPUClass cpu;
	PixelType filetype;
};

const AppOption OPTIONS[] = {
	{ "tv-in",     OptionType::OPTION_FALSE,     offsetof(AppContext, fullrange_in) },
	{ "pc-in",     OptionType::OPTION_TRUE,      offsetof(AppContext, fullrange_in) },
	{ "tv-out",    OptionType::OPTION_FALSE,     offsetof(AppContext, fullrange_out) },
	{ "pc-out",    OptionType::OPTION_TRUE,      offsetof(AppContext, fullrange_out) },
	{ "visualise", OptionType::OPTION_STRING,    offsetof(AppContext, visualise) },
	{ "times",     OptionType::OPTION_INTEGER,   offsetof(AppContext, times) },
	{ "cpu",       OptionType::OPTION_CPUCLASS,  offsetof(AppContext, cpu) },
    { "filetype",  OptionType::OPTION_PIXELTYPE, offsetof(AppContext, filetype) },
};

void usage()
{
	std::cout << "colorspace infile outfile w h csp_in csp_out [--tv-in | --pc-in] [--tv-out | --pc-out] [--visualise path] [--times n] [--cpu cpu] [--filetype type]\n";
	std::cout << "    infile               input file\n";
	std::cout << "    outfile              output file\n";
	std::cout << "    w                    image width\n";
	std::cout << "    h                    image height\n";
	std::cout << "    csp_in               input colorspace\n";
	std::cout << "    csp_out              output colorspace\n";
	std::cout << "    --tv-in | --pc-in    toggle TV vs PC range for input\n";
	std::cout << "    --tv-out | --pc-out  toggle TV vs PC range for output\n";
	std::cout << "    --visualise          path to BMP file for visualisation\n";
	std::cout << "    --times              number of cycles\n";
	std::cout << "    --cpu                select CPU type\n";
	std::cout << "    --filetype           pixel format of input/output files\n";
}

colorspace::MatrixCoefficients parse_matrix(const char *matrix)
{
	static const static_string_map<colorspace::MatrixCoefficients, 5> map{
		{ "rgb",      colorspace::MatrixCoefficients::MATRIX_RGB },
		{ "601",      colorspace::MatrixCoefficients::MATRIX_601 },
		{ "709",      colorspace::MatrixCoefficients::MATRIX_709 },
		{ "2020_ncl", colorspace::MatrixCoefficients::MATRIX_2020_NCL },
		{ "2020_cl",  colorspace::MatrixCoefficients::MATRIX_2020_CL },
	};
	auto it = map.find(matrix);
	return it == map.end() ? throw std::invalid_argument{ "bad matrix coefficients" } : it->second;
}

colorspace::TransferCharacteristics parse_transfer(const char *transfer)
{
	static const static_string_map<colorspace::TransferCharacteristics, 2> map{
		{ "linear", colorspace::TransferCharacteristics::TRANSFER_LINEAR },
		{ "709",    colorspace::TransferCharacteristics::TRANSFER_709 },
	};
	auto it = map.find(transfer);
	return it == map.end() ? throw std::invalid_argument{ "bad transfer characteristics" } : it->second;
}

colorspace::ColorPrimaries parse_primaries(const char *primaries)
{
	static const static_string_map<colorspace::ColorPrimaries, 3> map{
		{ "smpte_c", colorspace::ColorPrimaries::PRIMARIES_SMPTE_C },
		{ "709",     colorspace::ColorPrimaries::PRIMARIES_709 },
		{ "2020",    colorspace::ColorPrimaries::PRIMARIES_2020 }
	};
	auto it = map.find(primaries);
	return it == map.end() ? throw std::invalid_argument{ "bad primaries" } : it->second;
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

void execute(const colorspace::ColorspaceConversion2 &conv, const Frame &in, Frame &out, int times,
             bool fullrange_in, bool fullrange_out, bool yuv_in, bool yuv_out, PixelType filetype)
{
	int width = in.width();
	int height = in.height();

	Frame in_conv{ width, height, pixel_size(PixelType::FLOAT), 3 };
	Frame out_conv{ width, height, pixel_size(PixelType::FLOAT), 3 };

	convert_frame(in, in_conv, filetype, PixelType::FLOAT, fullrange_in, yuv_in);

	auto tmp = alloc_filter_tmp(conv, in_conv, out_conv);

	measure_time(times, [&]()
	{
		apply_filter(conv, in_conv, out_conv, tmp.data(), 0);
	});

	convert_frame(out_conv, out, PixelType::FLOAT, filetype, fullrange_out, yuv_out);
}

} // namespace


int colorspace_main(int argc, const char **argv)
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
	c.csp_in        = parse_csp(argv[5]);
	c.csp_out       = parse_csp(argv[6]);
	c.fullrange_in  = 0;
	c.fullrange_out = 0;
	c.visualise     = nullptr;
	c.times         = 1;
	c.filetype      = PixelType::FLOAT;
	c.cpu           = CPUClass::CPU_NONE;

	parse_opts(argv + 7, argv + argc, std::begin(OPTIONS), std::end(OPTIONS), &c, nullptr);

	int width = c.width;
	int height = c.height;
	int pxsize = pixel_size(c.filetype);

	bool yuv_in = c.csp_in.matrix != colorspace::MatrixCoefficients::MATRIX_RGB;
	bool yuv_out = c.csp_out.matrix != colorspace::MatrixCoefficients::MATRIX_RGB;

	Frame in{ width, height, pxsize, 3 };
	Frame out{ width, height, pxsize, 3 };

	read_frame_raw(in, c.infile);

	colorspace::ColorspaceConversion2 conv{ c.csp_in, c.csp_out, c.cpu };
	execute(conv, in, out, c.times, !!c.fullrange_in, !!c.fullrange_out, yuv_in, yuv_out, c.filetype);

	write_frame_raw(out, c.outfile);

	if (c.visualise) {
		Frame bmp{ width, height, 1, 3 };

		convert_frame(out, bmp, c.filetype, PixelType::BYTE, !!c.fullrange_out, yuv_out);
		write_frame_bmp(bmp, c.visualise);
	}

	return 0;
}
