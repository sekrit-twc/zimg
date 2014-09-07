#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include "align.h"
#include "bitmap.h"
#include "filter.h"
#include "resize.h"
#include "timer.h"

using namespace resize;

namespace {;

Filter *select_filter(const char *filter)
{
	if (!strcmp(filter, "point"))
		return new PointFilter{};
	else if (!strcmp(filter, "bilinear"))
		return new BilinearFilter{};
	else if (!strcmp(filter, "bicubic"))
		return new BicubicFilter(1.0 / 3.0, 1.0 / 3.0);
	else if (!strcmp(filter, "lanczos"))
		return new LanczosFilter{ 4 };
	else if (!strcmp(filter, "spline16"))
		return new Spline16Filter{};
	else if (!strcmp(filter, "spline36"))
		return new Spline36Filter{};
	else
		return new BilinearFilter{};
}

void load_plane_u8_u16(const uint8_t *src, uint16_t *dst, int width, int height, int src_stride, int dst_stride)
{
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			dst[i * dst_stride + j] = (uint16_t)(src[i * src_stride + j] << 8);
		}
	}
}

void load_plane_u8_f32(const uint8_t *src, float *dst, int width, int height, int src_stride, int dst_stride)
{
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			dst[i * dst_stride + j] = (float)src[i * src_stride + j] / (float)UINT8_MAX;
		}
	}
}

void store_plane_u16_u8(const uint16_t *src, uint8_t *dst, int width, int height, int src_stride, int dst_stride)
{
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			dst[i * dst_stride + j] = (uint8_t)(src[i * src_stride + j] >> 8);
		}
	}
}

void store_plane_f32_u8(const float *src, uint8_t *dst, int width, int height, int src_stride, int dst_stride)
{
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			float x = std::round(src[i * src_stride + j] * (float)UINT8_MAX);
			dst[i * dst_stride + j] = (uint8_t)std::min(std::max(x, 0.f), (float)UINT8_MAX);
		}
	}
}

void usage()
{
	std::cout << "TestApp infile outfile w h [--filter filter] [--shift-w shift] [--shift-h shift] [--sub-w w] [--sub-h h] [--times n] [--u16 | --f32] [--x86 | --no-x86]\n";
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
	std::cout << "    --u16               use 16-bit unsigned\n";
	std::cout << "    --f32               use single precision float\n";
	std::cout << "    --x86 / --no-x86    toggle x86 optimizations\n";
}

void execute_u16(const Resize &resize, const Bitmap &in, Bitmap &out, int times, bool x86)
{
	int src_width = in.width();
	int src_stride = align(in.width(), 16);
	int src_height = in.height();
	int src_plane_size = src_stride * src_height;
	int dst_width = out.width();
	int dst_stride = align(out.width(), 16);
	int dst_height = out.height();
	int dst_plane_size = dst_stride * dst_height;
	int planes = in.planes();

	AlignedVector<uint16_t> in_planes(src_stride * src_height * planes);
	AlignedVector<uint16_t> out_planes(dst_stride * dst_height * planes);
	AlignedVector<uint16_t> tmp(resize.tmp_size(PixelType::WORD));

	for (int p = 0; p < planes; ++p) {
		load_plane_u8_u16(in.data(p), in_planes.data() + p * src_plane_size, src_width, src_height, in.stride(), src_stride);
	}

	Timer timer;
	double min_time = INFINITY;
	double avg_time = 0.0;

	for (int n = 0; n < times; ++n) {
		timer.start();
		for (int p = 0; p < planes; ++p) {
			const uint16_t *src = in_planes.data() + p * src_plane_size;
			uint16_t *dst = out_planes.data() + p * dst_plane_size;

			resize.process_u16(src, dst, tmp.data(), src_stride, dst_stride);
		}
		timer.stop();

		double elapsed = timer.elapsed();

		std::cout << '#' << n << ": " << elapsed << '\n';
		avg_time += elapsed / times;
		min_time = std::min(min_time, elapsed);
	}
	std::cout << "average: " << avg_time << '\n';
	std::cout << "min: " << min_time << '\n';

	for (int p = 0; p < planes; ++p) {
		store_plane_u16_u8(out_planes.data() + p * dst_plane_size, out.data(p), dst_width, dst_height, dst_stride, out.stride());
	}
}

void execute_f32(const Resize &resize, const Bitmap &in, Bitmap &out, int times, bool x86)
{
	int src_width = in.width();
	int src_stride = align(in.width(), 8);
	int src_height = in.height();
	int src_plane_size = src_stride * src_height;
	int dst_width = out.width();
	int dst_stride = align(out.width(), 8);
	int dst_height = out.height();
	int dst_plane_size = dst_stride * dst_height;
	int planes = in.planes();

	AlignedVector<float> in_planes(src_stride * src_height * planes);
	AlignedVector<float> out_planes(dst_stride * dst_height * planes);
	AlignedVector<float> tmp(resize.tmp_size(PixelType::WORD));

	for (int p = 0; p < planes; ++p) {
		load_plane_u8_f32(in.data(p), in_planes.data() + p * src_plane_size, src_width, src_height, in.stride(), src_stride);
	}

	Timer timer;
	double min_time = INFINITY;
	double avg_time = 0.0;

	for (int n = 0; n < times; ++n) {
		timer.start();
		for (int p = 0; p < planes; ++p) {
			const float *src = in_planes.data() + p * src_plane_size;
			float *dst = out_planes.data() + p * dst_plane_size;

			resize.process_f32(src, dst, tmp.data(), src_stride, dst_stride);
		}
		timer.stop();

		double elapsed = timer.elapsed();

		std::cout << '#' << n << ": " << elapsed << '\n';
		avg_time += elapsed / times;
		min_time = std::min(min_time, elapsed);
	}
	std::cout << "average: " << avg_time << '\n';
	std::cout << "min: " << min_time << '\n';

	for (int p = 0; p < planes; ++p) {
		store_plane_f32_u8(out_planes.data() + p * dst_plane_size, out.data(p), dst_width, dst_height, dst_stride, out.stride());
	}
}

} // namespace


int main(int argc, const char **argv)
{
	if (argc < 5) {
		usage();
		return -1;
	}

	const char *ifile = argv[1];
	const char *ofile = argv[2];
	int width = std::atoi(argv[3]);
	int height = std::atoi(argv[4]);
	const char *filter_str = "bilinear";
	double shift_w = 0.0;
	double shift_h = 0.0;
	double sub_w = -1.0;
	double sub_h = -1.0;
	int times = 1;
	bool x86 = false;
	PixelType type = PixelType::FLOAT;

	for (int i = 5; i < argc; ++i) {
		if (!strcmp(argv[i], "--filter") && i + 1 < argc) {
			filter_str = argv[i + 1];
			++i;
		} else if (!strcmp(argv[i], "--shift-w") && i + 1 < argc) {
			shift_w = std::atof(argv[i + 1]);
			++i;
		} else if (!strcmp(argv[i], "--shift-h") && i + 1 < argc) {
			shift_h = std::atof(argv[i + 1]);
			++i;
		} else if (!strcmp(argv[i], "--sub-w") && i + 1 < argc) {
			sub_w = std::atof(argv[i + 1]);
			++i;
		} else if (!strcmp(argv[i], "--sub-h") && i + 1 < argc) {
			sub_h = std::atof(argv[i + 1]);
			++i;
		} else if (!strcmp(argv[i], "--times") && i + 1 < argc) {
			times = std::atoi(argv[i + 1]);
			++i;
		} else if (!strcmp(argv[i], "--u16")) {
			type = PixelType::WORD;
		} else if (!strcmp(argv[i], "--f32")) {
			type = PixelType::FLOAT;
		} else if (!strcmp(argv[i], "--x86")) {
			x86 = true;
		} else if (!strcmp(argv[i], "--no-x86")) {
			x86 = false;
		} else {
			std::cerr << "unknown argument: " << argv[i] << '\n';
		}
	}

	try {
		Bitmap in = read_bitmap(ifile);
		Bitmap out{ width, height, in.planes() == 4 };

		if (sub_w < 0.0)
			sub_w = in.width();
		if (sub_h < 0.0)
			sub_h = in.height();

		std::unique_ptr<Filter> filter{ select_filter(filter_str) };
		Resize resize{ *filter, in.width(), in.height(), width, height, shift_w, shift_h, sub_w, sub_h, x86 };

		if (type == PixelType::WORD)
			execute_u16(resize, in, out, times, x86);
		else if (type == PixelType::FLOAT)
			execute_f32(resize, in, out, times, x86);
		else
			throw std::runtime_error{ "unrecognized pixel type" };

		write_bitmap(out, ofile);
	} catch (std::exception &e) {
		std::cerr << e.what() << '\n';
		return -1;
	}

	return 0;
}
