#include <cstdlib>
#include <cstring>
#include <iostream>
#include "Unresize/unresize.h"
#include "apps.h"
#include "bitmap.h"

using namespace zimg;

namespace {;

void process(const Bitmap &in, Bitmap &out, float shift_w, float shift_h, int times, bool x86)
{
	unresize::Unresize u(in.width(), in.height(), out.width(), out.height(), shift_w, shift_h, x86);

	int src_stride = width_to_stride(in.width(), PixelType::FLOAT);
	int dst_stride = width_to_stride(out.width(), PixelType::FLOAT);
	int src_plane_size = (int)image_plane_size(src_stride, in.height(), PixelType::FLOAT);
	int dst_plane_size = (int)image_plane_size(dst_stride, out.height(), PixelType::FLOAT);

	auto src_planes = allocate_frame(src_stride, in.height(), 3, PixelType::FLOAT);
	auto dst_planes = allocate_frame(dst_stride, out.height(), 3, PixelType::FLOAT);
	auto tmp = allocate_buffer(u.tmp_size(PixelType::FLOAT), PixelType::FLOAT);

	for (int p = 0; p < 3; ++p) {
		convert_from_byte(PixelType::FLOAT, in.data(p), src_planes.data() + src_plane_size * p, in.width(), in.height(), in.stride(), src_stride);
	}

	measure_time(times, [&]()
	{
		for (int p = 0; p < 3; ++p) {
			const uint8_t *src_p = (const uint8_t *)(src_planes.data() + p * src_plane_size);
			uint8_t *dst_p = (uint8_t *)(dst_planes.data() + p * dst_plane_size);

			u.process(PixelType::FLOAT, src_p, dst_p, tmp.data(), src_stride, dst_stride);
		}
	});

	for (int p = 0; p < 3; ++p) {
		convert_to_byte(PixelType::FLOAT, dst_planes.data() + dst_plane_size * p, out.data(p), out.width(), out.height(), dst_stride, out.stride());
	}
}

void usage()
{
	std::cerr << "unresize infile outfile width height [--shift-w shift] [--shift-h shift] [--times n] [--x86 | --no-x86]\n";
	std::cerr << "    infile              input BMP file\n";
	std::cerr << "    outfile             output BMP file\n";
	std::cerr << "    width               output width\n";
	std::cerr << "    height              output height\n";
	std::cerr << "    --shift-w           horizontal shift\n";
	std::cerr << "    --shift-h           vertical shift\n";
	std::cerr << "    --times             number of cycles\n";
	std::cerr << "    --x86 / --no-x86    toggle x86 optimizations\n";
}

} // namespace


int unresize_main(int argc, const char **argv)
{
	if (argc < 5) {
		usage();
		return -1;
	}

	const char *ifile = argv[1];
	const char *ofile = argv[2];
	int dst_width = atoi(argv[3]);
	int dst_height = atoi(argv[4]);

	float shift_w = 0;
	float shift_h = 0;
	int times = 1;
	bool x86 = false;

	for (int i = 5; i < argc; ++i) {
		if (!strcmp(argv[i], "--shift-w") && i + 1 < argc) {
			shift_w = (float)atof(argv[i + 1]);
			++i;
		} else if (!strcmp(argv[i], "--shift-h") && i + 1 < argc) {
			shift_h = (float)atof(argv[i + 1]);
			++i;
		} else if (!strcmp(argv[i], "--times") && i + 1 < argc) {
			times = atoi(argv[i + 1]);
			++i;
		} else if (!strcmp(argv[i], "--x86")) {
			x86 = true;
		} else if (!strcmp(argv[i], "--no-x86")) {
			x86 = false;
		} else {
			std::cerr << "unknown argument: " << argv[i] << '\n';
		}
	}

	Bitmap in = read_bitmap(ifile);
	Bitmap out(dst_width, dst_height, false);

	process(in, out, shift_w, shift_h, times, x86);
	write_bitmap(out, ofile);

	return 0;
}
