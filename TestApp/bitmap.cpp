#include <cstdio>
#include <memory>
#include <stdexcept>
#include <Windows.h>
#include "Common/align.h"
#include "bitmap.h"

using namespace zimg;

namespace {;

// STL deleter for FILE pointers.
struct FileCloser {
	void operator()(void *p) { fclose((FILE *)p); }
};

// std::unique_ptr specialization for FILE pointers.
typedef std::unique_ptr<FILE, FileCloser> PFILE;

} // namespace


Bitmap::Bitmap(int width, int height, bool four_planes)
: m_planes(four_planes ? 4 : 3), m_stride(align(width, ALIGNMENT)), m_width(width), m_height(height)
{
	for (int i = 0; i < m_planes; ++i) {
		m_data[i].resize(m_stride * m_height);
	}
}

int Bitmap::planes() const
{
	return m_planes;
}

int Bitmap::stride() const
{
	return m_stride;
}

int Bitmap::width() const
{
	return m_width;
}

int Bitmap::height() const
{
	return m_height;
}

uint8_t *Bitmap::data(int plane)
{
	return m_data[plane].data();
}

const uint8_t *Bitmap::data(int plane) const
{
	return m_data[plane].data();
}


Bitmap read_bitmap(const char *filename)
{
	PFILE handle(fopen(filename, "rb"));
	FILE *f = handle.get();

	if (!f)
		throw std::runtime_error("error opening file");

	BITMAPFILEHEADER bfheader;
	BITMAPINFOHEADER biheader;

	if (fread(&bfheader, sizeof(BITMAPFILEHEADER), 1, f) != 1)
		throw std::runtime_error("error reading bitmap file header");
	if (fread(&biheader, sizeof(BITMAPINFOHEADER), 1, f) != 1)
		throw std::runtime_error("error reading bitmap info header");

	if (bfheader.bfType != 'MB')
		throw std::runtime_error("incorrect bitmap magic bytes");
	if (biheader.biBitCount != 24 && biheader.biBitCount != 32)
		throw std::runtime_error("unsupported bit depth");
	if (biheader.biCompression)
		throw std::runtime_error("unsupported compression");

	if (fseek(f, bfheader.bfOffBits, SEEK_SET))
		throw std::runtime_error("error seeking to bitmap data");

	int width = biheader.biWidth;
	int height = biheader.biHeight;
	int planes = biheader.biBitCount / 8;
	int bmp_stride = align(width * planes, 4);

	Bitmap bmp(width, height, planes == 4);
	std::unique_ptr<uint8_t[]> buf(new uint8_t[bmp_stride]);

	for (int i = height - 1; i>= 0; --i) {
		if (fread(buf.get(), 1, bmp_stride, f) != bmp_stride)
			throw std::runtime_error("error reading bitmap data");

		for (int j = 0; j < width; ++j) {
			for (int k = 0; k < planes; ++k) {
				bmp.data(k)[bmp.stride() * i + j] = buf[j * planes + k];
			}
		}
	}

	return bmp;
}

void write_bitmap(const Bitmap &bmp, const char *filename)
{
	PFILE handle(fopen(filename, "wb"));
	FILE *f = handle.get();

	if (!f)
		throw std::runtime_error("error opening file");

	BITMAPFILEHEADER bfheader = { 0 };
	BITMAPINFOHEADER biheader = { 0 };

	int planes = bmp.planes();
	int stride = bmp.stride();
	int width = bmp.width();
	int height = bmp.height();

	int bmp_stride = align(width * planes, 4);
	int bmp_data_size = bmp_stride *height;

	bfheader.bfType = 'MB';
	bfheader.bfSize = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + bmp_data_size;
	bfheader.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);

	biheader.biSize = sizeof(BITMAPINFOHEADER);
	biheader.biWidth = bmp.width();
	biheader.biHeight = bmp.height();
	biheader.biPlanes = 1;
	biheader.biBitCount = bmp.planes() * 8;

	if (fwrite(&bfheader, sizeof(BITMAPFILEHEADER), 1, f) != 1)
		throw std::runtime_error("error writing bitmap file header");
	if (fwrite(&biheader, sizeof(BITMAPINFOHEADER), 1, f) != 1)
		throw std::runtime_error("error writing bitmap info header");

	std::unique_ptr<uint8_t[]> buf(new uint8_t[bmp_stride]());
	for (int i = height - 1; i >= 0; --i) {
		for (int j = 0; j < width; ++j) {
			for (int k = 0; k < planes; ++k) {
				buf[j * planes + k] = bmp.data(k)[i * stride + j];
			}
		}
		if (fwrite(buf.get(), 1, bmp_stride, f) != bmp_stride)
			throw std::runtime_error("error writing bitmap data");
	}
}
