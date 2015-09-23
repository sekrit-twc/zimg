#if 0
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include "frame.h"

// From Windows.h
#define DWORD uint32_t
#define WORD  uint16_t
#define LONG  int32_t

#define BMP_MAGIC 0x4d42

#pragma pack(push, 2)
typedef struct tagBITMAPFILEHEADER {
	WORD  bfType;
	DWORD bfSize;
	WORD  bfReserved1;
	WORD  bfReserved2;
	DWORD bfOffBits;
} BITMAPFILEHEADER, *PBITMAPFILEHEADER;

typedef struct tagBITMAPINFOHEADER {
	DWORD biSize;
	LONG  biWidth;
	LONG  biHeight;
	WORD  biPlanes;
	WORD  biBitCount;
	DWORD biCompression;
	DWORD biSizeImage;
	LONG  biXPelsPerMeter;
	LONG  biYPelsPerMeter;
	DWORD biClrUsed;
	DWORD biClrImportant;
} BITMAPINFOHEADER, *PBITMAPINFOHEADER;
#pragma pack(pop)


using namespace zimg;

namespace {;

struct FileCloser {
	void operator()(void *p) { fclose((FILE *)p); }
};

} // namespace


Frame::Frame(int width, int height, int pxsize, int planes) :
	m_width{ width },
	m_height{ height },
	m_pxsize{ pxsize },
	m_stride{ align(width, ALIGNMENT / pxsize) },
	m_planes{ planes }
{
	size_t plane_sz = (size_t)m_stride * height * pxsize;

	for (int p = 0; p < planes; ++p) {
		m_data[p].resize(plane_sz);
	}
}

int Frame::width() const
{
	return m_width;
}

int Frame::height() const
{
	return m_height;
}

int Frame::pxsize() const
{
	return m_pxsize;
}

int Frame::stride() const
{
	return m_stride;
}

int Frame::planes() const
{
	return m_planes;
}

unsigned char *Frame::data(int plane)
{
	return m_data[plane].data();
}

const unsigned char *Frame::data(int plane) const
{
	return const_cast<Frame *>(this)->data(plane);
}

unsigned char *Frame::row_ptr(int plane, int row)
{
	return m_data[plane].data() + (ptrdiff_t)m_pxsize * m_stride * row;
}

const unsigned char *Frame::row_ptr(int plane, int row) const
{
	return const_cast<Frame *>(this)->row_ptr(plane, row);
}


Frame read_frame_bmp(const char *filename)
{
	std::unique_ptr<FILE, FileCloser> handle{ fopen(filename, "rb") };
	FILE *file = handle.get();

	if (!file)
		throw std::runtime_error{ "error opening file" };

	BITMAPFILEHEADER bfheader;
	BITMAPINFOHEADER biheader;

	if (fread(&bfheader, sizeof(BITMAPFILEHEADER), 1, file) != 1)
		throw std::runtime_error{ "error reading BITMAPFILEHEADER" };
	if (fread(&biheader, sizeof(BITMAPINFOHEADER), 1, file) != 1)
		throw std::runtime_error{ "error reading BITMAPINFOHEADER" };

	if (bfheader.bfType != BMP_MAGIC)
		throw std::runtime_error{ "incorrect bitmap magic bytes" };
	if (biheader.biBitCount != 24 && biheader.biBitCount != 32)
		throw std::runtime_error{ "unsupported bit depth" };
	if (biheader.biCompression)
		throw std::runtime_error{ "unsupported compression" };

	if (fseek(file, bfheader.bfOffBits, SEEK_SET))
		throw std::runtime_error{ "error seeking to bitmap data" };

	int width = biheader.biWidth;
	int height = biheader.biHeight;
	int channels = biheader.biBitCount == 32 ? 4 : 3;
	size_t bmp_rowsize = align(width * channels, 4);

	Frame frame{ width, height, 1, channels };
	AlignedVector<uint8_t> buf(bmp_rowsize);

	for (int i = height; i > 0; --i) {
		if (fread(buf.data(), 1, bmp_rowsize, file) != bmp_rowsize)
			throw std::runtime_error{ "error reading bitmap data" };

		for (int j = 0; j < width; ++j) {
			for (int p = 0; p < channels; ++p) {
				frame.row_ptr(p, i - 1)[j] = buf[j * channels + p];
			}
		}
	}

	return frame;
}

void read_frame_raw(Frame &frame, const char *filename)
{
	std::unique_ptr<FILE, FileCloser> handle{ fopen(filename, "rb") };
	FILE *file = handle.get();
	size_t width = frame.width();

	if (!file)
		throw std::runtime_error{ "error opening file" };

	for (int p = 0; p < frame.planes(); ++p) {
		for (int i = 0; i < frame.height(); ++i) {
			void *dst = frame.row_ptr(p, i);

			if (fread(dst, frame.pxsize(), width, file) != width)
				throw std::runtime_error{ "error reading frame" };
		}
	}
}

void write_frame_bmp(const Frame &frame, const char *filename)
{
	if (frame.pxsize() != 1)
		throw std::runtime_error{ "DIB files must be 8-bit" };
	if (frame.planes() != 3 && frame.planes() != 4)
		throw std::runtime_error{ "DIB files must be 3 or 4 planes" };

	std::unique_ptr<FILE, FileCloser> handle{ fopen(filename, "wb") };
	FILE *file = handle.get();

	if (!file)
		throw std::runtime_error{ "error opening file" };

	BITMAPFILEHEADER bfheader{};
	BITMAPINFOHEADER biheader{};

	DWORD bmp_rowsize = align(frame.width() * frame.planes(), 4);
	DWORD bmp_datasize = bmp_rowsize * frame.height();

	bfheader.bfType = BMP_MAGIC;
	bfheader.bfSize = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + bmp_datasize;
	bfheader.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);

	biheader.biSize = sizeof(BITMAPINFOHEADER);
	biheader.biWidth = frame.width();
	biheader.biHeight = frame.height();
	biheader.biPlanes = 1;
	biheader.biBitCount = frame.planes() * 8;

	if (fwrite(&bfheader, sizeof(BITMAPFILEHEADER), 1, file) != 1)
		throw std::runtime_error{ "error writing BITMAPFILEHEADER" };
	if (fwrite(&biheader, sizeof(BITMAPINFOHEADER), 1, file) != 1)
		throw std::runtime_error{ "error writing BITMAPINFOHEADER" };

	std::unique_ptr<uint8_t[]> buf{ new uint8_t[bmp_rowsize] };
	for (int i = frame.height(); i > 0; --i) {
		for (int j = 0; j < frame.width(); ++j) {
			buf[j * frame.planes() + 0] = frame.row_ptr(0, i - 1)[j];
			buf[j * frame.planes() + 1] = frame.row_ptr(1, i - 1)[j];
			buf[j * frame.planes() + 2] = frame.row_ptr(2, i - 1)[j];

			if (frame.planes() == 4)
				buf[j * frame.planes() + 3] = frame.row_ptr(3, i - 1)[j];
		}

		if (fwrite(buf.get(), 1, bmp_rowsize, file) != bmp_rowsize)
			throw std::runtime_error{ "error writing bitmap data" };
	}
}

void write_frame_raw(const Frame &frame, const char *filename)
{
	std::unique_ptr<FILE, FileCloser> handle{ fopen(filename, "wb") };
	FILE *file = handle.get();
	size_t width = frame.width();

	if (!file)
		throw std::runtime_error{ "error opening file" };

	for (int p = 0; p < frame.planes(); ++p) {
		for (int i = 0; i < frame.height(); ++i) {
			const void *src = frame.row_ptr(p, i);

			if (fwrite(src, frame.pxsize(), width, file) != width)
				throw std::runtime_error{ "error writing frame" };
		}
	}
}
#endif
