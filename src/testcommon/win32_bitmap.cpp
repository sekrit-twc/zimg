#include <cstdint>
#include <stdexcept>
#include <utility>
#include "mmap.h"
#include "win32_bitmap.h"

namespace {

// From Windows.h
typedef uint32_t DWORD;
typedef uint16_t WORD;
typedef int32_t LONG;

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

typedef enum {
	BI_RGB = 0x0000,
	BI_RLE8 = 0x0001,
	BI_RLE4 = 0x0002,
	BI_BITFIELDS = 0x0003,
	BI_JPEG = 0x0004,
	BI_PNG = 0x0005,
	BI_CMYK = 0x000B,
	BI_CMYKRLE8 = 0x000C,
	BI_CMYKRLE4 = 0x000D
} Compression;


size_t bitmap_row_size(int width, int bit_count) noexcept
{
	size_t row_size;

	row_size = static_cast<size_t>(width) * (bit_count / 8);
	row_size = row_size % 4 ? row_size + 4 - row_size % 4 : row_size;

	return row_size;
}

size_t bitmap_data_size(int width, int height, int bit_count) noexcept
{
	return bitmap_row_size(width, bit_count) * height;
}

struct BitmapFileData {
	BITMAPFILEHEADER *bfHeader;
	BITMAPINFOHEADER *biHeader;
	void *image_data;

	BitmapFileData() noexcept : bfHeader{}, biHeader{}, image_data{} {}

	BitmapFileData(size_t file_size, void *file_base, bool new_image)
	{
		unsigned char *ptr = static_cast<unsigned char *>(file_base);

		if (file_size < sizeof(BITMAPINFOHEADER) + sizeof(BITMAPFILEHEADER))
			throw BitmapDataError{ "file too short" };

		bfHeader = reinterpret_cast<BITMAPFILEHEADER *>(ptr);
		biHeader = reinterpret_cast<BITMAPINFOHEADER *>(ptr + sizeof(BITMAPFILEHEADER));

		if (new_image) {
			image_data = ptr + sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);
		} else {
			if (bfHeader->bfOffBits > file_size || bfHeader->bfSize > file_size)
				throw BitmapDataError{ "file too short" };

			image_data = ptr + bfHeader->bfOffBits;
		}
	}

	void init(size_t file_size, int width, int height, int bit_count)
	{
		if (bitmap_data_size(width, height, bit_count) + sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) > file_size)
			throw BitmapDataError{ "file too short" };

		*bfHeader = BITMAPFILEHEADER{};

		bfHeader->bfType = BMP_MAGIC;
		bfHeader->bfSize = static_cast<DWORD>(file_size);
		bfHeader->bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);

		*biHeader = BITMAPINFOHEADER{};

		biHeader->biSize = sizeof(BITMAPINFOHEADER);
		biHeader->biWidth = width;
		biHeader->biHeight = height;
		biHeader->biPlanes = 1;
		biHeader->biBitCount = bit_count;
	}

	void validate(size_t file_size) const
	{
		if (bfHeader->bfType != BMP_MAGIC)
			throw BitmapDataError{ "invalid magic bytes" };
		if (biHeader->biWidth < 0 || biHeader->biHeight < 0)
			throw BitmapDataError{ "invalid bitmap dimensions" };
		if (biHeader->biBitCount != 24 && biHeader->biBitCount != 32)
			throw BitmapDataError{ "unsupported biBitCount" };
		if (biHeader->biCompression != BI_RGB && biHeader->biCompression != BI_BITFIELDS)
			throw BitmapDataError{ "unsupported biCompression" };
		if (bfHeader->bfOffBits + bitmap_data_size(biHeader->biWidth, biHeader->biHeight, biHeader->biBitCount) > file_size)
			throw BitmapDataError{ "file too short" };

		if (biHeader->biCompression == BI_BITFIELDS) {
			if (biHeader->biBitCount != 32)
				throw BitmapDataError{ "BI_BITFIELDS only supported with 32 bpp" };
			if (bfHeader->bfOffBits < sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + sizeof(DWORD) * 4)
				throw BitmapDataError{ "BI_BITFIELDS mask missing" };

			DWORD alpha_mask = reinterpret_cast<const DWORD *>(reinterpret_cast<const unsigned char *>(biHeader) + sizeof(BITMAPINFOHEADER))[3];
			if (alpha_mask != 0xFF000000U)
				throw BitmapDataError{ "BI_BITFIELDS only supported with alpha in MSB" };
		}
	}
};

} // namespace


const WindowsBitmap::read_tag WindowsBitmap::READ_TAG{};
const WindowsBitmap::write_tag WindowsBitmap::WRITE_TAG{};

class WindowsBitmap::impl {
	MemoryMappedFile m_mmap;
	BitmapFileData m_bitmap;
	bool m_writable;

	size_t row_size() const noexcept
	{
		size_t row_size = static_cast<size_t>(width()) * (bit_count() / 8);
		row_size = (row_size % 4) ? row_size + 4 - row_size % 4 : row_size;
		return row_size;
	}
public:
	impl(MemoryMappedFile &&mmap, bool writable) : m_writable{ writable }
	{
		if (writable && !mmap.write_ptr())
			throw std::logic_error{ "mapped file not writable" };

		m_bitmap = BitmapFileData{ mmap.size(), const_cast<void *>(mmap.read_ptr()), false };
		m_bitmap.validate(mmap.size());
		m_mmap = std::move(mmap);
	}

	impl(const char *path, int width, int height, int bit_count) : m_writable{ true }
	{
		size_t file_size = 0;

		if (width < 0 || height < 0)
			throw std::invalid_argument{ "invalid bitmap dimensions" };
		if (bit_count != 24 && bit_count != 32)
			throw std::invalid_argument{ "unsupported biBitCount" };

		file_size += sizeof(BITMAPFILEHEADER);
		file_size += sizeof(BITMAPINFOHEADER);
		file_size += height * bitmap_row_size(width, bit_count);

		m_mmap = MemoryMappedFile{ path, file_size, MemoryMappedFile::CREATE_TAG };

		m_bitmap = BitmapFileData{ m_mmap.size(), m_mmap.write_ptr(), true };
		m_bitmap.init(m_mmap.size(), width, height, bit_count);
	}

	const unsigned char *read_ptr() const noexcept
	{
		const unsigned char *image_data = static_cast<const unsigned char *>(m_bitmap.image_data);
		return image_data + row_size() * (height() - 1);
	}

	unsigned char *write_ptr() noexcept
	{
		return m_writable ? const_cast<unsigned char *>(read_ptr()) : nullptr;
	}

	ptrdiff_t stride() const noexcept { return -static_cast<ptrdiff_t>(row_size()); }

	int width() const noexcept { return m_bitmap.biHeader->biWidth; }
	int height() const noexcept { return m_bitmap.biHeader->biHeight; }
	int bit_count() const noexcept { return m_bitmap.biHeader->biBitCount; }

	void flush() { m_mmap.flush(); }
	void close() { m_mmap.close(); }
};


WindowsBitmap::WindowsBitmap(WindowsBitmap &&other) noexcept = default;

WindowsBitmap::WindowsBitmap(const char *path, read_tag)
{
	MemoryMappedFile mmap{ path, MemoryMappedFile::READ_TAG };
	std::unique_ptr<impl> impl_{ new impl{ std::move(mmap), false } };
	m_impl = std::move(impl_);
}

WindowsBitmap::WindowsBitmap(const char *path, write_tag)
{
	MemoryMappedFile mmap{ path, MemoryMappedFile::WRITE_TAG };
	std::unique_ptr<impl> impl_{ new impl{ std::move(mmap), false } };
	m_impl = std::move(impl_);
}

WindowsBitmap::WindowsBitmap(const char *path, int width, int height, int bit_count) :
	m_impl{ new impl{ path, width, height, bit_count} }
{}

WindowsBitmap::~WindowsBitmap() = default;

WindowsBitmap &WindowsBitmap::operator=(WindowsBitmap &&other) noexcept = default;

ptrdiff_t WindowsBitmap::stride() const noexcept { return get_impl()->stride(); }

int WindowsBitmap::width() const noexcept { return get_impl()->width(); }

int WindowsBitmap::height() const noexcept { return get_impl()->height(); }

int WindowsBitmap::bit_count() const noexcept { return get_impl()->bit_count(); }

const unsigned char *WindowsBitmap::read_ptr() const noexcept { return get_impl()->read_ptr(); }

unsigned char *WindowsBitmap::write_ptr() noexcept { return get_impl()->write_ptr(); }

void WindowsBitmap::flush() { get_impl()->flush(); }

void WindowsBitmap::close() { get_impl()->close(); }
