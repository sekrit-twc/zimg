#ifndef BITMAP_H_
#define BITMAP_H_

#include <cstdint>
#include <vector>

/**
 * Container for unsigned 8-bit 4:4:4 planar image data with 3 or 4 channels.
 */
class Bitmap {
	std::vector<uint8_t> m_data[4];

	int m_planes;
	int m_stride;
	int m_width;
	int m_height;
public:
	/*
	 * Construct a bitmap with given dimensions.
	 * The bitmap contents are zero-initialized.
	 *
	 * @param width image width
	 * @param height image height
	 * @param four_planes allocate 4 planes instead of 3
	 */
	Bitmap(int width, int height, bool four_planes);

	/**
	 * @return the number of planes
	 */
	int planes() const;

	/**
	 * Return the image stride. Stride is the distance between adjacent rows.
	 * All planes have the same stride, which will be a multiple of 32 bytes.
	 *
	 * @return the image stride
	 */
	int stride() const;

	/**
	 * @return the image width
	 */
	int width() const;

	/**
	 * @return the image height
	 */
	int height() const;

	/**
	 * @param plane index of plane
	 * @return a pointer to the top-most scanline of plane
	 */
	uint8_t *data(int plane);
	
	/**
	 * @see Bitmap::data(int)
	 */
	const uint8_t *data(int plane) const;
};

/**
 * Read a Windows DIB (.bmp) into a Bitmap object.
 * Only uncompressed 24 or 32 bpp bitmaps are supported.
 *
 * @param filename name of DIB file
 */
Bitmap read_bitmap(const char *filename);

/**
 * Write a Bitmap object to a Windows DIB (.bmp).
 * The DIB will use a v3 BITMAPINFOHEADER.
 *
 * @param bmp Bitmap to write
 * @param filename name of DIB file
 */
void write_bitmap(const Bitmap &bmp, const char *filename);

#endif // BITMAP_H_
