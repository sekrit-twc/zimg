#pragma once

#ifndef FRAME_H_
#define FRAME_H_

#include <cstdint>
#include "Common/align.h"

/**
 * Container for planar byte-aligned image data with 3 or 4 channels.
 */
class Frame {
	zimg::AlignedVector<uint8_t> m_data[4];
	int m_width;
	int m_height;
	int m_pxsize;
	int m_stride;
	int m_planes;
public:
	/**
	 * Construct an empty frame.
	 */
	Frame() = default;

	/**
	 * Construct a frame with given dimensions.
	 * The planes are zero-initialized.
	 *
	 * @param width image width
	 * @param height image height
	 * @param pxsize bytes per pixel
	 * @param planes number of planes, up to 4
	 */
	Frame(int width, int height, int pxsize, int planes);

	/**
	 * @return image width
	 */
	int width() const;

	/**
	 * @return image height
	 */
	int height() const;

	/**
	 * @return bytes per pixel
	 */
	int pxsize() const;

	/**
	 * @return distance between lines in pixels
	 */
	int stride() const;

	/**
	 * @return number of planes
	 */
	int planes() const;

	/**
	 * Get a pointer to a given plane.
	 *
	 * @param plane number of plane
	 * @return pointer to plane
	 */
	unsigned char *data(int plane);

	/**
	 * @see Frame::data(int)
	 */
	const unsigned char *data(int plane) const;

	/**
	 * Get a pointer to a row in a plane.
	 *
	 * @param plane number of plane
	 * @param row row index, counting from the image top
	 * @return pointer to row
	 */
	unsigned char *row_ptr(int plane, int row);

	/**
	 * @see Frame::row_ptr(int, int)
	 */
	const unsigned char *row_ptr(int plane, int row) const;
};

/**
 * Read a frame from a Windows DIB (.bmp).
 *
 * @param filename name of DIB file
 * @return frame
 */
Frame read_frame_bmp(const char *filename);

/**
 * Read a frame from a raw file (.yuv).
 *
 * @param frame frame to receive image data
 * @param filename name of file
 */
void read_frame_raw(Frame &frame, const char *filename);

/**
 * Write a frame to a Windows DIB (.bmp).
 *
 * @param frame frame to write
 * @param filename name of DIB file
 */
void write_frame_bmp(const Frame &frame, const char *filename);

/**
 * Write a frame to a raw file (.yuv)
 *
 * @param frame frame to write
 * @param filename name of DIB file
 */
void write_frame_raw(const Frame &frame, const char *filename);

#endif // FRAME_H_
