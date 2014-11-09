#pragma once

#ifndef ZIMG_COMMON_PLANE_H_
#define ZIMG_COMMON_PLANE_H_

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include "pixel.h"

namespace zimg {;

/**
 * Wrapper for pointer to buffer containing image data.
 */
template <class T>
class ImagePlane {
	T *m_data;
	int m_width;
	int m_height;
	int m_byte_stride;
	PixelFormat m_format;
public:
	/**
	 * Default construct plane.
	 */
	ImagePlane() = default;

	/**
	 * If T is a const type, this is a conversion constructor for non-const to const plane.
	 * If T is a non-const type, this is a copy constructor.
	 *
	 * @param other plane to convert
	 */
	ImagePlane(const ImagePlane<typename std::remove_const<T>::type> &other) :
		ImagePlane(other.data(), other.width(), other.height(), other.stride(), other.format())
	{}

	/**
	 * Construct a plane using the default pixel format for a given pixel type.
	 *
	 * @param data pointer to image data
	 * @param width image width in pixels
	 * @param height image height in pixels
	 * @param stride image stride in pixels
	 * @param type pixel type of image
	 */
	ImagePlane(T *data, int width, int height, int stride, PixelType type) :
		ImagePlane(data, width, height, stride, default_pixel_format(type))
	{}

	/**
	 * Construct a plane using a given pixel format.
	 *
	 * @param data pointer to image data
	 * @param width image width in pixels
	 * @param height image height in pixels
	 * @param stride image stride in pixels
	 * @param format pixel format
	 */
	ImagePlane(T *data, int width, int height, int stride, const PixelFormat &format) :
		m_data{ data }, m_width{ width }, m_height{ height }, m_byte_stride{ stride * pixel_size(format.type) }, m_format(format)
	{}

	/**
	 * @return image width
	 */
	int width() const
	{
		return m_width;
	}

	/**
	 * @return image height
	 */
	int height() const
	{
		return m_height;
	}

	/**
	 * @return image stride
	 */
	int stride() const
	{
		return m_byte_stride / pixel_size(m_format.type);
	}

	/**
	 * @return pixel format
	 */
	const PixelFormat &format() const
	{
		return m_format;
	}

	/**
	 * @return pointer to image data
	 */
	T *data() const
	{
		return m_data;
	}

	T *operator[](size_t i) const
	{
		const char *byte_ptr = reinterpret_cast<const char *>(m_data);
		const char *line_ptr = byte_ptr + i  * m_byte_stride;

		return reinterpret_cast<T *>(const_cast<char *>(line_ptr));
	}
};

/**
 * Cast an ImagePlane to another pointer type.
 * Planes are convertible only if their pointer types are convertible.
 *
 * @param x plane to cast
 * @return converted plane
 */
template <class T, class U>
ImagePlane<T> plane_cast(const ImagePlane<U> &x)
{
	return{ static_cast<T *>(x.data()), x.width(), x.height(), x.stride(), x.format() };
}

/**
 * Create a copy of an image plane.
 * The planes must have identical dimensions and formats.
 *
 * @param src input plane
 * @param dst output plane
 */
template <class T>
inline void copy_image_plane(const ImagePlane<const T> &src, const ImagePlane<T> &dst)
{
	size_t byte_width = (size_t)src.width() * pixel_size(src.format().type);

	for (int i = 0; i < src.height(); ++i) {
		std::copy_n((const char *)src[i], byte_width, (char *)dst[i]);
	}
}

} // namespace zimg

#endif // ZIMG_COMMON_PLANE_H_
