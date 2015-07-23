#pragma once

#ifndef ZIMG_LINEBUFFER_H_
#define ZIMG_LINEBUFFER_H_

#include <algorithm>

namespace zimg {;

/**
 * Buffer interface providing a rolling window of k rows.
 * Accessing a a row (n) by index yields the (n & k)-th row in the buffer.
 * This grants effective access to a power of two number of rows.
 *
 * @param T data type contained in buffer
 */
template <class T>
class LineBuffer {
	void *m_ptr; // Must be void pointer to allow casting to the template parameter T.
	unsigned m_width;
	unsigned m_byte_stride;
	unsigned m_mask;

	const T *at_line(unsigned n) const
	{
		const char *byte_ptr = reinterpret_cast<const char *>(m_ptr);

		return reinterpret_cast<const T *>(byte_ptr + (size_t)(n & m_mask) * m_byte_stride);
	}
public:
	/**
	 * Default initialize LineBuffer.
	 */
	LineBuffer() = default;

	/**
	 * Initialize a LineBuffer with a given buffer.
	 *
	 * @param ptr pointer to buffer
	 * @param width width of line in pixels
	 * @param byte_stride distance between lines in bytes
	 * @param mask bit mask applied to row index
	 */
	LineBuffer(T *ptr, unsigned width, unsigned byte_stride, unsigned mask) :
		m_ptr{ ptr }, m_width{ width }, m_byte_stride{ byte_stride }, m_mask{ mask }
	{}

	/**
	 * Get width of line in pixels.
	 *
	 * @return width
	 */
	unsigned width() const
	{
		return m_width;
	}

	/**
	 * Get a pointer to a buffer row.
	 *
	 * @param n row index
	 * @return pointer to row
	 */
	T *operator[](unsigned n)
	{
		return const_cast<T *>(at_line(n));
	}

	/**
	 * @see LineBuffer::operator[](unsigned)
	 */
	const T *operator[](unsigned n) const
	{
		return at_line(n);
	}
};

/**
 * Cast a LineBuffer to another data type.
 * The returned reference points to the original buffer object.
 *
 * @param T new data type
 * @param U old data type
 * @param x original buffer
 * @return original buffer with new data type
 */
template <class T, class U>
LineBuffer<T> &buffer_cast(LineBuffer<U> &x)
{
	return reinterpret_cast<LineBuffer<T> &>(x);
}

/**
 * @see buffer_cast<T,U>(LineBuffer<U> &)
 */
template <class T, class U>
const LineBuffer<T> &buffer_cast(const LineBuffer<U> &x)
{
	return reinterpret_cast<const LineBuffer<T> &>(x);
}

/**
 * Copy a sequence of lines between buffers.
 *
 * @param T data type
 * @param src input buffer
 * @param dst output buffer
 * @param bytes number of bytes per line
 * @param first index of top line
 * @param last index of bottom line
 */
template <class T>
void copy_buffer_lines(const LineBuffer<T> &src, LineBuffer<T> &dst, unsigned bytes, unsigned first, unsigned last)
{
	for (unsigned n = first; n < last; ++n) {
		std::copy_n((const char *)src[n], bytes, (char *)dst[n]);
	}
}

} // namespace zimg

#endif // ZIMG_LINEBUFFER_H_
