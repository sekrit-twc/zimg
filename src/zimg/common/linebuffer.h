#pragma once

#ifndef ZIMG_LINEBUFFER_H_
#define ZIMG_LINEBUFFER_H_

#include <algorithm>
#include <type_traits>
#include "graph/ztypes.h"

namespace zimg {;

template <class T, bool Value>
struct add_const_if_const;

template <class T>
struct add_const_if_const<T, true> {
	typedef const T type;
};

template <class T>
struct add_const_if_const<T, false> {
	typedef T type;
};

template <class T, class U>
struct propagate_const {
	typedef typename add_const_if_const<U, std::is_const<T>::value>::type type;
};

/**
 * Buffer interface providing a rolling window of k rows.
 * Accessing a a row (n) by index yields the (n & k)-th row in the buffer.
 * This grants effective access to a power of two number of rows.
 *
 * @param T data type contained in buffer
 */
template <class T>
class LineBuffer {
	// Must be void pointer to allow casting to the template parameter T.
	typename propagate_const<T, void>::type *m_ptr;
	ptrdiff_t m_stride;
	unsigned m_mask;

	const T *at_line(unsigned n) const
	{
		const char *byte_ptr = reinterpret_cast<const char *>(m_ptr);

		return reinterpret_cast<const T *>(byte_ptr + static_cast<ptrdiff_t>(n & m_mask) * m_stride);
	}
public:
	/**
	 * Default initialize LineBuffer.
	 */
	LineBuffer() = default;

	explicit LineBuffer(const graph::ZimgImageBufferTemplate<typename propagate_const<T, void>::type> &buffer, unsigned plane = 0) :
		m_ptr{ buffer.data[plane] },
		m_stride{ buffer.stride[plane] },
		m_mask{ buffer.mask[plane] }
	{
	}

	/**
	 * Initialize a LineBuffer with a given buffer.
	 *
	 * @param ptr pointer to buffer
	 * @param width width of line in pixels
	 * @param byte_stride distance between lines in bytes
	 * @param mask bit mask applied to row index
	 */
	LineBuffer(T *ptr, ptrdiff_t stride, unsigned mask) :
		m_ptr{ ptr },
		m_stride{ stride },
		m_mask{ mask }
	{
	}

	/**
	 * Implicit conversion to corresponding const LineBuffer.
	 */
	operator const LineBuffer<const T> &() const
	{
		return reinterpret_cast<LineBuffer<const T> &>(*this);
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
 * @param first_line index of top line
 * @param last_line index of bottom line
 * @param first_byte offset of first byte in line
 * @param last_byte offset of last byte in line
 */
template <class T>
void copy_buffer_lines(const LineBuffer<const T> &src, LineBuffer<T> &dst, unsigned first_line, unsigned last_line, unsigned first_byte, unsigned last_byte)
{
	for (unsigned n = first_line; n < last_line; ++n) {
		const char *src_p = reinterpret_cast<const char *>(src[n]);
		char *dst_p = reinterpret_cast<char *>(dst[n]);

		std::copy(src_p + first_byte, src_p + last_byte, dst_p + first_byte);
	}
}

} // namespace zimg

#endif // ZIMG_LINEBUFFER_H_
