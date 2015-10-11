#pragma once

#ifndef ZIMG_GRAPH_IMAGE_BUFFER_H_
#define ZIMG_GRAPH_IMAGE_BUFFER_H_

#include <cstddef>
#include <limits>
#include <type_traits>
#include "common/propagate_const.h"

namespace zimg {;
namespace graph {;

const unsigned BUFFER_MAX = (unsigned)-1;

template <class T>
class ImageBuffer {
	typename propagate_const<T, void>::type *m_data;
	ptrdiff_t m_stride;
	unsigned m_mask;

	T *at_line(unsigned i) const
	{
		auto *byte_ptr = static_cast<typename propagate_const<T, char>::type *>(m_data);
		return reinterpret_cast<T *>(byte_ptr + static_cast<ptrdiff_t>(i & m_mask) * m_stride);
	}
public:
	ImageBuffer() = default;

	ImageBuffer(T *data, ptrdiff_t stride, unsigned mask) :
		m_data{ data },
		m_stride{ stride },
		m_mask{ mask }
	{
		static_assert(std::is_standard_layout<ImageBuffer>::value, "layout error");
	}

	template <class U>
	ImageBuffer(const ImageBuffer<U> &other, typename std::enable_if<std::is_convertible<U *, T *>::value>::type * = nullptr) :
		m_data{ other.data() },
		m_stride{ other.stride() },
		m_mask{ other.mask() }
	{
	}

	T *data() const
	{
		return at_line(0);
	}

	ptrdiff_t stride() const
	{
		return m_stride;
	}

	unsigned mask() const
	{
		return m_mask;
	}

	T *operator[](unsigned i) const
	{
		return at_line(i);
	}

	template <class U>
	const ImageBuffer<U> &static_buffer_cast() const
	{
		static_assert(std::is_standard_layout<decltype(static_cast<U *>(static_cast<T *>(m_data)))>::value,
		              "type not convertible by static_cast");

		return *reinterpret_cast<const ImageBuffer<U> *>(this);
	}
};

template <class T>
class ColorImageBuffer {
	ImageBuffer<T> m_buffer[3];
public:
	ColorImageBuffer() = default;

	ColorImageBuffer(const ImageBuffer<T> &buf1, const ImageBuffer<T> &buf2, const ImageBuffer<T> &buf3) :
		m_buffer{ buf1, buf2, buf3 }
	{
	}

	template <class U>
	ColorImageBuffer(const ColorImageBuffer<U> &other, typename std::enable_if<std::is_convertible<U *, T *>::value>::type * = nullptr) :
		m_buffer{ other[0], other[1], other[2] }
	{
	}

	operator const ImageBuffer<T> *() const
	{
		return m_buffer;
	}

	operator ImageBuffer<T> *()
	{
		return m_buffer;
	}

	template <class U>
	const ColorImageBuffer<U> &static_buffer_cast() const
	{
		static_assert(std::is_standard_layout<decltype(m_buffer->static_buffer_cast<U>())>::value,
		              "type not convertible by static_cast");

		return *reinterpret_cast<const ColorImageBuffer<U> *>(this);
	}
};

template <class U, class T>
const ImageBuffer<U> &static_buffer_cast(const ImageBuffer<T> &buf)
{
	return buf.template static_buffer_cast<U>();
}

template <class U, class T>
const ImageBuffer<U> *static_buffer_cast(const ImageBuffer<T> *buf)
{
	return &static_buffer_cast<U>(*buf);
}

template <class U, class T>
const ColorImageBuffer<U> &static_buffer_cast(const ColorImageBuffer<T> &buf)
{
	return buf.template static_buffer_cast<U>();
}

inline unsigned select_zimg_buffer_mask(unsigned count)
{
	const unsigned UINT_BITS = std::numeric_limits<unsigned>::digits;

	if (count != 0 && ((count - 1) & (1U << (UINT_BITS - 1))))
		return BUFFER_MAX;

	for (unsigned i = UINT_BITS - 1; i != 0; --i) {
		if ((count - 1) & (1U << (i - 1)))
			return (1U << i) - 1;
	}

	return 0;
}

} // namespace graph
} // namespace zimg

#endif // ZIMG_GRAPH_ZTYPES_H_
