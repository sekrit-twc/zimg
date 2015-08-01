#pragma once

#ifndef ZIMG_ALLOC_H_
#define ZIMG_ALLOC_H_

#include <cstddef>
#include "align.h"
#include "except.h"

namespace zimg {;

/**
 * Simple allocator that increments a base pointer.
 * This allocator is not STL compliant.
 */
class LinearAllocator {
	char *m_ptr;

	template <class T>
	T *increment_and_return(size_t n)
	{
		char *ptr = m_ptr;
		m_ptr += n;
		return reinterpret_cast<T *>(ptr);
	}
public:
	/**
	 * Initialize a LinearAllocator with a given buffer.
	 *
	 * @param ptr pointer to buffer
	 */
	LinearAllocator(void *ptr) : m_ptr{ reinterpret_cast<char *>(ptr) }
	{}

	/**
	 * Allocate a given number of bytes.
	 *
	 * @param T buffer type
	 * @param bytes number of bytes
	 * @return pointer to buffer
	 */
	template <class T = void>
	T *allocate(size_t bytes)
	{
		return increment_and_return<T>(align(bytes, ALIGNMENT));
	}

	/**
	 * Allocate a given number of objects.
	 *
	 * @param T buffer type
	 * @param count number of objects
	 * @return pointer to buffer
	 */
	template <class T>
	T *allocate_n(size_t count)
	{
		return allocate(count * sizeof(T));
	}
};

/**
 * Fake allocator that tracks the amount allocated.
 * The pointers returned must not be dereferenced or incremented.
 */
class FakeAllocator {
	size_t m_count;
public:
	/**
	 * Initialize a FakeAllocator.
	 */
	FakeAllocator() : m_count{}
	{}

	/**
	 * @see LinearAllocator::allocate
	 */
	template <class T = void>
	T *allocate(size_t bytes)
	{
		m_count += align(bytes, ALIGNMENT);
		return nullptr;
	}

	/**
	 * @see LinearAllocator::allocate_n
	 */
	template <class T>
	T *allocate_n(size_t count)
	{
		return allocate(count * sizeof(T));
	}

	/**
	 * Get the number of bytes allocated.
	 *
	 * @return allocated size
	 */
	size_t count() const
	{
		return m_count;
	}
};

} // namespace zimg

#endif // ZIMG_ALLOC_H_
