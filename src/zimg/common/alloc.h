#pragma once

#ifndef ZIMG_ALLOC_H_
#define ZIMG_ALLOC_H_

#include <cstddef>
#include <vector>
#include "align.h"

#ifdef _WIN32
  #include <malloc.h>
  inline void *_zimg_aligned_malloc(size_t size, size_t alignment) { return _aligned_malloc(size, alignment); }
  inline void _zimg_aligned_free(void *ptr) { _aligned_free(ptr); }
#else
  #include <stdlib.h>
  inline void *_zimg_aligned_malloc(size_t size, size_t alignment) { void *p; if (posix_memalign(&p, alignment, size)) return nullptr; else return p; }
  inline void _zimg_aligned_free(void *ptr) { free(ptr); }
#endif

namespace zimg {;

/**
 * Simple allocator that increments a base pointer.
 * This allocator is not STL compliant.
 */
class LinearAllocator {
	char *m_ptr;
	size_t m_count;

	template <class T>
	T *increment_and_return(size_t n)
	{
		char *ptr = m_ptr;
		m_ptr += n;
		m_count += n;
		return reinterpret_cast<T *>(ptr);
	}
public:
	/**
	 * Initialize a LinearAllocator with a given buffer.
	 *
	 * @param ptr pointer to buffer
	 */
	LinearAllocator(void *ptr) : m_ptr{ static_cast<char *>(ptr) }, m_count{}
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
		return increment_and_return<T>(ceil_n(bytes, ALIGNMENT));
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
		return allocate<T>(count * sizeof(T));
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
		m_count += ceil_n(bytes, ALIGNMENT);
		return nullptr;
	}

	/**
	 * @see LinearAllocator::allocate_n
	 */
	template <class T>
	T *allocate_n(size_t count)
	{
		return allocate<T>(count * sizeof(T));
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

/**
 * STL allocator class which returns aligned buffers.
 *
 * @param T type of object to allocate
 */
template <class T>
struct AlignedAllocator {
	typedef T value_type;

	AlignedAllocator() = default;

	template <class U>
	AlignedAllocator(const AlignedAllocator<U> &)
	{
	}

	T *allocate(size_t n) const
	{
		T *ptr = (T *)_zimg_aligned_malloc(n * sizeof(T), ALIGNMENT);

		if (!ptr)
			throw std::bad_alloc{};

		return ptr;
	}

	void deallocate(void *ptr, size_t) const
	{
		_zimg_aligned_free(ptr);
	}

	bool operator==(const AlignedAllocator &) const
	{
		return true;
	}

	bool operator!=(const AlignedAllocator &) const
	{
		return false;
	}
};

/**
 * std::vector specialization using AlignedAllocator.
 */
template <class T>
using AlignedVector = std::vector<T, AlignedAllocator<T>>;

} // namespace zimg

#endif // ZIMG_ALLOC_H_
