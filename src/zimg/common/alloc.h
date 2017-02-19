#pragma once

#ifndef ZIMG_ALLOC_H_
#define ZIMG_ALLOC_H_

#include <cstddef>
#include <stdexcept>
#include <vector>
#include "align.h"
#include "checked_int.h"
#include "except.h"

#ifdef _WIN32
  #include <malloc.h>
  inline void *zimg_x_aligned_malloc(size_t size, size_t alignment) { return _aligned_malloc(size, alignment); }
  inline void zimg_x_aligned_free(void *ptr) { _aligned_free(ptr); }
#else
  #include <stdlib.h>
  inline void *zimg_x_aligned_malloc(size_t size, size_t alignment) { void *p; if (posix_memalign(&p, alignment, size)) return nullptr; else return p; }
  inline void zimg_x_aligned_free(void *ptr) { free(ptr); }
#endif

namespace zimg {

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
	LinearAllocator(void *ptr) noexcept :
		m_ptr{ static_cast<char *>(ptr) },
		m_count{}
	{}

	/**
	 * Allocate a given number of bytes.
	 *
	 * @tparam T buffer type
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
	 * @tparam T buffer type
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
	size_t count() const noexcept
	{
		return m_count;
	}
};

/**
 * Fake allocator that tracks the amount allocated.
 * The pointers returned must not be dereferenced or incremented.
 */
class FakeAllocator {
	checked_size_t m_count;
public:
	/**
	 * Initialize a FakeAllocator.
	 */
	FakeAllocator() noexcept : m_count{} {}

	/**
	 * Allocate a given number of bytes.
	 *
	 * @tparam T buffer type
	 * @param bytes number of bytes
	 * @return nullptr
	 * @throw error::OutOfMemory if total allocation exceeds SIZE_MAX
	 */
	template <class T = void>
	T *allocate(checked_size_t bytes)
	{
		try {
			m_count += ceil_n(bytes, ALIGNMENT);
		} catch (const std::overflow_error &) {
			throw error::OutOfMemory{};
		}

		return nullptr;
	}

	/**
	 * Allocate a given number of objects.
	 *
	 * @tparam T buffer type
	 * @param count number of objects
	 * @return nullptr
	 * @throw error::OutOfMemory if total allocation exceeds SIZE_MAX
	 */
	template <class T>
	T *allocate_n(checked_size_t count)
	{
		return allocate<T>(count * sizeof(T));
	}

	/**
	 * Get the number of bytes allocated.
	 *
	 * @return allocated size
	 */
	size_t count() const noexcept
	{
		return m_count.get();
	}
};

/**
 * STL allocator class which returns aligned buffers.
 *
 * @tparam T type of object to allocate
 */
template <class T>
struct AlignedAllocator {
	typedef T value_type;

	AlignedAllocator() = default;

	template <class U>
	AlignedAllocator(const AlignedAllocator<U> &) noexcept {}

	T *allocate(size_t n) const
	{
		T *ptr = static_cast<T *>(zimg_x_aligned_malloc(n * sizeof(T), ALIGNMENT));

		if (!ptr)
			throw std::bad_alloc{};

		return ptr;
	}

	void deallocate(void *ptr, size_t) const noexcept
	{
		zimg_x_aligned_free(ptr);
	}

	bool operator==(const AlignedAllocator &) const noexcept { return true; }
	bool operator!=(const AlignedAllocator &) const noexcept { return false; }
};

/**
 * std::vector specialization using AlignedAllocator.
 */
template <class T>
using AlignedVector = std::vector<T, AlignedAllocator<T>>;

} // namespace zimg

#endif // ZIMG_ALLOC_H_
