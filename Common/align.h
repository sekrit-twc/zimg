#pragma once

#ifndef ZIMG_ALIGN_H_
#define ZIMG_ALIGN_H_

#include <vector>

#ifdef _WIN32
  #include <malloc.h>
  inline void *zimg_aligned_malloc(size_t size, size_t alignment) { return _aligned_malloc(size, alignment); }
  inline void zimg_aligned_free(void *ptr) { _aligned_free(ptr); }
#else
  #include <stdlib.h>
  inline void *zimg_aligned_malloc(size_t size, int alignment) { void *p; if (posix_memalign(&p, alignment, size)) return nullptr; else return p; }
  inline void zimg_aligned_free(void *ptr) { free(ptr); }
#endif

namespace zimg {;

/**
 * 32-byte alignment allows the use of instructions up to AVX.
 */
const int ALIGNMENT = 32;

/**
 * Round up the argument x to the nearest multiple of n.
 * x must be non-negative and n must be positive.
 */
template <class T, class U>
inline T align(T x, U n) { return x % n ? x + n - (x % n) : x; }

/**
 * Round down the argument x to the nearest multiple of n.
 * x must be non-negative and n must be positive.
 */
template <class T, class U>
inline T mod(T x, U n) { return x - (x % n); }

/**
 * Helper struct that computes alignment in units of object count.
 *
 * @param T type of object
 */
template <class T>
struct AlignmentOf {
	static const int value = ALIGNMENT / sizeof(T);
};

/**
 * STL allocator class which returns aligned buffers.
 *
 * @param T type of object to allocate
 */
template <class T>
struct AlignedAllocator {
	typedef T value_type;

	T *allocate(size_t n) const
	{
		T *ptr = (T *)zimg_aligned_malloc(n * sizeof(T), ALIGNMENT);

		if (!ptr)
			throw std::bad_alloc{};

		return ptr;
	}

	void deallocate(void *ptr, size_t) const { zimg_aligned_free(ptr); }

	bool operator==(const AlignedAllocator &) const { return true; }

	bool operator!=(const AlignedAllocator &) const { return false; }
};

/**
 * std::vector specialization using AlignedAllocator.
 */
template <class T>
using AlignedVector = std::vector<T, AlignedAllocator<T>>;

} // namespace zimg

#endif // ZIMG_ALIGN_H_
