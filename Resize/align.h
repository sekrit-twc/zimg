#ifndef ALIGN_H
#define ALIGN_H

#include <memory>
#include <vector>

#ifdef _WIN32
  #include <malloc.h>
#else
  #include <stdlib.h>
  inline void *_aligned_malloc(size_t size, int alignment) { void *p; if (posix_memalign(&p, alignment, size)) return nullptr; else return p; }
  inline void _aligned_free(void *ptr) { free(ptr); }
#endif

/**
 * 32-byte alignment allows the use of instructions up to AVX.
 */
const int ALIGNMENT = 32;

/**
 * Round up the argument x to the nearest multiple of n.
 * x must be non-negative and n must be positive.
 */
inline int align(int x, int n) { return x % n ? x + n - (x % n) : x; }

/**
 * Round down the argument x to the nearest multiple of n.
 * x must be non-negative and n must be positive.
 */
inline int mod(int x, int n) { return x - (x % n); }

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
		return (T *)_aligned_malloc(n * sizeof(T), ALIGNMENT);
	}

	void deallocate(void *ptr, size_t) const
	{
		_aligned_free(ptr);
	}

	bool operator==(const AlignedAllocator &) const { return true; }

	bool operator!=(const AlignedAllocator &) const { return false; }
};

/**
 * std::vector specialization using AlignedAllocator.
 */
template <class T>
using AlignedVector = std::vector<T, AlignedAllocator<T>>;

#endif // ALIGN_H
