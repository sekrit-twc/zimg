#pragma once

#ifndef ALIGNED_MALLOC_H_
#define ALIGNED_MALLOC_H_

#ifdef _WIN32
  #include <malloc.h>
  static void *aligned_malloc(size_t size, size_t alignment) { return _aligned_malloc(size, alignment); }
  static void aligned_free(void *ptr) { _aligned_free(ptr); }
#else
  #include <stdlib.h>
  static void *aligned_malloc(size_t size, size_t alignment) { void *p; if (posix_memalign(&p, alignment, size)) return 0; else return p; }
  static void aligned_free(void *ptr) { free(ptr); }
#endif

#endif /* ALIGNED_MALLOC_H_ */
