#pragma once

#ifndef ZIMG_CCDEP_H_
#define ZIMG_CCDEP_H_

#if defined(_MSC_VER)
  #define FORCE_INLINE __forceinline
#elif defined(__GNUC__)
  #define FORCE_INLINE __attribute__((always_inline))
#else
  #define FORCE_INLINE
#endif

#if defined(_MSC_VER) && _MSC_VER < 1900
  #define ALIGNED(type, name, alignment) __declspec(align(alignment)) type name
#else
  #define ALIGNED(type, name, alignment) type name alignas(alignment)
#endif

#if defined(_MSC_VER) || defined(__GNUC__)
  #define RESTRICT __restrict
#else
  #define RESTRICT
#endif

#if defined(_MSC_VER) && _MSC_VER < 1900
  #define THREAD_LOCAL __declspec(thread)
#else
  #define THREAD_LOCAL thread_local
#endif

#endif /* ZIMG_CCDEP_H_ */
