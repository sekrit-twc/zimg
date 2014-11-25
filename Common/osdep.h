#pragma once

#ifndef ZIMG_OSDEP_H_
#define ZIMG_OSDEP_H_

#if defined(_MSC_VER)
  #define FORCE_INLINE __forceinline
#elif defined(__GNUC__)
  #define FORCE_INLINE __attribute__((always_inline))
#else
  #define FORCE_INLINE
#endif

#if defined(_MSC_VER) || defined(__GNUC__)
  #define RESTRICT __restrict
#else
  #define RESTRICT
#endif

#ifdef _MSC_VER
  #define THREAD_LOCAL __declspec(thread)
#else
  #define THREAD_LOCAL thread_local
#endif

#endif // ZIMG_OSDEP_H_
