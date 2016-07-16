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

#if defined(_MSC_VER) || defined(__GNUC__)
  #define RESTRICT __restrict
#else
  #define RESTRICT
#endif

#if defined(_MSC_VER)
  #define ASSUME_CONDITION(x) __assume((x))
#elif defined(__GNUC__)
  #define ASSUME_CONDITION(x) do { if (!(x)) __builtin_unreachable(); } while (0)
#else
  #define ASSUME_CONDITION(x) ((void)0)
#endif

#ifdef __APPLE__
  #define thread_local __thread
#endif

#endif /* ZIMG_CCDEP_H_ */
