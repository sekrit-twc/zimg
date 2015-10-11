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

#endif /* ZIMG_CCDEP_H_ */
