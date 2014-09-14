#pragma once

#ifndef ZIMG_OSDEP_H_
#define ZIMG_OSDEP_H_

#ifdef _MSC_VER
  #define FORCE_INLINE __forceinline
  #define RESTRICT __restrict
#else
  #define FORCE_INLINE
  #define RESTRICT
#endif

#endif // ZIMG_OSDEP_H_
