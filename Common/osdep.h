#pragma once

#ifndef ZIMG_OSDEP_H_
#define ZIMG_OSDEP_H_

#ifdef _MSC_VER
  #define FORCE_INLINE __forceinline
  #define RESTRICT __restrict
  #define THREAD_LOCAL __declspec(thread)
#else
  #define FORCE_INLINE
  #define RESTRICT
  #define THREAD_LOCAL thread_local
#endif

#endif // ZIMG_OSDEP_H_
