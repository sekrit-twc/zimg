#pragma once

#ifndef ZIMG_ZASSERT_H_
#define ZIMG_ZASSERT_H_

#ifdef NDEBUG
  #define Z_NDEBUG_
  #undef NDEBUG
#endif

#include <assert.h>

#define _zassert(x, msg) assert((x) && (msg))

#ifdef Z_NDEBUG_
  #define _zassert_d(x, msg)
#else
  #define _zassert_d(x, msg) _zassert(x, msg)
#endif

#endif /* ZIMG_ZASSERT_H_ */
