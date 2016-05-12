#undef _zassert
#undef _zassert_d

#ifdef NDEBUG
  #define Z_NDEBUG
  #undef NDEBUG
#endif

#include <assert.h>

#define _zassert(x, msg) assert((x) && (msg))

#ifdef Z_NDEBUG
  #include "ccdep.h"
  #define _zassert_d(x, msg) ASSUME_CONDITION(x)
  #undef Z_NDEBUG
  #define NDEBUG
#else
  #define _zassert_d(x, msg) _zassert(x, msg)
#endif
