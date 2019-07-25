#undef zassert
#undef zassert_d

#ifdef NDEBUG
  #define Z_NDEBUG
  #undef NDEBUG
#endif

#include <assert.h>

#define zassert(x, msg) assert((x) && (msg))

#ifdef Z_NDEBUG
  #include "ccdep.h"
  #define zassert_d(x, msg) ASSUME_CONDITION(x)
  #define zassert_dfatal(msg)
  #undef Z_NDEBUG
  #define NDEBUG
#else
  #define zassert_d(x, msg) zassert(x, msg)
  #define zassert_dfatal(msg) zassert(false, msg)
#endif
