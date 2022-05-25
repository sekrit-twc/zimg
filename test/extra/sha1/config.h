#include <endian.h>

#ifndef LITTLE_ENDIAN
  #define LITTLE_ENDIAN 4321
#endif

#ifndef BIG_ENDIAN
  #define BIG_ENDIAN 1234
#endif

#ifndef BYTE_ORDER
#if defined(__BYTE_ORDER) && (__BYTE_ORDER == __BIG_ENDIAN)
  #define BYTE_ORDER BIG_ENDIAN
#else
  #define BYTE_ORDER LITTLE_ENDIAN
#endif
#endif
