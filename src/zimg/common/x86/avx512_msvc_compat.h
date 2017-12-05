#if defined(ZIMG_X86_AVX512) && defined(_MSC_VER) && !defined(__INTEL_COMPILER)
  #if _MSC_VER == 1912
    #define _MM_PERM_AAAA _MM_SHUFFLE(0, 0, 0, 0)
    #define _MM_PERM_BBBB _MM_SHUFFLE(1, 1, 1, 1)
    #define _MM_PERM_CCCC _MM_SHUFFLE(2, 2, 2, 2)
    #define _MM_PERM_DDDD _MM_SHUFFLE(3, 3, 3, 3)
  #endif
#endif
