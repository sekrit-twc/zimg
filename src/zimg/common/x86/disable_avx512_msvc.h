#ifdef ZIMG_X86_AVX512
  #if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
    #undef ZIMG_X86_AVX512
  #endif
#endif
