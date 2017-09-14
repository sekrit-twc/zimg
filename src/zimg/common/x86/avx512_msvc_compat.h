#if defined(ZIMG_X86_AVX512) && defined(_MSC_VER) && !defined(__INTEL_COMPILER)
  #if _MSC_VER == 1911
    #define _MM_PERM_AAAA _MM_SHUFFLE(0, 0, 0, 0)
    #define _MM_PERM_BBBB _MM_SHUFFLE(1, 1, 1, 1)
    #define _MM_PERM_CCCC _MM_SHUFFLE(2, 2, 2, 2)
    #define _MM_PERM_DDDD _MM_SHUFFLE(3, 3, 3, 3)
    #define _mm_mask_storeu_epi8(p, m, x) _mm512_mask_storeu_epi8((p), (m), _mm512_castsi128_si512((x)))
    #define _mm256_maskz_loadu_epi16(m, p) _mm512_castsi512_si256(_mm512_maskz_loadu_epi16((m), (p)))
    #define _mm256_mask_storeu_epi16(p, m, x) _mm512_mask_storeu_epi16((p), (m), _mm512_castsi256_si512((x)))
  #endif
#endif
