AC_INIT([zimg], [2.9.2], [https://github.com/sekrit-twc/zimg/pulls], [zimg], [https://github.com/sekrit-twc/zimg])
AC_CONFIG_MACRO_DIR([m4])

: ${CFLAGS=""}
: ${CXXFLAGS=""}

AM_INIT_AUTOMAKE([foreign no-dist-gzip dist-xz subdir-objects no-define])
AM_SILENT_RULES([yes])

LT_INIT([win32-dll])

AC_PROG_CXX

AC_CANONICAL_HOST


AC_ARG_ENABLE([testapp],   AS_HELP_STRING([--enable-testapp],   [Compile the test application. (default=no)]))
AC_ARG_ENABLE([example],   AS_HELP_STRING([--enable-example],   [Compile example programs. (default=no)]))
AC_ARG_ENABLE([unit-test], AS_HELP_STRING([--enable-unit-test], [Compile unit tests. May result in slower code. (default=no)]))
AC_ARG_ENABLE([debug],     AS_HELP_STRING([--enable-debug],     [Enable compilation options required for debugging. (default=no)]))
AC_ARG_ENABLE([simd],      AS_HELP_STRING([--disable-simd],     [Disable SIMD code. (default=no)]))

AC_LANG_PUSH([C++])
AS_IF([test "x$CXXSTD" = "x"],
      AX_CHECK_COMPILE_FLAG([-std=c++14], [CXXSTD="c++14"]))
AS_IF([test "x$CXXSTD" = "x"],
      AX_CHECK_COMPILE_FLAG([-std=c++11], [CXXSTD="c++11"]))
AS_IF([test "x$CXXSTD" = "x"], AC_MSG_ERROR([C++11 support required]))
AC_LANG_POP([C++])

AC_SUBST([CXXSTD], [$CXXSTD])

AX_CHECK_COMPILE_FLAG([-fvisibility=hidden],
                      [CFLAGS="-fvisibility=hidden $CFLAGS" CXXFLAGS="-fvisibility=hidden $CXXFLAGS"])

AX_PTHREAD(, AC_MSG_WARN([Unable to find pthread. Testapp will be single-threaded.]))
AS_IF([test "x$PTHREAD_CC" != "x"], [CC="$PTHREAD_CC"])

AS_IF([test "x$enable_unit_test" = "xyes"],
      [
        CFLAGS="$CFLAGS -fno-fast-math -ffp-contract=off" CXXFLAGS="$CXXFLAGS -fno-fast-math -ffp-contract=off"
        AS_CASE([$host_cpu],
                [i?86],   [CFLAGS="$CFLAGS -mfpmath=sse -msse2" CXXFLAGS="$CXXFLAGS -mfpmath=sse -msse2"],
                [x86_64], [],
                [AC_MSG_WARN([Could not determine how to enable IEEE-754 compliance on host. Tests may fail.])])

        AC_CHECK_PROG([CMAKE], [cmake], [cmake])
        AS_MKDIR_P([test/extra/googletest/build])
        AS_IF([(cd "test/extra/googletest/build" && $CMAKE -Dgtest_disable_pthreads=ON "$ac_abs_confdir/test/extra/googletest")],
              [], [AC_MSG_ERROR([CMake error])])
      ])


AS_IF([test "x$enable_debug" = "xyes"],
      [AC_SUBST([DEBUGCFLAGS], ["-O0 -g3 -ggdb"])],
      [AC_SUBST([DEBUGCFLAGS], ["-DNDEBUG"])])

AS_IF([test "x$STL_LIBS" = "x"], [STL_LIBS="-lstdc++"])
AC_SUBST([STL_LIBS], [$STL_LIBS])

X86="no"

AS_CASE([$host_cpu],
        [i?86],   [BITS="32" X86="yes"],
        [x86_64], [BITS="64" X86="yes"])

AS_IF([test "x$X86" = "xyes" && test "x$enable_simd" != "xno"],
      [
        AC_DEFINE([ZIMG_X86])
        enable_x86_simd=yes

        AX_CHECK_COMPILE_FLAG([-mtune=sandybridge], AC_SUBST([SNB_CFLAGS], [-mtune=sandybridge]))
        AX_CHECK_COMPILE_FLAG([-mtune=ivybridge], AC_SUBST([IVB_CFLAGS], [-mtune=ivybridge]))
        AX_CHECK_COMPILE_FLAG([-mtune=haswell], AC_SUBST([HSW_CFLAGS], [-mtune=haswell]))
        AX_CHECK_COMPILE_FLAG([-mtune=skylake-avx512], AC_SUBST([SKX_CFLAGS], [-mtune=skylake-avx512]))
        AX_CHECK_COMPILE_FLAG([-mtune=cascadelake], AC_SUBST([CLX_CFLAGS], [-mtune=cascadelake]))

        AX_CHECK_COMPILE_FLAG(
          [-mavx512f -mavx512pf -mavx512er -mavx512cd -mavx512vl -mavx512bw -mavx512dq -mavx512ifma -mavx512vbmi -mavx512vbmi2 -mavx512bitalg -mavx512vpopcntdq -mavx512vnni],
          [
            AC_DEFINE([ZIMG_X86_AVX512])
            enable_x86_simd_avx512=yes
          ], [], [],
          [AC_LANG_SOURCE([
#include <immintrin.h>
int main()
{
	volatile __m512 x = _mm512_setzero_ps();
	x = _mm512_add_ps(x, x);
#if defined(__GNUC__) && defined(__x86_64__)
	asm volatile ("" : : : "zmm16", "zmm17", "zmm18", "zmm19");
#endif
	return 0;
}
          ])])
      ])


AM_CONDITIONAL([TESTAPP],        [test "x$enable_testapp" = "xyes"])
AM_CONDITIONAL([EXAMPLES],       [test "x$enable_example" = "xyes"])
AM_CONDITIONAL([UNIT_TEST],      [test "x$enable_unit_test" = "xyes"])
AM_CONDITIONAL([X86SIMD],        [test "x$enable_x86_simd" = "xyes"])
AM_CONDITIONAL([X86SIMD_AVX512], [test "x$enable_x86_simd_avx512" = "xyes"])

AS_CASE([$host_os],
        [cygwin*], [AS_IF([test "x$BITS" = "x32"], [LDFLAGS="-Wl,--kill-at"])],
        [mingw*],  [AS_IF([test "x$BITS" = "x32"], [LDFLAGS="-Wl,--kill-at"])])

AC_CONFIG_FILES([Makefile zimg.pc])
AC_OUTPUT
