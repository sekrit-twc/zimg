AC_INIT([zimg], [3.0.6], [https://github.com/sekrit-twc/zimg/pulls], [zimg], [https://github.com/sekrit-twc/zimg])
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
      AX_CHECK_COMPILE_FLAG([-std=c++17], [CXXSTD="c++17"]))
AS_IF([test "x$CXXSTD" = "x"], AC_MSG_ERROR([C++17 support required]))
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
                [armv7*],  [],
                [aarch64], [],
                [i?86],    [CFLAGS="$CFLAGS -mfpmath=sse -msse2" CXXFLAGS="$CXXFLAGS -mfpmath=sse -msse2"],
                [x86_64],  [],
                [AC_MSG_WARN([Could not determine how to enable IEEE-754 compliance on host. Tests may fail.])])

        AC_CHECK_PROG([CMAKE], [cmake], [cmake])
        AS_MKDIR_P([test/extra/googletest/build])
        AS_IF([(cd "test/extra/googletest/build" && $CMAKE -Dgtest_disable_pthreads=ON "$ac_abs_confdir/test/extra/googletest")],
              [], [AC_MSG_ERROR([CMake error])])
      ])


AS_IF([test "x$enable_debug" = "xyes"],
      [AC_SUBST([DEBUGCFLAGS], ["-O0 -g3 -ggdb"])],
      [AC_SUBST([DEBUGCFLAGS], ["-DNDEBUG"])])

AC_LANG_PUSH([C++])
AC_MSG_CHECKING(for the C++ standard library)
AS_IF([test "x$STL_LIBS" = "x"],
      [
        AC_COMPILE_IFELSE([AC_LANG_PROGRAM([
          #include <cstddef>
          #ifndef _LIBCPP_VERSION
          #error libc++ not in use
          #endif
        ], [;])], [STL_LIBS="-lc++"], [STL_LIBS="-lstdc++"])
      ])
AC_LANG_POP([C++])
AC_MSG_RESULT($STL_LIBS)
AC_SUBST([STL_LIBS], [$STL_LIBS])

ARM="no"
X86="no"

AS_CASE([$host_cpu],
        [aarch64], [BITS="64" ARM="yes"],
        [i?86],    [BITS="32" X86="yes"],
        [x86_64],  [BITS="64" X86="yes"])

AS_IF([test "x$ARM" = "xyes" && test "x$enable_simd" != "xno"],
      [
        AC_DEFINE([ZIMG_ARM])
        AC_SUBST([NEON_CFLAGS], [$NEON_CFLAGS])
        enable_arm_simd=yes
      ])

AS_IF([test "x$X86" = "xyes" && test "x$enable_simd" != "xno"],
      [
        AC_DEFINE([ZIMG_X86])
        enable_x86_simd=yes

        AX_CHECK_COMPILE_FLAG([-mtune=haswell], AC_SUBST([HSW_CFLAGS], [-mtune=haswell]))
        AX_CHECK_COMPILE_FLAG([-mtune=skylake-avx512], AC_SUBST([SKX_CFLAGS], [-mtune=skylake-avx512]))
        AX_CHECK_COMPILE_FLAG([-mtune=cascadelake], AC_SUBST([CLX_CFLAGS], [-mtune=cascadelake]))
      ])


AM_CONDITIONAL([TESTAPP],        [test "x$enable_testapp" = "xyes"])
AM_CONDITIONAL([EXAMPLES],       [test "x$enable_example" = "xyes"])
AM_CONDITIONAL([UNIT_TEST],      [test "x$enable_unit_test" = "xyes"])
AM_CONDITIONAL([ARMSIMD],        [test "x$enable_arm_simd" = "xyes"])
AM_CONDITIONAL([X86SIMD],        [test "x$enable_x86_simd" = "xyes"])
AM_CONDITIONAL([X86SIMD_AVX512], [test "x$enable_x86_simd_avx512" = "xyes"])

AS_CASE([$host_os],
        [cygwin*], [AS_IF([test "x$BITS" = "x32"], [LDFLAGS="-Wl,--kill-at"])],
        [mingw*],  [AS_IF([test "x$BITS" = "x32"], [LDFLAGS="-Wl,--kill-at"])])

AC_CONFIG_FILES([Makefile zimg.pc])
AC_OUTPUT
