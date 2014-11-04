AC_INIT([zimg], [1], [https://github.com/sekrit-twc/zimg/pulls], [zimg], [https://github.com/sekrit-twc/zimg])

AM_INIT_AUTOMAKE([foreign no-dist-gzip dist-xz subdir-objects no-define])
AM_SILENT_RULES([yes])

LT_INIT([win32-dll])

AC_PROG_CXX

AC_CANONICAL_HOST


AC_ARG_ENABLE([x86simd], AS_HELP_STRING([--enable-x86simd], [Enable optimisations for x86 CPUs.]))

AS_IF([test "x$enable_x86simd" = "xyes"], [AC_DEFINE([ZIMG_X86])])

AM_CONDITIONAL([X86SIMD], [test "x$enable_x86simd" = "xyes"])


AS_CASE(
  [$host_cpu],
  [i?86],   [BITS="32"],
  [x86_64], [BITS="64"],
  [AS_IF([test "x$enable_x86simd" = "xyes"], [AC_MSG_ERROR([--enable_x86simd was passed but host CPU type is not x86.])])]
)

AS_CASE(
   [$host_os],
   [cygwin*],
   [
      AS_IF(
         [test "x$BITS" = "x32"],
         [LDFLAGS="-Wl,--kill-at"]
      )
   ],
   [mingw*],
   [
      AS_IF(
         [test "x$BITS" = "x32"],
         [LDFLAGS="-Wl,--kill-at"]
      )
   ]
)

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
