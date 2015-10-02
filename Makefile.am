ACLOCAL_AMFLAGS = -I m4

SUBDIRS = $(CONFIG_SUBDIRS)

warningflags = -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers
commonflags = -O2 $(warningflags) $(DEBUGCFLAGS)
AM_CXXFLAGS = -std=c++11 $(commonflags)
AM_CFLAGS = $(commonflags)


pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = zimg.pc

EXTRA_DIST = zimg.pc.in


noinst_PROGRAMS =
lib_LTLIBRARIES = libzimg.la
noinst_LTLIBRARIES =
pkglib_LTLIBRARIES = vszimg.la

libzimg_la_SOURCES = src/zimg/api/zimg.cpp \
					 src/zimg/colorspace/colorspace.cpp \
					 src/zimg/colorspace/colorspace.h \
					 src/zimg/colorspace/colorspace_param.cpp \
					 src/zimg/colorspace/colorspace_param.h \
					 src/zimg/colorspace/graph.cpp \
					 src/zimg/colorspace/graph.h \
					 src/zimg/colorspace/matrix3.cpp \
					 src/zimg/colorspace/matrix3.h \
					 src/zimg/colorspace/operation.cpp \
					 src/zimg/colorspace/operation.h \
					 src/zimg/colorspace/operation_impl.cpp \
					 src/zimg/colorspace/operation_impl.h \
					 src/zimg/common/align.h \
					 src/zimg/common/alloc.h \
					 src/zimg/common/copy_filter.h \
					 src/zimg/common/cpuinfo.h \
					 src/zimg/common/except.h \
					 src/zimg/common/filtergraph.h \
					 src/zimg/common/filtergraph.cpp \
					 src/zimg/common/libm_wrapper.h \
					 src/zimg/common/libm_wrapper.cpp \
					 src/zimg/common/linebuffer.h \
					 src/zimg/common/matrix.h \
					 src/zimg/common/mux_filter.cpp \
					 src/zimg/common/mux_filter.h \
					 src/zimg/common/osdep.h \
					 src/zimg/common/pixel.h \
					 src/zimg/depth/depth_convert.cpp \
					 src/zimg/depth/depth_convert.h \
					 src/zimg/depth/depth.cpp \
					 src/zimg/depth/depth.h \
					 src/zimg/depth/dither.cpp \
					 src/zimg/depth/dither.h \
					 src/zimg/depth/quantize.h \
					 src/zimg/resize/filter.cpp \
					 src/zimg/resize/filter.h \
					 src/zimg/resize/resize.cpp \
					 src/zimg/resize/resize.h \
					 src/zimg/resize/resize_impl.cpp \
					 src/zimg/resize/resize_impl.h \
					 src/zimg/unresize/bilinear.cpp \
					 src/zimg/unresize/bilinear.h \
					 src/zimg/unresize/plane.h \
					 src/zimg/unresize/unresize.cpp \
					 src/zimg/unresize/unresize.h \
					 src/zimg/unresize/unresize_impl.cpp \
					 src/zimg/unresize/unresize_impl.h

include_HEADERS = src/zimg/api/zimg.h \
				  src/zimg/api/zimg++.hpp

libzimg_la_CPPFLAGS = -I$(srcdir)/src/zimg
libzimg_la_LDFLAGS = -no-undefined -version-info 2


vszimg_la_SOURCES = src/vszimg/vszimg.c \
					src/vszimg/VapourSynth.h \
					src/vszimg/VSHelper.h

vszimg_la_CFLAGS = $(AM_CFLAGS) $(PTHREAD_CFLAGS) -I$(srcdir)/src/zimg/api

vszimg_la_LIBADD = $(PTHREAD_LIBS) libzimg.la

vszimg_la_LDFLAGS = -no-undefined -avoid-version -module

vszimg_la_LIBTOOLFLAGS = --tag=disable-static


if X86SIMD
noinst_LTLIBRARIES += libsse.la libsse2.la libavx2.la

libzimg_la_SOURCES += src/zimg/depth/depth_convert_x86.cpp \
					  src/zimg/depth/depth_convert_x86.h \
					  src/zimg/depth/dither_x86.cpp \
					  src/zimg/depth/dither_x86.h \
					  src/zimg/resize/resize_impl_x86.cpp \
					  src/zimg/resize/resize_impl_x86.h \
					  src/zimg/unresize/unresize_impl_x86.cpp \
					  src/zimg/unresize/unresize_impl_x86.h


libsse_la_SOURCES = src/zimg/resize/resize_impl_sse.cpp

libsse_la_CXXFLAGS = $(AM_CXXFLAGS) -msse
libsse_la_CPPFLAGS = -I$(srcdir)/src/zimg

libsse2_la_SOURCES = src/zimg/depth/dither_sse2.cpp \
					 src/zimg/resize/resize_impl_sse2.cpp \
					 src/zimg/unresize/unresize_impl_sse2.cpp

libsse2_la_CXXFLAGS = $(AM_CXXFLAGS) -msse2
libsse2_la_CPPFLAGS = -I$(srcdir)/src/zimg

libavx2_la_SOURCES = src/zimg/unresize/unresize_impl_avx2.cpp

libavx2_la_CXXFLAGS = $(AM_CXXFLAGS) -mavx2 -mfma -mf16c
libavx2_la_CPPFLAGS = -I$(srcdir)/src/zimg

libzimg_la_LIBADD = libsse.la libsse2.la libavx2.la
endif


libtestcommon_la_SOURCES = src/testcommon/aligned_malloc.h \
						   src/testcommon/argparse.cpp \
						   src/testcommon/argparse.h \
						   src/testcommon/mmap.cpp \
						   src/testcommon/mmap.h \
						   src/testcommon/timer.h \
						   src/testcommon/win32_bitmap.cpp \
						   src/testcommon/win32_bitmap.h


if EXAMPLES
noinst_PROGRAMS += example/alpha_example \
					example/api_example \
					example/api_example_c \
					example/interlace_example
noinst_LTLIBRARIES += libtestcommon.la

EXAMPLE_CPPFLAGS = -I$(srcdir)/src/zimg/api -I$(srcdir)/src/testcommon
EXAMPLE_LDADD = libtestcommon.la libzimg.la

example_alpha_example_CPPFLAGS = $(EXAMPLE_CPPFLAGS)
example_alpha_example_SOURCES = doc/example/alpha_example.cpp
example_alpha_example_LDADD = $(EXAMPLE_LDADD)

example_api_example_CPPFLAGS = $(EXAMPLE_CPPFLAGS)
example_api_example_SOURCES = doc/example/api_example.cpp
example_api_example_LDADD = $(EXAMPLE_LDADD)

example_api_example_c_CPPFLAGS = $(EXAMPLE_CPPFLAGS)
example_api_example_c_SOURCES = doc/example/api_example_c.c doc/example/.dummy.cpp
example_api_example_c_LDADD = $(EXAMPLE_LDADD)

example_interlace_example_CPPFLAGS = $(EXAMPLE_CPPFLAGS)
example_interlace_example_SOURCES = doc/example/interlace_example.cpp
example_interlace_example_LDADD = $(EXAMPLE_LDADD)
endif # EXAMPLES


if TESTAPP
noinst_PROGRAMS += zimg-test
noinst_LTLIBRARIES += libtestcommon.la
zimg_test_CPPFLAGS = -I$(srcdir)/src/zimg -I$(srcdir)/src/testcommon

zimg_test_SOURCES = src/testapp/apps.h \
					src/testapp/colorspaceapp.cpp \
					src/testapp/depthapp.cpp \
					src/testapp/frame.cpp \
					src/testapp/frame.h \
					src/testapp/main.cpp \
					src/testapp/pair_filter.cpp \
					src/testapp/pair_filter.h \
					src/testapp/resizeapp.cpp \
					src/testapp/unresizeapp.cpp \
					src/testapp/utils.cpp \
					src/testapp/utils.h

zimg_test_LDADD = libtestcommon.la libzimg.la
endif # TESTAPP


if UNIT_TEST
TESTS=test/unit_test

check_LTLIBRARIES = test/libmusl_m.la
check_PROGRAMS = test/unit_test

test_libmusl_m_la_CFLAGS = $(AM_CFLAGS) -Wno-sign-compare -Wno-unused-variable -Wno-unused-but-set-variable

test_libmusl_m_la_SOURCES = test/extra/musl-libm/__cos.c \
							test/extra/musl-libm/__rem_pio2.c \
							test/extra/musl-libm/__rem_pio2_large.c \
							test/extra/musl-libm/__sin.c \
							test/extra/musl-libm/cos.c \
							test/extra/musl-libm/fpu_wrapper.c \
							test/extra/musl-libm/libm.h \
							test/extra/musl-libm/mymath.h \
							test/extra/musl-libm/pow.c \
							test/extra/musl-libm/powf.c \
							test/extra/musl-libm/sin.c

test_unit_test_CPPFLAGS = -I$(srcdir)/src/zimg \
							-I$(srcdir)/test \
							-I$(srcdir)/test/extra \
							-I$(srcdir)/test/extra/googletest/googletest/include

test_unit_test_SOURCES = test/main.cpp \
							test/colorspace/colorspace_test.cpp \
							test/common/audit_buffer.cpp \
							test/common/audit_buffer.h \
							test/common/copy_filter_test.cpp \
							test/common/filter_validator.cpp \
							test/common/filter_validator.h \
							test/common/filtergraph_test.cpp \
							test/common/mock_filter.cpp \
							test/common/mock_filter.h \
							test/common/mux_filter_test.cpp \
							test/depth/depth_convert_test.cpp \
							test/depth/dither_test.cpp \
							test/extra/sha1/config.h \
							test/extra/sha1/sha1.c \
							test/extra/sha1/sha1.h \
							test/resize/resize_impl_test.cpp

if X86SIMD

test_unit_test_SOURCES += test/depth/dither_sse2_test.cpp \
							test/resize/resize_impl_sse_test.cpp \
							test/resize/resize_impl_sse2_test.cpp

endif #X86SIMD

test_unit_test_LDADD = test/extra/googletest/googletest/lib/libgtest.la \
						test/libmusl_m.la \
						libzimg.la
endif # UNIT_TEST
