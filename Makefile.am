ACLOCAL_AMFLAGS = -I m4

SUBDIRS = $(CONFIG_SUBDIRS)

warningflags = -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers
commonflags = -O2 $(warningflags) $(DEBUGCFLAGS)
AM_CXXFLAGS = -std=c++11 $(commonflags)
AM_CFLAGS = $(commonflags)


pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = zimg.pc

EXTRA_DIST = zimg.pc.in


exampledir = $(docdir)/example
examplemiscdir = $(exampledir)/misc


dist_doc_DATA = ChangeLog COPYING README.md

dist_example_DATA = \
	doc/example/alpha_example.cpp \
	doc/example/api_example.cpp \
	doc/example/api_example_c.c \
	doc/example/hdr_example.cpp \
	doc/example/interlace_example.cpp \
	doc/example/Makefile

dist_examplemisc_DATA = \
	src/testcommon/aligned_malloc.h \
	src/testcommon/argparse.cpp \
	src/testcommon/argparse.h \
	src/testcommon/mmap.cpp \
	src/testcommon/mmap.h \
	src/testcommon/win32_bitmap.cpp \
	src/testcommon/win32_bitmap.h


noinst_PROGRAMS =
lib_LTLIBRARIES = libzimg.la
noinst_LTLIBRARIES = libzimg_internal.la

include_HEADERS = \
	src/zimg/api/zimg.h \
	src/zimg/api/zimg++.hpp

libzimg_la_SOURCES = dummy.cpp
libzimg_la_LIBADD = libzimg_internal.la
libzimg_la_LDFLAGS = -no-undefined -version-info 2

libzimg_internal_la_SOURCES = \
	src/zimg/api/zimg.cpp \
	src/zimg/api/zimg.h \
	src/zimg/api/zimg++.hpp \
	src/zimg/colorspace/colorspace.cpp \
	src/zimg/colorspace/colorspace.h \
	src/zimg/colorspace/colorspace_param.cpp \
	src/zimg/colorspace/colorspace_param.h \
	src/zimg/colorspace/gamma.cpp \
	src/zimg/colorspace/gamma.h \
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
	src/zimg/common/builder.h \
	src/zimg/common/ccdep.h \
	src/zimg/common/checked_int.h \
	src/zimg/common/cpuinfo.cpp \
	src/zimg/common/cpuinfo.h \
	src/zimg/common/except.h \
	src/zimg/common/libm_wrapper.h \
	src/zimg/common/libm_wrapper.cpp \
	src/zimg/common/make_unique.h \
	src/zimg/common/matrix.cpp \
	src/zimg/common/matrix.h \
	src/zimg/common/pixel.h \
	src/zimg/common/static_map.h \
	src/zimg/common/zassert.h \
	src/zimg/depth/depth_convert.cpp \
	src/zimg/depth/depth_convert.h \
	src/zimg/depth/depth.cpp \
	src/zimg/depth/depth.h \
	src/zimg/depth/dither.cpp \
	src/zimg/depth/dither.h \
	src/zimg/depth/hexfloat.h \
	src/zimg/depth/quantize.h \
	src/zimg/depth/quantize.cpp \
	src/zimg/graph/basic_filter.cpp \
	src/zimg/graph/basic_filter.h \
	src/zimg/graph/filtergraph.h \
	src/zimg/graph/filtergraph.cpp \
	src/zimg/graph/graphbuilder.h \
	src/zimg/graph/graphbuilder.cpp \
	src/zimg/graph/image_buffer.h \
	src/zimg/graph/image_filter.h \
	src/zimg/resize/filter.cpp \
	src/zimg/resize/filter.h \
	src/zimg/resize/resize.cpp \
	src/zimg/resize/resize.h \
	src/zimg/resize/resize_impl.cpp \
	src/zimg/resize/resize_impl.h \
	src/zimg/unresize/bilinear.cpp \
	src/zimg/unresize/bilinear.h \
	src/zimg/unresize/unresize.cpp \
	src/zimg/unresize/unresize.h \
	src/zimg/unresize/unresize_impl.cpp \
	src/zimg/unresize/unresize_impl.h

libzimg_internal_la_CPPFLAGS = -I$(srcdir)/src/zimg


if X86SIMD
noinst_LTLIBRARIES += libsse.la libsse2.la libavx.la libf16c.la libavx2.la

libzimg_internal_la_SOURCES += \
	src/zimg/colorspace/operation_impl_x86.cpp \
	src/zimg/colorspace/operation_impl_x86.h \
	src/zimg/common/x86util.cpp \
	src/zimg/common/x86util.h \
	src/zimg/depth/depth_convert_x86.cpp \
	src/zimg/depth/depth_convert_x86.h \
	src/zimg/depth/dither_x86.cpp \
	src/zimg/depth/dither_x86.h \
	src/zimg/depth/f16c_x86.h \
	src/zimg/resize/resize_impl_x86.cpp \
	src/zimg/resize/resize_impl_x86.h


libsse_la_SOURCES = \
	src/zimg/colorspace/operation_impl_sse.cpp \
	src/zimg/resize/resize_impl_sse.cpp

libsse_la_CXXFLAGS = $(AM_CXXFLAGS) -msse
libsse_la_CPPFLAGS = -I$(srcdir)/src/zimg

libsse2_la_SOURCES = \
	src/zimg/colorspace/operation_impl_sse2.cpp \
	src/zimg/depth/depth_convert_sse2.cpp \
	src/zimg/depth/dither_sse2.cpp \
	src/zimg/depth/error_diffusion_sse2.cpp \
	src/zimg/depth/f16c_sse2.cpp \
	src/zimg/resize/resize_impl_sse2.cpp

libsse2_la_CXXFLAGS = $(AM_CXXFLAGS) -msse2
libsse2_la_CPPFLAGS = -I$(srcdir)/src/zimg

libavx_la_SOURCES = \
	src/zimg/colorspace/operation_impl_avx.cpp \
	src/zimg/resize/resize_impl_avx.cpp

libavx_la_CXXFLAGS = $(AM_CXXFLAGS) -mavx
libavx_la_CPPFLAGS = -I$(srcdir)/src/zimg

libf16c_la_SOURCES = src/zimg/depth/f16c_ivb.cpp

libf16c_la_CXXFLAGS = $(AM_CXXFLAGS) -mavx -mf16c
libf16c_la_CPPFLAGS = -I$(srcdir)/src/zimg

libavx2_la_SOURCES = \
	src/zimg/colorspace/operation_impl_avx2.cpp \
	src/zimg/depth/depth_convert_avx2.cpp \
	src/zimg/depth/dither_avx2.cpp \
	src/zimg/depth/error_diffusion_avx2.cpp \
	src/zimg/resize/resize_impl_avx2.cpp

libavx2_la_CXXFLAGS = $(AM_CXXFLAGS) -mavx2 -mf16c -mfma
libavx2_la_CPPFLAGS = -I$(srcdir)/src/zimg

libzimg_internal_la_LIBADD = libsse.la libsse2.la libavx.la libf16c.la libavx2.la
endif


libtestcommon_la_SOURCES = \
	src/testcommon/aligned_malloc.h \
	src/testcommon/argparse.cpp \
	src/testcommon/argparse.h \
	src/testcommon/json.cpp \
	src/testcommon/json.h \
	src/testcommon/mmap.cpp \
	src/testcommon/mmap.h \
	src/testcommon/timer.h \
	src/testcommon/win32_bitmap.cpp \
	src/testcommon/win32_bitmap.h


if EXAMPLES
noinst_PROGRAMS += \
	doc/example/alpha_example \
	doc/example/api_example \
	doc/example/api_example_c \
	doc/example/hdr_example \
	doc/example/interlace_example
noinst_LTLIBRARIES += libtestcommon.la

EXAMPLE_CPPFLAGS = -I$(srcdir)/src/zimg/api -I$(srcdir)/src/testcommon
EXAMPLE_LDADD = libtestcommon.la libzimg.la

doc_example_alpha_example_CPPFLAGS = $(EXAMPLE_CPPFLAGS)
doc_example_alpha_example_SOURCES = doc/example/alpha_example.cpp
doc_example_alpha_example_LDADD = $(EXAMPLE_LDADD)

doc_example_api_example_CPPFLAGS = $(EXAMPLE_CPPFLAGS)
doc_example_api_example_SOURCES = doc/example/api_example.cpp
doc_example_api_example_LDADD = $(EXAMPLE_LDADD)

doc_example_api_example_c_CPPFLAGS = $(EXAMPLE_CPPFLAGS)
doc_example_api_example_c_SOURCES = doc/example/api_example_c.c dummy.cpp
doc_example_api_example_c_LDADD = $(EXAMPLE_LDADD)

doc_example_hdr_example_CPPFLAGS = $(EXAMPLE_CPPFLAGS)
doc_example_hdr_example_SOURCES = doc/example/hdr_example.cpp
doc_example_hdr_example_LDADD = $(EXAMPLE_LDADD)

doc_example_interlace_example_CPPFLAGS = $(EXAMPLE_CPPFLAGS)
doc_example_interlace_example_SOURCES = doc/example/interlace_example.cpp
doc_example_interlace_example_LDADD = $(EXAMPLE_LDADD)
endif # EXAMPLES


if TESTAPP
noinst_PROGRAMS += testapp
noinst_LTLIBRARIES += libtestcommon.la
testapp_CPPFLAGS = -I$(srcdir)/src/zimg -I$(srcdir)/src/testcommon

testapp_SOURCES = \
	src/testapp/apps.h \
	src/testapp/colorspaceapp.cpp \
	src/testapp/depthapp.cpp \
	src/testapp/frame.cpp \
	src/testapp/frame.h \
	src/testapp/graphapp.cpp \
	src/testapp/main.cpp \
	src/testapp/pair_filter.cpp \
	src/testapp/pair_filter.h \
	src/testapp/resizeapp.cpp \
	src/testapp/table.cpp \
	src/testapp/table.h \
	src/testapp/unresizeapp.cpp \
	src/testapp/utils.cpp \
	src/testapp/utils.h

testapp_CXXFLAGS = $(AM_CXXFLAGS) $(PTHREAD_CFLAGS)
testapp_LDADD = $(PTHREAD_LIBS) libtestcommon.la libzimg_internal.la
endif # TESTAPP


if UNIT_TEST
TESTS = test/unit_test

check_LTLIBRARIES = test/libmusl_m.la
check_PROGRAMS = test/unit_test

test_libmusl_m_la_CFLAGS = $(AM_CFLAGS) -Wno-sign-compare -Wno-unused-variable -Wno-unused-but-set-variable

test_libmusl_m_la_SOURCES = \
	test/extra/musl-libm/__cos.c \
	test/extra/musl-libm/__rem_pio2.c \
	test/extra/musl-libm/__rem_pio2_large.c \
	test/extra/musl-libm/__sin.c \
	test/extra/musl-libm/cos.c \
	test/extra/musl-libm/fpu_wrapper.c \
	test/extra/musl-libm/expf.c \
	test/extra/musl-libm/libm.h \
	test/extra/musl-libm/logf.c \
	test/extra/musl-libm/mymath.h \
	test/extra/musl-libm/powf.c \
	test/extra/musl-libm/sin.c

test_unit_test_CPPFLAGS = \
	-I$(srcdir)/src/zimg \
	-I$(srcdir)/test \
	-I$(srcdir)/test/extra \
	-I$(srcdir)/test/extra/googletest/googletest/include

test_unit_test_SOURCES = \
	test/main.cpp \
	test/api/api_test.cpp \
	test/colorspace/colorspace_test.cpp \
	test/depth/depth_convert_test.cpp \
	test/depth/dither_test.cpp \
	test/extra/sha1/config.h \
	test/extra/sha1/sha1.c \
	test/extra/sha1/sha1.h \
	test/graph/audit_buffer.cpp \
	test/graph/audit_buffer.h \
	test/graph/basic_filter_test.cpp \
	test/graph/filter_validator.cpp \
	test/graph/filter_validator.h \
	test/graph/filtergraph_test.cpp \
	test/graph/mock_filter.cpp \
	test/graph/mock_filter.h \
	test/resize/resize_impl_test.cpp

if X86SIMD
test_unit_test_SOURCES += \
	test/colorspace/colorspace_avx_test.cpp \
	test/colorspace/colorspace_avx2_test.cpp \
	test/colorspace/colorspace_sse_test.cpp \
	test/colorspace/colorspace_sse2_test.cpp \
	test/colorspace/gamma_test.cpp \
	test/depth/depth_convert_avx2_test.cpp \
	test/depth/depth_convert_sse2_test.cpp \
	test/depth/dither_avx2_test.cpp \
	test/depth/dither_sse2_test.cpp \
	test/depth/error_diffusion_avx2_test.cpp \
	test/depth/error_diffusion_sse2_test.cpp \
	test/depth/f16c_ivb_test.cpp \
	test/depth/f16c_sse2_test.cpp \
	test/resize/resize_impl_avx_test.cpp \
	test/resize/resize_impl_avx2_test.cpp \
	test/resize/resize_impl_sse_test.cpp \
	test/resize/resize_impl_sse2_test.cpp
endif #X86SIMD

test_unit_test_LDADD = \
	test/extra/googletest/googletest/lib/libgtest.la \
	test/libmusl_m.la \
	libzimg_internal.la
endif # UNIT_TEST
