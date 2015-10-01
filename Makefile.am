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

libzimg_la_SOURCES = API/zimg.cpp \
					 Colorspace/colorspace.cpp \
					 Colorspace/colorspace.h \
					 Colorspace/colorspace_param.cpp \
					 Colorspace/colorspace_param.h \
					 Colorspace/colorspace2.cpp \
					 Colorspace/colorspace2.h \
					 Colorspace/graph.cpp \
					 Colorspace/graph.h \
					 Colorspace/matrix3.cpp \
					 Colorspace/matrix3.h \
					 Colorspace/operation.cpp \
					 Colorspace/operation.h \
					 Colorspace/operation_impl.cpp \
					 Colorspace/operation_impl.h \
					 Common/align.h \
					 Common/alloc.h \
					 Common/copy_filter.h \
					 Common/cpuinfo.h \
					 Common/except.h \
					 Common/filtergraph.h \
					 Common/filtergraph.cpp \
					 Common/libm_wrapper.h \
					 Common/libm_wrapper.cpp \
					 Common/linebuffer.h \
					 Common/matrix.h \
					 Common/mux_filter.cpp \
					 Common/mux_filter.h \
					 Common/osdep.h \
					 Common/pixel.h \
					 Depth/depth_convert.cpp \
					 Depth/depth_convert.h \
					 Depth/depth_convert2.cpp \
					 Depth/depth_convert2.h \
					 Depth/depth.cpp \
					 Depth/depth.h \
					 Depth/depth2.cpp \
					 Depth/depth2.h \
					 Depth/dither.cpp \
					 Depth/dither.h \
					 Depth/dither2.cpp \
					 Depth/dither2.h \
					 Depth/dither_impl.cpp \
					 Depth/dither_impl.h \
					 Depth/error_diffusion.cpp \
					 Depth/error_diffusion.h \
					 Depth/quantize.h \
					 Resize/filter.cpp \
					 Resize/filter.h \
					 Resize/resize.cpp \
					 Resize/resize.h \
					 Resize/resize2.cpp \
					 Resize/resize2.h \
					 Resize/resize_impl.cpp \
					 Resize/resize_impl.h \
					 Resize/resize_impl2.cpp \
					 Resize/resize_impl2.h \
					 Unresize/bilinear.cpp \
					 Unresize/bilinear.h \
					 Unresize/plane.h \
					 Unresize/unresize.cpp \
					 Unresize/unresize.h \
					 Unresize/unresize_impl.cpp \
					 Unresize/unresize_impl.h

include_HEADERS = API/zimg.h \
				  API/zimg++.hpp


libzimg_la_LDFLAGS = -no-undefined -version-info 2


vszimg_la_SOURCES = vszimg/vszimg.c \
					vszimg/VapourSynth.h \
					vszimg/VSHelper.h

vszimg_la_CFLAGS = $(AM_CFLAGS) $(PTHREAD_CFLAGS) -I$(srcdir)/API

vszimg_la_LIBADD = $(PTHREAD_LIBS) libzimg.la

vszimg_la_LDFLAGS = -no-undefined -avoid-version -module

vszimg_la_LIBTOOLFLAGS = --tag=disable-static


if X86SIMD
noinst_LTLIBRARIES += libsse.la libsse2.la libavx2.la

libzimg_la_SOURCES += Colorspace/operation_impl_x86.cpp \
					  Colorspace/operation_impl_x86.h \
					  Depth/depth_convert_x86.cpp \
					  Depth/depth_convert_x86.h \
					  Depth/depth_convert2_x86.cpp \
					  Depth/depth_convert2_x86.h \
					  Depth/dither_impl_x86.cpp \
					  Depth/dither_impl_x86.h \
					  Depth/dither2_x86.cpp \
					  Depth/dither2_x86.h \
					  Resize/resize_impl_x86.cpp \
					  Resize/resize_impl_x86.h \
					  Resize/resize_impl2_x86.cpp \
					  Resize/resize_impl2_x86.h \
					  Unresize/unresize_impl_x86.cpp \
					  Unresize/unresize_impl_x86.h


libsse_la_SOURCES = Resize/resize_impl2_sse.cpp

libsse_la_CXXFLAGS = $(AM_CXXFLAGS) -msse


libsse2_la_SOURCES = Colorspace/operation_impl_sse2.cpp \
					 Depth/depth_convert_sse2.cpp \
					 Depth/dither_impl_sse2.cpp \
					 Depth/dither2_sse2.cpp \
					 Depth/quantize_sse2.h \
					 Resize/resize_impl_sse2.cpp \
					 Resize/resize_impl2_sse2.cpp \
					 Unresize/unresize_impl_sse2.cpp

libsse2_la_CXXFLAGS = $(AM_CXXFLAGS) -msse2


libavx2_la_SOURCES = Colorspace/operation_impl_avx2.cpp \
					 Depth/depth_convert_avx2.cpp \
					 Depth/dither_impl_avx2.cpp \
					 Depth/quantize_avx2.h \
					 Resize/resize_impl_avx2.cpp \
					 Unresize/unresize_impl_avx2.cpp

libavx2_la_CXXFLAGS = $(AM_CXXFLAGS) -mavx2 -mfma -mf16c


libzimg_la_LIBADD = libsse.la libsse2.la libavx2.la
endif


libtestcommon_la_SOURCES = TestCommon/aligned_malloc.h \
						   TestCommon/argparse.cpp \
						   TestCommon/argparse.h \
						   TestCommon/mmap.cpp \
						   TestCommon/mmap.h \
						   TestCommon/timer.h \
						   TestCommon/win32_bitmap.cpp \
						   TestCommon/win32_bitmap.h


if EXAMPLES
noinst_PROGRAMS += Example/api_example \
				   Example/api_example_c \
				   Example/interlace_example
noinst_LTLIBRARIES += libtestcommon.la

EXAMPLE_CPPFLAGS = -I$(srcdir)/API -I$(srcdir)/TestCommon
EXAMPLE_LDADD = libtestcommon.la libzimg.la

Example_api_example_CPPFLAGS = $(EXAMPLE_CPPFLAGS)
Example_api_example_SOURCES = Example/api_example.cpp
Example_api_example_LDADD = $(EXAMPLE_LDADD)

Example_api_example_c_CPPFLAGS = $(EXAMPLE_CPPFLAGS)
Example_api_example_c_SOURCES = Example/api_example_c.c Example/.dummy.cpp
Example_api_example_c_LDADD = $(EXAMPLE_LDADD)

Example_interlace_example_CPPFLAGS = $(EXAMPLE_CPPFLAGS)
Example_interlace_example_SOURCES = Example/interlace_example.cpp
Example_interlace_example_LDADD = $(EXAMPLE_LDADD)
endif # EXAMPLES


if TESTAPP
noinst_PROGRAMS += zimg-test
noinst_LTLIBRARIES += libtestcommon.la
zimg_test_CPPFLAGS = -I$(srcdir)/TestCommon

zimg_test_SOURCES = TestApp/apps.h \
					TestApp/colorspaceapp.cpp \
					TestApp/depthapp.cpp \
					TestApp/frame.cpp \
					TestApp/frame.h \
					TestApp/main.cpp \
					TestApp/pair_filter.cpp \
					TestApp/pair_filter.h \
					TestApp/resizeapp.cpp \
					TestApp/unresizeapp.cpp \
					TestApp/utils.cpp \
					TestApp/utils.h

zimg_test_LDADD = libtestcommon.la libzimg.la
endif # TESTAPP


if UNIT_TEST
TESTS=UnitTest/unit_test

check_LTLIBRARIES = UnitTest/musl_m.la
check_PROGRAMS = UnitTest/unit_test

UnitTest_musl_m_la_CFLAGS = $(AM_CFLAGS) -Wno-sign-compare -Wno-unused-variable -Wno-unused-but-set-variable

UnitTest_musl_m_la_SOURCES = UnitTest/Extra/musl-libm/__cos.c \
								UnitTest/Extra/musl-libm/__rem_pio2.c \
								UnitTest/Extra/musl-libm/__rem_pio2_large.c \
								UnitTest/Extra/musl-libm/__sin.c \
								UnitTest/Extra/musl-libm/cos.c \
								UnitTest/Extra/musl-libm/fpu_wrapper.c \
								UnitTest/Extra/musl-libm/libm.h \
								UnitTest/Extra/musl-libm/mymath.h \
								UnitTest/Extra/musl-libm/pow.c \
								UnitTest/Extra/musl-libm/powf.c \
								UnitTest/Extra/musl-libm/sin.c

UnitTest_musl_m_la_LDFLAGS = -avoid-version -module

UnitTest_unit_test_CPPFLAGS = -I$(srcdir)/UnitTest \
								-I$(srcdir)/UnitTest/Extra \
								-I$(srcdir)/UnitTest/Extra/googletest/googletest/include

UnitTest_unit_test_SOURCES = UnitTest/main.cpp \
								UnitTest/Colorspace/colorspace2_test.cpp \
								UnitTest/Common/audit_buffer.cpp \
								UnitTest/Common/audit_buffer.h \
								UnitTest/Common/copy_filter_test.cpp \
								UnitTest/Common/filter_validator.cpp \
								UnitTest/Common/filter_validator.h \
								UnitTest/Common/filtergraph_test.cpp \
								UnitTest/Common/mock_filter.cpp \
								UnitTest/Common/mock_filter.h \
								UnitTest/Common/mux_filter_test.cpp \
								UnitTest/Depth/depth_convert2_test.cpp \
								UnitTest/Depth/dither2_test.cpp \
								UnitTest/Extra/sha1/config.h \
								UnitTest/Extra/sha1/sha1.c \
								UnitTest/Extra/sha1/sha1.h \
								UnitTest/Resize/resize_impl2_test.cpp

if X86SIMD

UnitTest_unit_test_SOURCES += UnitTest/Depth/dither2_sse2_test.cpp \
								UnitTest/Resize/resize_impl2_sse_test.cpp \
								UnitTest/Resize/resize_impl2_sse2_test.cpp

endif #X86SIMD

UnitTest_unit_test_LDADD = UnitTest/Extra/googletest/googletest/lib/libgtest.la UnitTest/musl_m.la libzimg.la
endif # UNIT_TEST
