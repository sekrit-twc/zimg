warningflags = -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers
commonflags = -O2 $(warningflags) $(DEBUGCFLAGS)
AM_CXXFLAGS = -std=c++11 $(commonflags)
AM_CFLAGS = $(commonflags)


pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = zimg.pc

EXTRA_DIST = zimg.pc.in


lib_LTLIBRARIES = libzimg.la

pkglib_LTLIBRARIES = vszimg.la

libzimg_la_SOURCES = API/zimg.cpp \
					 Colorspace/colorspace.cpp \
					 Colorspace/colorspace.h \
					 Colorspace/colorspace_param.cpp \
					 Colorspace/colorspace_param.h \
					 Colorspace/graph.cpp \
					 Colorspace/graph.h \
					 Colorspace/matrix3.cpp \
					 Colorspace/matrix3.h \
					 Colorspace/operation.cpp \
					 Colorspace/operation.h \
					 Colorspace/operation_impl.cpp \
					 Colorspace/operation_impl.h \
					 Common/align.h \
					 Common/cpuinfo.h \
					 Common/except.h \
					 Common/matrix.h \
					 Common/osdep.h \
					 Common/pixel.h \
					 Common/plane.h \
					 Depth/depth_convert.cpp \
					 Depth/depth_convert.h \
					 Depth/depth.cpp \
					 Depth/depth.h \
					 Depth/dither.cpp \
					 Depth/dither.h \
					 Depth/dither_impl.cpp \
					 Depth/dither_impl.h \
					 Depth/error_diffusion.cpp \
					 Depth/error_diffusion.h \
					 Depth/quantize.h \
					 Resize/filter.cpp \
					 Resize/filter.h \
					 Resize/resize.cpp \
					 Resize/resize.h \
					 Resize/resize_impl.cpp \
					 Resize/resize_impl.h \
					 Unresize/bilinear.cpp \
					 Unresize/bilinear.h \
					 Unresize/unresize.cpp \
					 Unresize/unresize.h \
					 Unresize/unresize_impl.cpp \
					 Unresize/unresize_impl.h

include_HEADERS = API/zimg.h \
				  API/zimg++.hpp


libzimg_la_LDFLAGS = -no-undefined -version-info 1


vszimg_la_SOURCES = vszimg/vszimg.c \
					vszimg/VapourSynth.h \
					vszimg/VSHelper.h

vszimg_la_CFLAGS = $(AM_CFLAGS) -I$(srcdir)/API

vszimg_la_LIBADD = libzimg.la

vszimg_la_LDFLAGS = -no-undefined -avoid-version -module

vszimg_la_LIBTOOLFLAGS = --tag=disable-static


if X86SIMD
noinst_LTLIBRARIES = libsse2.la libavx2.la

libzimg_la_SOURCES += Colorspace/operation_impl_x86.cpp \
					  Colorspace/operation_impl_x86.h \
					  Depth/depth_convert_x86.cpp \
					  Depth/depth_convert_x86.h \
					  Depth/dither_impl_x86.cpp \
					  Depth/dither_impl_x86.h \
					  Resize/resize_impl_x86.cpp \
					  Resize/resize_impl_x86.h \
					  Unresize/unresize_impl_x86.cpp \
					  Unresize/unresize_impl_x86.h


libsse2_la_SOURCES = Colorspace/operation_impl_sse2.cpp \
					 Depth/depth_convert_sse2.cpp \
					 Depth/dither_impl_sse2.cpp \
					 Depth/quantize_sse2.h \
					 Resize/resize_impl_sse2.cpp \
					 Unresize/unresize_impl_sse2.cpp

libsse2_la_CXXFLAGS = $(AM_CXXFLAGS) -msse2


libavx2_la_SOURCES = Colorspace/operation_impl_avx2.cpp \
					 Depth/depth_convert_avx2.cpp \
					 Depth/dither_impl_avx2.cpp \
					 Depth/quantize_avx2.h \
					 Resize/resize_impl_avx2.cpp \
					 Unresize/unresize_impl_avx2.cpp

libavx2_la_CXXFLAGS = $(AM_CXXFLAGS) -mavx2 -mfma -mf16c


libzimg_la_LIBADD = libsse2.la libavx2.la
endif


if TESTAPP
bin_PROGRAMS = zimg-test

zimg_test_SOURCES = TestApp/apps.h \
					TestApp/colorspaceapp.cpp \
					TestApp/depthapp.cpp \
					TestApp/frame.cpp \
					TestApp/frame.h \
					TestApp/main.cpp \
					TestApp/resizeapp.cpp \
					TestApp/timer.h \
					TestApp/unresizeapp.cpp \
					TestApp/utils.cpp \
					TestApp/utils.h

zimg_test_LDADD = libzimg.la
endif # TESTAPP
