commonflags = -O2 -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers
AM_CXXFLAGS = -std=c++11 $(commonflags)
AM_CFLAGS = $(commonflags)

lib_LTLIBRARIES = libzimg.la

pkglib_LTLIBRARIES = vszimg.la

libzimg_la_SOURCES = API/zimg.cpp \
					 Colorspace/colorspace.cpp \
					 Colorspace/colorspace_param.cpp \
					 Colorspace/graph.cpp \
					 Colorspace/matrix3.cpp \
					 Colorspace/operation.cpp \
					 Colorspace/operation_impl.cpp \
					 Depth/depth_convert.cpp \
					 Depth/depth.cpp \
					 Depth/dither.cpp \
					 Depth/dither_impl.cpp \
					 Depth/error_diffusion.cpp \
					 Resize/filter.cpp \
					 Resize/resize.cpp \
					 Resize/resize_impl.cpp

include_HEADERS = API/zimg.h \
				  API/zimg++.hpp


libzimg_la_LDFLAGS = -no-undefined -version-info 0 -export-symbols-regex '^zimg_'


vszimg_la_SOURCES = vszimg/vszimg.c

vszimg_la_CFLAGS = $(AM_CFLAGS) -I$(srcdir)/API

vszimg_la_LIBADD = libzimg.la

vszimg_la_LDFLAGS = -no-undefined -avoid-version -module

vszimg_la_LIBTOOLFLAGS = --tag=disable-static


if X86SIMD
noinst_LTLIBRARIES = libsse2.la libavx2.la libf16c.la

libzimg_la_SOURCES += Colorspace/operation_impl_x86.cpp \
					  Resize/resize_impl_x86.cpp


libsse2_la_SOURCES = Colorspace/operation_impl_sse2.cpp \
					 Resize/resize_impl_sse2.cpp

libsse2_la_CXXFLAGS = $(AM_CXXFLAGS) -msse2


libavx2_la_SOURCES = Colorspace/operation_impl_avx2.cpp \
					 Resize/resize_impl_avx2.cpp

libavx2_la_CXXFLAGS = $(AM_CXXFLAGS) -mavx2 -mfma -mf16c


libf16c_la_SOURCES = Colorspace/operation_impl_f16c.cpp

libf16c_la_CXXFLAGS = $(AM_CXXFLAGS) -mf16c


libzimg_la_LIBADD = libsse2.la libavx2.la libf16c.la
endif
