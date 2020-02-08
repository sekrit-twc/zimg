ZIMG_INCLUDES := $(shell pkg-config --cflags zimg)
ZIMG_LIBS := $(shell pkg-config --libs zimg)

MY_CFLAGS := -std=c89 -pthread $(CFLAGS)
MY_CXXFLAGS := -std=c++11 -pthread $(CXXFLAGS)
MY_CPPFLAGS := $(ZIMG_INCLUDES) -Imisc $(CPPFLAGS)
MY_LDFLAGS := $(LDFLAGS)
MY_LIBS := -pthread $(ZIMG_LIBS) $(LIBS)

misc_HDRS = \
	misc/aligned_malloc.h \
	misc/argparse.h \
	misc/mmap.h \
	misc/win32_bitmap.h

misc_OBJS = \
	misc/argparse.o \
	misc/mmap.o \
	misc/win32_bitmap.o

all: api_example api_example_c hdr_example interlace_example tile_example

api_example: api_example.o $(misc_OBJS)
	$(CXX) $(MY_LDFLAGS) $^ $(MY_LIBS) -o $@
api_example_c: api_example_c.o $(misc_OBJS)
	$(CXX) $(MY_LDFLAGS) $^ $(MY_LIBS) -o $@
hdr_example: hdr_example.o $(misc_OBJS)
	$(CXX) $(MY_LDFLAGS) $^ $(MY_LIBS) -o $@
interlace_example: interlace_example.o $(misc_OBJS)
	$(CXX) $(MY_LDFLAGS) $^ $(MY_LIBS) -o $@
tile_example: tile_example.o $(misc_OBJS)
	$(CXX) $(MY_LDFLAGS) $^ $(MY_LIBS) -o $@

%.o: %.c $(misc_HDRS)
	$(CC) -c $(MY_CFLAGS) $(MY_CPPFLAGS) $< -o $@
%.o: %.cpp $(misc_HDRS)
	$(CXX) -c $(MY_CXXFLAGS) $(MY_CPPFLAGS) $< -o $@

clean:
	rm -f *.a *.o misc/*.o api_example api_example_c hdr_example interlace_example tile_example

.PHONY: clean
