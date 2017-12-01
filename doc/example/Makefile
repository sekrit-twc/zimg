ZIMG_INCLUDES := $(shell pkg-config --cflags zimg)
ZIMG_LIBS := $(shell pkg-config --libs zimg)

MY_CFLAGS := -std=c89 $(CFLAGS)
MY_CXXFLAGS := -std=c++11 $(CXXFLAGS)
MY_CPPFLAGS := $(ZIMG_INCLUDES) -Imisc $(CPPFLAGS)
MY_LDFLAGS := $(LDFLAGS)
MY_LIBS := $(ZIMG_LIBS) $(LIBS)

misc_OBJS = \
	misc/argparse.o \
	misc/mmap.o \
	misc/win32_bitmap.o

all: alpha_example api_example api_example_c hdr_example interlace_example

alpha_example: alpha_example.o $(misc_OBJS)
	$(CXX) $(MY_LDFLAGS) $^ $(MY_LIBS) -o $@
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

%.o: %.c
	$(CC) -c $(MY_CFLAGS) $(MY_CPPFLAGS) $< -o $@
%.o: %.cpp
	$(CXX) -c $(MY_CXXFLAGS) $(MY_CPPFLAGS) $< -o $@

clean:
	rm -f *.a *.o misc/*.o alpha_example api_example api_example_c interlace_example

.PHONY: clean

