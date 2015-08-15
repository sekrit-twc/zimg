#pragma once

#ifndef ZIMG_ZTYPES_H_
#define ZIMG_ZTYPES_H_

#include <cstddef>

namespace zimg {;

const unsigned API_VERSION = 2;

struct ZimgImageBuffer {
	unsigned version = API_VERSION;
	void *data[3];
	ptrdiff_t stride[3];
	unsigned mask[3];
};

struct ZimgFilterFlags {
	unsigned version = API_VERSION;
	unsigned char has_state;
	unsigned char same_row;
	unsigned char in_place;
	unsigned char entire_row;
	unsigned char entire_plane;
	unsigned char color;
};

} // namespace zimg

#endif // ZIMG_ZTYPES_H_
