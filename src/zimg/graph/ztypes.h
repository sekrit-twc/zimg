#pragma once

#ifndef ZIMG_GRAPH_ZTYPES_H_
#define ZIMG_GRAPH_ZTYPES_H_

#include <cstddef>

namespace zimg {;
namespace graph {;

const unsigned API_VERSION = 2;

template <class T>
struct ZimgImageBufferTemplate {
	unsigned version = API_VERSION;
	T *data[3];
	ptrdiff_t stride[3];
	unsigned mask[3];

	operator const ZimgImageBufferTemplate<const T> &() const
	{
		return reinterpret_cast<const ZimgImageBufferTemplate<const T> &>(*this);
	}
};

typedef ZimgImageBufferTemplate<const void> ZimgImageBufferConst;
typedef ZimgImageBufferTemplate<void> ZimgImageBuffer;

struct ZimgFilterFlags {
	unsigned version = API_VERSION;
	unsigned char has_state;
	unsigned char same_row;
	unsigned char in_place;
	unsigned char entire_row;
	unsigned char entire_plane;
	unsigned char color;
};

} // namespace graph
} // namespace zimg

#endif // ZIMG_GRAPH_ZTYPES_H_
