#pragma once

#ifndef ZIMG_GRAPH_ZTYPES_H_
#define ZIMG_GRAPH_ZTYPES_H_

#include <cstddef>

namespace zimg {;
namespace graph {;

template <class T>
struct ZimgImageBufferTemplate {
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

} // namespace graph
} // namespace zimg

#endif // ZIMG_GRAPH_ZTYPES_H_
