#pragma once

#ifndef ZIMG_GRAPH_IMAGE_BUFFER_H_
#define ZIMG_GRAPH_IMAGE_BUFFER_H_

#include <cstddef>

namespace zimg {;
namespace graph {;

template <class T>
struct ImageBufferTemplate {
	T *data[3];
	ptrdiff_t stride[3];
	unsigned mask[3];

	operator const ImageBufferTemplate<const T> &() const
	{
		return reinterpret_cast<const ImageBufferTemplate<const T> &>(*this);
	}
};

typedef ImageBufferTemplate<const void> ImageBufferConst;
typedef ImageBufferTemplate<void> ImageBuffer;

} // namespace graph
} // namespace zimg

#endif // ZIMG_GRAPH_ZTYPES_H_
