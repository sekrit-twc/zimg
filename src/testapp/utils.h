#pragma once

#ifndef UTILS_H_
#define UTILS_H_

#include <memory>
#include <utility>
#include <vector>

namespace graphengine {
class Filter;
}


namespace zimg {
namespace graph {

class ImageFilter;

} // namespace graph
} // namespace zimg


class ImageFrame;

class FilterExecutor_GE {
public:
	static constexpr int ALL_PLANES = -1;
	static constexpr int CHROMA_PLANES = -2;
private:
	struct data;

	std::unique_ptr<data> m_data;
public:
	FilterExecutor_GE(const graphengine::Filter *filter, const ImageFrame *src_frame, ImageFrame *dst_frame);

	FilterExecutor_GE(const std::vector<std::pair<int, const graphengine::Filter *>> &filters, const ImageFrame *src_frame, ImageFrame *dst_frame);

	FilterExecutor_GE(FilterExecutor_GE &&other) noexcept;

	~FilterExecutor_GE();

	FilterExecutor_GE &operator=(FilterExecutor_GE &&other) noexcept;

	void operator()();
};

#endif // UTILS_H_
