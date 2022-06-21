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

class FilterExecutor {
public:
	static constexpr int ALL_PLANES = -1;
	static constexpr int CHROMA_PLANES = -2;
private:
	struct data;

	std::unique_ptr<data> m_data;
public:
	FilterExecutor(const graphengine::Filter *filter, const ImageFrame *src_frame, ImageFrame *dst_frame);

	FilterExecutor(const std::vector<std::pair<int, const graphengine::Filter *>> &filters, const ImageFrame *src_frame, ImageFrame *dst_frame);

	FilterExecutor(FilterExecutor &&other) noexcept;

	~FilterExecutor();

	FilterExecutor &operator=(FilterExecutor &&other) noexcept;

	void operator()();
};

#endif // UTILS_H_
