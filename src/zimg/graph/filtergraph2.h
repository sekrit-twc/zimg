#pragma once

#ifndef ZIMG_GRAPH_FILTERGRAPH2_H_
#define ZIMG_GRAPH_FILTERGRAPH2_H_

#include <memory>
#include "image_filter.h"

// Base class in global namespace for API export.
struct zimg_filter_graph {
	virtual inline ~zimg_filter_graph() = 0;
};

zimg_filter_graph::~zimg_filter_graph() = default;


namespace graphengine {
class Graph;
}


namespace zimg {
namespace graph {

class FilterGraph2 : public zimg_filter_graph {
	typedef int (*callback_type)(void *user, unsigned i, unsigned left, unsigned right);

	std::unique_ptr<graphengine::Graph> m_graph;
	std::shared_ptr<void> m_instance_data;
	int m_source_id;
	int m_sink_id;
	bool m_requires_64b;
public:
	FilterGraph2(std::unique_ptr<graphengine::Graph> graph, std::shared_ptr<void> instance_data, int source_id, int sink_id);

	~FilterGraph2();

	size_t get_tmp_size() const;

	unsigned get_input_buffering() const;

	unsigned get_output_buffering() const;

	bool requires_64b_alignment() const { return m_requires_64b; }

	void set_requires_64b_alignment() { m_requires_64b = true; }

	void process(const ColorImageBuffer<const void> &src, const ColorImageBuffer<void> &dst, void *tmp, callback_type unpack_cb, void *unpack_user, callback_type pack_cb, void *pack_user) const;
};

} // namespace graph
} // namespace zimg

#endif // ZIMG_GRAPH_FILTERGRAPH2_H_
