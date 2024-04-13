#pragma once

#ifndef ZIMG_GRAPH_FILTERGRAPH_H_
#define ZIMG_GRAPH_FILTERGRAPH_H_

#include <array>
#include <memory>
#include <utility>
#include "graphengine/types.h"

// Base class in global namespace for API export.
struct zimg_filter_graph {
	virtual inline ~zimg_filter_graph() = 0;
};

zimg_filter_graph::~zimg_filter_graph() = default;

struct zimg_subgraph {
	virtual inline ~zimg_subgraph() = 0;
};

zimg_subgraph::~zimg_subgraph() = default;


namespace graphengine {
class Graph;
class SubGraph;
}

namespace zimg::graph {

class FilterGraph : public zimg_filter_graph {
	typedef int (*callback_type)(void *user, unsigned i, unsigned left, unsigned right);

	std::unique_ptr<graphengine::Graph> m_graph;
	std::shared_ptr<void> m_instance_data;
	graphengine::node_id m_source_id;
	graphengine::node_id m_sink_id;
	bool m_requires_64b;
	bool m_source_greyalpha;
	bool m_sink_greyalpha;
public:
	FilterGraph(std::unique_ptr<graphengine::Graph> graph, std::shared_ptr<void> instance_data, graphengine::node_id source_id, graphengine::node_id sink_id);

	~FilterGraph();

	size_t get_tmp_size() const;

	unsigned get_input_buffering() const;

	unsigned get_output_buffering() const;

	unsigned get_tile_width() const;

	void set_tile_width(unsigned tile_width);

	bool requires_64b_alignment() const { return m_requires_64b; }

	void set_requires_64b_alignment() { m_requires_64b = true; }

	void set_source_greyalpha() { m_source_greyalpha = true; }

	void set_sink_greyalpha() { m_sink_greyalpha = true; }

	void process(const std::array<graphengine::BufferDescriptor, 4> &src, const std::array<graphengine::BufferDescriptor, 4> &dst, void *tmp, callback_type unpack_cb, void *unpack_user, callback_type pack_cb, void *pack_user) const;
};

class SubGraph : public zimg_subgraph {
	typedef std::array<graphengine::node_id, graphengine::NODE_MAX_PLANES> node_list;
	typedef std::array<graphengine::PlaneDescriptor, graphengine::NODE_MAX_PLANES> plane_desc_list;

	std::unique_ptr<graphengine::SubGraph> m_subgraph;
	std::shared_ptr<void> m_instance_data;
	plane_desc_list m_source_desc;
	node_list m_source_ids;
	node_list m_sink_ids;

	bool m_requires_64b;
public:
	SubGraph(std::unique_ptr<graphengine::SubGraph> subgraph, std::shared_ptr<void> instance_data, plane_desc_list source_desc, node_list source_ids, node_list sink_ids);

	~SubGraph();

	std::pair<unsigned, unsigned> get_endpoint_ids(int source_ids[], int sink_ids[]) const noexcept;

	const graphengine::SubGraph *get_subgraph() const noexcept { return m_subgraph.get(); }

	void set_requires_64b_alignment() { m_requires_64b = true; }

	std::unique_ptr<FilterGraph> build_full_graph() const;
};

} // namespace zimg::graph

#endif // ZIMG_GRAPH_FILTERGRAPH_H_
