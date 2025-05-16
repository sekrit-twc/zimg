#include <algorithm>
#include <climits>
#include <cstdint>
#include <tuple>
#include "common/align.h"
#include "common/except.h"
#include "common/zassert.h"
#include "graphengine/graph.h"
#include "graphengine/types.h"
#include "filtergraph.h"
#include "graphengine_except.h"

namespace zimg::graph {

FilterGraph::FilterGraph(std::unique_ptr<graphengine::Graph> graph, std::shared_ptr<void> instance_data, graphengine::node_id source_id, graphengine::node_id sink_id) :
	m_graph{ std::move(graph) },
	m_instance_data{ std::move(instance_data) },
	m_source_id{ source_id },
	m_sink_id{ sink_id },
	m_requires_64b{},
	m_source_greyalpha{},
	m_sink_greyalpha{}
{}

FilterGraph::~FilterGraph() = default;

const FilterGraph *FilterGraph::check_alignment(const std::array<graphengine::BufferDescriptor, 4> &src, const std::array<graphengine::BufferDescriptor, 4> &dst) const
{
#define POINTER_ALIGNMENT_ASSERT(x) zassert_d(!(x) || reinterpret_cast<uintptr_t>(x) % alignment == 0, "pointer not aligned")
#define STRIDE_ALIGNMENT_ASSERT(x) zassert_d((x) % alignment == 0, "stride not aligned")
	const int alignment = m_requires_64b ? zimg::ALIGNMENT : zimg::ALIGNMENT_RELAXED;

	POINTER_ALIGNMENT_ASSERT(src[0].ptr);
	POINTER_ALIGNMENT_ASSERT(src[1].ptr);
	POINTER_ALIGNMENT_ASSERT(src[2].ptr);
	POINTER_ALIGNMENT_ASSERT(src[3].ptr);

	STRIDE_ALIGNMENT_ASSERT(src[0].stride);
	STRIDE_ALIGNMENT_ASSERT(src[1].stride);
	STRIDE_ALIGNMENT_ASSERT(src[2].stride);
	STRIDE_ALIGNMENT_ASSERT(src[3].stride);

	POINTER_ALIGNMENT_ASSERT(dst[0].ptr);
	POINTER_ALIGNMENT_ASSERT(dst[1].ptr);
	POINTER_ALIGNMENT_ASSERT(dst[2].ptr);
	POINTER_ALIGNMENT_ASSERT(dst[3].ptr);

	STRIDE_ALIGNMENT_ASSERT(dst[0].stride);
	STRIDE_ALIGNMENT_ASSERT(dst[1].stride);
	STRIDE_ALIGNMENT_ASSERT(dst[2].stride);
	STRIDE_ALIGNMENT_ASSERT(dst[3].stride);
#undef POINTER_ALIGNMENT_ASSERT
#undef STRIDE_ALIGNMENT_ASSERT
	return this;
}

size_t FilterGraph::get_tmp_size() const try
{
	return m_graph->get_tmp_size();
} catch (const graphengine::Exception &e) {
	rethrow_graphengine_exception(e);
}

unsigned FilterGraph::get_input_buffering() const try
{
	graphengine::Graph::BufferingRequirement buffering = m_graph->get_buffering_requirement();
	auto it = std::find_if(buffering.begin(), buffering.end(), [=](const auto &entry) { return entry.id == m_source_id; });
	zassert(it != buffering.end(), "invalid node id");
	return std::min(it->mask, UINT_MAX - 1) + 1;
} catch (const graphengine::Exception &e) {
	rethrow_graphengine_exception(e);
}

unsigned FilterGraph::get_output_buffering() const try
{
	graphengine::Graph::BufferingRequirement buffering = m_graph->get_buffering_requirement();
	auto it = std::find_if(buffering.begin(), buffering.end(), [=](const auto &entry) { return entry.id == m_sink_id; });
	zassert(it != buffering.end(), "invalid node id");
	return std::min(it->mask, UINT_MAX - 1) + 1;
} catch (const graphengine::Exception &e) {
	rethrow_graphengine_exception(e);
}

unsigned FilterGraph::get_tile_width() const try
{
	return graphengine::GraphImpl::from(m_graph.get())->get_tile_width(false);
} catch (const graphengine::Exception &e) {
	rethrow_graphengine_exception(e);
}

void FilterGraph::set_tile_width(unsigned tile_width)
{
	graphengine::GraphImpl::from(m_graph.get())->set_tile_width(tile_width);
}

void FilterGraph::process(const std::array<graphengine::BufferDescriptor, 4> &src, const std::array<graphengine::BufferDescriptor, 4> &dst, void *tmp, callback_type unpack_cb, void *unpack_user, callback_type pack_cb, void *pack_user) const
{
	graphengine::Graph::Endpoint endpoints[] = {
		{ m_source_id, src.data(), { unpack_cb, unpack_user } },
		{ m_sink_id, dst.data(), { pack_cb, pack_user } },
	};

	graphengine::BufferDescriptor src_reorder[2];
	if (m_source_greyalpha) {
		src_reorder[0] = src[0];
		src_reorder[1] = src[3];
		endpoints[0].buffer = src_reorder;
	}

	graphengine::BufferDescriptor dst_reorder[2];
	if (m_sink_greyalpha) {
		dst_reorder[0] = dst[0];
		dst_reorder[1] = dst[3];
		endpoints[1].buffer = dst_reorder;
	}

	try {
		m_graph->run(endpoints, tmp);
	} catch (const graphengine::Exception &e) {
		rethrow_graphengine_exception(e);
	}
}

SubGraph::SubGraph(std::unique_ptr<graphengine::SubGraph> subgraph, std::shared_ptr<void> instance_data, plane_desc_list source_desc, node_list source_ids, node_list sink_ids) :
	m_subgraph(std::move(subgraph)),
	m_instance_data(std::move(instance_data)),
	m_source_desc(source_desc),
	m_source_ids(source_ids),
	m_sink_ids(sink_ids),
	m_requires_64b{}
{}

SubGraph::~SubGraph() = default;

std::pair<unsigned, unsigned> SubGraph::get_endpoint_ids(int source_ids[], int sink_ids[]) const noexcept
{
	unsigned num_sources = 0;
	unsigned num_sinks = 0;

	for (graphengine::node_id id : m_source_ids) {
		if (id != graphengine::null_node)
			source_ids[num_sources++] = id;
	}
	for (graphengine::node_id id : m_sink_ids) {
		if (id != graphengine::null_node)
			sink_ids[num_sinks++] = id;
	}

	return{ num_sources, num_sinks };
}

std::unique_ptr<FilterGraph> SubGraph::build_full_graph() const try
{
	std::unique_ptr<graphengine::Graph> graph = std::make_unique<graphengine::GraphImpl>();
	unsigned num_source_planes = 0;
	unsigned num_sink_planes = 0;
	int subgraph_source_ids[4];
	int subgraph_sink_ids[4];

	std::tie(num_source_planes, num_sink_planes) = get_endpoint_ids(subgraph_source_ids, subgraph_sink_ids);
	graphengine::node_id real_source_id = graph->add_source(num_source_planes, m_source_desc.data());

	graphengine::SubGraph::Mapping source_mapping[graphengine::NODE_MAX_PLANES];
	graphengine::SubGraph::Mapping sink_mapping[graphengine::NODE_MAX_PLANES];

	for (unsigned i = 0; i < num_source_planes; ++i) {
		source_mapping[i].internal_id = subgraph_source_ids[i];
		source_mapping[i].external_dep = { real_source_id, i };
	}

	m_subgraph->connect(graph.get(), num_source_planes, source_mapping, sink_mapping);

	graphengine::node_dep_desc real_sink_deps[graphengine::NODE_MAX_PLANES];
	for (unsigned i = 0; i < num_sink_planes; ++i) {
		graphengine::SubGraph::Mapping *mapping = std::find_if(sink_mapping, sink_mapping + graphengine::NODE_MAX_PLANES,
			[=](const auto &mapping) { return mapping.internal_id == subgraph_sink_ids[i]; });

		real_sink_deps[i] = mapping->external_dep;
	}

	graphengine::node_id real_sink_id = graph->add_sink(num_sink_planes, real_sink_deps);

	std::unique_ptr<FilterGraph> filtergraph = std::make_unique<FilterGraph>(std::move(graph), m_instance_data, real_source_id, real_sink_id);
	if (m_requires_64b)
		filtergraph->set_requires_64b_alignment();
	if (num_source_planes == 2)
		filtergraph->set_source_greyalpha();
	if (num_sink_planes == 2)
		filtergraph->set_sink_greyalpha();

	return filtergraph;
} catch (const graphengine::Exception &e) {
	rethrow_graphengine_exception(e);
} catch (const std::exception &e) {
	error::throw_<error::InternalError>(e.what());
}

} // namespace zimg::graph
