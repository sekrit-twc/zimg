#include <algorithm>
#include <climits>
#include "common/except.h"
#include "common/zassert.h"
#include "graphengine/graph.h"
#include "graphengine/types.h"
#include "filtergraph.h"

namespace zimg {
namespace graph {

FilterGraph::FilterGraph(std::unique_ptr<graphengine::Graph> graph, std::shared_ptr<void> instance_data, graphengine::node_id source_id, graphengine::node_id sink_id) :
	m_graph{ std::move(graph) },
	m_instance_data{ std::move(instance_data) },
	m_source_id{ source_id },
	m_sink_id{ sink_id },
	m_requires_64b{}
{}

FilterGraph::~FilterGraph() = default;

size_t FilterGraph::get_tmp_size() const try
{
	return m_graph->get_tmp_size();
} catch (const std::exception &e) {
	error::throw_<error::InternalError>(e.what());
}

unsigned FilterGraph::get_input_buffering() const try
{
	graphengine::Graph::BufferingRequirement buffering = m_graph->get_buffering_requirement();
	auto it = std::find_if(buffering.begin(), buffering.end(), [=](const auto &entry) { return entry.id == m_source_id; });
	zassert(it != buffering.end(), "invalid node id");
	return std::min(it->mask, UINT_MAX - 1) + 1;
} catch (const std::exception &e) {
	error::throw_<error::InternalError>(e.what());
}

unsigned FilterGraph::get_output_buffering() const try
{
	graphengine::Graph::BufferingRequirement buffering = m_graph->get_buffering_requirement();
	auto it = std::find_if(buffering.begin(), buffering.end(), [=](const auto &entry) { return entry.id == m_sink_id; });
	zassert(it != buffering.end(), "invalid node id");
	return std::min(it->mask, UINT_MAX - 1) + 1;
} catch (const std::exception &e) {
	error::throw_<error::InternalError>(e.what());
}

unsigned FilterGraph::get_tile_width() const try
{
	return graphengine::GraphImpl::from(m_graph.get())->get_tile_width(false);
} catch (const std::exception &e) {
	error::throw_<error::InternalError>(e.what());
}

void FilterGraph::set_tile_width(unsigned tile_width)
{
	graphengine::GraphImpl::from(m_graph.get())->set_tile_width(tile_width);
}

void FilterGraph::process(const std::array<graphengine::BufferDescriptor, 4> &src, const std::array<graphengine::BufferDescriptor, 4> &dst, void *tmp, callback_type unpack_cb, void *unpack_user, callback_type pack_cb, void *pack_user) const
{
	graphengine::Graph::EndpointConfiguration endpoints{};

	endpoints[0] = { m_source_id, {}, {unpack_cb, unpack_user} };
	std::copy_n(src.data(), 4, endpoints[0].buffer);

	endpoints[1] = { m_sink_id, {}, {pack_cb, pack_user} };
	std::copy_n(dst.data(), 4, endpoints[1].buffer);

	try {
		m_graph->run(endpoints, tmp);
	} catch (const graphengine::Graph::CallbackError &) {
		error::throw_<error::UserCallbackFailed>();
	} catch (const std::exception &e) {
		error::throw_<error::InternalError>(e.what());
	}
}

} // namespace graph
} // namespace zimg
