#include <algorithm>
#include <climits>
#include "common/except.h"
#include "common/zassert.h"
#include "graphengine/graph.h"
#include "graphengine/types.h"
#include "filtergraph2.h"

namespace zimg {
namespace graph {

FilterGraph2::FilterGraph2(std::unique_ptr<graphengine::Graph> graph, std::shared_ptr<void> instance_data, int source_id, int sink_id) :
	m_graph{ std::move(graph) },
	m_instance_data{ std::move(instance_data) },
	m_source_id{ source_id },
	m_sink_id{ sink_id },
	m_requires_64b{}
{}

FilterGraph2::~FilterGraph2() = default;

size_t FilterGraph2::get_tmp_size() const try
{
	return m_graph->get_tmp_size();
} catch (const std::exception &e) {
	error::throw_<error::InternalError>(e.what());
}

unsigned FilterGraph2::get_input_buffering() const try
{
	graphengine::Graph::BufferingRequirement buffering = m_graph->get_buffering_requirement();
	auto it = std::find_if(buffering.begin(), buffering.end(), [=](const auto &entry) { return entry.first == m_source_id; });
	zassert(it != buffering.end(), "invalid node id");
	return std::min(it->second, UINT_MAX - 1) + 1;
} catch (const std::exception &e) {
	error::throw_<error::InternalError>(e.what());
}

unsigned FilterGraph2::get_output_buffering() const try
{
	graphengine::Graph::BufferingRequirement buffering = m_graph->get_buffering_requirement();
	auto it = std::find_if(buffering.begin(), buffering.end(), [=](const auto &entry) { return entry.first == m_sink_id; });
	zassert(it != buffering.end(), "invalid node id");
	return std::min(it->second, UINT_MAX - 1) + 1;
} catch (const std::exception &e) {
	error::throw_<error::InternalError>(e.what());
}

unsigned FilterGraph2::get_tile_width() const try
{
	return m_graph->get_tile_width(false);
} catch (const std::exception &e) {
	error::throw_<error::InternalError>(e.what());
}

void FilterGraph2::set_tile_width(unsigned tile_width)
{
	m_graph->set_tile_width(tile_width);
}

void FilterGraph2::process(const ColorImageBuffer<const void> &src, const ColorImageBuffer<void> &dst, void *tmp, callback_type unpack_cb, void *unpack_user, callback_type pack_cb, void *pack_user) const
{
	graphengine::Graph::EndpointConfiguration endpoints{};

	endpoints[0] = {
		m_source_id,
		{
			{ const_cast<void *>(src[0].data()), src[0].stride(), src[0].mask() },
			{ const_cast<void *>(src[1].data()), src[1].stride(), src[1].mask() },
			{ const_cast<void *>(src[2].data()), src[2].stride(), src[2].mask() },
			{ const_cast<void *>(src[3].data()), src[3].stride(), src[3].mask() },
		},
		{ unpack_cb, unpack_user }
	};
	endpoints[1] = {
		m_sink_id,
		{
			{ dst[0].data(), dst[0].stride(), dst[0].mask() },
			{ dst[1].data(), dst[1].stride(), dst[1].mask() },
			{ dst[2].data(), dst[2].stride(), dst[2].mask() },
			{ dst[3].data(), dst[3].stride(), dst[3].mask() },
		},
		{ pack_cb, pack_user }
	};

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
