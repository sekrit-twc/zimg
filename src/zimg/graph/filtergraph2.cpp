#include <algorithm>
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>
#include "common/except.h"
#include "common/make_unique.h"
#include "common/pixel.h"
#include "common/zassert.h"
#include "copy_filter.h"
#include "filtergraph2.h"
#include "graphnode.h"

namespace zimg {
namespace graph {


class FilterGraph2::impl {
	std::vector<std::unique_ptr<GraphNode>> m_nodes;
	SimulationState::result m_interleaved_sim;
	SimulationState::result m_planar_sim[PLANE_NUM];
	GraphNode *m_source;
	GraphNode *m_sink;
	node_map m_output_nodes;
	size_t m_tmp_size;
	bool m_planar;

	node_id next_id() const { return static_cast<node_id>(m_nodes.size()); }

	node_map id_to_node(const id_map &ids) const
	{
		node_map nodes{};

		for (int p = 0; p < PLANE_NUM; ++p) {
			if (ids[p] < 0)
				continue;

			zassert_d(ids[p] < next_id(), "node index out of range");
			nodes[p] = m_nodes[ids[p]].get();
		}

		return nodes;
	}

	void add_ref(const node_map &nodes)
	{
		std::unordered_set<GraphNode *> unique{ nodes.begin(), nodes.end() };

		for (GraphNode *node : unique) {
			if (node)
				node->add_ref();
		}
	}

	void simulate_interleaved()
	{
		SimulationState sim{ m_nodes.size() };
		unsigned height = m_sink->get_image_attributes(PLANE_Y).height;
		unsigned step = 1U << m_sink->get_subsample_h();

		for (unsigned cursor = 0; cursor < height; cursor += step) {
			m_sink->simulate(&sim, cursor, cursor + step, PLANE_Y);
		}
		m_sink->simulate_alloc(&sim);

		m_interleaved_sim = sim.get_result(m_nodes);
		m_tmp_size = std::max(m_tmp_size, ExecutionState::calculate_tmp_size(m_interleaved_sim, m_nodes));
	}

	void simulate_planar()
	{
		if (!m_planar)
			return;

		for (int p = 0; p < PLANE_NUM; ++p) {
			if (!m_output_nodes[p])
				continue;

			SimulationState sim{ m_nodes.size() };
			unsigned height = m_output_nodes[p]->get_image_attributes(p).height;

			for (unsigned i = 0; i < height; ++i) {
				m_output_nodes[p]->simulate(&sim, i, i + 1, p);
			}
			m_output_nodes[p]->simulate_alloc(&sim);

			m_planar_sim[p] = sim.get_result(m_nodes);
			m_tmp_size = std::max(m_tmp_size, ExecutionState::calculate_tmp_size(m_planar_sim[p], m_nodes));
		}
	}

	void process_interleaved(const ImageBuffer<const void> src[], const ImageBuffer<void> dst[], void *tmp, callback unpack_cb, callback pack_cb) const
	{
		ExecutionState state{ m_interleaved_sim, m_nodes, m_source->cache_id(), m_sink->cache_id(), src, dst, unpack_cb, pack_cb, tmp };

		m_sink->init_context(&state);
		m_sink->generate(&state, m_sink->get_image_attributes(PLANE_Y).height, PLANE_Y);
	}

	void process_planar(const ImageBuffer<const void> src[], const ImageBuffer<void> dst[], void *tmp) const
	{
		for (int p = 0; p < PLANE_NUM; ++p) {
			if (!m_output_nodes[p])
				continue;

			ExecutionState state{ m_planar_sim[p], m_nodes, m_source->cache_id(), m_sink->cache_id(), src, dst, nullptr, nullptr, tmp };
			m_output_nodes[p]->init_context(&state);
			m_output_nodes[p]->generate(&state, m_output_nodes[p]->get_image_attributes(p).height, p);
		}
	}
public:
	impl() :
		m_interleaved_sim{},
		m_planar_sim{},
		m_source{},
		m_sink{},
		m_output_nodes{},
		m_tmp_size{},
		m_planar{ true }
	{}

	node_id add_source(const ImageFilter::image_attributes &attr, unsigned subsample_w, unsigned subsample_h, const plane_mask &planes)
	{
		zassert_d(!m_source, "source already defined");
		m_nodes.emplace_back(make_source_node(next_id(), attr, subsample_w, subsample_h, planes));
		m_source = m_nodes.back().get();
		return m_source->id();
	}

	node_id attach_filter(std::shared_ptr<ImageFilter> filter, const id_map &deps, const plane_mask &output_planes)
	{
		node_map parents = id_to_node(deps);
		add_ref(parents);

		plane_mask input_planes{};
		for (int p = 0; p < PLANE_NUM; ++p) {
			input_planes[p] = !!parents[p];
		}

		size_t input_plane_count = std::count(input_planes.begin(), input_planes.end(), true);
		size_t output_plane_count = std::count(output_planes.begin(), output_planes.end(), true);

		if (output_plane_count > 1 || input_plane_count > 1 || (input_plane_count > 0 && input_planes != output_planes))
			m_planar = false;

		m_nodes.emplace_back(make_filter_node(next_id(), std::move(filter), id_to_node(deps), output_planes));
		return m_nodes.back()->id();
	}

	void set_output(const id_map &deps)
	{
		zassert_d(!m_sink, "sink already defined");
		node_map parents = id_to_node(deps);

		for (int p = 0; p < PLANE_NUM; ++p) {
			GraphNode *node = parents[p];
			if (!node)
				continue;

			bool need_copy = false;

			// If the node is the source, then a copy is needed, because the source buffer is external.
			// If the node is not a terminal, then a copy is also needed.
			if (node->is_sourcesink() || node->ref_count() > 0) {
				need_copy = true;
			} else {
				// If the node produces planes that do not contribute to the output, then a copy is needed.
				plane_mask mask = node->get_plane_mask();

				for (int q = 0; q < PLANE_NUM; ++q) {
					if (mask[q] && parents[q] != node) {
						need_copy = true;
						break;
					}
				}
			}

			if (need_copy) {
				id_map deps = null_ids;
				deps[p] = node->id();

				plane_mask mask{};
				mask[p] = true;

				auto attr = node->get_image_attributes(p);
				node_id id = attach_filter(ztd::make_unique<CopyFilter>(attr.width, attr.height, attr.type), deps, mask);
				parents[p] = m_nodes[id].get();
			}
		}
		add_ref(parents);

		m_output_nodes = parents;
		m_nodes.emplace_back(make_sink_node(next_id(), m_output_nodes));
		m_sink = m_nodes.back().get();
		m_sink->add_ref();

		for (const auto &node : m_nodes) {
			node->try_inplace();
		}

		simulate_interleaved();
		simulate_planar();
	}

	size_t get_tmp_size() const { return m_tmp_size; }

	unsigned get_input_buffering() const
	{
		return m_interleaved_sim.node_result[m_source->id()].cache_lines;
	}

	unsigned get_output_buffering() const
	{
		return m_interleaved_sim.node_result[m_sink->id()].cache_lines;
	}

	void process(const ImageBuffer<const void> src[], const ImageBuffer<void> dst[], void *tmp, callback unpack_cb, callback pack_cb) const
	{
		if (!m_planar || unpack_cb || pack_cb)
			process_interleaved(src, dst, tmp, unpack_cb, pack_cb);
		else
			process_planar(src, dst, tmp);
	}
};


FilterGraph2::callback::callback(std::nullptr_t) : m_func{}, m_user{} {}

FilterGraph2::callback::callback(func_type func, void *user) : m_func{ func }, m_user{ user } {}

FilterGraph2::callback::operator bool() const { return m_func != nullptr; }

void FilterGraph2::callback::operator()(unsigned i, unsigned left, unsigned right) const
{
	int ret;

	try {
		ret = m_func(m_user, i, left, right);
	} catch (...) {
		ret = 1;
		zassert_dfatal("user callback must not throw");
	}

	if (ret)
		error::throw_<error::UserCallbackFailed>("user callback failed");
}


FilterGraph2::FilterGraph2() : m_impl(ztd::make_unique<impl>()) {}

FilterGraph2::FilterGraph2(FilterGraph2 &&other) noexcept = default;

FilterGraph2::~FilterGraph2() = default;

FilterGraph2 &FilterGraph2::operator=(FilterGraph2 &&other) noexcept = default;

node_id FilterGraph2::add_source(const ImageFilter::image_attributes &attr, unsigned subsample_w, unsigned subsample_h, const plane_mask &planes)
{
	return get_impl()->add_source(attr, subsample_w, subsample_h, planes);
}

node_id FilterGraph2::attach_filter(std::shared_ptr<ImageFilter> filter, const id_map &deps, const plane_mask &output_planes)
{
	return get_impl()->attach_filter(std::move(filter), deps, output_planes);
}

void FilterGraph2::set_output(const id_map &deps)
{
	get_impl()->set_output(deps);
}

size_t FilterGraph2::get_tmp_size() const
{
	return get_impl()->get_tmp_size();
}

unsigned FilterGraph2::get_input_buffering() const
{
	return get_impl()->get_input_buffering();
}

unsigned FilterGraph2::get_output_buffering() const
{
	return get_impl()->get_output_buffering();
}

void FilterGraph2::process(const ImageBuffer<const void> src[], const ImageBuffer<void> dst[], void *tmp, callback unpack_cb, callback pack_cb) const
{
	get_impl()->process(src, dst, tmp, unpack_cb, pack_cb);
}

} // namespace graph
} // namespace zimg
