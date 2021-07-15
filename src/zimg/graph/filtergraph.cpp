#include <algorithm>
#include <cmath>
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>
#include "common/align.h"
#include "common/checked_int.h"
#include "common/cpuinfo.h"
#include "common/except.h"
#include "common/pixel.h"
#include "common/zassert.h"
#include "basic_filter.h"
#include "filtergraph.h"
#include "graphnode.h"
#include "image_buffer.h"

namespace zimg {
namespace graph {

namespace {

constexpr unsigned TILE_WIDTH_MIN = 128;

unsigned calculate_tile_width(size_t cpu_cache, size_t footprint, unsigned width)
{
	unsigned tile = static_cast<unsigned>(std::lrint(width * std::min(static_cast<double>(cpu_cache) / footprint, 1.0)));

	if (tile > (width / 5) * 4)
		return width;
	else if (tile > width / 2)
		return ceil_n(width / 2, ALIGNMENT);
	else if (tile > width / 3)
		return ceil_n(width / 3, ALIGNMENT);
	else
		return std::max(floor_n(tile, ALIGNMENT), TILE_WIDTH_MIN);
}

} // namespace


class FilterGraph::impl {
	std::vector<std::unique_ptr<GraphNode>> m_nodes;
	SimulationState::result m_interleaved_sim;
	SimulationState::result m_planar_sim[PLANE_NUM];
	GraphNode *m_source;
	GraphNode *m_sink;
	node_map m_output_nodes;
	unsigned m_interleaved_tile_width;
	unsigned m_planar_tile_width[PLANE_NUM];
	size_t m_tmp_size;
	bool m_entire_row;
	bool m_planar;
	bool m_requires_64b_alignment;

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

	size_t calculate_cache_footprint(SimulationState::result &sim, int plane) const
	{
		const GraphNode *out = plane < 0 ? m_sink : m_output_nodes[plane];
		unsigned input_lines = sim.node_result[m_source->id()].cache_lines;
		unsigned output_lines = sim.node_result[out->id()].cache_lines;

		checked_size_t footprint = ExecutionState::calculate_tmp_size(sim, m_nodes);

		if (plane < 0) {
			auto input_attr = m_source->get_image_attributes(PLANE_Y);
			auto output_attr = out->get_image_attributes(PLANE_Y);
			plane_mask input_planes = m_source->get_plane_mask();
			plane_mask output_planes = out->get_plane_mask();

			for (int p = 0; p < PLANE_NUM; ++p) {
				if (input_planes[p]) {
					unsigned width = input_attr.width >> (p == PLANE_U || p == PLANE_V ? m_source->get_subsample_w() : 0);
					unsigned lines = input_lines >> (p == PLANE_U || p == PLANE_V ? m_source->get_subsample_h() : 0);

					footprint += ceil_n(static_cast<checked_size_t>(width) * pixel_size(input_attr.type), ALIGNMENT) * lines;
				}
				if (output_planes[p]) {
					unsigned width = output_attr.width >> (p == PLANE_U || p == PLANE_V ? out->get_subsample_w() : 0);
					unsigned lines = output_lines >> (p == PLANE_U || p == PLANE_V ? out->get_subsample_h() : 0);

					footprint += ceil_n(static_cast<checked_size_t>(width) * pixel_size(output_attr.type), ALIGNMENT) * lines;
				}
			}
		} else {
			if (m_source->get_plane_mask()[plane]) {
				auto input_attr = m_source->get_image_attributes(plane);
				footprint += ceil_n(static_cast<checked_size_t>(input_attr.width) * pixel_size(input_attr.type), ALIGNMENT) * input_lines;
			}
			if (m_sink->get_plane_mask()[plane]) {
				auto output_attr = m_sink->get_image_attributes(plane);
				footprint += ceil_n(static_cast<checked_size_t>(output_attr.width) * pixel_size(output_attr.type), ALIGNMENT) * output_lines;
			}
		}

		return footprint.get();
	}

	void simulate_interleaved()
	{
		SimulationState sim{ m_nodes };
		unsigned height = m_sink->get_image_attributes(PLANE_Y).height;
		unsigned step = 1U << m_sink->get_subsample_h();

		for (unsigned cursor = 0; cursor < height; cursor += step) {
			m_sink->simulate(&sim, cursor, cursor + step, PLANE_Y);
		}
		m_sink->simulate_alloc(&sim);

		m_interleaved_sim = sim.get_result(m_nodes);
		m_tmp_size = std::max(m_tmp_size, ExecutionState::calculate_tmp_size(m_interleaved_sim, m_nodes));

		if (!m_interleaved_tile_width) {
			if (!m_entire_row) {
				size_t cpu_cache = cpu_cache_size();
				size_t footprint = calculate_cache_footprint(m_interleaved_sim, -1);
				m_interleaved_tile_width = calculate_tile_width(cpu_cache, footprint, m_sink->get_image_attributes(PLANE_Y).width);
			} else {
				m_interleaved_tile_width = m_sink->get_image_attributes(PLANE_Y).width;
			}
		}
	}

	void simulate_planar()
	{
		if (!m_planar)
			return;

		for (int p = 0; p < PLANE_NUM; ++p) {
			if (!m_output_nodes[p])
				continue;

			SimulationState sim{ m_nodes };
			unsigned height = m_output_nodes[p]->get_image_attributes(p).height;

			for (unsigned i = 0; i < height; ++i) {
				m_output_nodes[p]->simulate(&sim, i, i + 1, p);
			}
			m_output_nodes[p]->simulate_alloc(&sim);

			m_planar_sim[p] = sim.get_result(m_nodes);
			m_tmp_size = std::max(m_tmp_size, ExecutionState::calculate_tmp_size(m_planar_sim[p], m_nodes));

			if (!m_planar_tile_width[p]) {
				if (!m_entire_row) {
					size_t cpu_cache = cpu_cache_size();
					size_t footprint = calculate_cache_footprint(m_planar_sim[p], p);
					m_planar_tile_width[p] = calculate_tile_width(cpu_cache, footprint, m_output_nodes[p]->get_image_attributes(p).width);
				} else {
					m_planar_tile_width[p] = m_output_nodes[p]->get_image_attributes(p).width;
				}
			}
		}
	}

	void process_interleaved(const ImageBuffer<const void> src[], const ImageBuffer<void> dst[], void *tmp, callback unpack_cb, callback pack_cb) const
	{
		ExecutionState state{ m_interleaved_sim, m_nodes, m_source->cache_id(), m_sink->cache_id(), src, dst, unpack_cb, pack_cb, tmp };
		auto attr = m_sink->get_image_attributes(PLANE_Y);

		for (unsigned j = 0; j < attr.width;) {
			unsigned j_end = j + std::min(m_interleaved_tile_width, attr.width - j);
			if (attr.width - j_end < TILE_WIDTH_MIN)
				j_end = attr.width;

			state.reset_initialized(m_nodes.size());
			m_sink->init_context(&state, 0, j, j_end, PLANE_Y);
			m_sink->generate(&state, attr.height, PLANE_Y);

			j = j_end;
		}
	}

	void process_planar(const ImageBuffer<const void> src[], const ImageBuffer<void> dst[], void *tmp) const
	{
		for (int p = 0; p < PLANE_NUM; ++p) {
			if (!m_output_nodes[p])
				continue;

			ExecutionState state{ m_planar_sim[p], m_nodes, m_source->cache_id(), m_sink->cache_id(), src, dst, nullptr, nullptr, tmp };
			auto attr = m_output_nodes[p]->get_image_attributes(p);

			for (unsigned j = 0; j < attr.width;) {
				unsigned j_end = j + std::min(m_planar_tile_width[p], attr.width - j);
				if (attr.width - j_end < TILE_WIDTH_MIN)
					j_end = attr.width;

				state.reset_initialized(m_nodes.size());
				m_output_nodes[p]->init_context(&state, 0, j, j_end, p);
				m_output_nodes[p]->generate(&state, attr.height, p);

				j = j_end;
			}
		}
	}
public:
	impl() :
		m_interleaved_sim{},
		m_planar_sim{},
		m_source{},
		m_sink{},
		m_output_nodes{},
		m_interleaved_tile_width{},
		m_planar_tile_width{},
		m_tmp_size{},
		m_entire_row{},
		m_planar{ true },
		m_requires_64b_alignment{}
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
		if (filter->get_flags().entire_row)
			m_entire_row = true;

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
				node_id id = attach_filter(std::make_unique<CopyFilter>(attr.width, attr.height, attr.type), deps, mask);
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
		zassert_d(m_sink, "complete graph required");
		unsigned lines = m_interleaved_sim.node_result[m_source->id()].cache_lines;
		return lines >= m_source->get_image_attributes(PLANE_Y).height ? BUFFER_MAX : lines;
	}

	unsigned get_output_buffering() const
	{
		zassert_d(m_sink, "complete graph required");
		unsigned lines = m_interleaved_sim.node_result[m_sink->id()].cache_lines;
		return lines >= m_sink->get_image_attributes(PLANE_Y).height ? BUFFER_MAX : lines;
	}

	unsigned get_tile_width() const
	{
		zassert_d(m_sink, "complete graph required");
		return m_planar ? m_planar_tile_width[PLANE_Y] : m_interleaved_tile_width;
	}

	void set_tile_width(unsigned tile_width)
	{
		zassert_d(m_sink, "complete graph required");
		if (m_entire_row)
			return;

		m_interleaved_tile_width = tile_width;

		for (int p = 0; p < PLANE_NUM; ++p) {
			if (!m_output_nodes[p])
				continue;
			m_planar_tile_width[p] = tile_width >> (p == PLANE_U || p == PLANE_V ? m_sink->get_subsample_w() : 0);
		}
	}

	bool requires_64b_alignment() const { return m_requires_64b_alignment; }

	void set_requires_64b_alignment() { m_requires_64b_alignment = true; }

	void process(const ImageBuffer<const void> src[], const ImageBuffer<void> dst[], void *tmp, callback unpack_cb, callback pack_cb) const
	{
		zassert_d(m_sink, "complete graph required");

		if (!m_planar || unpack_cb || pack_cb)
			process_interleaved(src, dst, tmp, unpack_cb, pack_cb);
		else
			process_planar(src, dst, tmp);
	}
};


FilterGraph::callback::callback(std::nullptr_t) : m_func{}, m_user{} {}

FilterGraph::callback::callback(func_type func, void *user) : m_func{ func }, m_user{ user } {}

FilterGraph::callback::operator bool() const { return m_func != nullptr; }

void FilterGraph::callback::operator()(unsigned i, unsigned left, unsigned right) const
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


FilterGraph::FilterGraph() : m_impl(std::make_unique<impl>()) {}

FilterGraph::FilterGraph(FilterGraph &&other) noexcept = default;

FilterGraph::~FilterGraph() = default;

FilterGraph &FilterGraph::operator=(FilterGraph &&other) noexcept = default;

node_id FilterGraph::add_source(const ImageFilter::image_attributes &attr, unsigned subsample_w, unsigned subsample_h, const plane_mask &planes)
{
	return get_impl()->add_source(attr, subsample_w, subsample_h, planes);
}

node_id FilterGraph::attach_filter(std::shared_ptr<ImageFilter> filter, const id_map &deps, const plane_mask &output_planes)
{
	return get_impl()->attach_filter(std::move(filter), deps, output_planes);
}

void FilterGraph::set_output(const id_map &deps)
{
	get_impl()->set_output(deps);
}

size_t FilterGraph::get_tmp_size() const
{
	return get_impl()->get_tmp_size();
}

unsigned FilterGraph::get_input_buffering() const
{
	return get_impl()->get_input_buffering();
}

unsigned FilterGraph::get_output_buffering() const
{
	return get_impl()->get_output_buffering();
}

unsigned FilterGraph::get_tile_width() const
{
	return m_impl->get_tile_width();
}

void FilterGraph::set_tile_width(unsigned tile_width)
{
	m_impl->set_tile_width(tile_width);
}

bool FilterGraph::requires_64b_alignment() const
{
	return m_impl->requires_64b_alignment();
}

void FilterGraph::set_requires_64b_alignment()
{
	m_impl->set_requires_64b_alignment();
}

void FilterGraph::process(const ImageBuffer<const void> src[], const ImageBuffer<void> dst[], void *tmp, callback unpack_cb, callback pack_cb) const
{
	get_impl()->process(src, dst, tmp, unpack_cb, pack_cb);
}

} // namespace graph
} // namespace zimg
