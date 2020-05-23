#include <algorithm>
#include <climits>
#include <type_traits>
#include "common/alloc.h"
#include "common/make_unique.h"
#include "common/pixel.h"
#include "common/zassert.h"
#include "image_buffer.h"
#include "graphnode.h"

namespace zimg {
namespace graph {

namespace {

constexpr plane_mask nodes_to_mask(const node_map &nodes)
{
	return{ !!nodes[0], !!nodes[1], !!nodes[2], !!nodes[3] };
}

void validate_plane_mask(const plane_mask &planes)
{
	if (!planes[PLANE_Y])
		error::throw_<error::InternalError>("luma plane is required");
	if (planes[PLANE_U] != planes[PLANE_V])
		error::throw_<error::InternalError>("both chroma planes must be present");
}


class SourceNode final : public GraphNode {
	image_attributes m_attr;
	unsigned m_subsample_w;
	unsigned m_subsample_h;
	plane_mask m_planes;
public:
	explicit SourceNode(node_id id, const image_attributes &attr, unsigned subsample_w, unsigned subsample_h, const plane_mask &planes) :
		GraphNode(id),
		m_attr(attr),
		m_subsample_w{ subsample_w },
		m_subsample_h{ subsample_h },
		m_planes(planes)
	{
		validate_plane_mask(m_planes);
	}

	bool is_sourcesink() const override { return true; }
	unsigned get_subsample_w() const override { return m_subsample_w; }
	unsigned get_subsample_h() const override { return m_subsample_h; }
	plane_mask get_plane_mask() const override { return m_planes; }

	image_attributes get_image_attributes(int plane) const override
	{
		zassert_d(m_planes[plane], "plane not present");

		if (plane == PLANE_Y || plane == PLANE_A)
			return m_attr;
		else
			return{ m_attr.width >> m_subsample_w, m_attr.height >> m_subsample_h, m_attr.type };
	}

	void simulate(SimulationState *state, unsigned first, unsigned last, int plane) const override
	{
		zassert_d(m_planes[plane], "plane not present");

		if (plane == PLANE_U || plane == PLANE_V) {
			first <<= m_subsample_h;
			last <<= m_subsample_h;
		}

		unsigned cursor = state->get_cursor(id(), 0);
		if (cursor >= last) {
			state->update(id(), cache_id(), first, last, PLANE_Y);
		} else {
			unsigned step = 1U << m_subsample_h;
			state->update(id(), cache_id(), floor_n(first, step), ceil_n(last, step), PLANE_Y);
		}
	}

	void simulate_alloc(SimulationState *) const override {}

	void try_inplace() override {}

	void request_external_cache(node_id) override {}

	void init_context(ExecutionState *state, unsigned top, unsigned left, unsigned right, int plane) const override
	{
		if (!state->is_initialized(id()))
			state->reset_tile_bounds(id());

		if (plane == PLANE_U || plane == PLANE_V) {
			left <<= m_subsample_w;
			right <<= m_subsample_w;
			top <<= m_subsample_h;
		} else {
			left = floor_n(left, 1U << m_subsample_w);
			right = ceil_n(right, 1U << m_subsample_w);
			top = floor_n(top, 1U << m_subsample_h);
		}

		ExecutionState::node_state *s = state->get_node_state(id());
		s->left = std::min(s->left, left);
		s->right = std::max(s->right, right);

		state->set_cursor(id(), std::min(top, state->get_cursor(id())));
		state->set_initialized(id());
	}

	void generate(ExecutionState *state, unsigned last, int plane) const override
	{
		zassert_d(m_planes[plane], "plane not present");
		if (!state->unpack_cb())
			return;

		if (plane == PLANE_U || plane == PLANE_V)
			last <<= m_subsample_h;

		ExecutionState::node_state *s = state->get_node_state(id());
		unsigned cursor = state->get_cursor(id());

		for (; cursor < last; cursor += (1U << m_subsample_h)) {
			if (state->unpack_cb()) {
				state->unpack_cb()(cursor, s->left, s->right);
				state->check_guard_pages();
			}
		}

		state->set_cursor(id(), cursor);
	}
};


class SinkNode final : public GraphNode {
	node_map m_parents;
	unsigned m_subsample_w;
	unsigned m_subsample_h;
	image_attributes m_attr;
public:
	explicit SinkNode(node_id id, const node_map &parents) :
		GraphNode(id),
		m_parents(parents),
		m_subsample_w{},
		m_subsample_h{},
		m_attr{}
	{
		validate_plane_mask(SinkNode::get_plane_mask());

		auto attr_y = m_parents[PLANE_Y]->get_image_attributes(PLANE_Y);
		m_attr = attr_y;

		if (m_parents[PLANE_U] && m_parents[PLANE_V]) {
			auto attr_u = m_parents[PLANE_U]->get_image_attributes(PLANE_U);
			auto attr_v = m_parents[PLANE_V]->get_image_attributes(PLANE_V);

			if (attr_u.width != attr_v.width || attr_u.height != attr_v.height || attr_u.type != attr_v.type)
				error::throw_<error::InternalError>("chroma planes must have same dimensions and type");

			for (int ss = 0; ss < 3; ++ss) {
				if (attr_u.width << ss == attr_y.width)
					m_subsample_w = ss;
				if (attr_u.height << ss == attr_y.height)
					m_subsample_h = ss;
			}
			if (attr_u.width << m_subsample_w != attr_y.width || attr_v.height << m_subsample_h != attr_y.height)
				error::throw_<error::InternalError>("unsupported subsampling factor");
		}

		if (m_parents[PLANE_A]) {
			auto attr_a = m_parents[PLANE_A]->get_image_attributes(PLANE_A);
			if (attr_a.width != attr_y.width || attr_a.height != attr_y.height)
				error::throw_<error::InternalError>("alpha plane must have same dimensions as image");
		}
	}

	bool is_sourcesink() const override { return true; }
	unsigned get_subsample_w() const override { return m_subsample_w; }
	unsigned get_subsample_h() const override { return m_subsample_h; }
	plane_mask get_plane_mask() const override { return nodes_to_mask(m_parents); }

	image_attributes get_image_attributes(int plane) const override
	{
		zassert_d(m_parents[plane], "plane not present");
		return m_parents[plane]->get_image_attributes(plane);
	}

	void try_inplace() override
	{
		for (GraphNode *node : m_parents) {
			if (node)
				node->request_external_cache(cache_id());
		}
	}

	void request_external_cache(node_id) override {}

	void simulate(SimulationState *state, unsigned first, unsigned last, int plane) const override
	{
		zassert_d(m_parents[plane], "plane not present");

		if (plane == PLANE_U || plane == PLANE_V) {
			first <<= m_subsample_h;
			last <<= m_subsample_h;
		}

		unsigned cursor = state->get_cursor(id(), 0);
		if (cursor >= last) {
			state->update(id(), cache_id(), first, last, PLANE_Y);
			return;
		}

		for (; cursor < last; cursor += (1U << m_subsample_h)) {
			if (m_parents[PLANE_Y])
				m_parents[PLANE_Y]->simulate(state, cursor, cursor + (1U << m_subsample_h), PLANE_Y);

			if (m_parents[PLANE_U] && m_parents[PLANE_V]) {
				m_parents[PLANE_U]->simulate(state, cursor >> m_subsample_h, (cursor >> m_subsample_h) + 1, PLANE_U);
				m_parents[PLANE_V]->simulate(state, cursor >> m_subsample_h, (cursor >> m_subsample_h) + 1, PLANE_V);
			}
			if (m_parents[PLANE_A]) {
				m_parents[PLANE_A]->simulate(state, cursor, cursor + (1U << m_subsample_h), PLANE_A);
			}
		}
		state->update(id(), cache_id(), first, cursor, PLANE_Y);
	}

	void simulate_alloc(SimulationState *state) const override
	{
		for (const GraphNode *node : m_parents) {
			if (node)
				node->simulate_alloc(state);
		}
	}

	void init_context(ExecutionState *state, unsigned top, unsigned left, unsigned right, int plane) const override
	{
		if (!state->is_initialized(id()))
			state->reset_tile_bounds(id());

		if (plane == PLANE_U || plane == PLANE_V) {
			top <<= m_subsample_h;
			left <<= m_subsample_w;
			right <<= m_subsample_w;
		}

		for (int p = 0; p < PLANE_NUM; ++p) {
			const GraphNode *node = m_parents[p];
			if (!node)
				continue;

			unsigned plane_top = top >> (p == PLANE_U || p == PLANE_V ? m_subsample_h : 0);
			unsigned plane_left = left >> (p == PLANE_U || p == PLANE_V ? m_subsample_w : 0);
			unsigned plane_right = right >> (p == PLANE_U || p == PLANE_V ? m_subsample_w : 0);
			node->init_context(state, plane_top, plane_left, plane_right, p);
		}

		ExecutionState::node_state *s = state->get_node_state(id());
		s->left = std::min(s->left, left);
		s->right = std::max(s->right, right);

		state->set_cursor(id(), std::min(state->get_cursor(id()), top));
		state->set_initialized(id());
	}

	void generate(ExecutionState *state, unsigned last, int plane) const override
	{
		zassert_d(m_parents[plane], "plane not present");
		if (plane == PLANE_U || plane == PLANE_V)
			last <<= m_subsample_h;

		ExecutionState::node_state *s = state->get_node_state(id());
		unsigned cursor = state->get_cursor(id());
		unsigned subsample_h = m_subsample_h;

		for (; cursor < last; cursor += (1U << subsample_h)) {
			unsigned next = cursor + (1U << subsample_h);

			m_parents[PLANE_Y]->generate(state, next, PLANE_Y);

			if (m_parents[PLANE_U]) {
				unsigned next_uv = next >> subsample_h;

				m_parents[PLANE_U]->generate(state, next_uv, PLANE_U);
				m_parents[PLANE_V]->generate(state, next_uv, PLANE_V);
			}

			if (m_parents[PLANE_A])
				m_parents[PLANE_A]->generate(state, next, PLANE_A);

			if (state->pack_cb()) {
				state->pack_cb()(cursor, s->left, s->right);
				state->check_guard_pages();
			}
		}

		state->set_cursor(id(), cursor);
	}
};


class FilterNodeBase : public GraphNode {
protected:
	std::shared_ptr<ImageFilter> m_filter;
	node_map m_parents;
	plane_mask m_output_planes;
	unsigned m_step;
	image_attributes m_attr;
public:
	FilterNodeBase(node_id id, std::shared_ptr<ImageFilter> filter, const node_map &parents, const plane_mask &output_planes) :
		GraphNode(id),
		m_filter{ std::move(filter) },
		m_parents(parents),
		m_output_planes(output_planes),
		m_step{ m_filter->get_simultaneous_lines() },
		m_attr(m_filter->get_image_attributes())
	{}

	bool is_sourcesink() const override { return false; }
	unsigned get_subsample_w() const override { return 0; }
	unsigned get_subsample_h() const override { return 0; }
	plane_mask get_plane_mask() const override { return m_output_planes; }

	image_attributes get_image_attributes(int plane) const override
	{
		zassert_d(m_output_planes[plane], "plane not present");
		return m_attr;
	}

	void simulate(SimulationState *state, unsigned first, unsigned last, int plane) const override
	{
		zassert_d(m_output_planes[plane], "plane not present");
		unsigned cursor = state->get_cursor(id(), 0);
		if (cursor >= last) {
			state->update(id(), cache_id(), first, last, plane);
			return;
		}

		for (; cursor < last; cursor += m_step) {
			auto range = m_filter->get_required_row_range(cursor);
			zassert_d(range.first < range.second, "bad row range");

			for (int p = 0; p < PLANE_NUM; ++p) {
				if (m_parents[p])
					m_parents[p]->simulate(state, range.first, range.second, p);
			}
		}
		state->update(id(), cache_id(), first, cursor, plane);
	}

	void simulate_alloc(SimulationState *state) const override
	{
		state->alloc_context(id(), m_filter->get_context_size());
		state->alloc_tmp(m_filter->get_tmp_size(0, m_attr.width));

		for (const GraphNode *node : m_parents) {
			if (node)
				node->simulate_alloc(state);
		}
	}

	void try_inplace() override
	{
		if (!m_filter->get_flags().in_place)
			return;

		for (int p = 0; p < PLANE_NUM; ++p) {
			if (!m_output_planes[p])
				continue;

			GraphNode *node = m_parents[p];
			if (!node || node->is_sourcesink() || node->ref_count() > 1)
				continue;

			plane_mask self_mask = get_plane_mask();
			plane_mask mask = node->get_plane_mask();
			bool ineligible = false;

			auto self_attr = get_image_attributes(p);
			auto attr = node->get_image_attributes(p);
			if (self_attr.width != attr.width || pixel_size(self_attr.type) != pixel_size(attr.type))
				continue;

			for (int q = 0; q < PLANE_NUM; ++q) {
				ineligible = ineligible || (mask[q] && !self_mask[q]);
			}
			if (ineligible)
				continue;

			node->request_external_cache(cache_id());
		}
	}

	void request_external_cache(node_id id) override
	{
		for (GraphNode *node : m_parents) {
			if (!node)
				continue;
			if (node->cache_id() == cache_id())
				node->request_external_cache(id);
		}

		set_cache_id(id);
	}

	void init_context(ExecutionState *state, unsigned top, unsigned left, unsigned right, int plane) const override
	{
		if (!state->is_initialized(id()))
			state->reset_tile_bounds(id());

		auto row_range = m_filter->get_required_row_range(top);
		auto col_range = m_filter->get_required_col_range(left, right);

		for (int p = 0; p < PLANE_NUM; ++p) {
			const GraphNode *node = m_parents[p];
			if (node)
				node->init_context(state, row_range.first, col_range.first, col_range.second, p);
		}

		ExecutionState::node_state *s = state->get_node_state(id());
		s->left = std::min(s->left, left);
		s->right = std::max(s->right, right);
		state->set_cursor(id(), std::min(state->get_cursor(id()), top));

		if (!state->is_initialized(id())) {
			unsigned seq = static_cast<unsigned>(std::find(m_output_planes.begin(), m_output_planes.end(), true) - m_output_planes.begin());
			m_filter->init_context(state->get_node_state(id())->context, seq);
		}

		state->set_initialized(id());
	}
};

template <int P0 = invalid_id, int P1 = invalid_id, int P2 = invalid_id, int P3 = invalid_id>
class FilterNodeColor final : public FilterNodeBase {
public:
	using FilterNodeBase::FilterNodeBase;

	void generate(ExecutionState *state, unsigned last, int plane) const override
	{
		zassert_d(m_output_planes[plane], "plane not present");
		unsigned cursor = state->get_cursor(id());
		if (cursor >= last)
			return;

		std::aligned_storage<sizeof(ImageBuffer<const void>), alignof(ImageBuffer<const void>)>::type src[PLANE_NUM];
		const ImageBuffer<void> *dst = state->get_buffer(cache_id());
		ExecutionState::node_state *node_state = state->get_node_state(id());
		void *tmp = state->get_shared_tmp();

		if (P0 == 1 || (P0 == invalid_id && m_parents[0]))
			new (&src[0]) ImageBuffer<const void>(state->get_buffer(m_parents[0]->cache_id())[0]);
		if (P1 == 1 || (P1 == invalid_id && m_parents[1]))
			new (&src[1]) ImageBuffer<const void>(state->get_buffer(m_parents[1]->cache_id())[1]);
		if (P2 == 1 || (P2 == invalid_id && m_parents[2]))
			new (&src[2]) ImageBuffer<const void>(state->get_buffer(m_parents[2]->cache_id())[2]);
		if (P3 == 1 || (P3 == invalid_id && m_parents[3]))
			new (&src[3]) ImageBuffer<const void>(state->get_buffer(m_parents[3]->cache_id())[3]);

		for (; cursor < last; cursor += m_step) {
			auto range = m_filter->get_required_row_range(cursor);
			zassert_d(range.first < range.second, "bad row range");

			if (P0 == 1 || (P0 == invalid_id && m_parents[0]))
				m_parents[0]->generate(state, range.second, 0);
			if (P1 == 1 || (P1 == invalid_id && m_parents[1]))
				m_parents[1]->generate(state, range.second, 1);
			if (P2 == 1 || (P2 == invalid_id && m_parents[2]))
				m_parents[2]->generate(state, range.second, 2);
			if (P3 == 1 || (P3 == invalid_id && m_parents[3]))
				m_parents[3]->generate(state, range.second, 3);

			m_filter->process(node_state->context, reinterpret_cast<ImageBuffer<const void> *>(src), dst, tmp, cursor, node_state->left, node_state->right);
			state->check_guard_pages();
		}

		state->set_cursor(id(), cursor);
	}
};

template <int Plane, bool Parent>
class FilterNodeGrey final : public FilterNodeBase {
public:
	using FilterNodeBase::FilterNodeBase;

	void generate(ExecutionState *state, unsigned last, int plane) const override
	{
		zassert_d(m_output_planes[plane], "plane not present");
		unsigned cursor = state->get_cursor(id());
		if (cursor >= last)
			return;

		const ImageBuffer<const void> *src = Parent ? static_buffer_cast<const void>(state->get_buffer(m_parents[Plane]->cache_id()) + Plane) : nullptr;
		const ImageBuffer<void> *dst = state->get_buffer(cache_id()) + Plane;
		ExecutionState::node_state *node_state = state->get_node_state(id());
		void *tmp = state->get_shared_tmp();

		for (; cursor < last; cursor += m_step) {
			auto range = m_filter->get_required_row_range(cursor);
			zassert_d(range.first < range.second, "bad row range");

			if (Parent)
				m_parents[Plane]->generate(state, range.second, Plane);
			m_filter->process(node_state->context, src, dst, tmp, cursor, node_state->left, node_state->right);
			state->check_guard_pages();
		}

		state->set_cursor(id(), cursor);
	}
};

} // namespace


SimulationState::SimulationState(const std::vector<std::unique_ptr<GraphNode>> &nodes) : m_state(nodes.size()), m_tmp{}
{
	for (const auto &node : nodes) {
		m_state[node->cache_id()].subsample_h = std::max(m_state[node->cache_id()].subsample_h, node->get_subsample_h());
	}
}

SimulationState::result SimulationState::get_result(const std::vector<std::unique_ptr<GraphNode>> &nodes) const
{
	zassert_d(nodes.size() == m_state.size(), "incorrect number of nodes");
	result res{ std::vector<result::s>(m_state.size()), m_tmp };

	for (const auto &node : nodes) {
		unsigned history = m_state[node->id()].cache_history;

		if (history > 0) {
			plane_mask planes = node->get_plane_mask();
			auto attr = node->get_image_attributes(static_cast<int>(std::find(planes.begin(), planes.end(), true) - planes.begin()));

			unsigned mask = select_zimg_buffer_mask(history);
			unsigned lines = mask == BUFFER_MAX ? BUFFER_MAX : mask + 1;

			res.node_result[node->id()].cache_lines = std::min(lines, attr.height);
			res.node_result[node->id()].mask = lines >= attr.height ? BUFFER_MAX : mask;
		}

		res.node_result[node->id()].context_size = m_state[node->id()].context_size;
	}

	return res;
}

void SimulationState::update(node_id id, node_id cache_id, unsigned first, unsigned last, unsigned plane)
{
	zassert_d(id >= 0, "invalid id");
	state &s = m_state[id];
	state &cache = m_state[cache_id];

	s.cursor = s.cursor_initialized ? std::max(s.cursor, last) : last;
	s.cursor_initialized = true;

	unsigned real_first = first << (plane == PLANE_U || plane == PLANE_V ? cache.subsample_h : 0);
	unsigned real_cursor = s.cursor << (plane == PLANE_U || plane == PLANE_V ? cache.subsample_h : 0);

	cache.cache_pos = std::max(cache.cache_pos, real_cursor);
	cache.cache_history = std::max(cache.cache_history, cache.cache_pos - real_first);
}

unsigned SimulationState::get_cursor(node_id id, unsigned initial_pos) const
{
	zassert_d(id >= 0, "invalid id");
	return m_state[id].cursor_initialized ? m_state[id].cursor : initial_pos;
}

void SimulationState::alloc_context(node_id id, size_t sz)
{
	zassert_d(id >= 0, "invalid id");
	m_state[id].context_size = std::max(m_state[id].context_size, sz);
}

void SimulationState::alloc_tmp(size_t sz)
{
	m_tmp = std::max(m_tmp, sz);
}


#ifndef NDEBUG
class ExecutionState::guard_page {
	static constexpr uint32_t pattern = 0xDEADBEEF;

	uint32_t page[4096 / sizeof(uint32_t)];
public:
	template <class Alloc>
	static void allocate(Alloc &alloc) { alloc.template allocate_n<guard_page>(1); }

	template <class Alloc>
	static void allocate(guard_page **&p, Alloc &alloc) { *p++ = new (alloc.template allocate_n<guard_page>(1)) guard_page{}; }

	guard_page() { std::fill(std::begin(page), std::end(page), pattern); }

	void assert_page() const
	{
		for (uint32_t x : page) {
			zassert(x == pattern, "buffer overflow detected");
		}
	}
};

constexpr uint32_t ExecutionState::guard_page::pattern;
#else
class ExecutionState::guard_page {
public:
	template <class Alloc> static void allocate(Alloc &) {}
	template <class Alloc> static void allocate(guard_page **&, Alloc&) {}
	void assert_page() const {}
};
#endif

size_t ExecutionState::calculate_tmp_size(const SimulationState::result &sim, const std::vector<std::unique_ptr<GraphNode>> &nodes)
{
	zassert_d(nodes.size() == sim.node_result.size(), "incorrect number of nodes");
	FakeAllocator alloc;

	alloc.allocate_n<ColorImageBuffer<void>>(sim.node_result.size()); // m_buffers
	alloc.allocate_n<unsigned>(nodes.size()); // m_cursors
	alloc.allocate_n<node_state>(nodes.size()); // m_state
	alloc.allocate_n<unsigned char>(ceil_n(nodes.size(), CHAR_BIT) / CHAR_BIT); // m_init_bitset
#ifndef NDEBUG
	alloc.allocate_n<guard_page *>(nodes.size() * 2 + 2 + 1); // m_guard_pages
#endif

	for (const auto &node : nodes) {
		guard_page::allocate(alloc);

		if (node->is_sourcesink())
			continue;

		plane_mask planes = node->get_plane_mask();
		unsigned cache_lines = sim.node_result[node->id()].cache_lines;

		for (int p = 0; p < PLANE_NUM; ++p) {
			if (!planes[p])
				continue;

			auto attr = node->get_image_attributes(p);
			unsigned shift_h = p == PLANE_U || p == PLANE_V ? node->get_subsample_h() : 0;

			checked_size_t stride = ceil_n(checked_size_t{ attr.width } * pixel_size(attr.type), ALIGNMENT);
			unsigned plane_lines = cache_lines >> shift_h;
			alloc.allocate(stride * plane_lines);
		}
	}

	for (const auto &node : nodes) {
		guard_page::allocate(alloc);
		alloc.allocate(sim.node_result[node->id()].context_size);
	}

	guard_page::allocate(alloc);
	alloc.allocate(sim.shared_tmp);
	guard_page::allocate(alloc);

	return alloc.count();
}

ExecutionState::ExecutionState(const SimulationState::result &sim, const std::vector<std::unique_ptr<GraphNode>> &nodes, node_id src_id, node_id dst_id, const ImageBuffer<const void> src[], const ImageBuffer<void> dst[], FilterGraph::callback unpack_cb, FilterGraph::callback pack_cb, void *buf) :
	m_unpack_cb{ unpack_cb },
	m_pack_cb{ pack_cb },
	m_buffers{},
	m_cursors{},
	m_state{},
	m_init_bitset{},
	m_tmp{},
	m_guard_pages{}
{
	zassert_d(nodes.size() == sim.node_result.size(), "incorrect number of nodes");

	LinearAllocator alloc{ buf };
	m_buffers = alloc.allocate_n<ColorImageBuffer<void>>(nodes.size());
	m_cursors = alloc.allocate_n<unsigned>(nodes.size());
	m_state = alloc.allocate_n<node_state>(nodes.size());
	m_init_bitset = alloc.allocate_n<unsigned char>(ceil_n(nodes.size(), CHAR_BIT) / CHAR_BIT);
#ifndef NDEBUG
	m_guard_pages = alloc.allocate_n<guard_page *>(nodes.size() * 2 + 2 + 1);
	std::fill_n(m_guard_pages, nodes.size() * 2 + 2 + 1, nullptr);
#endif

	guard_page **guard_pages = m_guard_pages;

	for (const auto &node : nodes) {
		guard_page::allocate(guard_pages, alloc);

		if (node->is_sourcesink())
			continue;

		plane_mask planes = node->get_plane_mask();
		ColorImageBuffer<void> &buffer = m_buffers[node->id()];
		const auto &node_sim = sim.node_result[node->id()];

		for (int p = 0; p < PLANE_NUM; ++p) {
			if (!planes[p])
				continue;

			auto attr = node->get_image_attributes(p);
			unsigned shift_h = p == PLANE_U || p == PLANE_V ? node->get_subsample_h() : 0;

			size_t stride = ceil_n(static_cast<size_t>(attr.width) * pixel_size(attr.type), ALIGNMENT);
			unsigned plane_lines = node_sim.cache_lines >> shift_h;
			unsigned mask = node_sim.mask == BUFFER_MAX ? BUFFER_MAX : node_sim.mask >> shift_h;
			buffer[p] = { alloc.allocate(stride * plane_lines), static_cast<ptrdiff_t>(stride), mask };
		}
	}

	for (const auto &node : nodes) {
		guard_page::allocate(guard_pages, alloc);
		m_state[node->id()].context = alloc.allocate(sim.node_result[node->id()].context_size);
	}

	for (int p = 0; p < PLANE_NUM; ++p) {
		m_buffers[src_id][p] = { const_cast<void *>(src[p].data()), src[p].stride(), src[p].mask() };
	}
	m_buffers[dst_id] = { dst[0], dst[1], dst[2], dst[3] };

	guard_page::allocate(guard_pages, alloc);
	m_tmp = alloc.allocate(sim.shared_tmp);
	guard_page::allocate(guard_pages, alloc);
}

void ExecutionState::reset_initialized(size_t max_id)
{
	std::fill_n(m_init_bitset, ceil_n(max_id, CHAR_BIT) / CHAR_BIT, 0);
}

bool ExecutionState::is_initialized(node_id id) const
{
	return !!(m_init_bitset[id / CHAR_BIT] & (1U << (id % CHAR_BIT)));
}

void ExecutionState::reset_tile_bounds(node_id id)
{
	get_node_state(id)->left = UINT_MAX;
	get_node_state(id)->right = 0;
	set_cursor(id, UINT_MAX);
}

void ExecutionState::set_initialized(node_id id)
{
	m_init_bitset[id / CHAR_BIT] |= 1U << (id % CHAR_BIT);
}

void ExecutionState::check_guard_pages() const
{
#ifndef NDEBUG
	for (guard_page **p = m_guard_pages; *p; ++p) {
		(*p)->assert_page();
	}
#endif
}


GraphNode::~GraphNode() = default;


std::unique_ptr<GraphNode> make_source_node(node_id id, const ImageFilter::image_attributes &attr, unsigned subsample_w, unsigned subsample_h, const plane_mask &planes)
{
	return ztd::make_unique<SourceNode>(id, attr, subsample_w, subsample_h, planes);
}

std::unique_ptr<GraphNode> make_sink_node(node_id id, const node_map &parents)
{
	return ztd::make_unique<SinkNode>(id, parents);
}

std::unique_ptr<GraphNode> make_filter_node(node_id id, std::shared_ptr<ImageFilter> filter, const node_map &parents, const plane_mask &output_planes)
{
	if (filter->get_flags().color) {
		if (parents[0] && parents[1] && parents[2] && parents[3])
			return ztd::make_unique<FilterNodeColor<1, 1, 1, 1>>(id, filter, parents, output_planes);
		else if (parents[0] && parents[1] && parents[2] && !parents[3])
			return ztd::make_unique<FilterNodeColor<1, 1, 1, 0>>(id, filter, parents, output_planes);
		else
			return ztd::make_unique<FilterNodeColor<>>(id, filter, parents, output_planes);
	} else {
		if (parents[0] && output_planes[0])
			return ztd::make_unique<FilterNodeGrey<0, true>>(id, filter, parents, output_planes);
		else if (parents[1] && output_planes[1])
			return ztd::make_unique<FilterNodeGrey<1, true>>(id, filter, parents, output_planes);
		else if (parents[2] && output_planes[2])
			return ztd::make_unique<FilterNodeGrey<2, true>>(id, filter, parents, output_planes);
		else if (parents[3] && output_planes[3])
			return ztd::make_unique<FilterNodeGrey<3, true>>(id, filter, parents, output_planes);
		else if (!parents[0] && output_planes[0])
			return ztd::make_unique<FilterNodeGrey<0, false>>(id, filter, parents, output_planes);
		else if (!parents[1] && output_planes[1])
			return ztd::make_unique<FilterNodeGrey<1, false>>(id, filter, parents, output_planes);
		else if (!parents[2] && output_planes[2])
			return ztd::make_unique<FilterNodeGrey<2, false>>(id, filter, parents, output_planes);
		else if (!parents[3] && output_planes[3])
			return ztd::make_unique<FilterNodeGrey<3, false>>(id, filter, parents, output_planes);
		else
			error::throw_<error::InternalError>("must produce output plane");
	}
}

} // namespace graph
} // namespace zimg
