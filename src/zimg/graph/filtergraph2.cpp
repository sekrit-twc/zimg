#include <algorithm>
#include <array>
#include <cstring>
#include <utility>
#include <vector>
#include "common/align.h"
#include "common/alloc.h"
#include "common/except.h"
#include "common/make_unique.h"
#include "common/pixel.h"
#include "common/zassert.h"
#include "filtergraph2.h"

namespace zimg {
namespace graph {

namespace {

typedef FilterGraph2::node_id node_id;
typedef FilterGraph2::image_attributes image_attributes;
typedef FilterGraph2::plane_mask plane_mask;
typedef FilterGraph2::id_map id_map;

class GraphNode;

typedef std::array<GraphNode *, PLANE_NUM> node_map;

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


class SimulationState {
	struct state {
		unsigned pos;
		unsigned history;
		bool initialized;
	};

	std::vector<state> m_state;
public:
	explicit SimulationState(size_t num_nodes) : m_state(num_nodes) {}

	void update(node_id id, unsigned first, unsigned last)
	{
		zassert_d(id >= 0, "invalid id");
		state &s = m_state[id];

		if (s.initialized) {
			s.pos = std::max(s.pos, last);
			s.history = std::max(s.history, s.pos - first);
		} else {
			s.pos = last;
			s.history = last - first;
		}

		s.initialized = true;
	}

	unsigned get_cursor(node_id id, unsigned initial_pos) const
	{
		zassert_d(id >= 0, "invalid id");
		return m_state[id].initialized ? m_state[id].pos : initial_pos;
	}

	std::pair<unsigned, unsigned> suggest_mask(node_id id, unsigned height) const
	{
		zassert_d(id >= 0, "invalid id");
		unsigned n = m_state[id].history;
		unsigned mask = n >= height ? BUFFER_MAX : select_zimg_buffer_mask(n);
		return{ mask == BUFFER_MAX ? height : mask + 1, mask };
	}
};


class ExecutionState {
	ImageBuffer<const void> m_src[PLANE_NUM];
	ImageBuffer<void> m_dst[PLANE_NUM];

	FilterGraph2::callback m_unpack_cb;
	FilterGraph2::callback m_pack_cb;

	std::vector<std::array<ImageBuffer<void>, PLANE_NUM>> m_buffers;
	std::vector<unsigned> m_cursor;
	std::vector<std::shared_ptr<void>> m_contexts;
	std::vector<std::shared_ptr<void>> m_buffer_allocs;
	std::pair<std::shared_ptr<void>, size_t> m_tmp;

	std::shared_ptr<void> alloc(size_t size)
	{
		return { zimg_x_aligned_malloc(size, ALIGNMENT), zimg_x_aligned_free };
	}
public:
	ExecutionState(size_t num_nodes, const ImageBuffer<const void> src[], const ImageBuffer<void> dst[], FilterGraph2::callback unpack_cb, FilterGraph2::callback pack_cb) :
		m_src{},
		m_dst{},
		m_unpack_cb{ unpack_cb },
		m_pack_cb{ pack_cb },
		m_buffers(num_nodes),
		m_cursor(num_nodes),
		m_contexts(num_nodes),
		m_buffer_allocs(num_nodes),
		m_tmp{ nullptr, 0 }
	{
		std::copy_n(src, PLANE_NUM, m_src);
		std::copy_n(dst, PLANE_NUM, m_dst);
	}

	const ImageBuffer<const void> *src() const { return m_src; }
	const ImageBuffer<void> *dst() const { return m_dst; }

	const FilterGraph2::callback &unpack_cb() const { return m_unpack_cb; }
	const FilterGraph2::callback &pack_cb() const { return m_pack_cb; }

	unsigned get_cursor(node_id id) const { return m_cursor[id]; }
	void set_cursor(node_id id, unsigned pos) { m_cursor[id] = pos; }

	void alloc_buffer(node_id id, const size_t rowsize[PLANE_NUM], const unsigned lines[PLANE_NUM], const unsigned mask[PLANE_NUM], const plane_mask &planes)
	{
		std::array<ImageBuffer<void>, PLANE_NUM> buffer = {};
		size_t sz = 0;

		for (int p = 0; p < PLANE_NUM; ++p) {
			if (!planes[p])
				continue;

			sz += rowsize[p] * lines[p];
		}

		m_buffer_allocs[id] = alloc(sz);
		unsigned char *base = static_cast<unsigned char *>(m_buffer_allocs[id].get());

		for (int p = 0; p < PLANE_NUM; ++p) {
			if (!planes[p])
				continue;

			buffer[p] = { base, static_cast<ptrdiff_t>(rowsize[p]), mask[p] };
			base += rowsize[p] * lines[p];
		}

		m_buffers[id] = buffer;
	}

	const ImageBuffer<void> *get_buffer(node_id id) { return m_buffers[id].data(); }

	void alloc_context(node_id id, size_t size) { m_contexts[id] = alloc(size); }
	void *get_context(node_id id) const { return m_contexts[id].get(); }

	void alloc_shared_tmp(size_t size) { if (size > m_tmp.second) m_tmp = { alloc(size), size }; }
	void *get_shared_tmp() const { return m_tmp.first.get(); }
};


class GraphNode {
	node_id m_id;
protected:
	explicit GraphNode(node_id id) : m_id{ id } {}
public:
	virtual ~GraphNode() = default;

	node_id id() const { return m_id; }

	virtual unsigned get_subsample_w() const = 0;

	virtual unsigned get_subsample_h() const = 0;

	virtual plane_mask get_plane_mask() const = 0;

	virtual image_attributes get_image_attributes(int plane) const = 0;

	virtual void simulate(SimulationState *state, unsigned first, unsigned last, int plane) const = 0;

	virtual void init_context(ExecutionState *state) const = 0;

	virtual void generate(ExecutionState *state, unsigned last, int plane) const = 0;
};

class SourceNode : public GraphNode {
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

	void init_context(ExecutionState *state) const override
	{
		image_attributes attr[PLANE_NUM];

		for (int p = 0; p < PLANE_NUM; ++p) {
			if (m_planes[p])
				attr[p] = get_image_attributes(p);
		}

		state->set_cursor(id(), 0);
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
			state->update(id(), first, last);
		} else {
			unsigned step = 1U << m_subsample_h;
			state->update(id(), floor_n(first, step), ceil_n(last, step));
		}
	}

	void generate(ExecutionState *state, unsigned last, int plane) const override
	{
		zassert_d(m_planes[plane], "plane not present");
		if (plane == PLANE_U || plane == PLANE_V)
			last <<= m_subsample_h;

		unsigned cursor = state->get_cursor(id());
		if (cursor >= last)
			return;

		const ImageBuffer<const void> *src = state->src();
		const ImageBuffer<void> *dst = state->get_buffer(id());

		size_t sz = static_cast<size_t>(m_attr.width) * pixel_size(m_attr.type);

		for (; cursor < last; cursor += (1U << m_subsample_h)) {
			if (state->unpack_cb())
				state->unpack_cb()(cursor, 0, m_attr.width);

			if (m_planes[PLANE_Y]) {
				for (unsigned i = cursor; i < cursor + (1U << m_subsample_h); ++i) {
					memcpy(dst[PLANE_Y][i], src[PLANE_Y][i], sz);
				}
			}
			if (m_planes[PLANE_U] && m_planes[PLANE_V]) {
				memcpy(dst[PLANE_U][cursor >> m_subsample_h], src[PLANE_U][cursor >> m_subsample_h], sz >> m_subsample_w);
				memcpy(dst[PLANE_V][cursor >> m_subsample_h], src[PLANE_V][cursor >> m_subsample_h], sz >> m_subsample_w);
			}
			if (m_planes[PLANE_A]) {
				for (unsigned i = cursor; i < cursor + (1U << m_subsample_h); ++i) {
					memcpy(dst[PLANE_A][i], src[PLANE_A][i], sz);
				}
			}
		}

		state->set_cursor(id(), cursor);
	}
};

class SinkNode : public GraphNode {
	node_map m_parents;
	unsigned m_subsample_w;
	unsigned m_subsample_h;
public:
	explicit SinkNode(node_id id, const node_map &parents) :
		GraphNode(id),
		m_parents(parents),
		m_subsample_w{},
		m_subsample_h{}
	{
		validate_plane_mask(SinkNode::get_plane_mask());

		auto attr_y = m_parents[PLANE_Y]->get_image_attributes(PLANE_Y);

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

	unsigned get_subsample_w() const override { return m_subsample_w; }
	unsigned get_subsample_h() const override { return m_subsample_h; }
	plane_mask get_plane_mask() const override { return nodes_to_mask(m_parents); }

	image_attributes get_image_attributes(int plane) const override
	{
		zassert_d(m_parents[plane], "plane not present");
		return m_parents[plane]->get_image_attributes(plane);
	}

	void init_context(ExecutionState *state) const override
	{
		state->set_cursor(id(), 0);
	}

	void simulate(SimulationState *state, unsigned first, unsigned last, int plane) const override
	{
		zassert_d(m_parents[plane], "plane not present");
		if (plane == PLANE_U || plane == PLANE_V) {
			first <<= m_subsample_h;
			last <<= m_subsample_h;
		}

		unsigned cursor = state->get_cursor(id(), 0);
		if (cursor >= last) {
			state->update(id(), first, last);
			return;
		}

		for (; cursor < last; cursor += (1U << m_subsample_h)) {
			if (m_parents[PLANE_Y]) {
				for (unsigned i = cursor; i < cursor + (1U << m_subsample_h); ++i) {
					m_parents[PLANE_Y]->simulate(state, i, i + 1, PLANE_Y);
				}
			}
			if (m_parents[PLANE_U] && m_parents[PLANE_V]) {
				m_parents[PLANE_U]->simulate(state, cursor >> m_subsample_h, (cursor >> m_subsample_h) + 1, PLANE_U);
				m_parents[PLANE_V]->simulate(state, cursor >> m_subsample_h, (cursor >> m_subsample_h) + 1, PLANE_V);
			}
			if (m_parents[PLANE_A]) {
				for (unsigned i = cursor; i < cursor + (1U << m_subsample_h); ++i) {
					m_parents[PLANE_A]->simulate(state, i, i + 1, PLANE_A);
				}
			}
		}
		state->update(id(), first, cursor);
	}

	void generate(ExecutionState *state, unsigned last, int plane) const override
	{
		zassert_d(m_parents[plane], "plane not present");
		if (plane == PLANE_U || plane == PLANE_V)
			last <<= m_subsample_h;

		unsigned cursor = state->get_cursor(id());
		if (cursor >= last)
			return;

		ImageBuffer<const void> src[PLANE_NUM];
		const ImageBuffer<void> *dst = state->dst();

		auto attr = get_image_attributes(PLANE_Y);
		size_t sz = static_cast<size_t>(attr.width) * pixel_size(attr.type);

		for (int p = 0; p < PLANE_NUM; ++p) {
			if (m_parents[p])
				src[p] = state->get_buffer(m_parents[p]->id())[p];
		}

		for (; cursor < last; cursor += (1U << m_subsample_h)) {
			if (m_parents[PLANE_Y]) {
				for (unsigned i = cursor; i < cursor + (1U << m_subsample_h); ++i) {
					m_parents[PLANE_Y]->generate(state, i + 1, PLANE_Y);
					memcpy(dst[PLANE_Y][i], src[PLANE_Y][i], sz);
				}
			}
			if (m_parents[PLANE_U] && m_parents[PLANE_V]) {
				m_parents[PLANE_U]->generate(state, (cursor >> m_subsample_h) + 1, PLANE_U);
				memcpy(dst[PLANE_U][cursor >> m_subsample_h], src[PLANE_U][cursor >> m_subsample_h], sz >> m_subsample_w);

				m_parents[PLANE_V]->generate(state, (cursor >> m_subsample_h) + 1, PLANE_V);
				memcpy(dst[PLANE_V][cursor >> m_subsample_h], src[PLANE_V][cursor >> m_subsample_h], sz >> m_subsample_w);
			}
			if (m_parents[PLANE_A]) {
				for (unsigned i = cursor; i < cursor + (1U << m_subsample_h); ++i) {
					m_parents[PLANE_A]->generate(state, i + 1, PLANE_A);
					memcpy(dst[PLANE_A][i], src[PLANE_A][i], sz);
				}
			}

			if (state->pack_cb())
				state->pack_cb()(cursor, 0, attr.width);
		}

		state->set_cursor(id(), cursor);
	}
};

class FilterNode : public GraphNode {
	std::shared_ptr<ImageFilter> m_filter;
	node_map m_parents;
	plane_mask m_output_planes;
public:
	explicit FilterNode(node_id id, std::shared_ptr<ImageFilter> filter, const node_map &parents, const plane_mask &output_planes) :
		GraphNode(id),
		m_filter{ std::move(filter) },
		m_parents(parents),
		m_output_planes(output_planes)
	{}

	unsigned get_subsample_w() const override { return 0; }
	unsigned get_subsample_h() const override { return 0; }
	plane_mask get_plane_mask() const override { return m_output_planes; }

	image_attributes get_image_attributes(int plane) const override
	{
		zassert_d(m_output_planes[plane], "plane not present");
		return m_filter->get_image_attributes();
	}

	void init_context(ExecutionState *state) const override
	{
		image_attributes attr[PLANE_NUM] = {};
		std::fill_n(attr, PLANE_NUM, m_filter->get_image_attributes());

		state->alloc_context(id(), m_filter->get_context_size());
		state->alloc_shared_tmp(m_filter->get_tmp_size(0, m_filter->get_image_attributes().width));

		unsigned seq = static_cast<unsigned>(std::find(m_output_planes.begin(), m_output_planes.end(), true) - m_output_planes.begin());
		m_filter->init_context(state->get_context(id()), seq);
	}

	void simulate(SimulationState *state, unsigned first, unsigned last, int plane) const override
	{
		zassert_d(m_output_planes[plane], "plane not present");
		unsigned cursor = state->get_cursor(id(), 0);
		if (cursor >= last) {
			state->update(id(), first, last);
			return;
		}

		unsigned step = m_filter->get_simultaneous_lines();

		for (; cursor < last; cursor += step) {
			auto range = m_filter->get_required_row_range(cursor);

			for (int p = 0; p < PLANE_NUM; ++p) {
				if (m_parents[p])
					m_parents[p]->simulate(state, range.first, range.second, p);
			}
		}
		state->update(id(), first, cursor);
	}

	void generate(ExecutionState *state, unsigned last, int plane) const override
	{
		zassert_d(m_output_planes[plane], "plane not present");
		unsigned cursor = state->get_cursor(id());
		if (cursor >= last)
			return;

		ImageBuffer<const void> src[PLANE_NUM];
		const ImageBuffer<void> *dst = state->get_buffer(id());

		for (int p = 0; p < PLANE_NUM; ++p) {
			if (m_parents[p])
				src[p] = state->get_buffer(m_parents[p]->id())[p];
		}

		auto flags = m_filter->get_flags();
		unsigned step = m_filter->get_simultaneous_lines();

		for (; cursor < last; cursor += step) {
			auto range = m_filter->get_required_row_range(cursor);

			if (flags.color) {
				for (int p = 0; p < PLANE_NUM; ++p) {
					if (m_parents[p])
						m_parents[p]->generate(state, range.second, p);
				}
				m_filter->process(state->get_context(id()), src, dst, state->get_shared_tmp(), cursor, 0, m_filter->get_image_attributes().width);
			} else {
				for (int p = 0; p < PLANE_NUM; ++p) {
					if (m_output_planes[p]) {
						if (m_parents[p])
							m_parents[p]->generate(state, range.second, p);
						m_filter->process(state->get_context(id()), src + p, dst + p, state->get_shared_tmp(), cursor, 0, m_filter->get_image_attributes().width);
					}
				}
			}
		}

		state->set_cursor(id(), cursor);
	}
};

} // namespace


class FilterGraph2::impl {
	std::vector<std::unique_ptr<GraphNode>> m_nodes;
	SimulationState m_simulation;
	SourceNode *m_source;
	SinkNode *m_sink;

	node_id next_id() const { return static_cast<node_id>(m_nodes.size()); }

	node_map id_to_node(const id_map &ids) const
	{
		node_map nodes = {};

		for (int p = 0; p < PLANE_NUM; ++p) {
			if (ids[p] < 0)
				continue;

			zassert_d(ids[p] < next_id(), "node index out of range");
			nodes[p] = m_nodes[ids[p]].get();
		}

		return nodes;
	}
public:
	impl() : m_simulation{ 0 }, m_source {}, m_sink{} {}

	node_id add_source(const image_attributes &attr, unsigned subsample_w, unsigned subsample_h, const plane_mask &planes)
	{
		zassert_d(!m_source, "source already defined");
		m_nodes.emplace_back(ztd::make_unique<SourceNode>(next_id(), attr, subsample_w, subsample_h, planes));
		m_source = static_cast<SourceNode *>(m_nodes.back().get());
		return m_source->id();
	}

	node_id attach_filter(std::shared_ptr<ImageFilter> filter, const id_map &deps, const plane_mask &output_planes)
	{
		m_nodes.emplace_back(ztd::make_unique<FilterNode>(next_id(), std::move(filter), id_to_node(deps), output_planes));
		return m_nodes.back()->id();
	}

	void set_output(const id_map &deps)
	{
		zassert_d(!m_sink, "sink already defined");
		m_nodes.emplace_back(ztd::make_unique<SinkNode>(next_id(), id_to_node(deps)));
		m_sink = static_cast<SinkNode *>(m_nodes.back().get());

		SimulationState sim{ m_nodes.size() };
		m_sink->simulate(&sim, 0, m_sink->get_image_attributes(PLANE_Y).height, PLANE_Y);
		m_simulation = std::move(sim);
	}

	void process(const ImageBuffer<const void> src[], const ImageBuffer<void> dst[], void *tmp, callback unpack_cb, callback pack_cb) const
	{
		ExecutionState state{ m_nodes.size(), src, dst, unpack_cb, pack_cb };

		for (const auto &node : m_nodes) {
			if (node.get() == m_sink)
				continue;

			plane_mask planes = node->get_plane_mask();
			size_t rowsize[PLANE_NUM] = {};
			unsigned lines[PLANE_NUM] = {};
			unsigned mask[PLANE_NUM] = {};

			for (int p = 0; p < PLANE_NUM; ++p) {
				if (!planes[p])
					continue;

				auto attr = node->get_image_attributes(p);
				auto tmp = m_simulation.suggest_mask(node->id(), attr.height);

				if (p == PLANE_U || p == PLANE_V) {
					tmp.first >>= node->get_subsample_h();
					tmp.second = tmp.second == BUFFER_MAX ? BUFFER_MAX : tmp.second >> node->get_subsample_h();
				}

				rowsize[p] = ceil_n(static_cast<size_t>(attr.width) * pixel_size(attr.type), ALIGNMENT);
				lines[p] = tmp.first;
				mask[p] = tmp.second;
			}

			state.alloc_buffer(node->id(), rowsize, lines, mask, planes);
			node->init_context(&state);
		}

		m_sink->generate(&state, m_sink->get_image_attributes(PLANE_Y).height, PLANE_Y);
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

node_id FilterGraph2::add_source(const image_attributes &attr, unsigned subsample_w, unsigned subsample_h, const plane_mask &planes)
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

void FilterGraph2::process(const ImageBuffer<const void> src[], const ImageBuffer<void> dst[], void *tmp, callback unpack_cb, callback pack_cb) const
{
	get_impl()->process(src, dst, tmp, unpack_cb, pack_cb);
}

} // namespace graph
} // namespace zimg
