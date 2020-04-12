#pragma once

#ifndef ZIMG_GRAPH_GRAPHNODE_H_
#define ZIMG_GRAPH_GRAPHNODE_H_

#include <array>
#include <memory>
#include <utility>
#include <vector>
#include "filtergraph.h"
#include "image_filter.h"

namespace zimg {
namespace graph {

class GraphNode;

class SimulationState {
public:
	struct result {
		struct s {
			unsigned cache_lines;
			unsigned mask;
			size_t context_size;
		};

		std::vector<s> node_result;
		size_t shared_tmp;
	};
private:
	struct state {
		size_t context_size;
		unsigned cache_pos;
		unsigned cache_history;
		unsigned cursor;
		unsigned subsample_h;
		bool cursor_initialized;
	};

	std::vector<state> m_state;
	size_t m_tmp;
public:
	explicit SimulationState(const std::vector<std::unique_ptr<GraphNode>> &nodes);

	result get_result(const std::vector<std::unique_ptr<GraphNode>> &nodes) const;

	void update(node_id id, node_id cache_id, unsigned first, unsigned last, unsigned plane);

	unsigned get_cursor(node_id id, unsigned initial_pos) const;

	void alloc_context(node_id id, size_t sz);

	void alloc_tmp(size_t sz);
};


class ExecutionState {
public:
	struct node_state {
		void *context;
		unsigned left;
		unsigned right;
	};
private:
	class guard_page;

	FilterGraph::callback m_unpack_cb;
	FilterGraph::callback m_pack_cb;

	ColorImageBuffer<void> *m_buffers;
	unsigned *m_cursors;
	node_state *m_state;
	unsigned char *m_init_bitset;
	void *m_tmp;

	guard_page **m_guard_pages;
public:
	static size_t calculate_tmp_size(const SimulationState::result &sim, const std::vector<std::unique_ptr<GraphNode>> &nodes);

	ExecutionState(const SimulationState::result &sim, const std::vector<std::unique_ptr<GraphNode>> &nodes, node_id src_id, node_id dst_id, const ImageBuffer<const void> src[], const ImageBuffer<void> dst[], FilterGraph::callback unpack_cb, FilterGraph::callback pack_cb, void *buf);

	const FilterGraph::callback &unpack_cb() const { return m_unpack_cb; }
	const FilterGraph::callback &pack_cb() const { return m_pack_cb; }

	unsigned get_cursor(node_id id) const { return m_cursors[id]; }
	void set_cursor(node_id id, unsigned pos) { m_cursors[id] = pos; }

	const ColorImageBuffer<void> &get_buffer(node_id id) { return m_buffers[id]; }
	node_state *get_node_state(node_id id) { return m_state + id; }
	void *get_shared_tmp() const { return m_tmp; }

	bool is_initialized(node_id id) const;
	void set_initialized(node_id id);

	void reset_tile_bounds(node_id id);

	void reset_initialized(size_t max_id);

	void check_guard_pages() const;
};


class GraphNode {
protected:
	typedef ImageFilter::image_attributes image_attributes;
private:
	node_id m_id;
	node_id m_cache_id;
	int m_ref_count;
protected:
	explicit GraphNode(node_id id) : m_id{ id }, m_cache_id{ id }, m_ref_count{} {}

	void set_cache_id(node_id id) { m_cache_id = id; }
public:
	virtual ~GraphNode() = 0;

	node_id id() const { return m_id; }

	node_id cache_id() const { return m_cache_id; }

	int ref_count() const { return m_ref_count; }

	void add_ref() { ++m_ref_count; }

	virtual bool is_sourcesink() const = 0;

	virtual unsigned get_subsample_w() const = 0;

	virtual unsigned get_subsample_h() const = 0;

	virtual plane_mask get_plane_mask() const = 0;

	virtual image_attributes get_image_attributes(int plane) const = 0;

	virtual void simulate(SimulationState *state, unsigned first, unsigned last, int plane) const = 0;

	virtual void simulate_alloc(SimulationState *state) const = 0;

	virtual void try_inplace() = 0;

	virtual void request_external_cache(node_id id) = 0;

	virtual void init_context(ExecutionState *state, unsigned top, unsigned left, unsigned right, int plane) const = 0;

	virtual void generate(ExecutionState *state, unsigned last, int plane) const = 0;
};



typedef std::array<GraphNode *, PLANE_NUM> node_map;

std::unique_ptr<GraphNode> make_source_node(node_id id, const ImageFilter::image_attributes &attr, unsigned subsample_w, unsigned subsample_h, const plane_mask &planes);

std::unique_ptr<GraphNode> make_sink_node(node_id id, const node_map &parents);

std::unique_ptr<GraphNode> make_filter_node(node_id id, std::shared_ptr<ImageFilter> filter, const node_map &parents, const plane_mask &output_planes);

} // namespace graph
} // namespace zimg

#endif // ZIMG_GRAPH_GRAPHNODE_H_
