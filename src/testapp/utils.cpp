#include <algorithm>
#include <memory>
#include <vector>
#include "common/alloc.h"
#include "common/pixel.h"
#include "common/zassert.h"
#include "graph/image_filter.h"
#include "graphengine/filter.h"
#include "graphengine/graph.h"

#include "frame.h"
#include "utils.h"

struct FilterExecutor::data {
	zimg::AlignedVector<char> ctx;
	zimg::AlignedVector<char> tmp;
};

void FilterExecutor::exec_grey(const zimg::graph::ImageFilter *filter, unsigned plane)
{
	auto src_buffer = m_src_frame->as_read_buffer(plane);
	auto dst_buffer = m_dst_frame->as_write_buffer(plane);

	auto attr = filter->get_image_attributes();
	unsigned step = filter->get_simultaneous_lines();

	filter->init_context(m_data->ctx.data(), plane);

	for (unsigned i = 0; i < attr.height; i += step) {
		filter->process(m_data->ctx.data(), &src_buffer, &dst_buffer, m_data->tmp.data(), i, 0, attr.width);
	}
}

void FilterExecutor::exec_color()
{
	auto attr = m_filter->get_image_attributes();
	unsigned step = m_filter->get_simultaneous_lines();

	m_filter->init_context(m_data->ctx.data(), 0);

	for (unsigned i = 0; i < attr.height; i += step) {
		m_filter->process(m_data->ctx.data(), m_src_frame->as_read_buffer(), m_dst_frame->as_write_buffer(), m_data->tmp.data(), i, 0, attr.width);
	}
}

FilterExecutor::FilterExecutor(const zimg::graph::ImageFilter *filter, const zimg::graph::ImageFilter *filter_uv, const ImageFrame *src_frame, ImageFrame *dst_frame) :
	m_data{ std::make_shared<data>() },
	m_filter{ filter },
	m_filter_uv{ filter_uv },
	m_src_frame{ src_frame },
	m_dst_frame{ dst_frame }
{
	filter_uv = filter_uv ? filter_uv : filter;

	m_data->ctx.resize(std::max(filter->get_context_size(), filter_uv->get_context_size()));
	m_data->tmp.resize(std::max(filter->get_tmp_size(0, dst_frame->width()),
	                            filter_uv->get_tmp_size(0, dst_frame->width())));
}

void FilterExecutor::operator()()
{
	zimg::graph::ImageFilter::filter_flags flags = m_filter->get_flags();

	if (!flags.color) {
		unsigned planes = m_dst_frame->planes();

		for (unsigned p = 0; p < planes; ++p) {
			const zimg::graph::ImageFilter *filter = (m_filter_uv && (p == 1 || p == 2)) ? m_filter_uv : m_filter;
			exec_grey(filter, p);
		}
	} else {
		exec_color();
	}
}

struct FilterExecutor_GE::data {
	graphengine::Graph graph;
	graphengine::Graph::EndpointConfiguration endpoints;
	zimg::AlignedVector<char> tmp;
};

FilterExecutor_GE::FilterExecutor_GE(const graphengine::Filter *filter, const ImageFrame *src_frame, ImageFrame *dst_frame) :
	FilterExecutor_GE(filter
		? std::vector<std::pair<int, const graphengine::Filter *>>{ { ALL_PLANES, filter } }
	    : std::vector<std::pair<int, const graphengine::Filter *>>{},
			src_frame, dst_frame)
{}

FilterExecutor_GE::FilterExecutor_GE(const std::vector<std::pair<int, const graphengine::Filter *>> &filters, const ImageFrame *src_frame, ImageFrame *dst_frame) :
	m_data{ std::make_unique<data>() }
{
	std::vector<graphengine::PlaneDescriptor> src_desc(src_frame->planes());
	for (unsigned p = 0; p < src_frame->planes(); ++p) {
		src_desc[p].width = src_frame->width(p);
		src_desc[p].height = src_frame->height(p);
		src_desc[p].bytes_per_sample = zimg::pixel_size(src_frame->pixel_type());
	}

	graphengine::node_id src_id = m_data->graph.add_source(src_frame->planes(), src_desc.data());

	std::vector<graphengine::node_dep_desc> ids(graphengine::NODE_MAX_PLANES);
	zassert(src_frame->planes() == dst_frame->planes(), "incompatible plane count");
	zassert(src_frame->planes() <= graphengine::NODE_MAX_PLANES, "incompatible plane count");

	for (unsigned p = 0; p < src_frame->planes(); ++p) {
		ids[p] = { src_id, p };
	}

	for (const auto &entry : filters) {
		const graphengine::FilterDescriptor &desc = entry.second->descriptor();

		if (desc.num_deps == 1 && desc.num_planes == 1) {
			std::vector<unsigned> planes;

			if (entry.first == ALL_PLANES) {
				for (unsigned p = 0; p < src_frame->planes(); ++p) {
					planes.push_back(p);
				}
			} else if (entry.first == CHROMA_PLANES) {
				planes = { 1, 2 };
			} else if (entry.first >= 0) {
				planes = { static_cast<unsigned>(entry.first) };
			}

			for (unsigned p : planes) {
				ids[p] = { m_data->graph.add_transform(entry.second, &ids[p]), 0 };
			}
		} else {
			zassert(entry.first == ALL_PLANES, "incompatible dependency type");

			graphengine::node_id id = m_data->graph.add_transform(entry.second, ids.data());
			for (unsigned p = 0; p < desc.num_planes; ++p) {
				ids[p] = { id, p };
			}
		}
	}
	graphengine::node_id sink_id = m_data->graph.add_sink(dst_frame->planes(), ids.data());

	m_data->endpoints[0].id = src_id;
	for (unsigned p = 0; p < src_frame->planes(); ++p) {
		m_data->endpoints[0].buffer[p] = { const_cast<void *>(src_frame->as_read_buffer(p).data()), src_frame->as_read_buffer(p).stride(), src_frame->as_read_buffer(p).mask() };
	}

	m_data->endpoints[1].id = sink_id;
	for (unsigned p = 0; p < dst_frame->planes(); ++p) {
		m_data->endpoints[1].buffer[p] = { dst_frame->as_write_buffer(p).data(), dst_frame->as_write_buffer(p).stride(), dst_frame->as_write_buffer(p).mask() };
	}

	m_data->tmp.resize(m_data->graph.get_tmp_size(false));
}

FilterExecutor_GE::~FilterExecutor_GE() = default;

FilterExecutor_GE::FilterExecutor_GE(FilterExecutor_GE &&) noexcept = default;

FilterExecutor_GE &FilterExecutor_GE::operator=(FilterExecutor_GE &&) noexcept = default;

void FilterExecutor_GE::operator()()
{
	m_data->graph.run(m_data->endpoints, m_data->tmp.data());
}
