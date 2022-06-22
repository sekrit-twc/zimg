#include <algorithm>
#include <memory>
#include <vector>
#include "common/alloc.h"
#include "common/pixel.h"
#include "common/zassert.h"
#include "graphengine/filter.h"
#include "graphengine/graph.h"

#include "frame.h"
#include "utils.h"

struct FilterExecutor::data {
	graphengine::GraphImpl graph;
	graphengine::Graph::EndpointConfiguration endpoints;
	zimg::AlignedVector<unsigned char> tmp;
};

FilterExecutor::FilterExecutor(const graphengine::Filter *filter, const ImageFrame *src_frame, ImageFrame *dst_frame) :
	FilterExecutor(filter
		? std::vector<std::pair<int, const graphengine::Filter *>>{ { ALL_PLANES, filter } }
	    : std::vector<std::pair<int, const graphengine::Filter *>>{},
			src_frame, dst_frame)
{}

FilterExecutor::FilterExecutor(const std::vector<std::pair<int, const graphengine::Filter *>> &filters, const ImageFrame *src_frame, ImageFrame *dst_frame) try :
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
		m_data->endpoints[0].buffer[p] = src_frame->as_buffer(p);
	}

	m_data->endpoints[1].id = sink_id;
	for (unsigned p = 0; p < dst_frame->planes(); ++p) {
		m_data->endpoints[1].buffer[p] = dst_frame->as_buffer(p);
	}

	m_data->tmp.resize(m_data->graph.get_tmp_size(false));
} catch (const graphengine::Exception &e) {
	throw std::runtime_error{ e.msg };
}

FilterExecutor::~FilterExecutor() = default;

FilterExecutor::FilterExecutor(FilterExecutor &&) noexcept = default;

FilterExecutor &FilterExecutor::operator=(FilterExecutor &&) noexcept = default;

void FilterExecutor::operator()()
{
	m_data->graph.run(m_data->endpoints, m_data->tmp.data());
}
