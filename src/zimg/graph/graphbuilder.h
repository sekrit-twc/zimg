#pragma once

#ifndef ZIMG_GRAPH_GRAPHBUILDER_H_
#define ZIMG_GRAPH_GRAPHBUILDER_H_

#include <memory>
#include <vector>
#include "common/pixel.h"
#include "colorspace/colorspace_param.h"

namespace zimg {;

enum class CPUClass;

namespace depth {;

enum class DitherType;

} // namespace depth

namespace resize {;

class Filter;

} // namespace resize


namespace graph {;

class FilterGraph;
class IZimgFilter;

class FilterFactory {
public:
	struct colorspace_params {
		unsigned width;
		unsigned height;
		colorspace::ColorspaceDefinition csp_in;
		colorspace::ColorspaceDefinition csp_out;
		CPUClass cpu;
	};

	struct depth_params {
		unsigned width;
		unsigned height;
		depth::DitherType type;
		PixelFormat format_in;
		PixelFormat format_out;
		CPUClass cpu;
	};

	struct resize_params {
		const resize::Filter *filter;
		PixelType type;
		unsigned depth;

		unsigned width_in;
		unsigned height_in;
		unsigned width_out;
		unsigned height_out;

		double shift_w;
		double shift_h;
		double subwidth;
		double subheight;

		CPUClass cpu;
	};

	typedef std::vector<std::unique_ptr<IZimgFilter>> filter_list;

	virtual ~FilterFactory() = 0;

	virtual filter_list create_colorspace(const colorspace_params &params) = 0;

	virtual filter_list create_depth(const depth_params &params) = 0;

	virtual filter_list create_resize(const resize_params &params) = 0;
};

class DefaultFilterFactory : public FilterFactory {
public:
	filter_list create_colorspace(const colorspace_params &params) override;

	filter_list create_depth(const depth_params &params) override;

	filter_list create_resize(const resize_params &params) override;
};


class GraphBuilder {
public:
	enum class ColorFamily {
		COLOR_GREY,
		COLOR_RGB,
		COLOR_YUV
	};

	enum class FieldParity {
		FIELD_PROGRESSIVE,
		FIELD_TOP,
		FIELD_BOTTOM
	};

	enum class ChromaLocationW {
		CHROMA_W_LEFT,
		CHROMA_W_CENTER
	};

	enum class ChromaLocationH {
		CHROMA_H_CENTER,
		CHROMA_H_TOP,
		CHROMA_H_BOTTOM
	};

	struct params {
		std::unique_ptr<const resize::Filter> filter;
		std::unique_ptr<const resize::Filter> filter_uv;
		depth::DitherType dither_type;
		CPUClass cpu;
	};

	struct state {
		unsigned width;
		unsigned height;
		PixelType type;
		unsigned subsample_w;
		unsigned subsample_h;

		ColorFamily color;
		colorspace::ColorspaceDefinition colorspace;

		unsigned depth;
		bool fullrange;

		FieldParity parity;
		ChromaLocationW chroma_location_w;
		ChromaLocationH chroma_location_h;
	};
private:
	struct resize_spec {
		unsigned width;
		unsigned height;
		unsigned subsample_w;
		unsigned subsample_h;
		double shift_w;
		double shift_h;
		double subwidth;
		double subheight;
		ChromaLocationW chroma_location_w;
		ChromaLocationH chroma_location_h;

		resize_spec() = default;

		explicit resize_spec(const state &state);
	};

	std::unique_ptr<FilterGraph> m_graph;
	FilterFactory *m_factory;
	state m_state;

	void attach_filter(std::unique_ptr<IZimgFilter> &&filter);

	void attach_filter_uv(std::unique_ptr<IZimgFilter> &&filter);

	void convert_colorspace(const colorspace::ColorspaceDefinition &colorspace, const params *params);

	void convert_depth(const PixelFormat &format, const params *params);

	void convert_resize(const resize_spec &spec, const params *params);
public:
	GraphBuilder() = default;

	~GraphBuilder();

	GraphBuilder &set_factory(FilterFactory *factory);

	GraphBuilder &set_source(const state &source);

	GraphBuilder &connect_graph(const state &target, const params *params);

	FilterGraph *complete_graph();
};

} // namespace graph
} // namespace zimg

#endif // ZIMG_GRAPH_GRAPHBUILDER_H_
