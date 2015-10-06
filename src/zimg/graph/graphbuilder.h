#pragma once

#ifndef ZIMG_GRAPH_GRAPHBUILDER_H_
#define ZIMG_GRAPH_GRAPHBUILDER_H_

#include <memory>
#include <vector>
#include "common/pixel.h"
#include "colorspace/colorspace.h"
#include "depth/depth.h"
#include "resize/resize.h"

namespace zimg {;

enum class CPUClass;

namespace graph {;

class FilterGraph;
class ImageFilter;

class FilterFactory {
public:
	typedef std::vector<std::unique_ptr<ImageFilter>> filter_list;

	virtual ~FilterFactory() = 0;

	virtual filter_list create_colorspace(const colorspace::ColorspaceConversion &conv) = 0;

	virtual filter_list create_depth(const depth::DepthConversion &conv) = 0;

	virtual filter_list create_resize(const resize::ResizeConversion &conv) = 0;
};

class DefaultFilterFactory : public FilterFactory {
public:
	filter_list create_colorspace(const colorspace::ColorspaceConversion &conv) override;

	filter_list create_depth(const depth::DepthConversion &conv) override;

	filter_list create_resize(const resize::ResizeConversion &conv) override;
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

	void attach_filter(std::unique_ptr<ImageFilter> &&filter);

	void attach_filter_uv(std::unique_ptr<ImageFilter> &&filter);

	void color_to_grey(colorspace::MatrixCoefficients matrix);

	void grey_to_color(ColorFamily color, colorspace::MatrixCoefficients matrix, unsigned subsample_w, unsigned subsample_h,
	                   ChromaLocationW chroma_location_w, ChromaLocationH chroma_location_h);

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
