#include <algorithm>
#include <cstdint>
#include <cstring>

#include "Common/align.h"
#include "Common/except.h"
#include "Common/filtergraph.h"
#include "Common/linebuffer.h"
#include "Common/pixel.h"
#include "Common/zfilter.h"

#include "gtest/gtest.h"

namespace {;

template <class T>
T splat_byte(unsigned char b)
{
	T ret;

	for (size_t i = 0; i < sizeof(ret); ++i) {
		reinterpret_cast<char *>(&ret)[i] = b;
	}
	return ret;
}

template <class T>
class AuditImage {
	zimg::AlignedVector<T> m_vector[3];
	unsigned m_width[3];
	unsigned m_height[3];
	ptrdiff_t m_stride[3];
	T *m_image_ptr[3];
	T m_test_val[3] = { splat_byte<T>(0xCD), splat_byte<T>(0xCD), splat_byte<T>(0xCD) };
	T m_guard_val = splat_byte<T>(0xFE);

	void add_guard_bytes()
	{
		for (unsigned p = 0; p < 3; ++p) {
			T *image_base = m_image_ptr[p];
			ptrdiff_t stride_x = m_stride[p] / sizeof(T);

			if (!image_base)
				continue;

			std::fill_n(image_base - stride_x, stride_x, m_guard_val);

			for (unsigned i = 0; i < m_height[p]; ++i) {
				T *line_base = image_base + i * stride_x;

				std::fill_n(line_base - zimg::AlignmentOf<T>::value, zimg::AlignmentOf<T>::value, m_guard_val);
				std::fill_n(line_base + m_width[p], stride_x - m_width[p], m_guard_val);
			}

			std::fill_n(image_base + m_height[p] * stride_x, stride_x, m_guard_val);
		}
	}

	void check_guard_bytes()
	{
		const auto pred = [=](T x) { return x != m_guard_val; };

		for (unsigned p = 0; p < 3; ++p) {
			T *image_base = m_image_ptr[p];
			ptrdiff_t stride_x = m_stride[p] / sizeof(T);

			if (!image_base)
				continue;

			auto it = std::find_if(image_base - stride_x, image_base, pred);
			ASSERT_TRUE(it == image_base) << "header guard bytes corrupted";

			for (unsigned i = 0; i < m_height[p]; ++i) {
				T *line_base = image_base + i * stride_x;

				it = std::find_if(line_base - zimg::AlignmentOf<T>::value, line_base, pred);
				ASSERT_TRUE(it == line_base) << "line guard header corrupted at: " << i;

				it = std::find_if(line_base + m_width[p], line_base + stride_x, pred);
				ASSERT_TRUE(it == line_base + stride_x) << "line guard footer corrupted at: " << i;
			}

			it = std::find_if(image_base + m_height[p] * stride_x, image_base + (m_height[p] + 1) * stride_x, pred);
			ASSERT_TRUE(it == image_base + (m_height[p] + 1) * stride_x) << "footer guard bytes corrupted";
		}
	}
public:
	AuditImage(unsigned width, unsigned height, unsigned subsample_w, unsigned subsample_h, bool color) :
		m_width{},
		m_height{},
		m_stride{},
		m_image_ptr{}
	{
		for (unsigned p = 0; p < (color ? 3U : 1U); ++p) {
			unsigned width_p = width >> (p ? subsample_w : 0);
			unsigned height_p = height >> (p ? subsample_h : 0);

			unsigned guarded_width = zimg::align(width_p, zimg::AlignmentOf<T>::value) + 2 * zimg::AlignmentOf<T>::value;
			unsigned guarded_height = height_p + 2;

			m_width[p] = width_p;
			m_height[p] = height_p;
			m_stride[p] = (ptrdiff_t)(guarded_width * sizeof(T));

			m_vector[p].resize((size_t)guarded_width * guarded_height);
			m_image_ptr[p] = m_vector[p].data() + m_stride[p] / sizeof(T);
		}

		add_guard_bytes();
	}

	void set_test_byte(unsigned char x, unsigned p)
	{
		m_test_val[p] = splat_byte<T>(x);
	}

	void set_test_byte(unsigned char x)
	{
		for (unsigned p = 0; p < 3; ++p) {
			set_test_byte(x, p);
		}
	}

	void validate()
	{
		for (unsigned p = 0; p < 3; ++p) {
			T *image_base = m_image_ptr[p];
			ptrdiff_t stride_x = m_stride[p] / sizeof(T);

			for (unsigned i = 0; i < m_height[p]; ++i) {
				T *line_base = image_base + i * stride_x;

				auto it = std::find_if(line_base, line_base + m_width[p], [=](T x) { return x != m_test_val[p]; });
				ASSERT_TRUE(it == line_base + m_width[p]) << "validation failure at: " << p << " (" << i << ", " << it - line_base << ")";
			}
		}
		check_guard_bytes();
	}

	void default_fill()
	{
		for (unsigned p = 0; p < 3; ++p) {
			T *image_base = m_image_ptr[p];
			ptrdiff_t stride_x = m_stride[p] / sizeof(T);

			for (unsigned i = 0; i < m_height[p]; ++i) {
				T *line_base = image_base + i * stride_x;

				std::fill_n(line_base, m_width[p], m_test_val[p]);
			}
		}
	}

	zimg::ZimgImageBuffer as_image_buffer()
	{
		zimg::ZimgImageBuffer buf;

		for (unsigned p = 0; p < 3; ++p) {
			buf.data[p] = m_image_ptr[p];
			buf.stride[p] = m_stride[p];
			buf.mask[p] = -1;
		}
		return buf;
	}
};

class MockFilter : public zimg::IZimgFilter {
protected:
	struct context {
		unsigned last_line = 0;
		unsigned last_left = 0;
		unsigned last_right = 0;
	};

	image_attributes m_attr;
	zimg::ZimgFilterFlags m_flags;
	mutable unsigned m_total_calls;
	unsigned m_simultaneous_lines;
	unsigned m_horizontal_support;
public:
	MockFilter(unsigned width, unsigned height, zimg::PixelType type, const zimg::ZimgFilterFlags &flags = {}) :
		m_attr{ width, height, type },
		m_flags(flags),
		m_total_calls{},
		m_simultaneous_lines{ 1 },
		m_horizontal_support{}
	{
	}

	unsigned get_total_calls() const
	{
		return m_total_calls;
	}

	void set_simultaneous_lines(unsigned n)
	{
		m_simultaneous_lines = n;
	}

	void set_horizontal_support(unsigned n)
	{
		m_horizontal_support = n;
	}

	zimg::ZimgFilterFlags get_flags() const override
	{
		return m_flags;
	}

	image_attributes get_image_attributes() const override
	{
		return m_attr;
	}

	pair_unsigned get_required_row_range(unsigned i) const override
	{
		EXPECT_LT(i, m_attr.height);

		if (m_flags.entire_plane)
			return{ 0, m_attr.height };
		else
			return{ i, std::min(i + m_simultaneous_lines, m_attr.height) };
	}

	pair_unsigned get_required_col_range(unsigned left, unsigned right) const override
	{
		EXPECT_LE(left, right);
		EXPECT_LE(right, m_attr.width);

		if (m_flags.entire_row)
			return{ 0, m_attr.width };
		else
			return{ std::max(left, m_horizontal_support) - left, std::min(right + m_horizontal_support, m_attr.width) };
	}

	unsigned get_simultaneous_lines() const override
	{
		return m_flags.entire_plane ? m_attr.height : m_simultaneous_lines;
	}

	unsigned get_max_buffering() const override
	{
		return m_flags.entire_plane ? (unsigned)-1 : 1;
	}

	size_t get_context_size() const override
	{
		return sizeof(context);
	}

	size_t get_tmp_size(unsigned left, unsigned right) const override
	{
		EXPECT_LE(left, right);
		EXPECT_LE(right, m_attr.width);
		return 0;
	}

	void init_context(void *ctx) const override
	{
		new (ctx) context{};
	}

	void process(void *ctx, const zimg::ZimgImageBufferConst &src, const zimg::ZimgImageBuffer &dst, void *tmp, unsigned i, unsigned left, unsigned right) const override
	{
		context *audit_ctx = reinterpret_cast<context *>(ctx);

		ASSERT_LT(i, m_attr.height);
		ASSERT_LT(left, right);
		ASSERT_LE(right, m_attr.width);

		if (m_flags.has_state && left == audit_ctx->last_left && right == audit_ctx->last_right)
			ASSERT_GE(i, audit_ctx->last_line);

		if (m_flags.entire_row) {
			ASSERT_EQ(0, left);
			ASSERT_EQ(m_attr.width, right);
		}

		if (m_flags.entire_plane)
			ASSERT_EQ(i, 0);

		for (unsigned p = 0; p < (m_flags.color ? 3U : 1U); ++p) {
			if (!m_flags.in_place)
				ASSERT_NE(src.data[p], dst.data[p]);
			if (m_flags.entire_plane) {
				ASSERT_EQ((unsigned)-1, src.mask[p]);
				ASSERT_EQ((unsigned)-1, dst.mask[p]);
			}
		}

		audit_ctx->last_line = i;
		audit_ctx->last_left = left;
		audit_ctx->last_right = right;
		++m_total_calls;
	}
};

template <class T>
class SplatFilter : public MockFilter {
	T m_src_val = 0xCD;
	T m_dst_val = 0xCD;

	void assert_pixel_size(zimg::PixelType type)
	{
		ASSERT_EQ(zimg::pixel_size(type), sizeof(T));
	}
public:
	SplatFilter(unsigned width, unsigned height, zimg::PixelType type, const zimg::ZimgFilterFlags &flags = {}) :
		MockFilter(width, height, type, flags)
	{
		assert_pixel_size(type);
	}

	void set_input_test_byte(unsigned char x)
	{
		m_src_val = splat_byte<T>(x);
	}

	void set_output_test_byte(unsigned char x)
	{
		m_dst_val = splat_byte<T>(x);
	}

	void process(void *ctx, const zimg::ZimgImageBufferConst &src, const zimg::ZimgImageBuffer &dst, void *tmp, unsigned i, unsigned left, unsigned right) const override
	{
		context *audit_ctx = reinterpret_cast<context *>(ctx);

		MockFilter::process(ctx, src, dst, tmp, i, left, right);
		pair_unsigned row_range = get_required_row_range(i);
		pair_unsigned col_range = get_required_col_range(left, right);

		for (unsigned p = 0; p < (m_flags.color ? 3U : 1U); ++p) {
			zimg::LineBuffer<const T> src_buf{ src, p };
			zimg::LineBuffer<T> dst_buf{ dst, p };

			for (unsigned ii = row_range.first; ii < row_range.second; ++ii) {
				auto it = std::find_if(src_buf[ii] + col_range.first, src_buf[ii] + col_range.second, [=](T x) { return x != m_src_val; });
				ASSERT_TRUE(it == src_buf[ii] + col_range.second) <<
					"found invalid value at position: (" << ii << ", " << it - src_buf[ii] << ")";
			}

			for (unsigned ii = i; ii < std::min(i + get_simultaneous_lines(), m_attr.height); ++ii) {
				std::fill(dst_buf[ii] + left, dst_buf[ii] + right, m_dst_val);
			}
		}
	}
};

} // namespace


TEST(FilterGraphTest, test_noop)
{
	const unsigned w = 640;
	const unsigned h = 480;
	const zimg::PixelType type = zimg::PixelType::BYTE;

	for (unsigned x = 0; x < 2; ++x) {
		SCOPED_TRACE(!!x);

		zimg::FilterGraph graph{ w, h, type, 0, 0, !!x };
		graph.complete();

		AuditImage<uint8_t> src_image{ w, h, 0, 0, !!x };
		AuditImage<uint8_t> dst_image{ w, h, 0, 0, !!x };
		zimg::AlignedVector<char> tmp(graph.get_tmp_size());

		src_image.default_fill();
		graph.process(src_image.as_image_buffer(), dst_image.as_image_buffer(), tmp.data(), nullptr, nullptr);

		SCOPED_TRACE("validating src");
		src_image.validate();
		SCOPED_TRACE("validating dst");
		dst_image.validate();
	}
}

TEST(FilterGraphTest, test_noop_subsampling)
{
	const unsigned w = 640;
	const unsigned h = 480;
	const zimg::PixelType type = zimg::PixelType::BYTE;

	for (unsigned sw = 0; sw < 3; ++sw) {
		for (unsigned sh = 0; sh < 3; ++sh) {
			SCOPED_TRACE(sw);
			SCOPED_TRACE(sh);

			zimg::FilterGraph graph{ w, h, type, sw, sh, true };
			graph.complete();

			AuditImage<uint8_t> src_image{ w, h, sw, sh, true };
			AuditImage<uint8_t> dst_image{ w, h, sw, sh, true };
			zimg::AlignedVector<char> tmp(graph.get_tmp_size());

			src_image.default_fill();
			graph.process(src_image.as_image_buffer(), dst_image.as_image_buffer(), tmp.data(), nullptr, nullptr);

			SCOPED_TRACE("validating src");
			src_image.validate();
			SCOPED_TRACE("validating dst");
			dst_image.validate();
		}
	}
}

TEST(FilterGraphTest, test_basic)
{
	const unsigned w = 640;
	const unsigned h = 480;
	const zimg::PixelType type = zimg::PixelType::WORD;

	const uint8_t test_byte1 = 0xCD;
	const uint8_t test_byte2 = 0xDD;
	const uint8_t test_byte3 = 0xDC;

	for (unsigned x = 0; x < 2; ++x) {
		SCOPED_TRACE(!!x);

		zimg::ZimgFilterFlags flags1{};
		flags1.has_state = true;
		flags1.entire_row = true;
		flags1.entire_plane = true;
		flags1.color = !!x;

		zimg::ZimgFilterFlags flags2{};
		flags2.has_state = true;
		flags2.entire_row = false;
		flags2.entire_plane = false;
		flags2.color = !!x;

		std::unique_ptr<SplatFilter<uint16_t>> filter1_uptr{ new SplatFilter<uint16_t>{ w, h, type, flags1 } };
		std::unique_ptr<SplatFilter<uint16_t>> filter2_uptr{ new SplatFilter<uint16_t>{ w, h, type, flags2 } };
		SplatFilter<uint16_t> *filter1 = filter1_uptr.get();
		SplatFilter<uint16_t> *filter2 = filter2_uptr.get();

		filter1->set_input_test_byte(test_byte1);
		filter1->set_output_test_byte(test_byte2);

		filter2->set_input_test_byte(test_byte2);
		filter2->set_output_test_byte(test_byte3);

		zimg::FilterGraph graph{ w, h, type, 0, 0, !!x };

		graph.attach_filter(filter1);
		filter1_uptr.release();
		graph.attach_filter(filter2);
		filter2_uptr.release();
		graph.complete();

		AuditImage<uint16_t> src_image{ w, h, 0, 0, !!x };
		AuditImage<uint16_t> dst_image{ w, h, 0, 0, !!x };
		zimg::AlignedVector<char> tmp(graph.get_tmp_size());

		src_image.set_test_byte(test_byte1);
		src_image.default_fill();
		graph.process(src_image.as_image_buffer(), dst_image.as_image_buffer(), tmp.data(), nullptr, nullptr);
		dst_image.set_test_byte(test_byte3);

		ASSERT_EQ(1U, filter1->get_total_calls());
		ASSERT_EQ(h, filter2->get_total_calls());

		SCOPED_TRACE("validating src");
		src_image.validate();
		SCOPED_TRACE("validating dst");
		dst_image.validate();
	}
}

TEST(FilterGraphTest, test_skip_plane)
{
	const unsigned w = 640;
	const unsigned h = 480;
	const zimg::PixelType type = zimg::PixelType::FLOAT;

	const uint8_t test_byte1 = 0xCD;
	const uint8_t test_byte2 = 0xDD;
	const uint8_t test_byte3 = 0xDC;

	for (unsigned x = 0; x < 2; ++x) {
		SCOPED_TRACE(!!x);

		zimg::ZimgFilterFlags flags1{};
		flags1.has_state = true;
		flags1.entire_row = true;
		flags1.color = true;

		zimg::ZimgFilterFlags flags2{};
		flags2.has_state = false;
		flags2.entire_row = true;
		flags2.color = false;

		std::unique_ptr<SplatFilter<float>> filter1_uptr{ new SplatFilter<float>{ w, h, type, flags1} };
		std::unique_ptr<SplatFilter<float>> filter2_uptr{ new SplatFilter<float>{ w, h, type, flags2} };
		SplatFilter<float> *filter1 = filter1_uptr.get();
		SplatFilter<float> *filter2 = filter2_uptr.get();

		filter1->set_input_test_byte(test_byte1);
		filter1->set_output_test_byte(test_byte2);

		filter2->set_input_test_byte(test_byte2);
		filter2->set_output_test_byte(test_byte3);

		zimg::FilterGraph graph{ w, h, type, 0, 0, true };

		graph.attach_filter(filter1);
		filter1_uptr.release();

		if (x)
			graph.attach_filter(filter2);
		else
			graph.attach_filter_uv(filter2);
		filter2_uptr.release();

		graph.complete();

		AuditImage<float> src_image{ w, h, 0, 0, true };
		AuditImage<float> dst_image{ w, h, 0, 0, true };
		zimg::AlignedVector<char> tmp(graph.get_tmp_size());

		src_image.set_test_byte(test_byte1);
		src_image.default_fill();

		graph.process(src_image.as_image_buffer(), dst_image.as_image_buffer(), tmp.data(), nullptr, nullptr);

		if (x) {
			dst_image.set_test_byte(test_byte3, 0);
			dst_image.set_test_byte(test_byte2, 1);
			dst_image.set_test_byte(test_byte2, 2);
		} else {
			dst_image.set_test_byte(test_byte2, 0);
			dst_image.set_test_byte(test_byte3, 1);
			dst_image.set_test_byte(test_byte3, 2);
		}

		ASSERT_EQ(h, filter1->get_total_calls());
		ASSERT_EQ(h * (x ? 1 : 2), filter2->get_total_calls());

		SCOPED_TRACE("validating src");
		src_image.validate();
		SCOPED_TRACE("validating dst");
		dst_image.validate();
	}
}

TEST(FilterGraphTest, test_support)
{
	const unsigned w = 1024;
	const unsigned h = 576;
	const zimg::PixelType type = zimg::PixelType::HALF;

	const uint8_t test_byte1 = 0xCD;
	const uint8_t test_byte2 = 0xDD;
	const uint8_t test_byte3 = 0xDC;

	for (unsigned x = 0; x < 2; ++x) {
		SCOPED_TRACE(!!x);

		std::unique_ptr<SplatFilter<uint16_t>> filter1_uptr{ new SplatFilter<uint16_t>{ w, h, type} };
		std::unique_ptr<SplatFilter<uint16_t>> filter2_uptr{ new SplatFilter<uint16_t>{ w, h, type} };
		SplatFilter<uint16_t> *filter1 = filter1_uptr.get();
		SplatFilter<uint16_t> *filter2 = filter2_uptr.get();

		filter1->set_input_test_byte(test_byte1);
		filter1->set_output_test_byte(test_byte2);

		filter2->set_input_test_byte(test_byte2);
		filter2->set_output_test_byte(test_byte3);

		if (x) {
			filter1->set_horizontal_support(5);
			filter1->set_simultaneous_lines(5);

			filter2->set_horizontal_support(3);
			filter2->set_simultaneous_lines(3);
		} else {
			filter1->set_horizontal_support(3);
			filter1->set_simultaneous_lines(3);

			filter2->set_horizontal_support(5);
			filter2->set_simultaneous_lines(5);
		}

		zimg::FilterGraph graph{ w, h, type, 0, 0, false };

		graph.attach_filter(filter1);
		filter1_uptr.release();
		graph.attach_filter(filter2);
		filter2_uptr.release();
		graph.complete();

		AuditImage<uint16_t> src_image{ w, h, 0, 0, false };
		AuditImage<uint16_t> dst_image{ w, h, 0, 0, false };
		zimg::AlignedVector<char> tmp(graph.get_tmp_size());

		src_image.set_test_byte(test_byte1);
		src_image.default_fill();

		if (x) {
			EXPECT_EQ(8U, graph.get_input_buffering());
			EXPECT_EQ(4U, graph.get_output_buffering());
		} else {
			EXPECT_EQ(4U, graph.get_input_buffering());
			EXPECT_EQ(8U, graph.get_output_buffering());
		}

		graph.process(src_image.as_image_buffer(), dst_image.as_image_buffer(), tmp.data(), nullptr, nullptr);
		dst_image.set_test_byte(test_byte3);

		SCOPED_TRACE("validating src");
		src_image.validate();
		SCOPED_TRACE("validating dst");
		dst_image.validate();

		if (x) {
			EXPECT_EQ(2 * 116U, filter1->get_total_calls());
			EXPECT_EQ(2 * 192U, filter2->get_total_calls());
		} else {
			EXPECT_EQ(2 * 192U, filter1->get_total_calls());
			EXPECT_EQ(2 * 116U, filter2->get_total_calls());
		}
	}
}

TEST(FilterGraphTest, test_callback)
{
	static const unsigned w = 1024;
	static const unsigned h = 576;
	const zimg::PixelType type = zimg::PixelType::BYTE;

	const uint8_t test_byte1 = 0xCD;
	const uint8_t test_byte2 = 0xFF;
	const uint8_t test_byte3 = 0xDC;

	struct callback_data {
		zimg::ZimgImageBuffer buffer;
		unsigned subsample_w;
		unsigned subsample_h;
		unsigned call_count;
		uint8_t byte_val;
	};

	auto cb = [](void *ptr, unsigned i, unsigned left, unsigned right) -> int
	{
		callback_data *xptr = reinterpret_cast<callback_data *>(ptr);

		EXPECT_LT(i, h);
		EXPECT_EQ(0U, i % (1 << xptr->subsample_h));
		EXPECT_LT(left, right);
		EXPECT_LE(right, w);

		for (unsigned ii = i; ii < i + (1 << xptr->subsample_h); ++ii) {
			zimg::LineBuffer<uint8_t> buf{ xptr->buffer, 0 };

			std::fill(buf[ii] + left, buf[ii] + right, xptr->byte_val);
		}
		for (unsigned p = 1; p < 3; ++p) {
			zimg::LineBuffer<uint8_t> chroma_buf{ xptr->buffer, p };
			unsigned i_chroma = i >> xptr->subsample_h;
			unsigned left_chroma = (left % 2 ? left - 1 : left) >> xptr->subsample_w;
			unsigned right_chroma = (right % 2 ? right + 1 : right) >> xptr->subsample_w;

			std::fill(chroma_buf[i_chroma] + left_chroma, chroma_buf[i_chroma] + right_chroma, xptr->byte_val);
		}

		++xptr->call_count;
		return HasFatalFailure();
	};

	for (unsigned sw = 0; sw < 3; ++sw) {
		for (unsigned sh = 0; sh < 3; ++sh) {
			for (unsigned x = 0; x < 2; ++x) {
				SCOPED_TRACE(sw);
				SCOPED_TRACE(sh);
				SCOPED_TRACE(!!x);

				zimg::ZimgFilterFlags flags{};
				flags.entire_row = !!x;
				flags.color = false;

				std::unique_ptr<SplatFilter<uint8_t>> filter1_uptr{ new SplatFilter<uint8_t>{ w, h, type, flags } };
				std::unique_ptr<SplatFilter<uint8_t>> filter2_uptr{ new SplatFilter<uint8_t>{ w >> sw, h >> sh, type, flags } };
				SplatFilter<uint8_t> *filter1 = filter1_uptr.get();
				SplatFilter<uint8_t> *filter2 = filter2_uptr.get();

				filter1->set_input_test_byte(test_byte1);
				filter1->set_output_test_byte(test_byte2);

				filter2->set_input_test_byte(test_byte1);
				filter2->set_output_test_byte(test_byte2);

				zimg::FilterGraph graph{ w, h, type, sw, sh, true };
				graph.attach_filter(filter1);
				filter1_uptr.release();
				graph.attach_filter_uv(filter2);
				filter2_uptr.release();
				graph.complete();

				AuditImage<uint8_t> src_image{ w, h, sw, sh, true };
				AuditImage<uint8_t> tmp_image{ w, h, sw, sh, true };
				AuditImage<uint8_t> dst_image{ w, h, sw, sh, true };
				zimg::AlignedVector<char> tmp(graph.get_tmp_size());

				callback_data cb1_data = { src_image.as_image_buffer(), sw, sh, 0, test_byte1 };
				callback_data cb2_data = { dst_image.as_image_buffer(), sw, sh, 0, test_byte3 };

				src_image.set_test_byte(test_byte1);
				tmp_image.set_test_byte(test_byte2);
				dst_image.set_test_byte(test_byte3);

				graph.process(src_image.as_image_buffer(), tmp_image.as_image_buffer(), tmp.data(), { cb, &cb1_data }, { cb, &cb2_data });

				SCOPED_TRACE("validating src");
				src_image.validate();
				SCOPED_TRACE("validating tmp");
				tmp_image.validate();
				SCOPED_TRACE("validating dst");
				dst_image.validate();

				EXPECT_EQ((h >> sh) * (x ? 1 : 2), cb1_data.call_count);
				EXPECT_EQ((h >> sh) * (x ? 1 : 2), cb2_data.call_count);
			}
		}
	}
}

TEST(FilterGraphTest, test_callback_failed)
{
	const unsigned w = 640;
	const unsigned h = 480;
	zimg::PixelType type = zimg::PixelType::BYTE;

	auto cb = [](void *, unsigned i, unsigned left, unsigned right) -> int
	{
		return 1;
	};

	zimg::FilterGraph graph{ w, h, type, 0, 0, false };
	graph.complete();

	AuditImage<uint8_t> src_image{ w, h, 0, 0, false };
	AuditImage<uint8_t> dst_image{ w, h, 0, 0, false };
	zimg::AlignedVector<char> tmp(graph.get_tmp_size());

	src_image.set_test_byte(255);
	dst_image.set_test_byte(0);

	src_image.default_fill();
	dst_image.default_fill();

	ASSERT_THROW(graph.process(src_image.as_image_buffer(), dst_image.as_image_buffer(), tmp.data(), { cb, nullptr }, nullptr), zimg::error::UserCallbackFailed);

	SCOPED_TRACE("validating dst");
	dst_image.validate();
}
