#include <algorithm>
#include <cassert>
#include <climits>
#include <iterator>
#include <locale>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include "json.h"

namespace json {

namespace {
namespace parser {

class Token {
public:
	enum tag_type {
		EMPTY,
		CURLY_BRACE_LEFT,
		CURLY_BRACE_RIGHT,
		SQUARE_BRACE_LEFT,
		SQUARE_BRACE_RIGHT,
		COMMA,
		COLON,
		MINUS,
		STRING_LITERAL,
		NUMBER_LITERAL,
		INFINITY_,
		NAN_,
		TRUE_,
		FALSE_,
		NULL_,
	};
private:
	tag_type m_tag;
	int m_line;
	int m_col;
	std::string_view m_str;
public:
	Token() noexcept : m_tag{ EMPTY }, m_line{}, m_col{} {}

	Token(tag_type tag, int line, int col, std::string_view str = {}) :
		m_tag{ tag },
		m_line{ line },
		m_col{ col },
		m_str{ str }
	{}

	tag_type tag() const noexcept { return m_tag; }
	int line() const noexcept { return m_line; }
	int col() const noexcept { return m_col; }
	std::string_view str() const noexcept { return m_str; }
};

template <class ForwardIt>
class TracingIterator {
public:
	typedef typename ForwardIt::difference_type difference_type;
	typedef typename ForwardIt::value_type value_type;
	typedef typename ForwardIt::reference reference;
	typedef typename ForwardIt::pointer pointer;
	typedef std::forward_iterator_tag iterator_category;
private:
	static constexpr int SPACES_PER_TAB = 4;

	ForwardIt m_it;
	int m_line;
	int m_col;

	void next()
	{
		if (*m_it == '\n') {
			++m_line;
			m_col = 0;
		} else if (*m_it == '\t') {
			m_col = m_col + SPACES_PER_TAB - m_col % SPACES_PER_TAB;
		} else {
			++m_col;
		}

		++m_it;
	}
public:
	explicit TracingIterator(ForwardIt it) : m_it(it), m_line{}, m_col{}
	{}

	ForwardIt base_iterator() const { return m_it; }

	int line() const noexcept { return m_line; }
	int col() const noexcept { return m_col; }

	reference operator*() const { return *m_it; }
	pointer operator->() const { return &*m_it; }

	TracingIterator &operator++()
	{
		next();
		return *this;
	}

	TracingIterator operator++(int)
	{
		TracingIterator ret = *this;
		next();
		return ret;
	}

	bool operator==(const TracingIterator &other) const { return m_it == other.m_it; }
	bool operator!=(const TracingIterator &other) const { return m_it != other.m_it; }
};

template <class ForwardIt, class Pred>
ForwardIt skip_while(ForwardIt first, ForwardIt last, Pred pred)
{
	while (first != last && pred(*first)) {
		++first;
	}
	return first;
}

template <class InputIt, class Difference>
InputIt skip_n(InputIt it, Difference n)
{
	std::advance(it, n);
	return it;
}

template <class ForwardIt1, class ForwardIt2>
bool prefix_match(ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2, ForwardIt2 last2)
{
	if (std::distance(first2, last2) < std::distance(first1, last1))
		return false;
	return std::equal(first1, last1, first2);
}

constexpr bool is_ascii(char c) noexcept
{
	return static_cast<signed char>(c) >= 0;
}

constexpr bool is_json_cntrl(char c) noexcept
{
	return static_cast<unsigned char>(c) <= 0x1F;
}

constexpr bool is_json_digit(char c) noexcept
{
	return c >= '0' && c <= '9';
}

constexpr bool is_json_space(char c) noexcept
{
	return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

constexpr bool is_number_or_keyword(Token::tag_type type) noexcept
{
	return type >= Token::NUMBER_LITERAL;
}

template <class ForwardIt>
std::pair<Token, TracingIterator<ForwardIt>> match_string_literal(TracingIterator<ForwardIt> first, TracingIterator<ForwardIt> last)
{
	TracingIterator<ForwardIt> pos = first;
	int line = first.line();
	int col = first.col();
	char prev_c = '\0';

	assert(*pos == '"');
	++pos;

	while (pos != last) {
		if (!is_ascii(*pos))
			throw JsonError{ "unicode not allowed", pos.line(), pos.col() };
		if (*pos == '"' && prev_c != '\\')
			break;

		prev_c = *pos;
		++pos;
	}
	if (pos == last)
		throw JsonError{ "expected matching '\"'", line, col };

	assert(*pos == '"');
	++pos;

	return{ { Token::STRING_LITERAL, line, col, { &*first, static_cast<size_t>(&*pos - &*first) }}, pos};
}

template <class ForwardIt>
std::pair<Token, TracingIterator<ForwardIt>> match_number_literal(TracingIterator<ForwardIt> first, TracingIterator<ForwardIt> last)
{
	TracingIterator<ForwardIt> pos = first;
	int line = first.line();
	int col = first.col();

	pos = skip_while(pos, last, is_json_digit);
	if (pos != last && *pos == '.') {
		++pos;
		pos = skip_while(pos, last, is_json_digit);
	}

	if (pos != last && (*pos == 'e' || *pos == 'E')) {
		++pos;
		if (pos != last && (*pos == '-' || *pos == '+'))
			++pos;

		if (pos == last)
			throw JsonError{ "expected digits following exponent indicator", line, col };

		pos = skip_while(first, last, is_json_digit);
	}

	return{ { Token::NUMBER_LITERAL, line, col, { &*first, static_cast<size_t>(&*pos - &*first) }}, pos };
}

template <class ForwardIt>
std::pair<Token, TracingIterator<ForwardIt>> match_keyword(TracingIterator<ForwardIt> first, TracingIterator<ForwardIt> last)
{
	static constexpr std::string_view keyword_infinity{ "Infinity" };
	static constexpr std::string_view keyword_nan{ "NaN" };
	static constexpr std::string_view keyword_true{ "true" };
	static constexpr std::string_view keyword_false{ "false" };
	static constexpr std::string_view keyword_null{ "null" };

	int line = first.line();
	int col = first.col();

	ForwardIt base_first = first.base_iterator();
	ForwardIt base_last = last.base_iterator();

#define MATCH(s, tag) do { \
  if (prefix_match(s.begin(), s.end(), base_first, base_last)) \
    return{ { tag, line, col }, skip_n(first, s.size()) }; \
  } while (0)

	MATCH(keyword_infinity, Token::INFINITY_);
	MATCH(keyword_nan, Token::NAN_);
	MATCH(keyword_true, Token::TRUE_);
	MATCH(keyword_false, Token::FALSE_);
	MATCH(keyword_null, Token::NULL_);

#undef MATCH
	throw JsonError{ "not a token", line, col };
}

template <class ForwardIt>
std::pair<Token, TracingIterator<ForwardIt>> match_next_token(TracingIterator<ForwardIt> first, TracingIterator<ForwardIt> last)
{
	char c = *first;
	int line = first.line();
	int col = first.col();

	if (!is_ascii(c))
		throw JsonError{ "unicode not allowed", line, col };
	if ((is_json_cntrl(c) && !is_json_space(c)) || c == 0x7F)
		throw JsonError{ "control character not allowed", line, col };

	if (is_json_space(c))
		return{ {}, skip_while(first, last, is_json_space) };
	else if (c == '{')
		return{ { Token::CURLY_BRACE_LEFT, line, col }, std::next(first) };
	else if (c == '}')
		return{ { Token::CURLY_BRACE_RIGHT, line, col }, std::next(first) };
	else if (c == '[')
		return{ { Token::SQUARE_BRACE_LEFT, line, col }, std::next(first) };
	else if (c == ']')
		return{ { Token::SQUARE_BRACE_RIGHT, line, col }, std::next(first) };
	else if (c == ',')
		return{ { Token::COMMA, line, col }, std::next(first) };
	else if (c == ':')
		return{ { Token::COLON, line, col }, std::next(first) };
	else if (c == '-')
		return{ { Token::MINUS, line, col }, std::next(first) };
	else if (c == '"')
		return match_string_literal(first, last);
	else if (is_json_digit(c))
		return match_number_literal(first, last);
	else
		return match_keyword(first, last);
}

template <class ForwardIt>
std::vector<Token> tokenize(ForwardIt first_, ForwardIt last_)
{
	std::vector<Token> tokens;
	Token::tag_type prev_tag = Token::EMPTY;

	TracingIterator<ForwardIt> first{ first_ };
	TracingIterator<ForwardIt> last{ last_ };

	while (first != last) {
		Token tok;

		std::tie(tok, first) = match_next_token(first, last);
		if (is_number_or_keyword(tok.tag()) && is_number_or_keyword(prev_tag))
			throw JsonError{ "not a token", tokens.back().line(), tokens.back().col() };

		if (tok.tag() != Token::EMPTY)
			tokens.emplace_back(std::move(tok));
	}
	return tokens;
}

template <class ForwardIt>
void expect_token(Token::tag_type type, ForwardIt first, ForwardIt last, const char *msg)
{
	if (first == last)
		throw JsonError{ "unexpected EOF" };
	if (first->tag() != type)
		throw JsonError{ msg, first->line(), first->col() };
}

char decode_escape_code(char c)
{
	switch (c) {
	case '"':
		return '"';
	case '\\':
		return '\\';
	case '/':
		return '/';
	case 'b':
		return '\b';
	case 'f':
		return '\f';
	case 'n':
		return '\n';
	case 'r':
		return '\r';
	case 't':
		return '\t';
	case 'u':
		throw JsonError{ "unicode not allowed" };
	default:
		throw JsonError{ "bad escape code" };
	}
}

std::string decode_string_literal(const Token &tok)
{
	std::string s;
	bool escaped = false;

	assert(tok.str().size() >= 2);
	assert(tok.str().front() == '"');
	assert(tok.str().back() == '"');

	s.reserve(tok.str().size() - 2);

	for (auto it = tok.str().begin() + 1; it != tok.str().end() - 1; ++it) {
		int col = tok.col() + static_cast<int>(it - tok.str().begin());
		char c = *it;

		if (!is_ascii(c))
			throw JsonError{ "unicode not allowed", tok.line(), col };
		if (is_json_cntrl(c))
			throw JsonError{ "control character not allowed", tok.line(), col };

		if (escaped) {
			try {
				s.push_back(decode_escape_code(c));
			} catch (const JsonError &e) {
				e.add_trace(tok.line(), col);
				throw;
			}
			escaped = false;
		} else if (c == '\\') {
			escaped = true;
		} else {
			s.push_back(c);
		}
	}
	return s;
}

double decode_number_literal(const Token &tok)
{
	std::istringstream ss{ std::string{ tok.str() } };
	double x = 0.0;

	ss.imbue(std::locale::classic());

	if (!(ss >> x))
		throw JsonError{ "bad number literal", tok.line(), tok.col() };
	if (ss.peek() != std::istringstream::traits_type::eof())
		throw JsonError{ "bad number literal", tok.line(), tok.col() };

	return x;
}

template <class ForwardIt>
std::pair<Value, ForwardIt> parse_value(ForwardIt first, ForwardIt last);

template <class ForwardIt>
std::pair<Value, ForwardIt> parse_number(ForwardIt first, ForwardIt last)
{
	int line = first->line();
	int col = first->col();

	double x = 0.0;
	bool negative = false;

	if (first->tag() == Token::MINUS) {
		negative = true;
		if (++first == last)
			throw JsonError{ "unexpected EOF after '-'", line, col };
	}

	switch (first->tag()) {
	case Token::NUMBER_LITERAL:
		x = decode_number_literal(*first);
		break;
	case Token::INFINITY_:
		x = INFINITY;
		break;
	case Token::NAN_:
		x = NAN;
		break;
	default:
		throw JsonError{ "bad number literal", first->line(), first->col() };
	}
	++first;

	return{ Value{ negative ? -x : x }, first };
}

template <class ForwardIt>
std::pair<Value, ForwardIt> parse_object(ForwardIt first, ForwardIt last)
{
	int line = first->line();
	int col = first->col();

	assert(first->tag() == Token::CURLY_BRACE_LEFT);
	++first;

	Object object;

	try {
		while (true) {
			if (first == last)
				throw JsonError{ "unexpected EOF" };
			if (first->tag() == Token::CURLY_BRACE_RIGHT)
				break;

			expect_token(Token::STRING_LITERAL, first, last, "expected member name");
			std::string name = decode_string_literal(*first++);

			expect_token(Token::COLON, first, last, "expected ':' after member name");
			++first;

			Value value;
			std::tie(value, first) = parse_value(first, last);

			object[std::move(name)] = std::move(value);

			if (first != last && first->tag() != Token::CURLY_BRACE_RIGHT) {
				expect_token(Token::COMMA, first, last, "expected ',' after object member");
				++first;
			}
		}

		assert(first->tag() == Token::CURLY_BRACE_RIGHT);
		++first;
	} catch (const JsonError &e) {
		e.add_trace(line, col);
		throw;
	}

	return{ Value{ std::move(object) }, first };
}

template <class ForwardIt>
std::pair<Value, ForwardIt> parse_array(ForwardIt first, ForwardIt last)
{
	int line = first->line();
	int col = first->col();

	assert(first->tag() == Token::SQUARE_BRACE_LEFT);
	++first;

	Array array;

	try {
		while (true) {
			if (first == last)
				throw JsonError{ "unexpected EOF" };
			if (first->tag() == Token::SQUARE_BRACE_RIGHT)
				break;

			Value value;
			std::tie(value, first) = parse_value(first, last);
			array.emplace_back(std::move(value));

			if (first != last && first->tag() != Token::SQUARE_BRACE_RIGHT) {
				expect_token(Token::COMMA, first, last, "expected ',' after array element");
				++first;
			}
		}

		assert(first->tag() == Token::SQUARE_BRACE_RIGHT);
		++first;
	} catch (const JsonError &e) {
		e.add_trace(line, col);
		throw;
	}

	return{ Value{ std::move(array) }, first };
}

template <class ForwardIt>
std::pair<Value, ForwardIt> parse_value(ForwardIt first, ForwardIt last)
{
	if (first == last)
		throw JsonError{ "empty document" };

	switch (first->tag()) {
	case Token::CURLY_BRACE_LEFT:
		return parse_object(first, last);
	case Token::SQUARE_BRACE_LEFT:
		return parse_array(first, last);
	case Token::STRING_LITERAL:
		return{ Value{ decode_string_literal(*first) }, std::next(first) };
	case Token::MINUS:
	case Token::NUMBER_LITERAL:
	case Token::NAN_:
	case Token::INFINITY_:
		return parse_number(first, last);
	case Token::TRUE_:
		return{ Value{ true }, std::next(first) };
	case Token::FALSE_:
		return{ Value{ false }, std::next(first) };
	case Token::NULL_:
		return{ Value{ nullptr }, std::next(first) };
	default:
		throw JsonError{ "unexpected token while parsing value", first->line(), first->col() };
	}
}

} // namespace parser


template <class T>
void uninitialized_move(void *src, void *dst) noexcept
{
	T *src_p = static_cast<T *>(src);
	T *dst_p = static_cast<T *>(dst);

	new (dst_p) T{ std::move(*src_p) };
	src_p->~T();
}

} // namespace


JsonError::JsonError(const char *msg, int line, int col) : std::runtime_error{ msg }
{
	add_trace(line, col);
}

JsonError::JsonError(const JsonError &other) : std::runtime_error{ other }
{
	try {
		m_stack_trace = other.m_stack_trace;
	} catch (const std::bad_alloc &) {
		// ...
	}
}

JsonError &JsonError::operator=(const JsonError &other) noexcept
{
	std::runtime_error::operator=(other);

	try {
		m_stack_trace = other.m_stack_trace;
	} catch (const std::bad_alloc &) {
		// ...
	}
	return *this;
}

void JsonError::add_trace(int line, int col) const noexcept
{
	try {
		m_stack_trace.emplace_back(line, col);
	} catch (const std::bad_alloc &) {
		// ...
	}
}

std::string JsonError::error_details() const noexcept
{
	std::string s;

	try {
		s += what();
		s += "\nbacktrace: \n";

		for (const auto &pos : m_stack_trace) {
			s += "\tat (l ";
			s += std::to_string(pos.first + 1);
			s += ", c ";
			s += std::to_string(pos.second + 1);
			s += ")\n";
		}
	} catch (const std::bad_alloc &) {
		// ...
	}

	return s;
}

Value parse_document(const std::string &str)
{
	std::vector<parser::Token> tokens = parser::tokenize(str.begin(), str.end());

	auto result = parser::parse_value(tokens.begin(), tokens.end());
	if (result.second != tokens.end())
		throw JsonError{ "unparsed document content", result.second->line(), result.second->col() };

	return std::move(result.first);
}

} // namespace json
