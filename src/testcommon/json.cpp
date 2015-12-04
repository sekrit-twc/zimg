#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>
#include "json.h"

using std::nullptr_t;

namespace {;

struct Token;

typedef std::string::const_iterator string_iterator;
typedef std::vector<Token>::const_iterator token_iterator;

struct Token {
	enum tag_type {
		NULL_,
		CURLY_BRACE_LEFT,
		CURLY_BRACE_RIGHT,
		SQUARE_BRACE_LEFT,
		SQUARE_BRACE_RIGHT,
		COMMA,
		COLON,
		STRING_LITERAL,
		NUMBER_LITERAL,
		TRUE_,
		FALSE_
	} tag;

	std::string str;

	Token(nullptr_t = nullptr) : tag{ NULL_ }
	{
	}

	Token(tag_type t, std::string s = {}) : tag{ t }, str{ s }
	{
	}

	explicit operator bool() const
	{
		return tag != NULL_;
	}
};

template <class InputIt1, class InputIt2>
bool equal_prefix(InputIt1 first1, InputIt1 last1, InputIt2 first2, InputIt2 last2)
{
	if (std::distance(first2, last2) < std::distance(first1, last1))
		return false;
	else
		return std::equal(first1, last1, first2);
}

template <class Pred>
string_iterator skip_if(string_iterator pos, string_iterator last, Pred pred)
{
	while (pos != last && pred(*pos)) {
		++pos;
	}
	return pos;
}

std::pair<Token, string_iterator> match_token_string_literal(string_iterator pos, string_iterator last)
{
	string_iterator first = pos;
	char prev_c;

	if ((prev_c = *pos++) != '"')
		throw std::invalid_argument{ "bad character" };

	while (pos != last && (*pos != '"' && prev_c != '\\')) {
		prev_c = *pos++;
	}

	if (pos == last)
		throw std::runtime_error{ "expected '\"' before end of string" };

	++pos;
	return{ { Token::STRING_LITERAL, { first, pos } }, pos };
}

std::pair<Token, string_iterator> match_token_number_literal(string_iterator pos, string_iterator last)
{
	string_iterator first = pos;

	if (*pos == '-')
		++pos;

	if (pos == last)
		throw std::runtime_error{ "expected a digit" };

	pos = skip_if(pos, last, isdigit);

	if (pos != last && (*pos == '.')) {
		++pos;
		pos = skip_if(pos, last, isdigit);
	}

	if (pos != last && (*pos == 'e' || *pos == 'E')) {
		++pos;

		if (pos != last && (*pos == '-' || *pos == '+'))
			++pos;
		if (pos == last)
			throw std::runtime_error{ "expected an exponent" };

		pos = skip_if(pos, last, isdigit);
	}

	return{ { Token::NUMBER_LITERAL, { first, pos } }, pos };
}

std::pair<Token, string_iterator> match_token_keyword(string_iterator pos, string_iterator last)
{
	std::string true_str{ "true" };
	std::string false_str{ "false" };
	std::string null_str{ "null" };

	if (equal_prefix(true_str.begin(), true_str.end(), pos, last))
		return{ Token::TRUE_, pos + true_str.size() };
	else if (equal_prefix(false_str.begin(), false_str.end(), pos, last))
		return{ Token::FALSE_, pos + false_str.size() };
	else if (equal_prefix(null_str.begin(), null_str.end(), pos, last))
		return{ Token::NULL_, pos + null_str.size() };
	else
		throw std::runtime_error{ "expected a keyword" };
}

std::pair<Token, string_iterator> match_token(string_iterator pos, string_iterator last)
{
	if (isspace(*pos))
		return{ nullptr, skip_if(pos, last, isspace) };
	else if (*pos == '{')
		return{ Token::CURLY_BRACE_LEFT, pos + 1 };
	else if (*pos == '}')
		return{ Token::CURLY_BRACE_RIGHT, pos + 1 };
	else if (*pos == '[')
		return{ Token::SQUARE_BRACE_LEFT, pos + 1 };
	else if (*pos == ']')
		return{ Token::SQUARE_BRACE_RIGHT, pos + 1 };
	else if (*pos == ',')
		return{ Token::COMMA, pos + 1 };
	else if (*pos == ':')
		return{ Token::COLON, pos + 1 };
	else if (*pos == '"')
		return match_token_string_literal(pos, last);
	else if (*pos == '-' || isdigit(*pos))
		return match_token_number_literal(pos, last);
	else
		return match_token_keyword(pos, last);
}

std::vector<Token> tokenize(const std::string &str)
{
	std::vector<Token> tokens;

	string_iterator pos = str.begin();
	string_iterator last = str.end();

	while (pos != last) {
		auto tok = match_token(pos, last);

		if (tok.first)
			tokens.emplace_back(std::move(tok.first));

		pos = tok.second;
	}

	return tokens;
}


std::string decode_string_literal(const std::string &str)
{
	std::string ret;
	bool is_escape = false;

	if (str.size() < 2)
		throw std::invalid_argument{ "bad string literal" };

	for (string_iterator it = str.begin() + 1; it != str.end() - 1; ++it) {
		if (is_escape) {
			char x;

			switch (*it) {
			case 'b':
				x = '\b';
				break;
			case 'f':
				x = '\f';
				break;
			case 'n':
				x = '\n';
				break;
			case 'r':
				x = '\r';
				break;
			case 't':
				x = '\t';
				break;
			case 'u':
				throw std::runtime_error{ "unicode not supported" };
			default:
				x = *it;
				break;
			}

			ret.push_back(x);
			is_escape = false;
		} else if (*it == '\\') {
			is_escape = true;
		} else {
			ret.push_back(*it);
		}
	}
	return ret;
}

double decode_number_literal(const std::string &str)
{
	try {
		size_t pos;
		double d = std::stod(str, &pos);

		if (pos != str.size())
			throw std::runtime_error{ "bad number literal" };

		return d;
	} catch (std::logic_error &e) {
		throw std::runtime_error{ e.what() };
	}
}

token_iterator expect_token(Token::tag_type type, token_iterator pos, token_iterator last)
{
	if (pos == last)
		throw std::runtime_error{ "expected a token" };
	if (pos->tag != type)
		throw std::runtime_error{ "unexpected token type" };
	return ++pos;
}

std::pair<JsonValue, token_iterator> parse_value(token_iterator pos, token_iterator last);

std::pair<JsonValue, token_iterator> parse_object(token_iterator pos, token_iterator last);

std::pair<JsonValue, token_iterator> parse_array(token_iterator pos, token_iterator last)
{
	std::vector<JsonValue> array;

	pos = expect_token(Token::SQUARE_BRACE_LEFT, pos, last);

	while (pos != last && pos->tag != Token::SQUARE_BRACE_RIGHT) {
		auto val = parse_value(pos, last);

		array.emplace_back(std::move(val.first));
		pos = val.second;

		if (pos == last)
			throw std::runtime_error{ "expected ']' before end of string" };
		if (pos->tag == Token::SQUARE_BRACE_RIGHT)
			break;

		pos = expect_token(Token::COMMA, pos, last);
	};
	pos = expect_token(Token::SQUARE_BRACE_RIGHT, pos, last);

	return{ JsonValue{ std::move(array) }, pos };
}

std::pair<JsonValue, token_iterator> parse_object(token_iterator pos, token_iterator last)
{
	JsonObject obj;

	pos = expect_token(Token::CURLY_BRACE_LEFT, pos, last);

	while (pos != last && pos->tag != Token::CURLY_BRACE_RIGHT) {
		expect_token(Token::STRING_LITERAL, pos, last);

		std::string name = decode_string_literal((pos++)->str);
		pos = expect_token(Token::COLON, pos, last);

		std::tie(obj[name], pos) = parse_value(pos, last);

		if (pos == last)
			throw std::runtime_error{ "expected '}' before end of string" };
		if (pos->tag == Token::CURLY_BRACE_RIGHT)
			break;

		pos = expect_token(Token::COMMA, pos, last);
	};
	pos = expect_token(Token::CURLY_BRACE_RIGHT, pos, last);

	return{ JsonValue{ std::move(obj) }, pos };
}

std::pair<JsonValue, token_iterator> parse_value(token_iterator pos, token_iterator last)
{
	if (pos == last)
		throw std::runtime_error{ "expected a token" };

	switch (pos->tag) {
	case Token::NUMBER_LITERAL:
		return{ decode_number_literal(pos->str), pos + 1 };
	case Token::STRING_LITERAL:
		return{ decode_string_literal(pos->str), pos + 1 };
	case Token::SQUARE_BRACE_LEFT:
		return parse_array(pos, last);
	case Token::CURLY_BRACE_LEFT:
		return parse_object(pos, last);
	case Token::TRUE_:
		return{ true, pos + 1 };
	case Token::FALSE_:
		return{ false, pos + 1 };
	case Token::NULL_:
		return{ nullptr, pos + 1 };
	default:
		throw std::runtime_error{ "expected a value" };
	}
}

} // namespace


void JsonValue::union_move(union_type &src, tag_type &src_tag, union_type &dst, tag_type &dst_tag)
{
	switch (src_tag) {
	case NULL_:
		new (&dst) nullptr_t{};
		break;
	case NUMBER:
		new (&dst) number_type{ std::move(reinterpret_cast<number_type &>(src)) };
		break;
	case STRING:
		new (&dst) string_type{ std::move(reinterpret_cast<string_type &>(src)) };
		break;
	case ARRAY:
		new (&dst) array_type{ std::move(reinterpret_cast<array_type &>(src)) };
		break;
	case OBJECT:
		new (&dst) object_type{ std::move(reinterpret_cast<object_type &>(src)) };
		break;
	case BOOL_:
		new (&dst) boolean_type{ std::move(reinterpret_cast<boolean_type &>(src)) };
		break;
	}

	dst_tag = src_tag;
	src_tag = NULL_;
}

JsonValue::JsonValue(const JsonValue &other) :
	m_tag{ NULL_ }
{
	switch (other.get_type()) {
	case NULL_:
		construct(nullptr);
		break;
	case NUMBER:
		construct(other.number());
		break;
	case STRING:
		construct(other.string());
		break;
	case ARRAY:
		construct(other.array());
		break;
	case OBJECT:
		construct(other.object());
		break;
	case BOOL_:
		construct(other.boolean());
		break;
	}

	m_tag = other.m_tag;
}

JsonValue::JsonValue(JsonValue &&other) :
	m_tag{ NULL_ }
{
	swap(other);
}

JsonValue::~JsonValue()
{
	switch (get_type()) {
	case NULL_:
		destroy<nullptr_t>();
		break;
	case NUMBER:
		destroy<number_type>();
		break;
	case STRING:
		destroy<string_type>();
		break;
	case ARRAY:
		destroy<array_type>();
		break;
	case OBJECT:
		destroy<object_type>();
		break;
	case BOOL_:
		destroy<boolean_type>();
		break;
	}
}

JsonValue &JsonValue::operator=(JsonValue other)
{
	swap(other);
	return *this;
}

void JsonValue::swap(JsonValue &other)
{
	union_type tmp;
	tag_type tmp_type;

	union_move(other.m_data, other.m_tag, tmp, tmp_type);
	union_move(m_data, m_tag, other.m_data, other.m_tag);
	union_move(tmp, tmp_type, m_data, m_tag);
}

const JsonValue &JsonObject::operator[](const std::string &key) const
{
	static const JsonValue null_value{};

	auto it = find(key);
	return it == end() ? null_value : it->second;
}


namespace json {;

JsonValue parse_document(const std::string &str)
{
	std::vector<Token> tokens = tokenize(str);

	auto val = parse_value(tokens.begin(), tokens.end());

	if (val.second != tokens.end())
		throw std::runtime_error{ "unparsed document content" };

	return std::move(val.first);
}

} // namespace json
