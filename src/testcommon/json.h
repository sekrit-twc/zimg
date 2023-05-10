#pragma once

#ifndef JSON_H_
#define JSON_H_

#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace json {

class JsonError : public std::runtime_error {
	mutable std::vector<std::pair<int, int>> m_stack_trace;
public:
	using std::runtime_error::runtime_error;

	JsonError(const char *msg, int line, int col);

	JsonError(const JsonError &other);

	JsonError &operator=(const JsonError &other) noexcept;

	void add_trace(int line, int col) const noexcept;

	std::string error_details() const noexcept;
};

class Value;

class Object : private std::map<std::string, Value> {
public:
	using map::iterator;
	using map::const_iterator;

	using map::operator[];
	using map::at;
	using map::find;

	using map::begin;
	using map::end;

	using map::cbegin;
	using map::cend;

	using map::empty;
	using map::size;

	using map::clear;
	using map::erase;
	using map::swap;

	const Value &operator[](const std::string &key) const noexcept;

	friend void swap(Object &a, Object &b) noexcept;
};

typedef std::vector<Value> Array;

class Value {
public:
	enum tag_type {
		UNDEFINED,
		NULL_,
		NUMBER,
		STRING,
		OBJECT,
		ARRAY,
		BOOL_,
	};
private:
	std::variant<
		std::monostate,
		std::nullptr_t,
		double,
		std::string,
		Object,
		Array,
		bool
	> m_variant;

	void check_tag(tag_type tag) const
	{
		if (m_variant.index() != tag)
			throw std::invalid_argument{ "access as wrong type" };
	}
public:
	constexpr Value() noexcept {}

	explicit Value(std::nullptr_t) noexcept : m_variant{ nullptr } {}
	explicit Value(double val) noexcept : m_variant{ val } {}
	explicit Value(std::string val) : m_variant{ std::move(val) } {}
	explicit Value(Array val) : m_variant{ std::move(val) } {}
	explicit Value(Object val) : m_variant{ std::move(val) } {}
	explicit Value(bool val) noexcept : m_variant{ val } {}

	template <class T, class = std::enable_if_t<std::is_integral_v<T>>>
	explicit Value(T val) noexcept : Value{ static_cast<double>(val) } {}

	explicit operator bool() const noexcept { return !is_undefined() && !is_null(); }

	bool is_undefined() const noexcept { return get_type() == UNDEFINED; }
	bool is_null() const noexcept { return get_type() == NULL_; }

	tag_type get_type() const noexcept { return static_cast<tag_type>(m_variant.index()); }

#define JSON_VALUE_GET_SET(T, name, tag) \
  const T &name() const { check_tag(tag); return std::get<T>(m_variant); } \
  T &name() { check_tag(tag); return std::get<T>(m_variant); }

	JSON_VALUE_GET_SET(double, number, NUMBER)
	JSON_VALUE_GET_SET(std::string, string, STRING)
	JSON_VALUE_GET_SET(Array, array, ARRAY)
	JSON_VALUE_GET_SET(Object, object, OBJECT)
	JSON_VALUE_GET_SET(bool, boolean, BOOL_)

#undef JSON_VALUE_GET_SET

	long long integer() const { return std::llrint(number()); }

	friend void swap(Value &a, Value &b) noexcept { std::swap(a.m_variant, b.m_variant); }
};

inline const Value &Object::operator[](const std::string &key) const noexcept
{
	static const Value null_value;

	auto it = find(key);
	return it == end() ? null_value : it->second;
}

inline void swap(Object &a, Object &b) noexcept { a.swap(b); }


Value parse_document(const std::string &str);

} // namespace json

#endif // JSON_H_
