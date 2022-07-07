#pragma once

#ifndef JSON_H_
#define JSON_H_

#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace json {

class JsonError : public std::runtime_error {
	mutable std::vector<std::pair<int, int>> m_stack_trace;
public:
	using std::runtime_error::runtime_error;

	JsonError(const char *msg, int line, int col) noexcept;

	JsonError(const JsonError &other) noexcept;

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
	struct static_initializer_tag {};

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
	typedef std::aligned_union_t<0,
		std::nullptr_t,
		double,
		std::string,
		Array,
		Object
	> union_type;

	tag_type m_tag;
	union_type m_union;

	static void move_helper(tag_type &src_tag, union_type &src_union,
	                        tag_type &dst_tag, union_type &dst_union) noexcept;

	template <class T>
	void construct(T x) { new (&m_union) T{ std::move(x) }; }

	template <class T>
	void destroy() noexcept { reinterpret_cast<T &>(m_union).~T(); }

	template <class T>
	T &as() noexcept { return reinterpret_cast<T &>(m_union); }

	template <class T>
	const T &as() const noexcept { return reinterpret_cast<const T &>(m_union); }

	void check_tag(tag_type tag) const
	{
		if (get_type() != tag)
			throw std::invalid_argument{ "access as wrong type" };
	}

public:
	constexpr explicit Value(static_initializer_tag) noexcept : m_tag{ UNDEFINED }, m_union{} {}

	Value() noexcept : m_tag{ UNDEFINED } {}

	explicit Value(std::nullptr_t) noexcept : m_tag{ NULL_ } {}
	explicit Value(double val) noexcept : m_tag{ NUMBER } { construct(val); }
	explicit Value(std::string val) : m_tag{ STRING } { construct(std::move(val)); }
	explicit Value(Array val) : m_tag{ ARRAY } { construct(std::move(val)); }
	explicit Value(Object val) : m_tag{ OBJECT } { construct(std::move(val)); }
	explicit Value(bool val) noexcept : m_tag{ BOOL_ } { construct(val); }

	template <class T, class = std::enable_if_t<std::is_integral<T>::value>>
	explicit Value(T val) noexcept : Value{ static_cast<double>(val) } {}

	Value(const Value &other);
	Value(Value &&other) noexcept;

	~Value();

	Value &operator=(const Value &other);
	Value &operator=(Value &&other) noexcept;

	explicit operator bool() const noexcept { return !is_undefined() && !is_null(); }

	bool is_undefined() const noexcept { return get_type() == UNDEFINED; }
	bool is_null() const noexcept { return get_type() == NULL_; }

	tag_type get_type() const noexcept { return m_tag; }

#define JSON_VALUE_GET_SET(T, name, tag) \
  const T &name() const { check_tag(tag); return as<T>(); } \
  T &name() { check_tag(tag); return as<T>(); }

	JSON_VALUE_GET_SET(double, number, NUMBER)
	JSON_VALUE_GET_SET(std::string, string, STRING)
	JSON_VALUE_GET_SET(Array, array, ARRAY)
	JSON_VALUE_GET_SET(Object, object, OBJECT)
	JSON_VALUE_GET_SET(bool, boolean, BOOL_)

#undef JSON_VALUE_GET_SET

	long long integer() const { return std::llrint(number()); }

	friend void swap(Value &a, Value &b) noexcept;
};

inline const Value &Object::operator[](const std::string &key) const noexcept
{
	static const Value null_value{ Value::static_initializer_tag{} };

	auto it = find(key);
	return it == end() ? null_value : it->second;
}

inline void swap(Object &a, Object &b) noexcept { a.swap(b); }


Value parse_document(const std::string &str);

} // namespace json

#endif // JSON_H_
