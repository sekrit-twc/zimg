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

	JsonError(const char *msg, int line, int col);

	JsonError(const JsonError &other);

	JsonError &operator=(const JsonError &other);

	void add_trace(int line, int col) const;

	std::string error_details() const;
};

class Value;

class Object : private std::map<std::string, Value> {
	typedef std::map<std::string, Value> base_type;
public:
	using base_type::iterator;
	using base_type::const_iterator;

	using base_type::operator[];
	using base_type::at;
	using base_type::find;

	using base_type::begin;
	using base_type::end;

	using base_type::cbegin;
	using base_type::cend;

	using base_type::empty;
	using base_type::size;

	using base_type::clear;
	using base_type::erase;
	using base_type::swap;

	const Value &operator[](const std::string &key) const;

	friend void swap(Object &a, Object &b);
};

typedef std::vector<Value> Array;

class Value {
public:
	enum tag_type {
		NULL_,
		NUMBER,
		STRING,
		OBJECT,
		ARRAY,
		BOOL_,
	};
private:
#if defined(__GNUC__) && !(__GNUC__ > 5 || (__GNUC__ == 5 && __GNUC_MINOR__ >= 1))
	typedef std::aligned_storage<64>::type union_type;
#else
	typedef std::aligned_union<0,
		std::nullptr_t,
		double,
		std::string,
		Array,
		Object>::type union_type;
#endif

	tag_type m_tag;
	union_type m_union;

	static void move_helper(tag_type &src_tag, union_type &src_union,
	                        tag_type &dst_tag, union_type &dst_union);

	template <class T>
	void construct(T x) { new (&m_union) T{ std::move(x) }; }

	template <class T>
	void destroy() { reinterpret_cast<T &>(m_union).~T(); }

	template <class T>
	T &as() { return reinterpret_cast<T &>(m_union); }

	template <class T>
	const T &as() const { return reinterpret_cast<const T &>(m_union); }

	void check_tag(tag_type tag) const
	{
		if (get_type() != tag)
			throw std::invalid_argument{ "access as wrong type" };
	}
public:
	Value(std::nullptr_t = nullptr) : m_tag{ NULL_ } {}

	explicit Value(double val) : m_tag{ NUMBER } { construct(val); }
	explicit Value(std::string val) : m_tag{ STRING } { construct(std::move(val)); }
	explicit Value(Array val) : m_tag{ ARRAY } { construct(std::move(val)); }
	explicit Value(Object val) : m_tag{ OBJECT } { construct(std::move(val)); }
	explicit Value(bool val) : m_tag{ BOOL_ } { construct(val); }

	Value(const Value &other);
	Value(Value &&other);

	~Value();

	Value &operator=(Value other);

	explicit operator bool() const { return !is_null(); }

	bool is_null() const { return get_type() == NULL_; }

	tag_type get_type() const { return m_tag; }

#define JSON_VALUE_GET_SET(T, name, tag) \
  const T &name() const { check_tag(tag); return as<T>(); } \
  T &name() { check_tag(tag); return as<T>(); } \
  static_assert(true, "")

	JSON_VALUE_GET_SET(double, number, NUMBER);
	JSON_VALUE_GET_SET(std::string, string, STRING);
	JSON_VALUE_GET_SET(Array, array, ARRAY);
	JSON_VALUE_GET_SET(Object, object, OBJECT);
	JSON_VALUE_GET_SET(bool, boolean, BOOL_);

#undef JSON_VALUE_GET_SET

	long long integer() const { return std::llrint(number()); }

	friend void swap(Value &a, Value &b);
};

inline const Value &Object::operator[](const std::string &key) const
{
	static Value null_value{};

	auto it = find(key);
	return it == end() ? null_value : it->second;
}

inline void swap(Object &a, Object &b)
{
	a.swap(b);
}


Value parse_document(const std::string &str);

} // namespace json

#endif // JSON_H_
