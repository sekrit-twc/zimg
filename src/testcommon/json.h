#pragma once

#ifndef JSON_H_
#define JSON_H_

#include <cmath>
#include <cstddef>
#include <map>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

class JsonValue;
class JsonObject;

class JsonObject : private std::map<std::string, JsonValue> {
	typedef std::map<std::string, JsonValue> base_type;
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

	const JsonValue &operator[](const std::string &key) const;
};

class JsonValue {
public:
	typedef double number_type;
	typedef std::string string_type;
	typedef JsonObject object_type;
	typedef std::vector<JsonValue> array_type;
	typedef bool boolean_type;

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
		number_type,
		string_type,
		array_type,
		object_type>::type union_type;
#endif
	union_type m_data;
	tag_type m_tag;

	static void union_move(union_type &src, tag_type &src_tag, union_type &dst, tag_type &dst_tag);

	void check_type(tag_type type) const
	{
		if (m_tag != type)
			throw std::invalid_argument{ "access as wrong type" };
	}

	template <class T>
	void construct(T val)
	{
		new (&m_data) T{ std::move(val) };
	}

	template <class T>
	void destroy()
	{
		reinterpret_cast<T *>(&m_data)->~T();
	}

	template <class T>
	const T &as() const
	{
		return *reinterpret_cast<const T *>(&m_data);
	}

	template <class T>
	T &as()
	{
		return *reinterpret_cast<T *>(&m_data);
	}
public:
	JsonValue(std::nullptr_t = nullptr) : m_tag{ NULL_ }
	{
		construct(nullptr);
	}

	JsonValue(bool x) : m_tag{ BOOL_ }
	{
		construct(x);
	}

	JsonValue(number_type x) : m_tag{ NUMBER }
	{
		construct(x);
	}

	JsonValue(string_type str) : m_tag{ STRING }
	{
		construct(std::move(str));
	}

	explicit JsonValue(array_type array) : m_tag{ ARRAY }
	{
		construct(std::move(array));
	}

	explicit JsonValue(object_type object) : m_tag{ OBJECT }
	{
		construct(std::move(object));
	}

	JsonValue(const JsonValue &other);

	JsonValue(JsonValue &&other);

	~JsonValue();

	JsonValue &operator=(JsonValue other);

	explicit operator bool() const
	{
		return !is_null();
	}

	tag_type get_type() const
	{
		return m_tag;
	}

	bool is_null() const
	{
		return get_type() == NULL_;
	}

	const number_type &number() const
	{
		check_type(NUMBER);
		return as<number_type>();
	}

	number_type &number()
	{
		check_type(NUMBER);
		return as<number_type>();
	}

	long long integer() const
	{
		return llrint(number());
	}

	const string_type &string() const
	{
		check_type(STRING);
		return as<string_type>();
	}

	string_type &string()
	{
		check_type(STRING);
		return as<string_type>();
	}

	const object_type &object() const
	{
		check_type(OBJECT);
		return as<object_type>();
	}

	object_type &object()
	{
		check_type(OBJECT);
		return as<object_type>();
	}

	const array_type &array() const
	{
		check_type(ARRAY);
		return as<array_type>();
	}

	array_type &array()
	{
		check_type(ARRAY);
		return as<array_type>();
	}

	const boolean_type &boolean() const
	{
		check_type(BOOL_);
		return as<boolean_type>();
	}

	boolean_type &boolean()
	{
		check_type(BOOL_);
		return as<boolean_type>();
	}

	void swap(JsonValue &other);
};


namespace json {

JsonValue parse_document(const std::string &str);

} // namespace json

#endif // JSON_H_
