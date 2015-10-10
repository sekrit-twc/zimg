#pragma once

#ifndef ZIMG_STATIC_MAP_H_
#define ZIMG_STATIC_MAP_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include "zassert.h"

namespace zimg {;

template <class Key, class T, class Compare = std::less<Key>, size_t Sz = 32>
class static_map {
public:
	typedef Key key_type;
	typedef T mapped_type;
	typedef std::pair<Key, T> value_type;
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;
	typedef Compare key_compare;
	typedef const value_type *const_iterator;
private:
	struct value_compare {
		key_compare comp;

		bool operator()(const value_type &a, const value_type &b)
		{
			return comp(a.first, b.first);
		}
	};

	struct data_member : private Compare {
		std::array<value_type, Sz> array;

		data_member(const Compare &comp) : Compare(comp)
		{
		}

		key_compare get_key_comp() const
		{
			return *this;
		}

		value_compare get_value_comp() const
		{
			return{ *this };
		}
	};

	data_member m_head;
	size_t m_size;

	bool equiv(const Key &a, const Key &b) const
	{
		key_compare comp = m_head.get_key_comp();
		return !comp(a, b) && !comp(b, a);
	}
public:
	explicit static_map(std::initializer_list<value_type> init, const Compare &comp = Compare()) :
		m_head{ comp },
		m_size{ init.size() }
	{
		_zassert(init.size() <= Sz, "list size incorrect");

		std::copy(init.begin(), init.end(), m_head.array.begin());
		std::sort(m_head.array.begin(), m_head.array.begin() + init.size(), m_head.get_value_comp());
	}

	size_type size() const
	{
		return m_size;
	}

	const_iterator begin() const
	{
		return m_head.array.data();
	}

	const_iterator end() const
	{
		return begin() + m_size;
	}

	const T &at(const Key &key) const
	{
		const auto it = find(key);

		if (it == end())
			throw std::out_of_range{ "key not found" };

		return it->second;
	}

	const T &operator[](const Key &key) const
	{
		return at(key);
	}

	const_iterator find(const Key &key) const
	{
		auto it = std::lower_bound(begin(), end(), value_type{ key, mapped_type{} }, m_head.get_value_comp());

		if (it == end() || !equiv(it->first, key))
			return end();
		else
			return it;
	}
};

struct strcmp_less {
	bool operator()(const char *a, const char *b) const
	{
		return strcmp(a, b) < 0;
	}
};

template <class T, size_t Sz>
using static_string_map = static_map<const char *, T, strcmp_less, Sz>;

template <class Key, class T, size_t Sz>
using static_enum_map = static_map<Key, T, std::less<Key>, Sz>;

} // namespace zimg

#endif // ZIMG_STATIC_MAP_H_
