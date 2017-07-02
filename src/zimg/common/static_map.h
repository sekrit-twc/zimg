#pragma once

#ifndef ZIMG_STATIC_MAP_H_
#define ZIMG_STATIC_MAP_H_

#include <cstddef>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <type_traits>
#include <utility>

#if __cplusplus >= 201402L || (defined(_MSVC_LANG) && _MSVC_LANG >= 201402L)
  #define SM_CONSTEXPR_14 constexpr
#else
  #define SM_CONSTEXPR_14
#endif

// ICL is unable to evaluate C++14 constexpr.
#ifdef __INTEL_COMPILER
  #undef SM_CONSTEXPR_14
  #define SM_CONSTEXPR_14
#endif

namespace zimg {

template <class...>
using _static_map_void_t = void;

template <class Comp, class K, class = void>
struct _static_map_is_transparent : std::false_type {};

template <class Comp, class K>
struct _static_map_is_transparent<Comp, K, _static_map_void_t<typename Comp::is_transparent>> : std::true_type {};

// Read-only associative container that contains key-value pairs with multiple
// entries with the same key permitted. Unlike std::map and std::unordered_map,
// static_map supports constexpr initialization.
template <class Key, class T, std::size_t N, class Compare = std::less<Key>>
class static_map : private Compare {
public:
	typedef Key key_type;
	typedef T mapped_type;
	typedef std::pair<const Key, T> value_type;
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;
	typedef Compare key_compare;
	typedef value_type &reference;
	typedef const value_type &const_reference;
	typedef value_type *pointer;
	typedef const value_type *const_pointer;
	typedef pointer iterator;
	typedef const_pointer const_iterator;

	// ztd::static_map::value_compare
	//
	// ztd::static_map::value_compare is a function object that compares objects of type ztd::static_map::value_type
	// (key-value pairs) by comparing of the first components of the pairs.
	class value_compare {
	protected:
		// The stored comparator.
		Compare comp;

		// Initializes the internal instance of the comparator to c.
		SM_CONSTEXPR_14 value_compare(Compare c);
	public:
		// Compares lhs.first and rhs.first by calling the stored comparator.
		SM_CONSTEXPR_14 bool operator()(const value_type &lhs, const value_type &rhs) const;

		friend class static_map;
	};
private:
	struct xcompare;

	typedef std::pair<Key, T> stored_value_type;
	static_assert(offsetof(value_type, first) == offsetof(stored_value_type, first), "wrong layout");
	static_assert(offsetof(value_type, second) == offsetof(stored_value_type, second), "wrong layout");

	size_type m_size;
	stored_value_type m_array[N ? N : 1];
public:
	// Constructs the container with the contents of the initializer list init.
	SM_CONSTEXPR_14 explicit static_map(std::initializer_list<value_type> init, Compare comp = Compare());

	static_map(const static_map &) = delete;
	static_map &operator=(const static_map &) = delete;

	// Returns a reference to the mapped value of the element with key equivalent to key.
	const T &at(const Key &key) const;
	const T &operator[](const Key &key) const;

	// Returns an iterator to the first element of the container.
	const_iterator begin() const noexcept;
	const_iterator cbegin() const noexcept;

	// Returns an iterator to the element following the last element of the container.
	const_iterator end() const noexcept;
	const_iterator cend() const noexcept;

	// Checks if the container has no elements, i.e. whether begin() == end().
	SM_CONSTEXPR_14 bool empty() const noexcept;

	// Returns the number of elements in the container, i.e. std::distance(begin(), end()).
	SM_CONSTEXPR_14 size_type size() const noexcept;

	// Returns the maximum number of elements the container is able to hold.
	SM_CONSTEXPR_14 size_type max_size() const noexcept;

	// (1) Returns the number of elements with key |key|.
	size_type count(const Key &key) const;
	// (2) Returns the number of elements with key that compares equivalent to the value |x|.
	template <class K>
	typename std::enable_if<_static_map_is_transparent<Compare, K>::value, size_type>::type
	count(const K &key) const;

	// (1) Finds an element with key equivalent to |key|.
	const_iterator find(const Key &key) const;
	// (2) Finds an element with key that compares equivalent to the value |x|.
	template <class K>
	typename std::enable_if<_static_map_is_transparent<Compare, K>::value, const_iterator>::type
	find(const K &x) const;

	// (1) Compares the keys to |key|.
	std::pair<const_iterator, const_iterator> equal_range(const Key &key) const;
	// (2) Compares the keys to the value |x|.
	template <class K>
	typename std::enable_if<_static_map_is_transparent<Compare, K>::value, std::pair<const_iterator, const_iterator>>::type
	equal_range(const K &x) const;

	// Returns the function object that compares the keys.
	SM_CONSTEXPR_14 key_compare key_comp() const;

	// Returns a function object that compares objects of type ztd::static_map::value_type.
	SM_CONSTEXPR_14 value_compare value_comp() const;
};

struct _static_map_strcmp {
	bool operator()(const char *lhs, const char *rhs) const
	{
		return std::strcmp(lhs, rhs) < 0;
	}
};

template <class T, size_t N>
using static_string_map = static_map<const char *, T, N, _static_map_strcmp>;

} // namespace zimg


// Implementation details follow.
#include <algorithm>
#include <stdexcept>

namespace zimg {

template <class Key, class T, std::size_t N, class Compare>
struct static_map<Key, T, N, Compare>::xcompare {
	Compare comp;

	xcompare(Compare c) : comp(c) {}

	template <class K>
	bool operator()(const value_type &lhs, const K &rhs) const
	{
		return comp(lhs.first, rhs);
	}

	template <class K>
	bool operator()(const K &lhs, const value_type &rhs) const
	{
		return comp(lhs, rhs.first);
	}
};

// ztd::static_map<Key, T, N, Compare>::value_compare::value_compare
//
// Initializes the internal instance of the comparator to c.
//
// Parameters
// c - comparator to assign
template <class Key, class T, std::size_t N, class Compare>
SM_CONSTEXPR_14 static_map<Key, T, N, Compare>::value_compare::value_compare(Compare c) : comp(c) {}

// ztd::static_map<Key, T, N, Compare>::value_compare::operator()
//
// Compares lhs.first and rhs.first by calling the stored comparator.
//
// Parameters
// lhs, rhs - values to compare
//
// Return value
// comp(lhs.first, rhs.first).
template <class Key, class T, std::size_t N, class Compare>
SM_CONSTEXPR_14 bool static_map<Key, T, N, Compare>::value_compare::operator()(
	const value_type &lhs, const value_type &rhs) const
{
	return comp(lhs.first, rhs.first);
}

// ztd::static_map::static_map
//
// Constructs the container with the contents of the initializer list init and optionally using user supplied comparison
// function object comp.
//
// Parameters
// comp - comparison function object to use for all comparisons of keys
// init - initializer list to initialize the elements of the container with
//
// Exceptions
// std::out_of_range if init contains more than N elements.
template <class Key, class T, std::size_t N, class Compare>
SM_CONSTEXPR_14 static_map<Key, T, N, Compare>::static_map(std::initializer_list<value_type> init, Compare comp) :
	Compare(comp),
	m_size(init.size()),
	m_array{}
{
	if (init.size() > N)
		throw std::out_of_range("");

	size_type i = 0;
	for (auto v = init.begin(); v != init.end(); ++v) {
		m_array[i].first = v->first;
		m_array[i].second = v->second;
		++i;
	}

	for (size_type i = 1; i < size(); ++i) {
		for (size_type j = i; j > 0; --j) {
			if (!comp(m_array[j].first, m_array[j - 1].first))
				break;

			stored_value_type v(std::move(m_array[j]));
			m_array[j].first = std::move(m_array[j - 1].first);
			m_array[j].second = std::move(m_array[j - 1].second);
			m_array[j - 1].first = std::move(v.first);
			m_array[j - 1].second = std::move(v.second);
		}
	}
}

// ztd::static_map::at, ztd::static_map::operator[]
//
// Returns a reference to the mapped value of the element with key equivalent to key. If no such element exists, an
// exception of type std::out_of_range is thrown.
//
// Parameters
// key - the key of the element to find
//
// Return value
// Reference to the mapped value of the requested element
//
// Exceptions
// std::out_of_range if the container does not have an element with the specified key
//
// Complexity
// Logarithmic in the size of the container.
template <class Key, class T, std::size_t N, class Compare>
const T &static_map<Key, T, N, Compare>::at(const Key &key) const
{
	const_iterator it = find(key);
	if (it == end())
		throw std::out_of_range("");
	return it->second;
}

template <class Key, class T, std::size_t N, class Compare>
const T &static_map<Key, T, N, Compare>::operator[](const Key &key) const
{
	return at(key);
}

// ztd::static_map::begin, ztd::static_map::cbegin
//
// Returns an iterator to the first element of the container.
// If the container is empty, the returned iterator will be equal to end().
//
// Return value
// Iterator to the first element
//
// Complexity
// Constant
template <class Key, class T, std::size_t N, class Compare>
auto static_map<Key, T, N, Compare>::begin() const noexcept -> const_iterator
{
	return cbegin();
}

template <class Key, class T, std::size_t N, class Compare>
auto static_map<Key, T, N, Compare>::cbegin() const noexcept -> const_iterator
{
	return reinterpret_cast<const value_type *>(m_array);
}

// ztd::static_map::end, ztd::static_map::cend
//
// Returns an iterator to the element following the last element of the container.
// This element acts as a placeholder; attempting to access it results in undefined behavior.
//
// Return value
// Iterator to the element following the last element.
//
// Complexity
// Constant.
template <class Key, class T, std::size_t N, class Compare>
auto static_map<Key, T, N, Compare>::end() const noexcept -> const_iterator
{
	return cend();
}

template <class Key, class T, std::size_t N, class Compare>
auto static_map<Key, T, N, Compare>::cend() const noexcept -> const_iterator
{
	return reinterpret_cast<const value_type *>(m_array) + size();
}

// ztd::static_map::empty
//
// Checks if the container has no elements, i.e. whether begin() == end().
//
// Return value
// true if the container is empty, false otherwise
//
// Complexity
// Constant.
template <class Key, class T, std::size_t N, class Compare>
SM_CONSTEXPR_14 bool static_map<Key, T, N, Compare>::empty() const noexcept
{
	return size() == 0;
}

// ztd::static_map::size
//
// Returns the number of elements in the container, i.e. std::distance(begin(), end()).
//
// Return value
// The number of elements in the container.
//
// Complexity
// Constant.
template <class Key, class T, std::size_t N, class Compare>
SM_CONSTEXPR_14 auto static_map<Key, T, N, Compare>::size() const noexcept -> size_type
{
	return m_size;
}

// ztd::static_map::max_size
//
// Returns the maximum number of elements the container is able to hold due to system or library implementation
// limitations, i.e. std::distance(begin(), end()) for the largest container.
//
// Return value
// Maximum number of elements.
//
// Complexity
// Constant.
template <class Key, class T, std::size_t N, class Compare>
SM_CONSTEXPR_14 auto static_map<Key, T, N, Compare>::max_size() const noexcept -> size_type
{
	return N;
}

// ztd::static_map::count
//
// Returns the number of elements with key that compares equivalent to the specified argument.
// (1) Returns the number of elements with key key.
// (2) Returns the number of elements with key that compares equivalent to the value x.
//     This overload only participates in overload resolution if the qualified-id Compare::is_transparent is valid and
//     denotes a type. They allow calling this function without constructing an instance of Key.
//
// Parameters
// key - key value of the elements to count
// x   - alternative value to compare to the keys
//
// Return value
// Number of elements with key that compares equivalent to key or x, that is either 1 or 0.
//
// Complexity
// Logarithmic in the size of the container.

// (1)
template <class Key, class T, std::size_t N, class Compare>
auto static_map<Key, T, N, Compare>::count(const Key &key) const -> size_type
{
	auto range = equal_range(key);
	return range.second - range.first;
}

// (2)
template <class Key, class T, std::size_t N, class Compare>
template <class K>
auto static_map<Key, T, N, Compare>::count(const K &x) const ->
	typename std::enable_if<_static_map_is_transparent<Compare, K>::value, size_type>::type
{
	auto range = equal_range(x);
	return range.second - range.first;
}

// ztd::static_map::find
//
// (1) Finds an element with key equivalent to key.
// (2) Finds an element with key that compares equivalent to the value x.
//     This overload only participates in overload resolution if the qualified-id Compare::is_transparent is valid and
//     denotes a type. It allows calling this function without constructing an instance of Key.
// Parameters
// key - key value of the element to search for
// x   - a value of any type that can be transparently compared with a key
//
// Return value
// Iterator to an element with key equivalent to key. If no such element is found, past-the-end (see end()) iterator is
// returned.
//
// Complexity
// Logarithmic in the size of the container.

// (1)
template <class Key, class T, std::size_t N, class Compare>
auto static_map<Key, T, N, Compare>::find(const Key &key) const -> const_iterator
{
	const_iterator it = std::lower_bound(begin(), end(), key, xcompare(key_comp()));
	key_compare comp = key_comp();

	return (it == end() || comp(it->first, key) || comp(key, it->first)) ? end() : it;
}

// (2)
template <class Key, class T, std::size_t N, class Compare>
template <class K>
auto static_map<Key, T, N, Compare>::find(const K &x) const ->
	typename std::enable_if<_static_map_is_transparent<Compare, K>::value, const_iterator>::type
{
	const_iterator it = std::lower_bound(begin(), end(), x, xcompare(key_comp()));
	key_compare comp = key_comp();

	return (it == end() || comp(it->first, x) || comp(x, it->first)) ? end() : it;
}

// ztd::static_map::equal_range
//
// Returns a range containing all elements with the given key in the container. The range is defined by two iterators,
// one pointing to the first element that is not less than key and another pointing to the first element greater than
// key.
// (1) Compares the keys to key.
// (2) Compares the keys to the value x.
//     This overload only participates in overload resolution if the qualified-id Compare::is_transparent is valid and
//     denotes a type. They allow calling this function without constructing an instance of Key.
//
// Parameters
// key - key value to compare the elements to
// x   - alternative value that can be compared to Key
//
// Return value
// std::pair containing a pair of iterators defining the wanted range: the first pointing to the first element that is
// not less than key and the second pointing to the first element greater than key.
//
// If there are no elements not less than key, past-the-end (see end()) iterator is returned as the first element.
// Similarly if there are no elements greater than key, past-the-end iterator is returned as the second element.
//
// Complexity
// Logarithmic in the size of the container.

// (1)
template <class Key, class T, std::size_t N, class Compare>
auto static_map<Key, T, N, Compare>::equal_range(const Key &key) const -> std::pair<const_iterator, const_iterator>
{
	const_iterator first = std::lower_bound(begin(), end(), key, xcompare(key_comp()));
	const_iterator last = std::upper_bound(first, end(), key, xcompare(key_comp()));
	return{ first, last };
}

// (2)
template <class Key, class T, std::size_t N, class Compare>
template <class K>
auto static_map<Key, T, N, Compare>::equal_range(const K &x) const ->
	typename std::enable_if<_static_map_is_transparent<Compare, K>::value, std::pair<const_iterator, const_iterator>>::type
{
	const_iterator first = std::lower_bound(begin(), end(), x, xcompare(key_comp()));
	const_iterator last = std::upper_bound(first, end(), x, xcompare(key_comp()));
	return{ first, last };
}

// ztd::static_map::key_comp
//
// Returns the function object that compares the keys, which is a copy of this container's constructor argument comp.
//
// Return value
// The key comparison function object.
//
// Complexity
// Constant.
template <class Key, class T, std::size_t N, class Compare>
SM_CONSTEXPR_14 auto static_map<Key, T, N, Compare>::key_comp() const -> key_compare
{
	return Compare(*this);
}

// ztd::static_map::value_comp
//
// Returns a function object that compares objects of type ztd::static_map::value_type (key-value pairs) by using
// key_comp to compare the first components of the pairs.
//
// Return value
// The value comparison function object.
//
// Complexity
// Constant.
template <class Key, class T, std::size_t N, class Compare>
SM_CONSTEXPR_14 auto static_map<Key, T, N, Compare>::value_comp() const -> value_compare
{
	return key_comp();
}

} // namespace zimg

#endif // ZIMG_STATIC_MAP_H_
