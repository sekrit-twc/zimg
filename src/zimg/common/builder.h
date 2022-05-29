#ifndef BUILDER_MEMBER
#define BUILDER_MEMBER(type, name) \
  auto &set_##name(type const &val) { name = val; return *this; } \
  template <class T> \
  auto &set_##name(T &&val) { name = std::forward<T>(val); return *this; } \
  type name;
#endif // BUILDER_MEMBER
