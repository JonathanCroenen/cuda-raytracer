#pragma once

#include "utils/cuda.h"
#include <stdint.h>
#include <utility>

namespace rtx::utils {

template <size_t Length, size_t Alignment>
struct AlignedStorage {
  alignas(Alignment) unsigned char data[Length];
};

template <size_t Arg, size_t... Args>
struct MaxSize;

template <size_t Arg>
struct MaxSize<Arg> {
  static constexpr size_t value = Arg;
};

template <size_t Arg, size_t... Args>
struct MaxSize {
  static constexpr size_t value = Arg > MaxSize<Args...>::value ? Arg
                                                                : MaxSize<Args...>::value;
};

template <uint8_t ID, typename... Ts>
struct VariantHelperRec;

template <uint8_t ID, typename T, typename... Ts>
struct VariantHelperRec<ID, T, Ts...> {
  static constexpr uint8_t type_index = ID;

  inline static void destroy(uint8_t id, void* data) {
    if (id == ID) {
      reinterpret_cast<T*>(data)->~T();
    } else {
      VariantHelperRec<ID + 1, Ts...>::destroy(id, data);
    }
  }

  inline static void move(uint8_t id, void* src, void* dst) {
    if (id == ID) {
      new (dst) T(std::move(*reinterpret_cast<T*>(src)));
    } else {
      VariantHelperRec<ID + 1, Ts...>::move(id, src, dst);
    }
  }

  inline static void copy(uint8_t id, const void* src, void* dst) {
    if (id == ID) {
      new (dst) T(*reinterpret_cast<const T*>(src));
    } else {
      VariantHelperRec<ID + 1, Ts...>::copy(id, src, dst);
    }
  }
};

template <uint8_t ID>
struct VariantHelperRec<ID> {
  static constexpr uint8_t type_index = ID;
  inline static void destroy(uint8_t, void*) {}
  inline static void move(uint8_t, void*, void*) {}
  inline static void copy(uint8_t, const void*, void*) {}
};

template <typename... Ts>
struct VariantHelper;

template <typename T, typename... Ts>
struct VariantHelper<T, Ts...> {
  static constexpr uint8_t type_index = VariantHelperRec<1, T, Ts...>::type_index;

  inline static void destroy(uint8_t id, void* data) {
    VariantHelperRec<1, T, Ts...>::destroy(id, data);
  }

  inline static void move(uint8_t id, void* src, void* dst) {
    VariantHelperRec<1, T, Ts...>::move(id, src, dst);
  }

  inline static void copy(uint8_t id, const void* src, void* dst) {
    VariantHelperRec<1, T, Ts...>::copy(id, src, dst);
  }
};

template <>
struct VariantHelper<> {
  static constexpr uint8_t type_index = 0;

  inline static void destroy(uint8_t, void*) {}
  inline static void move(uint8_t, void*, void*) {}
  inline static void copy(uint8_t, const void*, void*) {}
};

template <uint8_t ID, typename U, typename... Ts>
struct VariantTypeIndex;

template <uint8_t ID, typename U, typename T, typename... Ts>
struct VariantTypeIndex<ID, U, T, Ts...> {
  static constexpr uint8_t value = (std::is_same<U, T>::value)
                                       ? ID
                                       : VariantTypeIndex<ID + 1, U, Ts...>::value;
};

template <uint8_t ID, typename U>
struct VariantTypeIndex<ID, U> {
  static constexpr uint8_t value = 0;
};

template <typename... Ts>
struct Variant {
private:
  static constexpr size_t data_size = MaxSize<sizeof(Ts)...>::value;
  static constexpr size_t data_alignment = MaxSize<alignof(Ts)...>::value;

  using Storage = AlignedStorage<data_size, data_alignment>;
  using Helper = VariantHelper<Ts...>;
  template <typename U>
  using TypeIndex = VariantTypeIndex<1, U, Ts...>;
  template <typename U>
  using Supported = std::enable_if_t<TypeIndex<U>::value != 0, bool>;

public:
  Variant() : _id(0) {}

  template <typename T, Supported<T> = true>
  Variant(const T& value) {
    _id = TypeIndex<T>::value;
    new (&_data) T(value);
  }

  template <typename T, Supported<T> = true>
  Variant(T&& value) {
    _id = TypeIndex<T>::value;
    new (&_data) T(std::move(value));
  }

  template <typename T, typename... Args, Supported<T> = true>
  Variant(std::in_place_type_t<T>, Args&&... args) {
    _id = TypeIndex<T>::value;
    new (&_data) T(std::forward<Args>(args)...);
  }

  Variant(const Variant& other) : _id(other._id) {
    Helper::copy(other._id, &other._data, &_data);
  }

  Variant(Variant&& other) : _id(other._id) {
    Helper::move(other._id, &other._data, &_data);
  }

  ~Variant() { Helper::destroy(_id, &_data); }

  Variant& operator=(const Variant& other) {
    Helper::destroy(_id, &_data);
    _id = other._id;
    Helper::copy(other._id, &other._data, &_data);
    return *this;
  }

  Variant& operator=(Variant&& other) {
    Helper::destroy(_id, &_data);
    _id = other._id;
    Helper::move(other._id, &other._data, &_data);
    return *this;
  }

  template <typename T, Supported<T> = true>
  Variant& operator=(const T& value) {
    Helper::destroy(_id, &_data);
    _id = TypeIndex<T>::value;
    new (&_data) T(value);
    return *this;
  }

  template <typename T, Supported<T> = true>
  Variant& operator=(T&& value) {
    Helper::destroy(_id, &_data);
    _id = TypeIndex<T>::value;
    new (&_data) T(std::move(value));
    return *this;
  }

  template <typename T, Supported<T> = true>
  CPU_GPU_FUNC inline bool is() const {
    return _id == TypeIndex<T>::value;
  }

  CPU_GPU_FUNC inline bool valid() const { return _id != 0; }

  CPU_GPU_FUNC inline uint8_t type_id() const { return _id; };

  template<typename T, Supported<T> = true>
  CPU_GPU_FUNC static inline constexpr uint8_t type_id_of() { return TypeIndex<T>::value; }

  template <typename T, typename... Args, Supported<T> = true>
  void set(Args&&... args) {
    Helper::destroy(_id, &_data);
    _id = TypeIndex<T>::value;
    new (&_data) T(std::forward<Args>(args)...);
  }

  template <typename T, Supported<T> = true>
  CPU_GPU_FUNC inline T& get_unchecked() {
    return *reinterpret_cast<T*>(&_data);
  }

  template <typename T, Supported<T> = true>
  CPU_GPU_FUNC inline const T& get_unchecked() const {
    return *reinterpret_cast<const T*>(&_data);
  }

  template <typename T, Supported<T> = true>
  CPU_GPU_FUNC inline T& get() {
#ifdef __CUDA_ARCH__
    return get_unchecked<T>();
#else
    if (!is<T>()) {
      throw std::bad_cast();
    }
    return get_unchecked<T>();
#endif
  }

  template <typename T, Supported<T> = true>
  CPU_GPU_FUNC inline const T& get() const {
#ifdef __CUDA_ARCH__
    return get_unchecked<T>();
#else
    if (!is<T>()) {
      throw std::bad_cast();
    }
    return get_unchecked<T>();
#endif
  }

private:
  uint8_t _id;
  Storage _data;
};

} // namespace rtx::utils
