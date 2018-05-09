/*
   Copyright 2018 - The OPRECOMP Project Consortium, Universitat Jaume I,
                    IBM Research GmbH. All rights reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#ifndef FLOATX_FLOATX_HPP_
#define FLOATX_FLOATX_HPP_


#ifdef __CUDA_ARCH__
#include "cuda_runtime.h"
#endif  // __CUDA_ARCH__

#include <climits>

#if CHAR_BIT != 8
#error Expecting 8 bits in a char!
#endif  // ?CHAR_BIT

#include <cassert>
#include <cmath>
#include <cstdint>

#include <bitset>
#include <istream>
#include <ostream>
#include <string>
#include <type_traits>


#ifdef __CUDA_ARCH__
#define FLOATX_ATTRIBUTES __host__ __device__
#define FLOATX_INLINE __forceinline__
#else  // __CUDA_ARCH__
#define FLOATX_ATTRIBUTES
#define FLOATX_INLINE inline
#endif  // __CUDA_ARCH__


#define USE_BUILTINS


namespace flx {


namespace detail {


static constexpr int bits_in_byte = CHAR_BIT;


template <typename>
class floatx_base;


template <int size>
struct bits_type {};


#define ENABLE_STANDARD_BITS_TYPE(_size)     \
    template <>                              \
    struct bits_type<_size / bits_in_byte> { \
        using type = std::uint##_size##_t;   \
    }

ENABLE_STANDARD_BITS_TYPE(8);
ENABLE_STANDARD_BITS_TYPE(16);
ENABLE_STANDARD_BITS_TYPE(32);
ENABLE_STANDARD_BITS_TYPE(64);

#undef ENABLE_STANDARD_BITS_TYPE


}  // namespace detail


#define FLOATX_USE_DEFAULT_TRAITS(_type)                                 \
    static const auto sig_pos = 0;                                       \
    static const auto exp_pos = float_traits<_type>::sig_bits;           \
    static const auto sgn_pos = exp_pos + float_traits<_type>::exp_bits; \
    static const auto sig_mask =                                         \
        (UINT64_C(1) << float_traits<_type>::sig_bits) - UINT64_C(1);    \
    static const auto exp_mask =                                         \
        (UINT64_C(1) << float_traits<_type>::exp_bits) - UINT64_C(1);    \
    static const auto sgn_mask = UINT64_C(1);                            \
    static const auto bias = exp_mask >> 1;                              \
    using bits_type = typename detail::bits_type<sizeof(_type)>::type


template <typename T, typename = void>
struct float_traits {};

template <typename T>
struct float_traits<T,
                    typename std::enable_if<std::is_integral<T>::value>::type> {
    static const bool is_floatx = false;
    static const bool is_runtime = false;
    static const int exp_bits = 0;
    static const int sig_bits = 0;
    using backend_float = T;

    FLOATX_USE_DEFAULT_TRAITS(T);
};

template <>
struct float_traits<float, void> {
    static const bool is_floatx = false;
    static const bool is_runtime = false;
    static const int exp_bits = 8;
    static const int sig_bits = 23;
    using backend_float = float;

    FLOATX_USE_DEFAULT_TRAITS(float);
};

template <>
struct float_traits<double, void> {
    static const bool is_floatx = false;
    static const bool is_runtime = false;
    static const int exp_bits = 11;
    static const int sig_bits = 52;
    using backend_float = double;

    FLOATX_USE_DEFAULT_TRAITS(double);
};


#define ENABLE_PROPERTY(_prop)                                              \
    template <typename Float>                                               \
    FLOATX_ATTRIBUTES FLOATX_INLINE constexpr auto get_##_prop(             \
        const Float&) noexcept->                                            \
        typename std::enable_if<!float_traits<Float>::is_runtime,           \
                                decltype(float_traits<Float>::_prop)>::type \
    {                                                                       \
        return float_traits<Float>::_prop;                                  \
    }                                                                       \
    template <typename RuntimeFloat>                                        \
    FLOATX_ATTRIBUTES FLOATX_INLINE constexpr auto get_##_prop(             \
        const RuntimeFloat& f) noexcept->                                   \
        typename std::enable_if<float_traits<RuntimeFloat>::is_runtime,     \
                                decltype(f.get_##_prop())>::type            \
    {                                                                       \
        return f.get_##_prop();                                             \
    }

ENABLE_PROPERTY(exp_bits);  // get_exp_bits(f)
ENABLE_PROPERTY(sig_bits);  // get_sig_bits(f)

#undef ENABLE_PROPERTY


template <int ExpBits, int SigBits, typename BackendFloat = double>
class floatx
    : public detail::floatx_base<floatx<ExpBits, SigBits, BackendFloat>> {
private:
    using backend_float = typename float_traits<floatx>::backend_float;

public:
    FLOATX_ATTRIBUTES floatx() noexcept
        : detail::floatx_base<floatx>(backend_float(0.0))
    {
        this->initialize();
    }

    template <typename T>
    FLOATX_ATTRIBUTES floatx(const T& other) noexcept
        : detail::floatx_base<floatx>(backend_float(other))
    {
        this->initialize();
    }

    // Default copy/move constructors/assignment operators are OK here

    template <typename T>
    FLOATX_ATTRIBUTES floatx& operator=(const T& other) noexcept
    {
        return *this = floatx(other);
    }
};


template <int ExpBits, int SigBits, typename BackendFloat>
struct float_traits<floatx<ExpBits, SigBits, BackendFloat>, void> {
    static const bool is_floatx = true;
    static const bool is_runtime = false;
    static const int exp_bits = ExpBits;
    static const int sig_bits = SigBits;
    using backend_float = BackendFloat;

    FLOATX_USE_DEFAULT_TRAITS(backend_float);
};


template <typename BackendFloat = double, typename MetadataType = short>
class floatxr
    : public detail::floatx_base<floatxr<BackendFloat, MetadataType>> {
private:
    using backend_float = typename float_traits<floatxr>::backend_float;

public:
    using metadata_type = MetadataType;

    FLOATX_ATTRIBUTES
    floatxr(metadata_type exp_bits, metadata_type sig_bits) noexcept
        : detail::floatx_base<floatxr>(backend_float(0.0)),
          exp_bits_(exp_bits),
          sig_bits_(sig_bits)
    {
        this->initialize();
    }

    // Default copy/move constructors are OK

    template <typename T>
    FLOATX_ATTRIBUTES floatxr(metadata_type exp_bits, metadata_type sig_bits,
                              const T& other) noexcept
        : detail::floatx_base<floatxr>(backend_float(other)),
          exp_bits_(exp_bits),
          sig_bits_(sig_bits)
    {
        this->initialize();
    }

    template <typename T>
    FLOATX_ATTRIBUTES floatxr(const T& other) noexcept
        : detail::floatx_base<floatxr>(backend_float(other)),
          exp_bits_(flx::get_exp_bits(other)),
          sig_bits_(flx::get_sig_bits(other))
    {
        /* already initialized */
    }

    // Assignment needs to preserve the format of the result
    template <typename T>
    FLOATX_ATTRIBUTES floatxr& operator=(const T& other) noexcept
    {
        return *this = floatxr(flx::get_exp_bits(*this),
                               flx::get_sig_bits(*this), backend_float(other));
    }

    FLOATX_ATTRIBUTES void set_precision(metadata_type exp_bits,
                                         metadata_type sig_bits)
    {
        exp_bits_ = exp_bits;
        sig_bits_ = sig_bits;
        this->initialize();
    }

    FLOATX_ATTRIBUTES constexpr metadata_type get_exp_bits() const noexcept
    {
        return exp_bits_;
    }

    FLOATX_ATTRIBUTES constexpr metadata_type get_sig_bits() const noexcept
    {
        return sig_bits_;
    }

private:
    metadata_type exp_bits_;
    metadata_type sig_bits_;
};


template <typename BackendFloat, typename MetadataType>
struct float_traits<floatxr<BackendFloat, MetadataType>, void> {
    static const bool is_floatx = true;
    static const bool is_runtime = true;
    static const int exp_bits = float_traits<BackendFloat>::exp_bits;
    static const int sig_bits = float_traits<BackendFloat>::sig_bits;
    using backend_float = BackendFloat;

    FLOATX_USE_DEFAULT_TRAITS(backend_float);
};


template <typename FloatX1, typename FloatX2, typename BackendFloat>
struct supertype {
private:
    static constexpr int max(int x, int y) { return (x > y) ? x : y; }

public:
#ifdef FLOATX_NO_TYPE_RESOLUTION
    static_assert(std::is_same<FloatX1, FloatX2>::value,
                  "Common type detection is disabled by the user"
                  " [FLOATX_NO_TYPE_RESOLUTION]");
#endif  // FLOATX_NO_TYPE_RESOLUTION

    using type = typename std::enable_if<
        float_traits<FloatX1>::is_floatx || float_traits<FloatX2>::is_floatx,
        typename std::conditional<float_traits<FloatX1>::is_runtime ||
                                      float_traits<FloatX2>::is_runtime,
                                  floatxr<BackendFloat>,
                                  floatx<max(float_traits<FloatX1>::exp_bits,
                                             float_traits<FloatX2>::exp_bits),
                                         max(float_traits<FloatX1>::sig_bits,
                                             float_traits<FloatX2>::sig_bits),
                                         BackendFloat>>::type>::type;
    static constexpr int max_exp_bits(FloatX1 x, FloatX2 y)
    {
        return max(get_exp_bits(x), get_exp_bits(y));
    }
    static constexpr int max_sig_bits(FloatX1 x, FloatX2 y)
    {
        return max(get_sig_bits(x), get_sig_bits(y));
    }
};


#define ENABLE_RELATIONAL_OPERATOR(_op)                              \
    template <typename Float1, typename Float2>                      \
    FLOATX_ATTRIBUTES FLOATX_INLINE                                  \
        typename std::enable_if<float_traits<Float1>::is_floatx ||   \
                                    float_traits<Float2>::is_floatx, \
                                bool>::type                          \
        operator _op(const Float1& x, const Float2& y)               \
    {                                                                \
        return typename float_traits<Float1>::backend_float(x) _op   \
            typename float_traits<Float2>::backend_float(y);         \
    }

ENABLE_RELATIONAL_OPERATOR(==)
ENABLE_RELATIONAL_OPERATOR(!=)
ENABLE_RELATIONAL_OPERATOR(<)
ENABLE_RELATIONAL_OPERATOR(>)
ENABLE_RELATIONAL_OPERATOR(<=)
ENABLE_RELATIONAL_OPERATOR(>=)

#undef ENABLE_RELATIONAL_OPERATOR


#define ENABLE_ARITHMETIC_OPERATOR(_op)                                        \
    template <typename Float1, typename Float2>                                \
    FLOATX_ATTRIBUTES FLOATX_INLINE typename std::enable_if<                   \
        (float_traits<Float1>::is_floatx ||                                    \
         float_traits<Float2>::is_floatx) &&                                   \
            !float_traits<Float1>::is_runtime &&                               \
            !float_traits<Float2>::is_runtime,                                 \
        typename supertype<                                                    \
            Float1, Float2,                                                    \
            decltype(typename float_traits<Float1>::backend_float() _op        \
                     typename float_traits<Float2>::backend_float())>::type>:: \
        type                                                                   \
        operator _op(const Float1& x, const Float2& y)                         \
    {                                                                          \
        using bf = decltype(typename float_traits<Float1>::backend_float(      \
            x) _op typename float_traits<Float2>::backend_float(y));           \
        using st = typename supertype<Float1, Float2, bf>::type;               \
        return st(bf(x) _op bf(y));                                            \
    }                                                                          \
                                                                               \
    template <typename Float1, typename Float2>                                \
    FLOATX_ATTRIBUTES FLOATX_INLINE typename std::enable_if<                   \
        float_traits<Float1>::is_runtime || float_traits<Float2>::is_runtime,  \
        typename supertype<                                                    \
            Float1, Float2,                                                    \
            decltype(typename float_traits<Float1>::backend_float() _op        \
                     typename float_traits<Float2>::backend_float())>::type>:: \
        type                                                                   \
        operator _op(const Float1& x, const Float2& y)                         \
    {                                                                          \
        using bf = decltype(typename float_traits<Float1>::backend_float(      \
            x) _op typename float_traits<Float2>::backend_float(y));           \
        using st = supertype<Float1, Float2, bf>;                              \
        return typename st::type(st::max_exp_bits(x, y),                       \
                                 st::max_sig_bits(x, y), bf(x) _op bf(y));     \
    }                                                                          \
                                                                               \
    template <typename Float1, typename Float2>                                \
    FLOATX_ATTRIBUTES FLOATX_INLINE                                            \
        typename std::enable_if<float_traits<Float1>::is_floatx ||             \
                                    float_traits<Float2>::is_floatx,           \
                                Float1&>::type                                 \
        operator _op##=(Float1& x, const Float2& y)                            \
    {                                                                          \
        return x = Float1(x _op y);                                            \
    }

ENABLE_ARITHMETIC_OPERATOR(+)
ENABLE_ARITHMETIC_OPERATOR(-)
ENABLE_ARITHMETIC_OPERATOR(*)
ENABLE_ARITHMETIC_OPERATOR(/)

#undef ENABLE_ARITHMETIC_OPERATOR


template <typename FloatX>
FLOATX_INLINE typename std::enable_if<float_traits<FloatX>::is_floatx,
                                      std::ostream&>::type&
operator<<(std::ostream& os, const FloatX& f) noexcept
{
    return os << typename float_traits<FloatX>::backend_float(f);
}


template <typename FloatX>
FLOATX_INLINE typename std::enable_if<float_traits<FloatX>::is_floatx,
                                      std::istream&>::type
operator>>(std::istream& is, FloatX& f) noexcept
{
    typename float_traits<FloatX>::backend_float tmp;
    is >> tmp;
    f = tmp;
    return is;
}


template <typename Float>
FLOATX_ATTRIBUTES FLOATX_INLINE
    std::bitset<sizeof(typename float_traits<Float>::backend_float)>
    bits(const Float& x) noexcept
{
    using bf = typename float_traits<Float>::backend_float;
    using bitset = std::bitset<sizeof(bf)>;
    bf val = bf(x);
    return *reinterpret_cast<bitset*>(&val);
}


namespace detail {


template <typename Float>
constexpr FLOATX_ATTRIBUTES FLOATX_INLINE
    typename float_traits<Float>::bits_type
    reinterpret_as_bits(Float val)
{
    return *reinterpret_cast<const typename float_traits<Float>::bits_type*>(
        &val);
}


template <typename Float>
constexpr FLOATX_ATTRIBUTES FLOATX_INLINE Float
reinterpret_bits_as(typename float_traits<Float>::bits_type bits)
{
    return *reinterpret_cast<const Float*>(&bits);
}


template <typename SignificandType>
constexpr FLOATX_ATTRIBUTES FLOATX_INLINE SignificandType
get_round_nearest_correction(SignificandType sig, SignificandType lsb_mask,
                             SignificandType after_lsb_mask,
                             SignificandType rest_mask)
{
    return (sig & after_lsb_mask) && ((sig & rest_mask) || (sig & lsb_mask));
}


FLOATX_ATTRIBUTES FLOATX_INLINE constexpr uint64_t
generate_rest_mask_fast_shift_less64(uint64_t MASK_AFTER_LSB)
{
    return (MASK_AFTER_LSB >= 1) ? (MASK_AFTER_LSB - UINT64_C(0x1))
                                 : UINT64_C(0x0000000000000000);
}


FLOATX_ATTRIBUTES FLOATX_INLINE uint64_t round_nearest(uint64_t mant,
                                                       uint16_t SHIFT)
{
    if (SHIFT >= 64)
        SHIFT = 63;  // that works to cover the case of down-shifts if the bit
                     // number 63 is never set. (since DATA >> 64 is all zero
                     // which is in that case equivalent to DATA >> 63)
    assert(!(mant & (UINT64_C(0x1) << 63)));

    // fast, no additional cases and simpler MASK generation.
    const uint64_t MASK_LSB = UINT64_C(0x0000000000000001) << SHIFT;
    const uint64_t MASK_AFTER_LSB = UINT64_C(0x0000000000000001) << (SHIFT - 1);
    const uint64_t MASK_REST =
        generate_rest_mask_fast_shift_less64(MASK_AFTER_LSB);

    uint64_t mant_res = mant >> SHIFT;

    if ((mant & MASK_AFTER_LSB) && ((mant & MASK_REST) || (mant & MASK_LSB))) {
        // round up if the bit after the lsb is set (>=0.5) and the number is
        // indeed bigger than >0.5 or if it is =0.5 and the TiesToEven rule
        // requires to round up.
        mant_res += 0x1;
    }

    mant_res = mant_res << SHIFT;

    return mant_res;
}


// CONSTANTS USED FOR BACKEND = DOUBLE
const uint64_t MASK_MANTISSA = UINT64_C(0x000FFFFFFFFFFFFF);
const uint64_t MASK_EXPONENT = UINT64_C(0x7FF0000000000000);
const uint64_t MASK_SIGN = UINT64_C(0x8000000000000000);
const uint64_t MASK_MANTISSA_OVERFLOW = UINT64_C(0x0010000000000000);
const uint64_t POS_INF_PATTERN = UINT64_C(0x7ff0000000000000);
const uint64_t NEG_INF_PATTERN = UINT64_C(0xfff0000000000000);
const uint64_t BACKEND_BIAS =
    UINT64_C(1023);  // that value is 2^(BACKEND_E-1)-1.
const int BACKEND_E = 11;
const int BACKEND_M = 52;


template <typename BitsType>
constexpr FLOATX_ATTRIBUTES FLOATX_INLINE bool is_nan_or_inf(BitsType number)
{
    return (number & MASK_EXPONENT) == MASK_EXPONENT;
}


template <typename BitsType, typename ExpType>
constexpr FLOATX_ATTRIBUTES FLOATX_INLINE bool is_small(BitsType e,
                                                        ExpType emin)
{
    return e < emin;
}


FLOATX_ATTRIBUTES FLOATX_INLINE uint64_t
convert_nan_or_inf_to_backend(const uint64_t number, const uint8_t M)
{
    // The following line delets any additional information that might be coded
    // in NAN bits. NAN bits towards the MSB of the mantissa that fit into the
    // target type are preserved.
    const uint64_t MASK_DELETE = UINT64_C(0xFFFFFFFFFFFFFFFF)
                                 << (BACKEND_M - M);

    // fix the nan (note that the following line does not affects +INF and -INF
    // by construction)
    return (number & MASK_DELETE);
}

FLOATX_ATTRIBUTES FLOATX_INLINE void convert_subnormal_mantissa_and_exp(
    const uint64_t number, const uint8_t M, const int16_t emin, const int e,
    uint64_t& mant, uint64_t& exp)
{
    int t = emin - e;

    // the hidden one might have a influence in rounding, hence add the hidden
    // one to the mantissa.
    mant = mant | MASK_MANTISSA_OVERFLOW;

    // Perform IEEE 754 rounding with TiesToEven.
    mant = round_nearest(mant, BACKEND_M - M + t);

    // Handle the case where the number is rounded to exact 0
    // since it is smaller (after rounding) than the smallest Subnormal / 2
    if (mant == 0x0) {
        exp = 0x0;
    }

    // remove the hidden one from the mantissa
    mant = mant & ~MASK_MANTISSA_OVERFLOW;
}

FLOATX_ATTRIBUTES FLOATX_INLINE void fix_too_large_mantissa(const int M, int& e,
                                                            uint64_t& mant,
                                                            uint64_t& exp)
{
    e += 1;
    // The following is the formula for the new exponent in the case the
    // mantissa was rounded up to a value that does not fit into the MANTISSA
    // field.
    exp = ((uint64_t)e + BACKEND_BIAS) << BACKEND_M;
    mant = UINT64_C(0x0000000000000000);
}

FLOATX_ATTRIBUTES FLOATX_INLINE uint64_t assemble_regular_number(
    const uint64_t sign_pattern, const uint64_t mant, const uint64_t exp)
{
    // ensure that the mantissa and exp fields to not contain bits at wrong
    // locations.
    assert((mant & ~MASK_MANTISSA) == 0x0);
    assert((exp & ~MASK_EXPONENT) == 0x0);

    // Assemble the number from the original sign and the current exp and mant
    // field.
    return (sign_pattern | exp | mant);
}

FLOATX_ATTRIBUTES FLOATX_INLINE uint64_t
assemble_inf_number(const uint64_t sign_pattern)
{
    // The code of rounding operates on the magnitude, here we still need to at
    // the right sign for the final number
    return sign_pattern | POS_INF_PATTERN;
}

// // functionality to get bit representations.
// bool get_sign_from_backend(const double d);

// // That functions return the bit representation embedded in a data word. A
// backend representation of type <E,M>
// // will return the full representation of 1+E+M bits in the LSB : LSB+1+E+M
// bit positions of the embedding dataword (e.g. uint64_t). uint16_t
// get_exponent_from_backend(const double d, const uint8_t E, const uint8_t M);
// uint64_t get_mantissa_from_backend(const double d, const uint8_t E, const
// uint8_t M); uint64_t get_fullbit_representation(const double d, const uint8_t
// E, const uint8_t M);

// // The reverse operation generates constructs a given number of exponent and
// mantissa bits.
// // Note, that the input is encoded into the embedding type as follows:
// //  exp:  bits (E-1) downto 0
// //  mant: bits (M-1) downto 0
// //  -> bits at higher positions are required to be 0. (?) or neglected?
// double construct_number(bool sign, uint16_t exp, uint64_t mant, const uint8_t
// E, const uint8_t M);

// functionality to get bit representations.
FLOATX_ATTRIBUTES FLOATX_INLINE bool get_sign_from_backend(const double d)
{
    uint64_t number = flx::detail::reinterpret_as_bits(d);
    return (number & MASK_SIGN);
}

FLOATX_ATTRIBUTES FLOATX_INLINE constexpr bool is_zero_or_nan_or_inf_exp(
    const uint64_t exp)
{
    return ((exp == 0x0) || (exp == MASK_EXPONENT));
}

FLOATX_ATTRIBUTES FLOATX_INLINE uint16_t
get_exponent_zero_or_nan_or_inf_exp(const uint64_t exp, const uint8_t E)
{
    uint16_t target_exp = (uint16_t)(exp >> BACKEND_M);
    // if it is an inf or nan delete any additional ones in the format.
    // (exponent requires E 1's)
    target_exp = target_exp & ((0x1 << E) - 1);

    // assert no bits are set at positions 15:E.
    // information is encoded only at positons E-1:0.
    assert(target_exp < (0x1 << E));
    return target_exp;
}

FLOATX_ATTRIBUTES FLOATX_INLINE uint16_t
get_exponent_regular_backend_exp(const uint64_t exp, const uint8_t E)
{
    // That is the double exponent.
    // Interpret the exponent.
    uint16_t target_exp = 0x0;
    int e = (exp >> BACKEND_M) - BACKEND_BIAS;

    // TARGET FORMAT (emax and emin depends on E)
    // IEEE 754 STANDARD
    int16_t emax = (0x1 << (E - 1)) - 1;
    int16_t emin = 1 - emax;

    // Target bias is the same as emax.
    if (e < emin) {
        // a regular case in the backend, but a subnormal in the target format.
        target_exp = 0x0;  // subnormals have a zero exponent.
    } else {
        // Encode the exponent in target format.
        target_exp = (uint16_t)(e + emax);
    }

    // assert no bits are set at positions 15:E.
    // information is encoded only at positons E-1:0.
    assert(target_exp < (0x1 << E));
    return target_exp;
}

FLOATX_ATTRIBUTES FLOATX_INLINE uint64_t
get_mantissa_zero_or_nan_or_inf_exp(const uint64_t mant, const uint8_t M)
{
    uint64_t ret = mant >> (BACKEND_M - M);

    assert(ret < (UINT64_C(0x1) << M));
    return ret;
}

FLOATX_ATTRIBUTES FLOATX_INLINE uint64_t get_mantissa_regular_backend_exp(
    const uint64_t exp, const uint64_t mant, const uint8_t E, const uint8_t M)
{
    // That is the double exponent.
    // Interpret the exponent.
    int e = (exp >> BACKEND_M) - BACKEND_BIAS;

    // TARGET FORMAT (emax and emin depends on E)
    // IEEE 754 STANDARD
    int16_t emax = (0x1 << (E - 1)) - 1;
    int16_t emin = 1 - emax;
    // Target bias is the same as emax.

    uint64_t ret;

    if (e < emin) {
        int t = emin - e;
        // Subnormal. The backend mantissa needs the hidden 1 that is visible in
        // the subnormal representation of the target format.
        ret = (mant | MASK_MANTISSA_OVERFLOW) >> (BACKEND_M - M + t);
    } else {
        ret = mant >> (BACKEND_M - M);
    }

    assert(ret < (UINT64_C(0x1) << M));
    return ret;
}

FLOATX_ATTRIBUTES FLOATX_INLINE uint16_t
get_exponent_from_backend(const double d, const uint8_t E)
{
    uint64_t number = flx::detail::reinterpret_as_bits(d);
    uint64_t exp = number & MASK_EXPONENT;

    // detects, zero, denormals, infs and nans in the backend double.
    if (is_zero_or_nan_or_inf_exp(exp)) {
        return get_exponent_zero_or_nan_or_inf_exp(exp, E);
    } else {
        return get_exponent_regular_backend_exp(exp, E);
    }
}

FLOATX_ATTRIBUTES FLOATX_INLINE uint64_t
get_mantissa_from_backend(const double d, const uint8_t E, const uint8_t M)
{
    uint64_t number = flx::detail::reinterpret_as_bits(d);
    uint64_t exp = number & MASK_EXPONENT;
    uint64_t mant = number & MASK_MANTISSA;

    if (is_zero_or_nan_or_inf_exp(exp)) {
        return get_mantissa_zero_or_nan_or_inf_exp(mant, M);
    } else {
        return get_mantissa_regular_backend_exp(exp, mant, E, M);
    }
}

FLOATX_ATTRIBUTES FLOATX_INLINE uint8_t
count_leading_zeros(const uint64_t data) noexcept
{
#ifdef USE_BUILTINS
#ifdef __CUDA_ARCH__
    return __clzll(data);
#else   // !__CUDA_ARCH__
    return __builtin_clzl(data);
#endif  // ?__CUDA_ARCH__
#else   // !USE_BUILTINS
    uint8_t t = 0u;  // t will be the number of zero bits on the left
    for (t = 0u; t < 64u; ++t) {
        if (data & (UINT64_C(0x1) << (63u - t))) {
            break;
        }
    }
    return t;
#endif  // ?USE_BUILTINS
}


FLOATX_ATTRIBUTES FLOATX_INLINE void construct_number_subormal(
    uint64_t& backend_exp, uint64_t& mant, const int16_t emin, const uint8_t M)
{
    // Zero and Subnormal.
    if (mant == UINT64_C(0x0)) {
        // real zero.
        backend_exp = 0x0;
        mant = 0x0;
    } else {
        // a subnormal in the target fromat, but result in a regular number in
        // the backend fromat.
        uint8_t t = count_leading_zeros(mant);
        t = t - (63 - M);

        // interpret exponent in the <E,M> format.

        int e = emin - t;

        // rewrite the exponent in the backend format.
        backend_exp = ((uint64_t)e + BACKEND_BIAS) << BACKEND_M;

        // mantissa.
        mant = mant << (BACKEND_M - M + t);
        mant = mant & ~MASK_MANTISSA_OVERFLOW;
    }
}

FLOATX_ATTRIBUTES FLOATX_INLINE void construct_number_nan_or_inf(
    uint64_t& backend_exp, uint64_t& mant, const uint8_t M)
{
    if (mant == 0x0) {
        // Inf
        backend_exp = MASK_EXPONENT;  // encode a backend inf.
    } else {
        // Nan
        backend_exp = MASK_EXPONENT;  // encode nan
        mant = mant << (BACKEND_M - M);
    }
}

FLOATX_ATTRIBUTES FLOATX_INLINE void construct_number_regular(
    uint64_t& backend_exp, uint64_t& mant, uint16_t const exp, int16_t emax,
    const uint8_t M)
{
    mant = mant << (BACKEND_M - M);

    // interpret exponent in the <E,M> format.
    int e = exp - emax;

    // rewrite the exponent in the backend format.
    backend_exp = ((uint64_t)e + BACKEND_BIAS) << BACKEND_M;
}

FLOATX_ATTRIBUTES FLOATX_INLINE double construct_number(bool sign, uint16_t exp,
                                                        uint64_t mant,
                                                        const uint8_t E,
                                                        const uint8_t M)
{
    uint64_t backend_exp = 0x0;

    // use emax as bias for the <E,M> format.
    int16_t emax = (0x1 << (E - 1)) - 1;
    int16_t emin = 1 - emax;

    if (exp == 0x0) {
        construct_number_subormal(backend_exp, mant, emin, M);
    } else if (exp == ((0x1 << E) - 0x1)) {
        construct_number_nan_or_inf(backend_exp, mant, M);
    } else {
        construct_number_regular(backend_exp, mant, exp, emax, M);
    }

    uint64_t sign_bit = MASK_SIGN;
    sign_bit *= sign;

    uint64_t number = sign_bit | backend_exp | mant;
    double res = reinterpret_bits_as<double>(number);
    return res;
}

FLOATX_ATTRIBUTES FLOATX_INLINE double construct_number(
    uint64_t fullbit_representation, const uint8_t E, const uint8_t M)
{
    bool sign = (fullbit_representation & (UINT64_C(0x1) << (E + M)));
    uint64_t exp =
        fullbit_representation & (((UINT64_C(0x1) << E) - UINT64_C(1)) << M);
    exp = exp >> M;
    uint64_t mant =
        fullbit_representation & ((UINT64_C(0x1) << M) - UINT64_C(1));
    return construct_number(sign, (uint16_t)exp, mant, E, M);
}

FLOATX_ATTRIBUTES FLOATX_INLINE uint64_t construct_fullbit_representation(
    bool sign, uint16_t exp, uint64_t mant, const uint8_t E, const uint8_t M)
{
    assert(exp < (0x1 << E));
    assert(mant < (UINT64_C(0x1) << M));

    uint64_t sign_bit = UINT64_C(0x1) << (E + M);
    sign_bit *= sign;

    uint64_t target_exp = (uint64_t)exp;
    target_exp = target_exp << M;

    // Note that the words have information encoded at different positions
    // [63:E+M+1]   free
    // E+M          sign_bit
    // E+M-1:M      target_exp
    // M-1:0        mantissa
    mant = sign_bit | target_exp | mant;

    return mant;
}

// That functions return the bit representation embedded in a data word. A
// backend representation of type <E,M> will return the full representation of
// 1+E+M bits in the LSB : LSB+1+E+M bit positions of the embedding dataword
// (e.g. uint64_t). Encoding of the result: [63:E+M+1]   free E+M
// sign_bit E+M-1:M      target_exp M-1:0        mantissa
FLOATX_ATTRIBUTES FLOATX_INLINE uint64_t
get_fullbit_representation(const double d, const uint8_t E, const uint8_t M)
{
    return construct_fullbit_representation(
        get_sign_from_backend(d), get_exponent_from_backend(d, E),
        get_mantissa_from_backend(d, E, M), E, M);
}

// Bitset wrappers.
template <uint8_t E>
FLOATX_ATTRIBUTES FLOATX_INLINE std::bitset<E> get_exponent_from_backend_BS(
    const double d)
{
    return std::bitset<E>(get_exponent_from_backend(d, E));
}

template <uint8_t E, uint8_t M>
FLOATX_ATTRIBUTES FLOATX_INLINE std::bitset<M> get_mantissa_from_backend_BS(
    const double d)
{
    return std::bitset<M>(get_mantissa_from_backend(d, E, M));
}

template <uint8_t E, uint8_t M>
FLOATX_ATTRIBUTES FLOATX_INLINE std::bitset<1 + E + M>
get_fullbit_representation_BS(const double d)
{
    return std::bitset<1 + E + M>(get_fullbit_representation(d, E, M));
}

template <uint8_t E, uint8_t M>
FLOATX_ATTRIBUTES FLOATX_INLINE double construct_number(bool sign,
                                                        std::bitset<E> exp,
                                                        std::bitset<M> mant)
{
    return construct_number(sign, exp.to_ulong(), mant.to_ulong(), E, M);
}

template <uint8_t E, uint8_t M>
FLOATX_ATTRIBUTES FLOATX_INLINE double construct_number(
    std::bitset<1 + E + M> fullbit_representation)
{
    return construct_number(fullbit_representation.to_ulong(), E, M);
}

#define ENABLE_EXTRACT_PART(_part)                                   \
    template <typename T>                                            \
    FLOATX_ATTRIBUTES FLOATX_INLINE uint64_t extract_##_part##_bits( \
        const T& val) noexcept                                       \
    {                                                                \
        return (*reinterpret_cast<const uint64_t*>(&val) >>          \
                float_traits<T>::_part##_pos) &                      \
               float_traits<T>::_part##_mask;                        \
    }

ENABLE_EXTRACT_PART(sgn);
ENABLE_EXTRACT_PART(exp);
ENABLE_EXTRACT_PART(sig);

#undef ENABLE_EXTRACT_PART

template <typename ConcreteFloatX>
class floatx_base {
private:
    using backend_float = typename float_traits<ConcreteFloatX>::backend_float;
    using bits_type = typename float_traits<ConcreteFloatX>::bits_type;

public:
    FLOATX_ATTRIBUTES floatx_base(const backend_float& value) noexcept
        : value_(value)
    {}

    FLOATX_ATTRIBUTES void initialize() noexcept
    {
        value_ = this->enforce_rounding(value_);
    }

    FLOATX_ATTRIBUTES constexpr operator backend_float() const noexcept
    {
        return value_;
    }

    template <typename T>
    FLOATX_ATTRIBUTES constexpr operator T() const noexcept
    {
        return T(value_);
    }

private:
    FLOATX_ATTRIBUTES const ConcreteFloatX& self() const noexcept
    {
        return *static_cast<const ConcreteFloatX*>(this);
    }

    FLOATX_ATTRIBUTES ConcreteFloatX& self() noexcept
    {
        return *static_cast<ConcreteFloatX*>(this);
    }

    static constexpr auto backend_sig_pos =
        float_traits<backend_float>::sig_pos;
    static constexpr auto backend_exp_pos =
        float_traits<backend_float>::exp_pos;
    static constexpr auto backend_sgn_pos =
        float_traits<backend_float>::sgn_pos;
    static constexpr auto backend_sig_mask =
        float_traits<backend_float>::sig_mask << backend_sig_pos;
    static constexpr auto backend_exp_mask =
        float_traits<backend_float>::exp_mask << backend_exp_pos;
    static constexpr auto backend_sig_bits =
        float_traits<backend_float>::sig_bits;
    static constexpr auto backend_exp_bits =
        float_traits<backend_float>::exp_bits;
    static constexpr auto backend_bias = float_traits<backend_float>::bias;
    static constexpr auto backend_sig_overflow_mask =
        (float_traits<backend_float>::sig_mask + 1) << backend_sig_pos;
    static constexpr auto backend_sgn_mask =
        float_traits<backend_float>::sgn_mask << backend_sgn_pos;

    FLOATX_ATTRIBUTES
    backend_float enforce_rounding(backend_float value) const noexcept
    {
        const auto exp_bits = get_exp_bits(self());
        const auto sig_bits = get_sig_bits(self());
        bits_type bits = reinterpret_as_bits(value);
        auto sig = (bits & backend_sig_mask) >> backend_sig_pos;
        auto raw_exp = bits & backend_exp_mask;
        const auto sgn = bits & backend_sgn_mask;

        int exp = (raw_exp >> backend_exp_pos) - backend_bias;

        const int emax = (1 << (exp_bits - 1)) - 1;
        const int emin = 1 - emax;

        if (is_nan_or_inf(bits)) {
            bits = convert_nan_or_inf_to_backend(bits, sig_bits);
        } else {
            if (is_small(exp, emin)) {
                convert_subnormal_mantissa_and_exp(bits, sig_bits, emin, exp,
                                                   sig, raw_exp);
            } else {
                sig = round_nearest(sig, backend_sig_bits - sig_bits);
            }
            if (significand_is_out_of_range(sig)) {
                fix_too_large_mantissa(sig_bits, exp, sig, raw_exp);
            }
            if (exponent_is_out_of_range(exp, emax)) {
                bits = assemble_inf_number(sgn);
            } else {
                bits = assemble_regular_number(sgn, sig, raw_exp);
            }
        }

        return reinterpret_bits_as<backend_float>(bits);
    }

    static constexpr FLOATX_ATTRIBUTES bits_type
    reinterpret_as_bits(backend_float val)
    {
        return *reinterpret_cast<const bits_type*>(&val);
    }

    static constexpr FLOATX_ATTRIBUTES bool significand_is_out_of_range(
        bits_type sig)
    {
        return sig >= backend_sig_overflow_mask;
    }

    static constexpr FLOATX_ATTRIBUTES bool exponent_is_out_of_range(int exp,
                                                                     int emax)
    {
        return exp > emax;
    }

protected:
    backend_float value_;
};


}  // namespace detail


template <typename Float>
FLOATX_ATTRIBUTES FLOATX_INLINE std::string bitstring(const Float& x) noexcept
{
    using bf = typename float_traits<Float>::backend_float;
    const uint64_t one = UINT64_C(1);
    const char map[] = {'0', '1'};
    const int eb = get_exp_bits(x);
    const int sb = get_sig_bits(x);
    const int beb = get_exp_bits(bf(x));
    const int bsb = get_sig_bits(bf(x));

    std::string s(sb + eb + 3, '-');
    auto sgn = detail::extract_sgn_bits(bf(x));
    auto exp = detail::extract_exp_bits(bf(x));
    auto sig = detail::extract_sig_bits(bf(x));

    int i = 0;
    s[i++] = map[bool(sgn & UINT64_C(1))];  // sign bit
    ++i;  // leave '-' between sign and exponent parts
    s[i++] = map[bool(exp & (one << (beb - 1)))];  // bias bit
    for (auto mask = (one << (eb - 2)); mask > 0; mask >>= 1) {
        s[i++] = map[bool(exp & mask)];
    }
    ++i;  // leave '-' between exponent and significand parts
    for (auto mask = (one << (bsb - 1)); i < s.size(); mask >>= 1) {
        s[i++] = map[bool(sig & mask)];
    }
    return s;
}


};  // namespace flx


#undef FLOATX_ATTRIBUTES
#undef FLOATX_INLINE
#undef USE_BUILTINS


#endif  // FLOATX_FLOATX_HPP_
