FloatX (Float eXtended)
=======================

FloatX is a header-only C++ library which extends floating point types beyond
the native single and double (and on some hardware half) precision types. It
provides template types which allow the user to select the number of bits used
for the exponent and significand parts of the floating point number.
The idea of FloatX is based on the FlexFloat library, but, instead of
implementing the functionality in C and providing C++ wrappers, FloatX is
written completely in C++, which makes it more natural to the end user.
In addition, FloatX provides a superset of FlexFloat's functionalities.

Features
--------

This section lists the functionalities provided by FloatX. Functionalities that
are also provided by FlexFloat have (_flexfloat_) appended to the description.
In addition, functionalities that are planned, but are not yet implemented are
also listed and have (__TODO__) appended.

*   header-only library, without a compiled component, and heavy inlining,
    resulting in relatively high performance
*   `floatx<exp_bits, sig_bits, backend_float>` class template, which allows
    emulation of non-native types with `exp_bits` exponent bits and `sig_bits`
    significand bits using a natively supported `backend_float` type to perform
    arithmetic operations (_flexfloat_ - provides a similar functionality in
    the C++ wrapper, but the memory consumption of the flexfloat C++ class was
    suboptimal).
    (__TODO__: the only supported backend native type is double at the moment.)
*   `floatxr<backend_float>` class template, which provides the same
    functionality as `floatx`, but allows changing the precision of the type
    at runtime. This class is easier to experiment with, but is not as
    efficient as `floatx` in both the performance, as well as the memory
    consumption. (_flexfloat_ - provides a type that has a comparable memory
    consumption with the precision selectable at runtime in the C library only)
*   conversions between builtin types and `floatx`
    (_flexfloat_ - had a bug where NaN can be cast to Inf during conversion)
*   assignments on `floatx` and `floatxr` types (_flexfloat_)
*   relational operations on `floatx` and `floatxr` types
    (_flexfloat_ - did not handle NaN properly)
*   relational operations between different types
*   arithmetic operations on `floatx` and `floatxr` types (_flexfloat_)
*   arithmetic operations between different types with implicit type promotion
*   `std::ostream& operator <<(std::ostream&, floatx[r])` (_flexfloat_)
*   `std::istream& operator >>(std::istream&, floatx[r])`
*   CUDA support
*   conversion to `std::bitset` (__TODO__)
*   conversion to `std::string` (__TODO__)
*   optional operation counters (requires a compiled runtime library)
    (__TODO__)
*   automatic deduction of the smallest native type which can fit the requested
    number of exponent and significand bits (__TODO__)
*   optimized performance in case the requested type matches the backend type
    (_flexfloat_ - was only enabled for single, inconsistently with the rest of
    the library) (__TODO__)
*   compressed storage which allows to reduce memory footprint (__TODO__)
*   rounding modes other than "round to nearest, ties to even" (__TODO__)

What FloatX is NOT
------------------

FloatX does not implement arbitrary floating point types. The only supported
types are "subtypes" of those natively supported by the hardware.
In case you need implementations of larger types, consider using the SoftFloat
library.

FloatX __emulates__ the types of custom precision, subject to the constraints
above, and, while trying to achieve as high performance as possible, it is
__not__ capable of magically delivering better performance than natively
supported types. Thus, do not expect `floatx<3, 3>` to consume less memory, or
be faster than e.g. `float`, though `floatx<11, 52>` should deliver similar
performance as `double`.

That being said, it is not likely that FloatX will be useful in production
codes. On the other hand, it can be handy in research projects which aim to
study the effects of using different precisions.

Installation
------------

To use the library, just make sure that a directory containing `floatx.hpp` is
in your include path (here, it is in `src/` subdirectory).

Alternatively, if you are using CMake, a `CMakeLists.txt` file is provided.
You can download the repository into your project and use the following code to
depend on the floatx target:

```
add_subdirectory(floatx)
target_add_library(my_target PRIVATE floatx)
```

### Building the examples / unit tests

A standard CMake command line sequence should do:

```
mkdir build && cd build && cmake .. && make
```

To run all the tests:

```
make test
```

This will (hopefully) output a summary of the form:

```
test_<testname>............ Passed
```

To run only one of the tests (and see more detail output):

```
./test/<testname>
```


Examples
--------

TODO

Acknowledgments
---------------

TODO
