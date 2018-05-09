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

#include <gtest/gtest.h>
#include <floatx.hpp>


#include <algorithm>
#include <complex>
#include <tuple>
#include <vector>


namespace {


using doublex = flx::floatx<11, 52>;
using floatx = flx::floatx<8, 23>;


TEST(Tuple, CanCreateFloatXTuple)
{
    auto tpl = std::make_tuple(doublex{3.2}, floatx{5.2});

    ASSERT_NEAR(std::get<0>(tpl), 3.2, 1e-15);
    ASSERT_NEAR(std::get<1>(tpl), 5.2, 1e-6);
}


TEST(TupleVector, CanCreateVectorOfTuples)
{
    std::vector<std::tuple<doublex, floatx>> vec{
        std::make_tuple(doublex{3.2}, floatx{5.2}),
        std::make_tuple(doublex{0.5}, floatx{1.2})};

    ASSERT_NEAR(std::get<0>(vec[0]), 3.2, 1e-15);
    ASSERT_NEAR(std::get<1>(vec[0]), 5.2, 1e-6);
    ASSERT_NEAR(std::get<0>(vec[1]), 0.5, 1e-15);
    ASSERT_NEAR(std::get<1>(vec[1]), 1.2, 1e-6);
}


TEST(TupleVector, CanIterateThroughVector)
{
    std::vector<std::tuple<doublex, floatx>> vec{
        std::make_tuple(doublex{3.2}, floatx{5.2}),
        std::make_tuple(doublex{0.5}, floatx{1.2})};

    for (auto& elem : vec) {
        std::get<0>(elem) += 1;
    }

    ASSERT_NEAR(std::get<0>(vec[0]), 4.2, 1e-15);
    ASSERT_NEAR(std::get<1>(vec[0]), 5.2, 1e-6);
    ASSERT_NEAR(std::get<0>(vec[1]), 1.5, 1e-15);
    ASSERT_NEAR(std::get<1>(vec[1]), 1.2, 1e-6);
}


TEST(TupleVector, CanSortTupleVector)
{
    std::vector<std::tuple<doublex, floatx>> vec{
        std::make_tuple(doublex{3.2}, floatx{5.2}),
        std::make_tuple(doublex{0.5}, floatx{1.2})};

    std::sort(begin(vec), end(vec));

    ASSERT_NEAR(std::get<0>(vec[0]), 0.5, 1e-15);
    ASSERT_NEAR(std::get<1>(vec[0]), 1.2, 1e-6);
    ASSERT_NEAR(std::get<0>(vec[1]), 3.2, 1e-15);
    ASSERT_NEAR(std::get<1>(vec[1]), 5.2, 1e-6);
}


// NOTE: this is non-standard behavior, a conformant implementation is allowed
// to have undefined behavior for std::complex<flx::floatx<exp, sig>>
TEST(Complex, CanCreateComplexFloatX)
{
    std::complex<floatx> a(3.2, 2.5);

    ASSERT_NEAR(a.real(), 3.2, 1e-7);
    ASSERT_NEAR(a.imag(), 2.5, 1e-7);
}


TEST(Complex, CanAddComplexFloatX)
{
    std::complex<floatx> a(3.2, 2.5);
    std::complex<floatx> b(2.3, 1.4);

    auto res = a + b;

    ASSERT_NEAR(res.real(), 5.5, 1e-7);
    ASSERT_NEAR(res.imag(), 3.9, 1e-7);
}


TEST(Complex, CanSubstractComplexFloatX)
{
    std::complex<floatx> a(3.2, 2.5);
    std::complex<floatx> b(2.3, 1.4);

    auto res = a - b;

    ASSERT_NEAR(res.real(), 0.9, 1e-7);
    ASSERT_NEAR(res.imag(), 1.1, 1e-7);
}


TEST(Complex, CanMultiplyComplexFloatX)
{
    std::complex<floatx> a(3.0, 2.0);
    std::complex<floatx> b(2.0, 1.0);

    auto res = a * b;

    ASSERT_NEAR(res.real(), 4.0, 1e-7);
    ASSERT_NEAR(res.imag(), 7.0, 1e-7);
}


TEST(Complex, CanDivideComplexFloatX)
{
    std::complex<floatx> a(3.0, 2.0);
    std::complex<floatx> b(2.0, 1.0);

    auto res = a / b;

    ASSERT_NEAR(res.real(), 1.6, 1e-7);
    ASSERT_NEAR(res.imag(), 0.2, 1e-7);
}


}  // namespace
