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

// #include <cmath> // defines NAN
// #define nan NAN
#define nan double(0.0 / 0.0)
#define inf double(1.0 / 0.0)

namespace {

void show(double d)
{
    printf("%.20e\t", d);
    uint64_t u = flx::detail::reinterpret_as_bits(d);
    printf("0x%016llx\t", static_cast<long long int>(u));
    std::cout << std::bitset<64>(u) << std::endl;
}


// System representation of nan's.
TEST(FloatxNanInfTest, system_nans)
{
    double constnan = 0.0 / 0.0;
    printf("constnan: ");
    show((double)constnan);

    double zero;
    // try to prevent the compiler to figure out what
    // dynamicnan should be in the compile time
    *(double*)memset(&zero, ~0, sizeof(zero)) = 0.0;
    double dynamicnan = zero / zero;
    printf("dynamicnan: ");
    show((double)dynamicnan);

    EXPECT_NE(constnan, dynamicnan);  // holds only for NANs
    EXPECT_NE(constnan, nan);         // holds only for NANs
    EXPECT_NE(dynamicnan, nan);       // holds only for NANs
}

// See Intel 64 and IA-32 Architectures Software Developer's Manual
// Vol. 1, Appendix E Sect. 4.2.2 Table E-1 for a discussion of a
// type of NaN returned for an invalid operation (e.g., 0/0).  It
// seems that always a particular encoding ("QNaN indefinite") is
// used in such cases, but what happens generally (see TODOs below)?

// A NAN CASE
TEST(FloatxNanInfTest, cast_nans)
{
    using T1 = flx::floatx<2, 3>;
    using T2 = flx::floatx<10, 50>;
    T1 a = 0.0 / 0.0;
    T2 b = 0.0;
    b = a;

    double constnan =
        nan;  // note, the way how that nan is generated is relevant!

    EXPECT_NE(a, a);    // holds only for NANs
    EXPECT_NE(a, nan);  // holds only for NANs
    // TODO: is the following expectation true generally?
    EXPECT_EQ(*reinterpret_cast<uint64_t*>(&a),
              *reinterpret_cast<uint64_t*>(&constnan));

    EXPECT_NE(b, b);    // holds only for NANs
    EXPECT_NE(b, nan);  // holds only for NANs
    // TODO: is the following expectation true generally?
    EXPECT_EQ(*reinterpret_cast<uint64_t*>(&b),
              *reinterpret_cast<uint64_t*>(&constnan));

    // Differnt bit represenations for nans
    // TODO: is the following expectation true generally?
    EXPECT_EQ(*reinterpret_cast<uint64_t*>(&b),
              *reinterpret_cast<uint64_t*>(&a));
}

// A NAN CASE
TEST(FloatxNanInfTest, DIV_2_47_simple)
{
    using T = flx::floatx<2, 47>;
    T a = -(
        7.105427e-15 / 2 -
        1e-17);  // a bit smaller than half of the smallest subnormal in <2,47>
    T b = -(7.105427e-15 / 12.0);
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.00000000000000000000);
    EXPECT_EQ(double(b), 0.00000000000000000000);
    EXPECT_NE(c, c);    // holds only for NANs
    EXPECT_NE(c, nan);  // holds only for NANs

    double zero;
    // try to prevent the compiler to figure out what
    // dynamicnan should be in the compile time
    *(double*)memset(&zero, ~0, sizeof(zero)) = 0.0;
    double dynamicnan = zero / zero;

    EXPECT_NE(c, dynamicnan);  // holds only for NANs
    // TODO: is the following expectation true generally?
    EXPECT_EQ(*reinterpret_cast<uint64_t*>(&c),
              *reinterpret_cast<uint64_t*>(&dynamicnan));
}

// A REGULAR CASE (fixing in subnormal does not cause the inf case here)
TEST(FloatxNanInfTest, DIV_3_3_simple)
{
    using T = flx::floatx<3, 3>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 3.43750000000000000000e-01);
    EXPECT_EQ(double(b), 1.25000000000000000000e-01);
    EXPECT_EQ(double(c), 2.7500000000000000000e-00);
}

// A INF CASE.
TEST(FloatxNanInfTest, DIV_3_3_simple_inf)
{
    using T = flx::floatx<3, 3>;
    T a = 0.33333333333333331483;
    T b =
        (0.03125000000000000000 / 2 -
         1e-17);  // a bit smaller than half of the smallest subnormal in <3,3>
    T c = 0;
    c = a / b;

    // printf("a: 			"); show((double)a);
    // printf("b: 			"); show((double)b);
    // printf("c:   		"); show((double)c);
    // printf("inf: 		"); show((double)inf);

    EXPECT_EQ(double(a), 3.43750000000000000000e-01);
    EXPECT_EQ(double(b), 00000000000000000000);
    EXPECT_EQ(double(c), inf);
}

}  // namespace
