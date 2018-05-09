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

#include <bitset>


#include <gtest/gtest.h>
#include <floatx.hpp>


namespace {


TEST(FloatxConversionTest, PreservesDoublePrecision)
{
    const double val = 1.0 + 1e-15;
    EXPECT_EQ(val, double(flx::floatx<11, 52>(val)));
}

TEST(FloatxConversionTest, LowersPrecision)
{
    const double val = 1.0 + 1e-15;
    EXPECT_NE(val, double(flx::floatx<8, 23>(val)));  // round to float
}


TEST(FloatxConversionTest, HandlesDenormals)
{
    EXPECT_EQ(0.25, double(flx::floatx<2, 3>(0.25)));
    EXPECT_EQ(0.75, double(flx::floatx<2, 3>(0.75)));
}


TEST(FloatxConversionTest, ConvertsBetweenFloatX)
{
    const double val = 1.0 + 1e-15;
    flx::floatx<11, 52> d_val(val);
    flx::floatx<8, 23> s_val(d_val);
    EXPECT_NE(val, double(s_val));
    EXPECT_EQ(float(val), float(s_val));
}

TEST(FloatxConversionTest, ConvertsToBits)
{
    const double val = 1.0 + 1e-15;
    ::testing::StaticAssertTypeEq<std::bitset<sizeof(val)>,
                                  decltype(bits(flx::floatx<11, 52>(val)))>();
    ::testing::StaticAssertTypeEq<std::bitset<sizeof(val)>,
                                  decltype(bits(flx::floatx<8, 23>(val)))>();
    EXPECT_EQ(flx::bits(val), bits(flx::floatx<11, 52>(val)));
    EXPECT_NE(flx::bits(val), bits(flx::floatx<8, 23>(val)));
}

TEST(FloatxConversionTest, ConvertsToString)
{
    flx::floatx<4, 5> val1 = 1.0;
    EXPECT_EQ("0-0111-00000", bitstring(val1));
    flx::floatx<3, 2> val2 = 1.75;
    EXPECT_EQ("0-011-11", bitstring(val2));
    flx::floatx<5, 7> val3 = 0.0;
    EXPECT_EQ("0-00000-0000000", bitstring(val3));
}

TEST(FloatxrConversionTest, PreservesDoublePrecision)
{
    const double val = 1.0 + 1e-15;
    EXPECT_EQ(val, double(flx::floatxr<>(11, 52, val)));
}

TEST(FloatxrConversionTest, LowersPrecision)
{
    const double val = 1.0 + 1e-15;
    EXPECT_NE(val, double(flx::floatxr<>(8, 23, val)));  // round to float
}

TEST(FloatxrConversionTest, InheritsPrecision)
{
    const double val = 1.0 + 1e-15;
    EXPECT_EQ(val, double(flx::floatxr<>(val)));
}

TEST(FloatxrConversionTest, ChangesPrecision)
{
    const double val = 1.0 + 1e-15;
    flx::floatxr<> fxr_val(val);
    fxr_val.set_precision(8, 23);
    EXPECT_NE(val, double(fxr_val));
    EXPECT_EQ(float(val), float(fxr_val));
}

TEST(FloatxrConversionTest, ConvertsBetweenFloatX)
{
    const double val = 1.0 + 1e-15;
    flx::floatx<11, 52> d_val(val);
    flx::floatxr<> s_val(8, 23, d_val);
    EXPECT_NE(val, double(s_val));
    EXPECT_EQ(float(val), float(s_val));
}

TEST(FloatxrConversionTest, ConvertsToBits)
{
    const double val = 1.0 + 1e-15;
    ::testing::StaticAssertTypeEq<std::bitset<sizeof(val)>,
                                  decltype(bits(flx::floatxr<>(val)))>();
    EXPECT_EQ(flx::bits(val), bits(flx::floatxr<>(val)));
    EXPECT_NE(flx::bits(val), bits(flx::floatxr<>(8, 23, val)));
}


TEST(FloatxrConversionTest, ConvertsToString)
{
    flx::floatxr<> val1(4, 5, 1.0);
    EXPECT_EQ("0-0111-00000", bitstring(val1));
    flx::floatxr<> val2(3, 2, 1.75);
    EXPECT_EQ("0-011-11", bitstring(val2));
    flx::floatxr<> val3(5, 7, 0.0);
    EXPECT_EQ("0-00000-0000000", bitstring(val3));
}

}  // namespace
