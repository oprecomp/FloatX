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


namespace {


TEST(FloatxRelOpsTest, Equal)
{
    using doublex = flx::floatx<11, 52>;
    using floatx = flx::floatx<8, 23>;
    const double val1 = 1.0 + 1e-15;
    const double val2 = 1.0 + 2e-15;
    EXPECT_TRUE(doublex(val1) == doublex(val1));
    EXPECT_FALSE(doublex(val1) == doublex(val2));
    EXPECT_TRUE(floatx(val1) == floatx(val1));
    EXPECT_TRUE(floatx(val1) == floatx(val2));    // due to rounding
    EXPECT_FALSE(floatx(val1) == doublex(val1));  // due to rounding
    EXPECT_FALSE(floatx(val1) == doublex(val2));  // due to rounding
}


TEST(FloatxRelOpsTest, NotEqual)
{
    using doublex = flx::floatx<11, 52>;
    using floatx = flx::floatx<8, 23>;
    const double val1 = 1.0 + 1e-15;
    const double val2 = 1.0 + 2e-15;
    EXPECT_FALSE(doublex(val1) != doublex(val1));
    EXPECT_TRUE(doublex(val1) != doublex(val2));
    EXPECT_FALSE(floatx(val1) != floatx(val1));
    EXPECT_FALSE(floatx(val1) != floatx(val2));  // due to rounding
    EXPECT_TRUE(floatx(val1) != doublex(val1));  // due to rounding
    EXPECT_TRUE(floatx(val1) != doublex(val2));  // due to rounding
}


TEST(FloatxRelOpsTest, LessThan)
{
    using doublex = flx::floatx<11, 52>;
    using floatx = flx::floatx<8, 23>;
    const double val1 = 1.0 + 1e-15;
    const double val2 = 1.0 + 2e-15;
    EXPECT_FALSE(doublex(val1) < doublex(val1));
    EXPECT_FALSE(doublex(val2) < doublex(val1));
    EXPECT_TRUE(doublex(val1) < doublex(val2));
    EXPECT_FALSE(floatx(val1) < floatx(val1));
    EXPECT_FALSE(floatx(val2) < floatx(val1));
    EXPECT_FALSE(floatx(val1) < floatx(val2));  // due to rounding
    EXPECT_TRUE(floatx(val1) < doublex(val1));  // due to rounding
    EXPECT_TRUE(floatx(val2) < doublex(val1));  // due to rounding
    EXPECT_TRUE(floatx(val1) < doublex(val2));  // due to rounding
}


TEST(FloatxRelOpsTest, LessOrEqual)
{
    using doublex = flx::floatx<11, 52>;
    using floatx = flx::floatx<8, 23>;
    const double val1 = 1.0 + 1e-15;
    const double val2 = 1.0 + 2e-15;
    EXPECT_TRUE(doublex(val1) <= doublex(val1));
    EXPECT_FALSE(doublex(val2) <= doublex(val1));
    EXPECT_TRUE(doublex(val1) <= doublex(val2));
    EXPECT_TRUE(floatx(val1) <= floatx(val1));
    EXPECT_TRUE(floatx(val2) <= floatx(val1));  // due to rounding
    EXPECT_TRUE(floatx(val1) <= floatx(val2));
    EXPECT_TRUE(floatx(val1) <= doublex(val1));
    EXPECT_TRUE(floatx(val2) <= doublex(val1));  // due to rounding
    EXPECT_TRUE(floatx(val1) <= doublex(val2));
}


TEST(FloatxRelOpsTest, GreaterThan)
{
    using doublex = flx::floatx<11, 52>;
    using floatx = flx::floatx<8, 23>;
    const double val1 = 1.0 + 1e-15;
    const double val2 = 1.0 + 2e-15;
    EXPECT_FALSE(doublex(val1) > doublex(val1));
    EXPECT_TRUE(doublex(val2) > doublex(val1));
    EXPECT_FALSE(doublex(val1) > doublex(val2));
    EXPECT_FALSE(floatx(val1) > floatx(val1));
    EXPECT_FALSE(floatx(val2) > floatx(val1));  // due to rounding
    EXPECT_FALSE(floatx(val1) > floatx(val2));
    EXPECT_FALSE(floatx(val1) > doublex(val1));
    EXPECT_FALSE(floatx(val2) > doublex(val1));  // due to rounding
    EXPECT_FALSE(floatx(val1) > doublex(val2));
}


TEST(FloatxRelOpsTest, GreaterOrEqual)
{
    using doublex = flx::floatx<11, 52>;
    using floatx = flx::floatx<8, 23>;
    const double val1 = 1.0 + 1e-15;
    const double val2 = 1.0 + 2e-15;
    EXPECT_TRUE(doublex(val1) >= doublex(val1));
    EXPECT_TRUE(doublex(val2) >= doublex(val1));
    EXPECT_FALSE(doublex(val1) >= doublex(val2));
    EXPECT_TRUE(floatx(val1) >= floatx(val1));
    EXPECT_TRUE(floatx(val2) >= floatx(val1));    // due to rounding
    EXPECT_TRUE(floatx(val1) >= floatx(val2));    // due to rounding
    EXPECT_FALSE(floatx(val1) >= doublex(val1));  // due to rounding
    EXPECT_FALSE(floatx(val2) >= doublex(val1));  // due to rounding
    EXPECT_FALSE(floatx(val1) >= doublex(val2));
}


TEST(FloatxrRelOpsTest, Equal)
{
    auto doublex = [](double a) { return flx::floatxr<>(11, 52, a); };
    auto floatx = [](double a) { return flx::floatxr<>(8, 23, a); };
    const double val1 = 1.0 + 1e-15;
    const double val2 = 1.0 + 2e-15;
    EXPECT_TRUE(doublex(val1) == doublex(val1));
    EXPECT_FALSE(doublex(val1) == doublex(val2));
    EXPECT_TRUE(floatx(val1) == floatx(val1));
    EXPECT_TRUE(floatx(val1) == floatx(val2));    // due to rounding
    EXPECT_FALSE(floatx(val1) == doublex(val1));  // due to rounding
    EXPECT_FALSE(floatx(val1) == doublex(val2));  // due to rounding
}


TEST(FloatxrRelOpsTest, NotEqual)
{
    auto doublex = [](double a) { return flx::floatxr<>(11, 52, a); };
    auto floatx = [](double a) { return flx::floatxr<>(8, 23, a); };
    const double val1 = 1.0 + 1e-15;
    const double val2 = 1.0 + 2e-15;
    EXPECT_FALSE(doublex(val1) != doublex(val1));
    EXPECT_TRUE(doublex(val1) != doublex(val2));
    EXPECT_FALSE(floatx(val1) != floatx(val1));
    EXPECT_FALSE(floatx(val1) != floatx(val2));  // due to rounding
    EXPECT_TRUE(floatx(val1) != doublex(val1));  // due to rounding
    EXPECT_TRUE(floatx(val1) != doublex(val2));  // due to rounding
}


TEST(FloatxrRelOpsTest, LessThan)
{
    auto doublex = [](double a) { return flx::floatxr<>(11, 52, a); };
    auto floatx = [](double a) { return flx::floatxr<>(8, 23, a); };
    const double val1 = 1.0 + 1e-15;
    const double val2 = 1.0 + 2e-15;
    EXPECT_FALSE(doublex(val1) < doublex(val1));
    EXPECT_FALSE(doublex(val2) < doublex(val1));
    EXPECT_TRUE(doublex(val1) < doublex(val2));
    EXPECT_FALSE(floatx(val1) < floatx(val1));
    EXPECT_FALSE(floatx(val2) < floatx(val1));
    EXPECT_FALSE(floatx(val1) < floatx(val2));  // due to rounding
    EXPECT_TRUE(floatx(val1) < doublex(val1));  // due to rounding
    EXPECT_TRUE(floatx(val2) < doublex(val1));  // due to rounding
    EXPECT_TRUE(floatx(val1) < doublex(val2));  // due to rounding
}


TEST(FloatxrRelOpsTest, LessOrEqual)
{
    auto doublex = [](double a) { return flx::floatxr<>(11, 52, a); };
    auto floatx = [](double a) { return flx::floatxr<>(8, 23, a); };
    const double val1 = 1.0 + 1e-15;
    const double val2 = 1.0 + 2e-15;
    EXPECT_TRUE(doublex(val1) <= doublex(val1));
    EXPECT_FALSE(doublex(val2) <= doublex(val1));
    EXPECT_TRUE(doublex(val1) <= doublex(val2));
    EXPECT_TRUE(floatx(val1) <= floatx(val1));
    EXPECT_TRUE(floatx(val2) <= floatx(val1));  // due to rounding
    EXPECT_TRUE(floatx(val1) <= floatx(val2));
    EXPECT_TRUE(floatx(val1) <= doublex(val1));
    EXPECT_TRUE(floatx(val2) <= doublex(val1));  // due to rounding
    EXPECT_TRUE(floatx(val1) <= doublex(val2));
}


TEST(FloatxrRelOpsTest, GreaterThan)
{
    auto doublex = [](double a) { return flx::floatxr<>(11, 52, a); };
    auto floatx = [](double a) { return flx::floatxr<>(8, 23, a); };
    const double val1 = 1.0 + 1e-15;
    const double val2 = 1.0 + 2e-15;
    EXPECT_FALSE(doublex(val1) > doublex(val1));
    EXPECT_TRUE(doublex(val2) > doublex(val1));
    EXPECT_FALSE(doublex(val1) > doublex(val2));
    EXPECT_FALSE(floatx(val1) > floatx(val1));
    EXPECT_FALSE(floatx(val2) > floatx(val1));  // due to rounding
    EXPECT_FALSE(floatx(val1) > floatx(val2));
    EXPECT_FALSE(floatx(val1) > doublex(val1));
    EXPECT_FALSE(floatx(val2) > doublex(val1));  // due to rounding
    EXPECT_FALSE(floatx(val1) > doublex(val2));
}


TEST(FloatxrRelOpsTest, GreaterOrEqual)
{
    auto doublex = [](double a) { return flx::floatxr<>(11, 52, a); };
    auto floatx = [](double a) { return flx::floatxr<>(8, 23, a); };
    const double val1 = 1.0 + 1e-15;
    const double val2 = 1.0 + 2e-15;
    EXPECT_TRUE(doublex(val1) >= doublex(val1));
    EXPECT_TRUE(doublex(val2) >= doublex(val1));
    EXPECT_FALSE(doublex(val1) >= doublex(val2));
    EXPECT_TRUE(floatx(val1) >= floatx(val1));
    EXPECT_TRUE(floatx(val2) >= floatx(val1));    // due to rounding
    EXPECT_TRUE(floatx(val1) >= floatx(val2));    // due to rounding
    EXPECT_FALSE(floatx(val1) >= doublex(val1));  // due to rounding
    EXPECT_FALSE(floatx(val2) >= doublex(val1));  // due to rounding
    EXPECT_FALSE(floatx(val1) >= doublex(val2));
}


}  // namespace
