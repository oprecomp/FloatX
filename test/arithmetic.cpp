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


TEST(FloatxArithmeticTest, ResultHasCorrectType)
{
    using doublex = flx::floatx<11, 52>;
    using floatx = flx::floatx<8, 23>;

    ::testing::StaticAssertTypeEq<doublex, decltype(doublex() + doublex())>();
    ::testing::StaticAssertTypeEq<doublex, decltype(doublex() - doublex())>();
    ::testing::StaticAssertTypeEq<doublex, decltype(doublex() * doublex())>();
    ::testing::StaticAssertTypeEq<doublex, decltype(doublex() / doublex())>();

    ::testing::StaticAssertTypeEq<floatx, decltype(floatx() + floatx())>();
    ::testing::StaticAssertTypeEq<floatx, decltype(floatx() - floatx())>();
    ::testing::StaticAssertTypeEq<floatx, decltype(floatx() * floatx())>();
    ::testing::StaticAssertTypeEq<floatx, decltype(floatx() / floatx())>();

    doublex dlhs;
    ::testing::StaticAssertTypeEq<doublex&, decltype(dlhs += doublex())>();
    ::testing::StaticAssertTypeEq<doublex&, decltype(dlhs -= doublex())>();
    ::testing::StaticAssertTypeEq<doublex&, decltype(dlhs *= doublex())>();
    ::testing::StaticAssertTypeEq<doublex&, decltype(dlhs /= doublex())>();
    floatx flhs;
    ::testing::StaticAssertTypeEq<floatx&, decltype(flhs += floatx())>();
    ::testing::StaticAssertTypeEq<floatx&, decltype(flhs -= floatx())>();
    ::testing::StaticAssertTypeEq<floatx&, decltype(flhs *= floatx())>();
    ::testing::StaticAssertTypeEq<floatx&, decltype(flhs /= floatx())>();
}


TEST(FloatxArithmeticTest, PromotesTypes)
{
    using flx1 = flx::floatx<9, 7>;
    using flx2 = flx::floatx<6, 13>;
    using supertype = flx::floatx<9, 13>;
    ::testing::StaticAssertTypeEq<supertype, decltype(flx1() + flx2())>();
    ::testing::StaticAssertTypeEq<supertype, decltype(flx1() - flx2())>();
    ::testing::StaticAssertTypeEq<supertype, decltype(flx1() * flx2())>();
    ::testing::StaticAssertTypeEq<supertype, decltype(flx1() / flx2())>();

    flx1 flhs;
    ::testing::StaticAssertTypeEq<flx1&, decltype(flhs += flx2())>();
    ::testing::StaticAssertTypeEq<flx1&, decltype(flhs -= flx2())>();
    ::testing::StaticAssertTypeEq<flx1&, decltype(flhs *= flx2())>();
    ::testing::StaticAssertTypeEq<flx1&, decltype(flhs /= flx2())>();

    using flx3 = flx::floatx<9, 23>;
    ::testing::StaticAssertTypeEq<flx3, decltype(flx1() + float())>();
    ::testing::StaticAssertTypeEq<flx3, decltype(flx1() - float())>();
    ::testing::StaticAssertTypeEq<flx3, decltype(flx1() * float())>();
    ::testing::StaticAssertTypeEq<flx3, decltype(flx1() / float())>();

    ::testing::StaticAssertTypeEq<flx1&, decltype(flhs += float())>();
    ::testing::StaticAssertTypeEq<flx1&, decltype(flhs -= float())>();
    ::testing::StaticAssertTypeEq<flx1&, decltype(flhs *= float())>();
    ::testing::StaticAssertTypeEq<flx1&, decltype(flhs /= float())>();

    using doublex = flx::floatx<11, 52>;
    ::testing::StaticAssertTypeEq<doublex, decltype(flx1() + double())>();
    ::testing::StaticAssertTypeEq<doublex, decltype(flx1() - double())>();
    ::testing::StaticAssertTypeEq<doublex, decltype(flx1() * double())>();
    ::testing::StaticAssertTypeEq<doublex, decltype(flx1() / double())>();

    ::testing::StaticAssertTypeEq<flx1&, decltype(flhs += double())>();
    ::testing::StaticAssertTypeEq<flx1&, decltype(flhs -= double())>();
    ::testing::StaticAssertTypeEq<flx1&, decltype(flhs *= double())>();
    ::testing::StaticAssertTypeEq<flx1&, decltype(flhs /= double())>();

    ::testing::StaticAssertTypeEq<flx1, decltype(flx1() + int())>();
    ::testing::StaticAssertTypeEq<flx1, decltype(flx1() - int())>();
    ::testing::StaticAssertTypeEq<flx1, decltype(flx1() * int())>();
    ::testing::StaticAssertTypeEq<flx1, decltype(flx1() / int())>();

    ::testing::StaticAssertTypeEq<flx1&, decltype(flhs += int())>();
    ::testing::StaticAssertTypeEq<flx1&, decltype(flhs -= int())>();
    ::testing::StaticAssertTypeEq<flx1&, decltype(flhs *= int())>();
    ::testing::StaticAssertTypeEq<flx1&, decltype(flhs /= int())>();
}


TEST(FloatxrArithmeticTest, ResultHasCorrectType)
{
    auto fxr = []() { return flx::floatxr<>(8, 23); };
    ::testing::StaticAssertTypeEq<flx::floatxr<>, decltype(fxr() + fxr())>();
    ::testing::StaticAssertTypeEq<flx::floatxr<>, decltype(fxr() - fxr())>();
    ::testing::StaticAssertTypeEq<flx::floatxr<>, decltype(fxr() * fxr())>();
    ::testing::StaticAssertTypeEq<flx::floatxr<>, decltype(fxr() / fxr())>();

    flx::floatxr<> dlhs(8, 23);
    ::testing::StaticAssertTypeEq<flx::floatxr<>&, decltype(dlhs += fxr())>();
    ::testing::StaticAssertTypeEq<flx::floatxr<>&, decltype(dlhs -= fxr())>();
    ::testing::StaticAssertTypeEq<flx::floatxr<>&, decltype(dlhs *= fxr())>();
    ::testing::StaticAssertTypeEq<flx::floatxr<>&, decltype(dlhs /= fxr())>();
}


TEST(FloatxrArithmeticTest, PromotesTypes)
{
    auto fxr = []() { return flx::floatxr<>(8, 23); };
    using floatx = flx::floatx<9, 12>;
    ::testing::StaticAssertTypeEq<flx::floatxr<>, decltype(fxr() + floatx())>();
    ::testing::StaticAssertTypeEq<flx::floatxr<>, decltype(fxr() - floatx())>();
    ::testing::StaticAssertTypeEq<flx::floatxr<>, decltype(fxr() * floatx())>();
    ::testing::StaticAssertTypeEq<flx::floatxr<>, decltype(fxr() / floatx())>();

    flx::floatxr<> dlhs(8, 23);
    ::testing::StaticAssertTypeEq<flx::floatxr<>&,
                                  decltype(dlhs += floatx())>();
    ::testing::StaticAssertTypeEq<flx::floatxr<>&,
                                  decltype(dlhs -= floatx())>();
    ::testing::StaticAssertTypeEq<flx::floatxr<>&,
                                  decltype(dlhs *= floatx())>();
    ::testing::StaticAssertTypeEq<flx::floatxr<>&,
                                  decltype(dlhs /= floatx())>();

    ::testing::StaticAssertTypeEq<flx::floatxr<>, decltype(fxr() + double())>();
    ::testing::StaticAssertTypeEq<flx::floatxr<>, decltype(fxr() - double())>();
    ::testing::StaticAssertTypeEq<flx::floatxr<>, decltype(fxr() * double())>();
    ::testing::StaticAssertTypeEq<flx::floatxr<>, decltype(fxr() / double())>();

    ::testing::StaticAssertTypeEq<flx::floatxr<>&,
                                  decltype(dlhs += double())>();
    ::testing::StaticAssertTypeEq<flx::floatxr<>&,
                                  decltype(dlhs -= double())>();
    ::testing::StaticAssertTypeEq<flx::floatxr<>&,
                                  decltype(dlhs *= double())>();
    ::testing::StaticAssertTypeEq<flx::floatxr<>&,
                                  decltype(dlhs /= double())>();

    ::testing::StaticAssertTypeEq<flx::floatxr<>, decltype(fxr() + int())>();
    ::testing::StaticAssertTypeEq<flx::floatxr<>, decltype(fxr() - int())>();
    ::testing::StaticAssertTypeEq<flx::floatxr<>, decltype(fxr() * int())>();
    ::testing::StaticAssertTypeEq<flx::floatxr<>, decltype(fxr() / int())>();

    ::testing::StaticAssertTypeEq<flx::floatxr<>&, decltype(dlhs += int())>();
    ::testing::StaticAssertTypeEq<flx::floatxr<>&, decltype(dlhs -= int())>();
    ::testing::StaticAssertTypeEq<flx::floatxr<>&, decltype(dlhs *= int())>();
    ::testing::StaticAssertTypeEq<flx::floatxr<>&, decltype(dlhs /= int())>();
}


}  // namespace
