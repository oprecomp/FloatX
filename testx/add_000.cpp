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

#define nan double(0.0 / 0.0)
#define inf double(1.0 / 0.0)

namespace {

TEST(FloatxOperationsTest, ADD_1_1_simple)
{
    using T = flx::floatx<1, 1>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.00000000000000000000);
    EXPECT_EQ(double(b), 0.00000000000000000000);
    EXPECT_EQ(double(c), 0.00000000000000000000);
}

TEST(FloatxOperationsTest, ADD_1_2_simple)
{
    using T = flx::floatx<1, 2>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.50000000000000000000);
    EXPECT_EQ(double(b), 0.00000000000000000000);
    EXPECT_EQ(double(c), 0.50000000000000000000);
}

TEST(FloatxOperationsTest, ADD_1_3_simple)
{
    using T = flx::floatx<1, 3>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.25000000000000000000);
    EXPECT_EQ(double(b), 0.00000000000000000000);
    EXPECT_EQ(double(c), 0.25000000000000000000);
}

TEST(FloatxOperationsTest, ADD_1_4_simple)
{
    using T = flx::floatx<1, 4>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.37500000000000000000);
    EXPECT_EQ(double(b), 0.12500000000000000000);
    EXPECT_EQ(double(c), 0.50000000000000000000);
}

TEST(FloatxOperationsTest, ADD_1_5_simple)
{
    using T = flx::floatx<1, 5>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.31250000000000000000);
    EXPECT_EQ(double(b), 0.12500000000000000000);
    EXPECT_EQ(double(c), 0.43750000000000000000);
}

TEST(FloatxOperationsTest, ADD_1_6_simple)
{
    using T = flx::floatx<1, 6>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.34375000000000000000);
    EXPECT_EQ(double(b), 0.12500000000000000000);
    EXPECT_EQ(double(c), 0.46875000000000000000);
}

TEST(FloatxOperationsTest, ADD_1_7_simple)
{
    using T = flx::floatx<1, 7>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.32812500000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 0.43750000000000000000);
}

TEST(FloatxOperationsTest, ADD_1_8_simple)
{
    using T = flx::floatx<1, 8>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33593750000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 0.44531250000000000000);
}

TEST(FloatxOperationsTest, ADD_1_9_simple)
{
    using T = flx::floatx<1, 9>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33203125000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 0.44140625000000000000);
}

TEST(FloatxOperationsTest, ADD_1_10_simple)
{
    using T = flx::floatx<1, 10>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33398437500000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 0.44531250000000000000);
}

TEST(FloatxOperationsTest, ADD_1_11_simple)
{
    using T = flx::floatx<1, 11>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33300781250000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 0.44433593750000000000);
}

TEST(FloatxOperationsTest, ADD_1_12_simple)
{
    using T = flx::floatx<1, 12>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33349609375000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 0.44482421875000000000);
}

TEST(FloatxOperationsTest, ADD_1_13_simple)
{
    using T = flx::floatx<1, 13>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33325195312500000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 0.44433593750000000000);
}

TEST(FloatxOperationsTest, ADD_1_14_simple)
{
    using T = flx::floatx<1, 14>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33337402343750000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 0.44445800781250000000);
}

TEST(FloatxOperationsTest, ADD_1_15_simple)
{
    using T = flx::floatx<1, 15>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33331298828125000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 0.44439697265625000000);
}

TEST(FloatxOperationsTest, ADD_1_16_simple)
{
    using T = flx::floatx<1, 16>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33334350585937500000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 0.44445800781250000000);
}

TEST(FloatxOperationsTest, ADD_1_17_simple)
{
    using T = flx::floatx<1, 17>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33332824707031250000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 0.44444274902343750000);
}

TEST(FloatxOperationsTest, ADD_1_18_simple)
{
    using T = flx::floatx<1, 18>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333587646484375000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 0.44445037841796875000);
}

TEST(FloatxOperationsTest, ADD_1_19_simple)
{
    using T = flx::floatx<1, 19>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333206176757812500);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 0.44444274902343750000);
}

TEST(FloatxOperationsTest, ADD_1_20_simple)
{
    using T = flx::floatx<1, 20>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333396911621093750);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 0.44444465637207031250);
}

TEST(FloatxOperationsTest, ADD_1_21_simple)
{
    using T = flx::floatx<1, 21>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333301544189453125);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 0.44444370269775390625);
}

TEST(FloatxOperationsTest, ADD_1_22_simple)
{
    using T = flx::floatx<1, 22>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333349227905273438);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 0.44444465637207031250);
}

TEST(FloatxOperationsTest, ADD_1_23_simple)
{
    using T = flx::floatx<1, 23>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333325386047363281);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 0.44444441795349121094);
}

TEST(FloatxOperationsTest, ADD_1_24_simple)
{
    using T = flx::floatx<1, 24>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333337306976318359);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 0.44444453716278076172);
}

TEST(FloatxOperationsTest, ADD_1_25_simple)
{
    using T = flx::floatx<1, 25>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333331346511840820);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 0.44444441795349121094);
}

TEST(FloatxOperationsTest, ADD_1_26_simple)
{
    using T = flx::floatx<1, 26>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333334326744079590);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 0.44444444775581359863);
}

TEST(FloatxOperationsTest, ADD_1_27_simple)
{
    using T = flx::floatx<1, 27>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333332836627960205);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 0.44444443285465240479);
}

TEST(FloatxOperationsTest, ADD_1_28_simple)
{
    using T = flx::floatx<1, 28>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333581686019897);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 0.44444444775581359863);
}

TEST(FloatxOperationsTest, ADD_1_29_simple)
{
    using T = flx::floatx<1, 29>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333209156990051);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 0.44444444403052330017);
}

TEST(FloatxOperationsTest, ADD_1_30_simple)
{
    using T = flx::floatx<1, 30>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333395421504974);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 0.44444444589316844940);
}

TEST(FloatxOperationsTest, ADD_1_31_simple)
{
    using T = flx::floatx<1, 31>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333302289247513);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 0.44444444403052330017);
}

TEST(FloatxOperationsTest, ADD_1_32_simple)
{
    using T = flx::floatx<1, 32>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333348855376244);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 0.44444444449618458748);
}

TEST(FloatxOperationsTest, ADD_1_33_simple)
{
    using T = flx::floatx<1, 33>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333325572311878);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 0.44444444426335394382);
}

TEST(FloatxOperationsTest, ADD_1_34_simple)
{
    using T = flx::floatx<1, 34>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333337213844061);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 0.44444444449618458748);
}

TEST(FloatxOperationsTest, ADD_1_35_simple)
{
    using T = flx::floatx<1, 35>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333331393077970);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 0.44444444443797692657);
}

TEST(FloatxOperationsTest, ADD_1_36_simple)
{
    using T = flx::floatx<1, 36>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333334303461015);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 0.44444444446708075702);
}

TEST(FloatxOperationsTest, ADD_1_37_simple)
{
    using T = flx::floatx<1, 37>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333332848269492);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 0.44444444443797692657);
}

TEST(FloatxOperationsTest, ADD_1_38_simple)
{
    using T = flx::floatx<1, 38>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333575865254);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 0.44444444444525288418);
}

TEST(FloatxOperationsTest, ADD_1_39_simple)
{
    using T = flx::floatx<1, 39>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333212067373);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 0.44444444444161490537);
}

TEST(FloatxOperationsTest, ADD_1_40_simple)
{
    using T = flx::floatx<1, 40>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333393966313);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 0.44444444444525288418);
}

TEST(FloatxOperationsTest, ADD_1_41_simple)
{
    using T = flx::floatx<1, 41>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333303016843);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 0.44444444444434338948);
}

TEST(FloatxOperationsTest, ADD_1_42_simple)
{
    using T = flx::floatx<1, 42>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333348491578);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 0.44444444444479813683);
}

TEST(FloatxOperationsTest, ADD_1_43_simple)
{
    using T = flx::floatx<1, 43>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333325754211);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 0.44444444444434338948);
}

TEST(FloatxOperationsTest, ADD_1_44_simple)
{
    using T = flx::floatx<1, 44>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333337122895);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 0.44444444444445707632);
}

TEST(FloatxOperationsTest, ADD_1_45_simple)
{
    using T = flx::floatx<1, 45>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333331438553);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 0.44444444444440023290);
}

TEST(FloatxOperationsTest, ADD_1_46_simple)
{
    using T = flx::floatx<1, 46>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333334280724);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 0.44444444444445707632);
}

TEST(FloatxOperationsTest, ADD_1_47_simple)
{
    using T = flx::floatx<1, 47>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333332859638);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 0.44444444444444286546);
}

TEST(FloatxOperationsTest, ADD_1_48_simple)
{
    using T = flx::floatx<1, 48>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333570181);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 0.44444444444444997089);
}

TEST(FloatxOperationsTest, ADD_1_49_simple)
{
    using T = flx::floatx<1, 49>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333214910);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 0.44444444444444286546);
}

TEST(FloatxOperationsTest, ADD_1_50_simple)
{
    using T = flx::floatx<1, 50>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333392545);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 0.44444444444444464182);
}

TEST(FloatxOperationsTest, ADD_1_51_simple)
{
    using T = flx::floatx<1, 51>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333303727);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 0.44444444444444375364);
}

TEST(FloatxOperationsTest, ADD_1_52_simple)
{
    using T = flx::floatx<1, 52>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333348136);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 0.44444444444444464182);
}

TEST(FloatxOperationsTest, ADD_2_1_simple)
{
    using T = flx::floatx<2, 1>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.50000000000000000000);
    EXPECT_EQ(double(b), 0.00000000000000000000);
    EXPECT_EQ(double(c), 0.50000000000000000000);
}

TEST(FloatxOperationsTest, ADD_2_2_simple)
{
    using T = flx::floatx<2, 2>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.25000000000000000000);
    EXPECT_EQ(double(b), 0.00000000000000000000);
    EXPECT_EQ(double(c), 0.25000000000000000000);
}

TEST(FloatxOperationsTest, ADD_2_3_simple)
{
    using T = flx::floatx<2, 3>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.37500000000000000000);
    EXPECT_EQ(double(b), 0.12500000000000000000);
    EXPECT_EQ(double(c), 0.50000000000000000000);
}

TEST(FloatxOperationsTest, ADD_2_4_simple)
{
    using T = flx::floatx<2, 4>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.31250000000000000000);
    EXPECT_EQ(double(b), 0.12500000000000000000);
    EXPECT_EQ(double(c), 0.43750000000000000000);
}

TEST(FloatxOperationsTest, ADD_2_5_simple)
{
    using T = flx::floatx<2, 5>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.34375000000000000000);
    EXPECT_EQ(double(b), 0.12500000000000000000);
    EXPECT_EQ(double(c), 0.46875000000000000000);
}

TEST(FloatxOperationsTest, ADD_2_6_simple)
{
    using T = flx::floatx<2, 6>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.32812500000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 0.43750000000000000000);
}

TEST(FloatxOperationsTest, ADD_2_7_simple)
{
    using T = flx::floatx<2, 7>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33593750000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 0.44531250000000000000);
}

TEST(FloatxOperationsTest, ADD_2_8_simple)
{
    using T = flx::floatx<2, 8>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33203125000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 0.44140625000000000000);
}

TEST(FloatxOperationsTest, ADD_2_9_simple)
{
    using T = flx::floatx<2, 9>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33398437500000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 0.44531250000000000000);
}

TEST(FloatxOperationsTest, ADD_2_10_simple)
{
    using T = flx::floatx<2, 10>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33300781250000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 0.44433593750000000000);
}

TEST(FloatxOperationsTest, ADD_2_11_simple)
{
    using T = flx::floatx<2, 11>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33349609375000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 0.44482421875000000000);
}

TEST(FloatxOperationsTest, ADD_2_12_simple)
{
    using T = flx::floatx<2, 12>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33325195312500000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 0.44433593750000000000);
}

TEST(FloatxOperationsTest, ADD_2_13_simple)
{
    using T = flx::floatx<2, 13>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33337402343750000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 0.44445800781250000000);
}

TEST(FloatxOperationsTest, ADD_2_14_simple)
{
    using T = flx::floatx<2, 14>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33331298828125000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 0.44439697265625000000);
}

TEST(FloatxOperationsTest, ADD_2_15_simple)
{
    using T = flx::floatx<2, 15>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33334350585937500000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 0.44445800781250000000);
}

TEST(FloatxOperationsTest, ADD_2_16_simple)
{
    using T = flx::floatx<2, 16>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33332824707031250000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 0.44444274902343750000);
}

TEST(FloatxOperationsTest, ADD_2_17_simple)
{
    using T = flx::floatx<2, 17>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333587646484375000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 0.44445037841796875000);
}

TEST(FloatxOperationsTest, ADD_2_18_simple)
{
    using T = flx::floatx<2, 18>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333206176757812500);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 0.44444274902343750000);
}

TEST(FloatxOperationsTest, ADD_2_19_simple)
{
    using T = flx::floatx<2, 19>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333396911621093750);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 0.44444465637207031250);
}

TEST(FloatxOperationsTest, ADD_2_20_simple)
{
    using T = flx::floatx<2, 20>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333301544189453125);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 0.44444370269775390625);
}

TEST(FloatxOperationsTest, ADD_2_21_simple)
{
    using T = flx::floatx<2, 21>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333349227905273438);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 0.44444465637207031250);
}

TEST(FloatxOperationsTest, ADD_2_22_simple)
{
    using T = flx::floatx<2, 22>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333325386047363281);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 0.44444441795349121094);
}

TEST(FloatxOperationsTest, ADD_2_23_simple)
{
    using T = flx::floatx<2, 23>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333337306976318359);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 0.44444453716278076172);
}

TEST(FloatxOperationsTest, ADD_2_24_simple)
{
    using T = flx::floatx<2, 24>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333331346511840820);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 0.44444441795349121094);
}

TEST(FloatxOperationsTest, ADD_2_25_simple)
{
    using T = flx::floatx<2, 25>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333334326744079590);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 0.44444444775581359863);
}

TEST(FloatxOperationsTest, ADD_2_26_simple)
{
    using T = flx::floatx<2, 26>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333332836627960205);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 0.44444443285465240479);
}

TEST(FloatxOperationsTest, ADD_2_27_simple)
{
    using T = flx::floatx<2, 27>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333581686019897);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 0.44444444775581359863);
}

TEST(FloatxOperationsTest, ADD_2_28_simple)
{
    using T = flx::floatx<2, 28>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333209156990051);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 0.44444444403052330017);
}

TEST(FloatxOperationsTest, ADD_2_29_simple)
{
    using T = flx::floatx<2, 29>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333395421504974);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 0.44444444589316844940);
}

TEST(FloatxOperationsTest, ADD_2_30_simple)
{
    using T = flx::floatx<2, 30>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333302289247513);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 0.44444444403052330017);
}

TEST(FloatxOperationsTest, ADD_2_31_simple)
{
    using T = flx::floatx<2, 31>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333348855376244);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 0.44444444449618458748);
}

TEST(FloatxOperationsTest, ADD_2_32_simple)
{
    using T = flx::floatx<2, 32>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333325572311878);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 0.44444444426335394382);
}

TEST(FloatxOperationsTest, ADD_2_33_simple)
{
    using T = flx::floatx<2, 33>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333337213844061);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 0.44444444449618458748);
}

TEST(FloatxOperationsTest, ADD_2_34_simple)
{
    using T = flx::floatx<2, 34>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333331393077970);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 0.44444444443797692657);
}

TEST(FloatxOperationsTest, ADD_2_35_simple)
{
    using T = flx::floatx<2, 35>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333334303461015);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 0.44444444446708075702);
}

TEST(FloatxOperationsTest, ADD_2_36_simple)
{
    using T = flx::floatx<2, 36>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333332848269492);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 0.44444444443797692657);
}

TEST(FloatxOperationsTest, ADD_2_37_simple)
{
    using T = flx::floatx<2, 37>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333575865254);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 0.44444444444525288418);
}

TEST(FloatxOperationsTest, ADD_2_38_simple)
{
    using T = flx::floatx<2, 38>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333212067373);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 0.44444444444161490537);
}

TEST(FloatxOperationsTest, ADD_2_39_simple)
{
    using T = flx::floatx<2, 39>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333393966313);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 0.44444444444525288418);
}

TEST(FloatxOperationsTest, ADD_2_40_simple)
{
    using T = flx::floatx<2, 40>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333303016843);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 0.44444444444434338948);
}

TEST(FloatxOperationsTest, ADD_2_41_simple)
{
    using T = flx::floatx<2, 41>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333348491578);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 0.44444444444479813683);
}

TEST(FloatxOperationsTest, ADD_2_42_simple)
{
    using T = flx::floatx<2, 42>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333325754211);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 0.44444444444434338948);
}

TEST(FloatxOperationsTest, ADD_2_43_simple)
{
    using T = flx::floatx<2, 43>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333337122895);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 0.44444444444445707632);
}

TEST(FloatxOperationsTest, ADD_2_44_simple)
{
    using T = flx::floatx<2, 44>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333331438553);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 0.44444444444440023290);
}

TEST(FloatxOperationsTest, ADD_2_45_simple)
{
    using T = flx::floatx<2, 45>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333334280724);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 0.44444444444445707632);
}

TEST(FloatxOperationsTest, ADD_2_46_simple)
{
    using T = flx::floatx<2, 46>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333332859638);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 0.44444444444444286546);
}

TEST(FloatxOperationsTest, ADD_2_47_simple)
{
    using T = flx::floatx<2, 47>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333570181);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 0.44444444444444997089);
}

TEST(FloatxOperationsTest, ADD_2_48_simple)
{
    using T = flx::floatx<2, 48>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333214910);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 0.44444444444444286546);
}

TEST(FloatxOperationsTest, ADD_2_49_simple)
{
    using T = flx::floatx<2, 49>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333392545);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 0.44444444444444464182);
}

TEST(FloatxOperationsTest, ADD_2_50_simple)
{
    using T = flx::floatx<2, 50>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333303727);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 0.44444444444444375364);
}

TEST(FloatxOperationsTest, ADD_2_51_simple)
{
    using T = flx::floatx<2, 51>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333348136);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 0.44444444444444464182);
}

TEST(FloatxOperationsTest, ADD_2_52_simple)
{
    using T = flx::floatx<2, 52>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333325932);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 0.44444444444444441977);
}

TEST(FloatxOperationsTest, ADD_3_1_simple)
{
    using T = flx::floatx<3, 1>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.37500000000000000000);
    EXPECT_EQ(double(b), 0.12500000000000000000);
    EXPECT_EQ(double(c), 0.50000000000000000000);
}

TEST(FloatxOperationsTest, ADD_3_2_simple)
{
    using T = flx::floatx<3, 2>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.31250000000000000000);
    EXPECT_EQ(double(b), 0.12500000000000000000);
    EXPECT_EQ(double(c), 0.43750000000000000000);
}

TEST(FloatxOperationsTest, ADD_3_3_simple)
{
    using T = flx::floatx<3, 3>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.34375000000000000000);
    EXPECT_EQ(double(b), 0.12500000000000000000);
    EXPECT_EQ(double(c), 0.46875000000000000000);
}

TEST(FloatxOperationsTest, ADD_3_4_simple)
{
    using T = flx::floatx<3, 4>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.32812500000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 0.43750000000000000000);
}

TEST(FloatxOperationsTest, ADD_3_5_simple)
{
    using T = flx::floatx<3, 5>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33593750000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 0.44531250000000000000);
}

TEST(FloatxOperationsTest, ADD_3_6_simple)
{
    using T = flx::floatx<3, 6>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33203125000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 0.44140625000000000000);
}

TEST(FloatxOperationsTest, ADD_3_7_simple)
{
    using T = flx::floatx<3, 7>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33398437500000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 0.44531250000000000000);
}

TEST(FloatxOperationsTest, ADD_3_8_simple)
{
    using T = flx::floatx<3, 8>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33300781250000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 0.44433593750000000000);
}

TEST(FloatxOperationsTest, ADD_3_9_simple)
{
    using T = flx::floatx<3, 9>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33349609375000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 0.44482421875000000000);
}

TEST(FloatxOperationsTest, ADD_3_10_simple)
{
    using T = flx::floatx<3, 10>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33325195312500000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 0.44433593750000000000);
}

TEST(FloatxOperationsTest, ADD_3_11_simple)
{
    using T = flx::floatx<3, 11>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33337402343750000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 0.44445800781250000000);
}

TEST(FloatxOperationsTest, ADD_3_12_simple)
{
    using T = flx::floatx<3, 12>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33331298828125000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 0.44439697265625000000);
}

TEST(FloatxOperationsTest, ADD_3_13_simple)
{
    using T = flx::floatx<3, 13>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33334350585937500000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 0.44445800781250000000);
}

TEST(FloatxOperationsTest, ADD_3_14_simple)
{
    using T = flx::floatx<3, 14>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33332824707031250000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 0.44444274902343750000);
}

TEST(FloatxOperationsTest, ADD_3_15_simple)
{
    using T = flx::floatx<3, 15>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333587646484375000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 0.44445037841796875000);
}

TEST(FloatxOperationsTest, ADD_3_16_simple)
{
    using T = flx::floatx<3, 16>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333206176757812500);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 0.44444274902343750000);
}

TEST(FloatxOperationsTest, ADD_3_17_simple)
{
    using T = flx::floatx<3, 17>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333396911621093750);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 0.44444465637207031250);
}

TEST(FloatxOperationsTest, ADD_3_18_simple)
{
    using T = flx::floatx<3, 18>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333301544189453125);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 0.44444370269775390625);
}

TEST(FloatxOperationsTest, ADD_3_19_simple)
{
    using T = flx::floatx<3, 19>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333349227905273438);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 0.44444465637207031250);
}

TEST(FloatxOperationsTest, ADD_3_20_simple)
{
    using T = flx::floatx<3, 20>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333325386047363281);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 0.44444441795349121094);
}

TEST(FloatxOperationsTest, ADD_3_21_simple)
{
    using T = flx::floatx<3, 21>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333337306976318359);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 0.44444453716278076172);
}

TEST(FloatxOperationsTest, ADD_3_22_simple)
{
    using T = flx::floatx<3, 22>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333331346511840820);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 0.44444441795349121094);
}

TEST(FloatxOperationsTest, ADD_3_23_simple)
{
    using T = flx::floatx<3, 23>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333334326744079590);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 0.44444444775581359863);
}

TEST(FloatxOperationsTest, ADD_3_24_simple)
{
    using T = flx::floatx<3, 24>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333332836627960205);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 0.44444443285465240479);
}

TEST(FloatxOperationsTest, ADD_3_25_simple)
{
    using T = flx::floatx<3, 25>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333581686019897);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 0.44444444775581359863);
}

TEST(FloatxOperationsTest, ADD_3_26_simple)
{
    using T = flx::floatx<3, 26>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333209156990051);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 0.44444444403052330017);
}

TEST(FloatxOperationsTest, ADD_3_27_simple)
{
    using T = flx::floatx<3, 27>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333395421504974);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 0.44444444589316844940);
}

TEST(FloatxOperationsTest, ADD_3_28_simple)
{
    using T = flx::floatx<3, 28>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333302289247513);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 0.44444444403052330017);
}

TEST(FloatxOperationsTest, ADD_3_29_simple)
{
    using T = flx::floatx<3, 29>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333348855376244);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 0.44444444449618458748);
}

TEST(FloatxOperationsTest, ADD_3_30_simple)
{
    using T = flx::floatx<3, 30>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333325572311878);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 0.44444444426335394382);
}

TEST(FloatxOperationsTest, ADD_3_31_simple)
{
    using T = flx::floatx<3, 31>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333337213844061);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 0.44444444449618458748);
}

TEST(FloatxOperationsTest, ADD_3_32_simple)
{
    using T = flx::floatx<3, 32>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333331393077970);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 0.44444444443797692657);
}

TEST(FloatxOperationsTest, ADD_3_33_simple)
{
    using T = flx::floatx<3, 33>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333334303461015);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 0.44444444446708075702);
}

TEST(FloatxOperationsTest, ADD_3_34_simple)
{
    using T = flx::floatx<3, 34>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333332848269492);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 0.44444444443797692657);
}

TEST(FloatxOperationsTest, ADD_3_35_simple)
{
    using T = flx::floatx<3, 35>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333575865254);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 0.44444444444525288418);
}

TEST(FloatxOperationsTest, ADD_3_36_simple)
{
    using T = flx::floatx<3, 36>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333212067373);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 0.44444444444161490537);
}

TEST(FloatxOperationsTest, ADD_3_37_simple)
{
    using T = flx::floatx<3, 37>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333393966313);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 0.44444444444525288418);
}

TEST(FloatxOperationsTest, ADD_3_38_simple)
{
    using T = flx::floatx<3, 38>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333303016843);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 0.44444444444434338948);
}

TEST(FloatxOperationsTest, ADD_3_39_simple)
{
    using T = flx::floatx<3, 39>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333348491578);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 0.44444444444479813683);
}

TEST(FloatxOperationsTest, ADD_3_40_simple)
{
    using T = flx::floatx<3, 40>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333325754211);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 0.44444444444434338948);
}

TEST(FloatxOperationsTest, ADD_3_41_simple)
{
    using T = flx::floatx<3, 41>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333337122895);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 0.44444444444445707632);
}

TEST(FloatxOperationsTest, ADD_3_42_simple)
{
    using T = flx::floatx<3, 42>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333331438553);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 0.44444444444440023290);
}

TEST(FloatxOperationsTest, ADD_3_43_simple)
{
    using T = flx::floatx<3, 43>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333334280724);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 0.44444444444445707632);
}

TEST(FloatxOperationsTest, ADD_3_44_simple)
{
    using T = flx::floatx<3, 44>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333332859638);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 0.44444444444444286546);
}

TEST(FloatxOperationsTest, ADD_3_45_simple)
{
    using T = flx::floatx<3, 45>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333570181);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 0.44444444444444997089);
}

TEST(FloatxOperationsTest, ADD_3_46_simple)
{
    using T = flx::floatx<3, 46>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333214910);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 0.44444444444444286546);
}

TEST(FloatxOperationsTest, ADD_3_47_simple)
{
    using T = flx::floatx<3, 47>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333392545);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 0.44444444444444464182);
}

TEST(FloatxOperationsTest, ADD_3_48_simple)
{
    using T = flx::floatx<3, 48>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333303727);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 0.44444444444444375364);
}

TEST(FloatxOperationsTest, ADD_3_49_simple)
{
    using T = flx::floatx<3, 49>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333348136);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 0.44444444444444464182);
}

TEST(FloatxOperationsTest, ADD_3_50_simple)
{
    using T = flx::floatx<3, 50>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333325932);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 0.44444444444444441977);
}

TEST(FloatxOperationsTest, ADD_3_51_simple)
{
    using T = flx::floatx<3, 51>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333325932);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 0.44444444444444441977);
}

TEST(FloatxOperationsTest, ADD_3_52_simple)
{
    using T = flx::floatx<3, 52>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333331483);
    EXPECT_EQ(double(b), 0.11111111111111110494);
    EXPECT_EQ(double(c), 0.44444444444444441977);
}

TEST(FloatxOperationsTest, ADD_4_1_simple)
{
    using T = flx::floatx<4, 1>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.37500000000000000000);
    EXPECT_EQ(double(b), 0.12500000000000000000);
    EXPECT_EQ(double(c), 0.50000000000000000000);
}

TEST(FloatxOperationsTest, ADD_4_2_simple)
{
    using T = flx::floatx<4, 2>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.31250000000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 0.43750000000000000000);
}

TEST(FloatxOperationsTest, ADD_4_3_simple)
{
    using T = flx::floatx<4, 3>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.34375000000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 0.43750000000000000000);
}

TEST(FloatxOperationsTest, ADD_4_4_simple)
{
    using T = flx::floatx<4, 4>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.32812500000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 0.43750000000000000000);
}

TEST(FloatxOperationsTest, ADD_4_5_simple)
{
    using T = flx::floatx<4, 5>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33593750000000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 0.44531250000000000000);
}

TEST(FloatxOperationsTest, ADD_4_6_simple)
{
    using T = flx::floatx<4, 6>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33203125000000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 0.44531250000000000000);
}

TEST(FloatxOperationsTest, ADD_4_7_simple)
{
    using T = flx::floatx<4, 7>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33398437500000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 0.44531250000000000000);
}

TEST(FloatxOperationsTest, ADD_4_8_simple)
{
    using T = flx::floatx<4, 8>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33300781250000000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 0.44433593750000000000);
}

TEST(FloatxOperationsTest, ADD_4_9_simple)
{
    using T = flx::floatx<4, 9>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33349609375000000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 0.44433593750000000000);
}

TEST(FloatxOperationsTest, ADD_4_10_simple)
{
    using T = flx::floatx<4, 10>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33325195312500000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 0.44433593750000000000);
}

TEST(FloatxOperationsTest, ADD_4_11_simple)
{
    using T = flx::floatx<4, 11>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33337402343750000000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 0.44445800781250000000);
}

TEST(FloatxOperationsTest, ADD_4_12_simple)
{
    using T = flx::floatx<4, 12>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33331298828125000000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 0.44445800781250000000);
}

TEST(FloatxOperationsTest, ADD_4_13_simple)
{
    using T = flx::floatx<4, 13>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33334350585937500000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 0.44445800781250000000);
}

TEST(FloatxOperationsTest, ADD_4_14_simple)
{
    using T = flx::floatx<4, 14>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33332824707031250000);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 0.44444274902343750000);
}

TEST(FloatxOperationsTest, ADD_4_15_simple)
{
    using T = flx::floatx<4, 15>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333587646484375000);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 0.44444274902343750000);
}

TEST(FloatxOperationsTest, ADD_4_16_simple)
{
    using T = flx::floatx<4, 16>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333206176757812500);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 0.44444274902343750000);
}

TEST(FloatxOperationsTest, ADD_4_17_simple)
{
    using T = flx::floatx<4, 17>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333396911621093750);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 0.44444465637207031250);
}

TEST(FloatxOperationsTest, ADD_4_18_simple)
{
    using T = flx::floatx<4, 18>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333301544189453125);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 0.44444465637207031250);
}

TEST(FloatxOperationsTest, ADD_4_19_simple)
{
    using T = flx::floatx<4, 19>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333349227905273438);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 0.44444465637207031250);
}

TEST(FloatxOperationsTest, ADD_4_20_simple)
{
    using T = flx::floatx<4, 20>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333325386047363281);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 0.44444441795349121094);
}

TEST(FloatxOperationsTest, ADD_4_21_simple)
{
    using T = flx::floatx<4, 21>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333337306976318359);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 0.44444441795349121094);
}

TEST(FloatxOperationsTest, ADD_4_22_simple)
{
    using T = flx::floatx<4, 22>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333331346511840820);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 0.44444441795349121094);
}

TEST(FloatxOperationsTest, ADD_4_23_simple)
{
    using T = flx::floatx<4, 23>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333334326744079590);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 0.44444444775581359863);
}

TEST(FloatxOperationsTest, ADD_4_24_simple)
{
    using T = flx::floatx<4, 24>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333332836627960205);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 0.44444444775581359863);
}

TEST(FloatxOperationsTest, ADD_4_25_simple)
{
    using T = flx::floatx<4, 25>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333581686019897);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 0.44444444775581359863);
}

TEST(FloatxOperationsTest, ADD_4_26_simple)
{
    using T = flx::floatx<4, 26>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333209156990051);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 0.44444444403052330017);
}

TEST(FloatxOperationsTest, ADD_4_27_simple)
{
    using T = flx::floatx<4, 27>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333395421504974);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 0.44444444403052330017);
}

TEST(FloatxOperationsTest, ADD_4_28_simple)
{
    using T = flx::floatx<4, 28>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333302289247513);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 0.44444444403052330017);
}

TEST(FloatxOperationsTest, ADD_4_29_simple)
{
    using T = flx::floatx<4, 29>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333348855376244);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 0.44444444449618458748);
}

TEST(FloatxOperationsTest, ADD_4_30_simple)
{
    using T = flx::floatx<4, 30>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333325572311878);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 0.44444444449618458748);
}

TEST(FloatxOperationsTest, ADD_4_31_simple)
{
    using T = flx::floatx<4, 31>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333337213844061);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 0.44444444449618458748);
}

TEST(FloatxOperationsTest, ADD_4_32_simple)
{
    using T = flx::floatx<4, 32>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333331393077970);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 0.44444444443797692657);
}

TEST(FloatxOperationsTest, ADD_4_33_simple)
{
    using T = flx::floatx<4, 33>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333334303461015);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 0.44444444443797692657);
}

TEST(FloatxOperationsTest, ADD_4_34_simple)
{
    using T = flx::floatx<4, 34>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333332848269492);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 0.44444444443797692657);
}

TEST(FloatxOperationsTest, ADD_4_35_simple)
{
    using T = flx::floatx<4, 35>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333575865254);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 0.44444444444525288418);
}

TEST(FloatxOperationsTest, ADD_4_36_simple)
{
    using T = flx::floatx<4, 36>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333212067373);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 0.44444444444525288418);
}

TEST(FloatxOperationsTest, ADD_4_37_simple)
{
    using T = flx::floatx<4, 37>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333393966313);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 0.44444444444525288418);
}

TEST(FloatxOperationsTest, ADD_4_38_simple)
{
    using T = flx::floatx<4, 38>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333303016843);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 0.44444444444434338948);
}

TEST(FloatxOperationsTest, ADD_4_39_simple)
{
    using T = flx::floatx<4, 39>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333348491578);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 0.44444444444434338948);
}

TEST(FloatxOperationsTest, ADD_4_40_simple)
{
    using T = flx::floatx<4, 40>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333325754211);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 0.44444444444434338948);
}

TEST(FloatxOperationsTest, ADD_4_41_simple)
{
    using T = flx::floatx<4, 41>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333337122895);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 0.44444444444445707632);
}

TEST(FloatxOperationsTest, ADD_4_42_simple)
{
    using T = flx::floatx<4, 42>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333331438553);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 0.44444444444445707632);
}

TEST(FloatxOperationsTest, ADD_4_43_simple)
{
    using T = flx::floatx<4, 43>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333334280724);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 0.44444444444445707632);
}

TEST(FloatxOperationsTest, ADD_4_44_simple)
{
    using T = flx::floatx<4, 44>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333332859638);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 0.44444444444444286546);
}

TEST(FloatxOperationsTest, ADD_4_45_simple)
{
    using T = flx::floatx<4, 45>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333570181);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 0.44444444444444286546);
}

TEST(FloatxOperationsTest, ADD_4_46_simple)
{
    using T = flx::floatx<4, 46>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333214910);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 0.44444444444444286546);
}

TEST(FloatxOperationsTest, ADD_4_47_simple)
{
    using T = flx::floatx<4, 47>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333392545);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 0.44444444444444464182);
}

TEST(FloatxOperationsTest, ADD_4_48_simple)
{
    using T = flx::floatx<4, 48>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333303727);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 0.44444444444444464182);
}

TEST(FloatxOperationsTest, ADD_4_49_simple)
{
    using T = flx::floatx<4, 49>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333348136);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 0.44444444444444464182);
}

TEST(FloatxOperationsTest, ADD_4_50_simple)
{
    using T = flx::floatx<4, 50>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333325932);
    EXPECT_EQ(double(b), 0.11111111111111110494);
    EXPECT_EQ(double(c), 0.44444444444444441977);
}

TEST(FloatxOperationsTest, ADD_4_51_simple)
{
    using T = flx::floatx<4, 51>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333325932);
    EXPECT_EQ(double(b), 0.11111111111111110494);
    EXPECT_EQ(double(c), 0.44444444444444441977);
}

TEST(FloatxOperationsTest, ADD_4_52_simple)
{
    using T = flx::floatx<4, 52>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333331483);
    EXPECT_EQ(double(b), 0.11111111111111110494);
    EXPECT_EQ(double(c), 0.44444444444444441977);
}

TEST(FloatxOperationsTest, ADD_5_1_simple)
{
    using T = flx::floatx<5, 1>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.37500000000000000000);
    EXPECT_EQ(double(b), 0.12500000000000000000);
    EXPECT_EQ(double(c), 0.50000000000000000000);
}

TEST(FloatxOperationsTest, ADD_5_2_simple)
{
    using T = flx::floatx<5, 2>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.31250000000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 0.43750000000000000000);
}

TEST(FloatxOperationsTest, ADD_5_3_simple)
{
    using T = flx::floatx<5, 3>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.34375000000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 0.43750000000000000000);
}

TEST(FloatxOperationsTest, ADD_5_4_simple)
{
    using T = flx::floatx<5, 4>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.32812500000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 0.43750000000000000000);
}

TEST(FloatxOperationsTest, ADD_5_5_simple)
{
    using T = flx::floatx<5, 5>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33593750000000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 0.44531250000000000000);
}

TEST(FloatxOperationsTest, ADD_5_6_simple)
{
    using T = flx::floatx<5, 6>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33203125000000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 0.44531250000000000000);
}

TEST(FloatxOperationsTest, ADD_5_7_simple)
{
    using T = flx::floatx<5, 7>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33398437500000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 0.44531250000000000000);
}

TEST(FloatxOperationsTest, ADD_5_8_simple)
{
    using T = flx::floatx<5, 8>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33300781250000000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 0.44433593750000000000);
}

TEST(FloatxOperationsTest, ADD_5_9_simple)
{
    using T = flx::floatx<5, 9>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33349609375000000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 0.44433593750000000000);
}

TEST(FloatxOperationsTest, ADD_5_10_simple)
{
    using T = flx::floatx<5, 10>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33325195312500000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 0.44433593750000000000);
}

TEST(FloatxOperationsTest, ADD_5_11_simple)
{
    using T = flx::floatx<5, 11>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33337402343750000000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 0.44445800781250000000);
}

TEST(FloatxOperationsTest, ADD_5_12_simple)
{
    using T = flx::floatx<5, 12>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33331298828125000000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 0.44445800781250000000);
}

TEST(FloatxOperationsTest, ADD_5_13_simple)
{
    using T = flx::floatx<5, 13>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33334350585937500000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 0.44445800781250000000);
}

TEST(FloatxOperationsTest, ADD_5_14_simple)
{
    using T = flx::floatx<5, 14>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33332824707031250000);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 0.44444274902343750000);
}

TEST(FloatxOperationsTest, ADD_5_15_simple)
{
    using T = flx::floatx<5, 15>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333587646484375000);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 0.44444274902343750000);
}

TEST(FloatxOperationsTest, ADD_5_16_simple)
{
    using T = flx::floatx<5, 16>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333206176757812500);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 0.44444274902343750000);
}

TEST(FloatxOperationsTest, ADD_5_17_simple)
{
    using T = flx::floatx<5, 17>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333396911621093750);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 0.44444465637207031250);
}

TEST(FloatxOperationsTest, ADD_5_18_simple)
{
    using T = flx::floatx<5, 18>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333301544189453125);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 0.44444465637207031250);
}

TEST(FloatxOperationsTest, ADD_5_19_simple)
{
    using T = flx::floatx<5, 19>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333349227905273438);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 0.44444465637207031250);
}

TEST(FloatxOperationsTest, ADD_5_20_simple)
{
    using T = flx::floatx<5, 20>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333325386047363281);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 0.44444441795349121094);
}

TEST(FloatxOperationsTest, ADD_5_21_simple)
{
    using T = flx::floatx<5, 21>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333337306976318359);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 0.44444441795349121094);
}

TEST(FloatxOperationsTest, ADD_5_22_simple)
{
    using T = flx::floatx<5, 22>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333331346511840820);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 0.44444441795349121094);
}

TEST(FloatxOperationsTest, ADD_5_23_simple)
{
    using T = flx::floatx<5, 23>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333334326744079590);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 0.44444444775581359863);
}

TEST(FloatxOperationsTest, ADD_5_24_simple)
{
    using T = flx::floatx<5, 24>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333332836627960205);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 0.44444444775581359863);
}

TEST(FloatxOperationsTest, ADD_5_25_simple)
{
    using T = flx::floatx<5, 25>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333581686019897);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 0.44444444775581359863);
}

TEST(FloatxOperationsTest, ADD_5_26_simple)
{
    using T = flx::floatx<5, 26>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333209156990051);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 0.44444444403052330017);
}

TEST(FloatxOperationsTest, ADD_5_27_simple)
{
    using T = flx::floatx<5, 27>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333395421504974);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 0.44444444403052330017);
}

TEST(FloatxOperationsTest, ADD_5_28_simple)
{
    using T = flx::floatx<5, 28>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333302289247513);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 0.44444444403052330017);
}

TEST(FloatxOperationsTest, ADD_5_29_simple)
{
    using T = flx::floatx<5, 29>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333348855376244);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 0.44444444449618458748);
}

TEST(FloatxOperationsTest, ADD_5_30_simple)
{
    using T = flx::floatx<5, 30>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333325572311878);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 0.44444444449618458748);
}

TEST(FloatxOperationsTest, ADD_5_31_simple)
{
    using T = flx::floatx<5, 31>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333337213844061);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 0.44444444449618458748);
}

TEST(FloatxOperationsTest, ADD_5_32_simple)
{
    using T = flx::floatx<5, 32>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333331393077970);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 0.44444444443797692657);
}

TEST(FloatxOperationsTest, ADD_5_33_simple)
{
    using T = flx::floatx<5, 33>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333334303461015);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 0.44444444443797692657);
}

TEST(FloatxOperationsTest, ADD_5_34_simple)
{
    using T = flx::floatx<5, 34>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333332848269492);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 0.44444444443797692657);
}

TEST(FloatxOperationsTest, ADD_5_35_simple)
{
    using T = flx::floatx<5, 35>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333575865254);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 0.44444444444525288418);
}

TEST(FloatxOperationsTest, ADD_5_36_simple)
{
    using T = flx::floatx<5, 36>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333212067373);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 0.44444444444525288418);
}

TEST(FloatxOperationsTest, ADD_5_37_simple)
{
    using T = flx::floatx<5, 37>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333393966313);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 0.44444444444525288418);
}

TEST(FloatxOperationsTest, ADD_5_38_simple)
{
    using T = flx::floatx<5, 38>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333303016843);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 0.44444444444434338948);
}

TEST(FloatxOperationsTest, ADD_5_39_simple)
{
    using T = flx::floatx<5, 39>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333348491578);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 0.44444444444434338948);
}

TEST(FloatxOperationsTest, ADD_5_40_simple)
{
    using T = flx::floatx<5, 40>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333325754211);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 0.44444444444434338948);
}

TEST(FloatxOperationsTest, ADD_5_41_simple)
{
    using T = flx::floatx<5, 41>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333337122895);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 0.44444444444445707632);
}

TEST(FloatxOperationsTest, ADD_5_42_simple)
{
    using T = flx::floatx<5, 42>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333331438553);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 0.44444444444445707632);
}

TEST(FloatxOperationsTest, ADD_5_43_simple)
{
    using T = flx::floatx<5, 43>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333334280724);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 0.44444444444445707632);
}

TEST(FloatxOperationsTest, ADD_5_44_simple)
{
    using T = flx::floatx<5, 44>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333332859638);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 0.44444444444444286546);
}

TEST(FloatxOperationsTest, ADD_5_45_simple)
{
    using T = flx::floatx<5, 45>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333570181);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 0.44444444444444286546);
}

TEST(FloatxOperationsTest, ADD_5_46_simple)
{
    using T = flx::floatx<5, 46>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333214910);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 0.44444444444444286546);
}

TEST(FloatxOperationsTest, ADD_5_47_simple)
{
    using T = flx::floatx<5, 47>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333392545);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 0.44444444444444464182);
}

TEST(FloatxOperationsTest, ADD_5_48_simple)
{
    using T = flx::floatx<5, 48>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333303727);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 0.44444444444444464182);
}

TEST(FloatxOperationsTest, ADD_5_49_simple)
{
    using T = flx::floatx<5, 49>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333348136);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 0.44444444444444464182);
}

TEST(FloatxOperationsTest, ADD_5_50_simple)
{
    using T = flx::floatx<5, 50>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333325932);
    EXPECT_EQ(double(b), 0.11111111111111110494);
    EXPECT_EQ(double(c), 0.44444444444444441977);
}

TEST(FloatxOperationsTest, ADD_5_51_simple)
{
    using T = flx::floatx<5, 51>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333325932);
    EXPECT_EQ(double(b), 0.11111111111111110494);
    EXPECT_EQ(double(c), 0.44444444444444441977);
}

TEST(FloatxOperationsTest, ADD_5_52_simple)
{
    using T = flx::floatx<5, 52>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333331483);
    EXPECT_EQ(double(b), 0.11111111111111110494);
    EXPECT_EQ(double(c), 0.44444444444444441977);
}

TEST(FloatxOperationsTest, ADD_6_1_simple)
{
    using T = flx::floatx<6, 1>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.37500000000000000000);
    EXPECT_EQ(double(b), 0.12500000000000000000);
    EXPECT_EQ(double(c), 0.50000000000000000000);
}

TEST(FloatxOperationsTest, ADD_6_2_simple)
{
    using T = flx::floatx<6, 2>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.31250000000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 0.43750000000000000000);
}

TEST(FloatxOperationsTest, ADD_6_3_simple)
{
    using T = flx::floatx<6, 3>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.34375000000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 0.43750000000000000000);
}

TEST(FloatxOperationsTest, ADD_6_4_simple)
{
    using T = flx::floatx<6, 4>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.32812500000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 0.43750000000000000000);
}

TEST(FloatxOperationsTest, ADD_6_5_simple)
{
    using T = flx::floatx<6, 5>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33593750000000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 0.44531250000000000000);
}

TEST(FloatxOperationsTest, ADD_6_6_simple)
{
    using T = flx::floatx<6, 6>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33203125000000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 0.44531250000000000000);
}

TEST(FloatxOperationsTest, ADD_6_7_simple)
{
    using T = flx::floatx<6, 7>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33398437500000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 0.44531250000000000000);
}

TEST(FloatxOperationsTest, ADD_6_8_simple)
{
    using T = flx::floatx<6, 8>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33300781250000000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 0.44433593750000000000);
}

TEST(FloatxOperationsTest, ADD_6_9_simple)
{
    using T = flx::floatx<6, 9>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33349609375000000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 0.44433593750000000000);
}

TEST(FloatxOperationsTest, ADD_6_10_simple)
{
    using T = flx::floatx<6, 10>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33325195312500000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 0.44433593750000000000);
}

TEST(FloatxOperationsTest, ADD_6_11_simple)
{
    using T = flx::floatx<6, 11>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33337402343750000000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 0.44445800781250000000);
}

TEST(FloatxOperationsTest, ADD_6_12_simple)
{
    using T = flx::floatx<6, 12>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33331298828125000000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 0.44445800781250000000);
}

TEST(FloatxOperationsTest, ADD_6_13_simple)
{
    using T = flx::floatx<6, 13>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33334350585937500000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 0.44445800781250000000);
}

TEST(FloatxOperationsTest, ADD_6_14_simple)
{
    using T = flx::floatx<6, 14>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33332824707031250000);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 0.44444274902343750000);
}

TEST(FloatxOperationsTest, ADD_6_15_simple)
{
    using T = flx::floatx<6, 15>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333587646484375000);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 0.44444274902343750000);
}

TEST(FloatxOperationsTest, ADD_6_16_simple)
{
    using T = flx::floatx<6, 16>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333206176757812500);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 0.44444274902343750000);
}

TEST(FloatxOperationsTest, ADD_6_17_simple)
{
    using T = flx::floatx<6, 17>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333396911621093750);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 0.44444465637207031250);
}

TEST(FloatxOperationsTest, ADD_6_18_simple)
{
    using T = flx::floatx<6, 18>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333301544189453125);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 0.44444465637207031250);
}

TEST(FloatxOperationsTest, ADD_6_19_simple)
{
    using T = flx::floatx<6, 19>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333349227905273438);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 0.44444465637207031250);
}

TEST(FloatxOperationsTest, ADD_6_20_simple)
{
    using T = flx::floatx<6, 20>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333325386047363281);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 0.44444441795349121094);
}

TEST(FloatxOperationsTest, ADD_6_21_simple)
{
    using T = flx::floatx<6, 21>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333337306976318359);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 0.44444441795349121094);
}

TEST(FloatxOperationsTest, ADD_6_22_simple)
{
    using T = flx::floatx<6, 22>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333331346511840820);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 0.44444441795349121094);
}

TEST(FloatxOperationsTest, ADD_6_23_simple)
{
    using T = flx::floatx<6, 23>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333334326744079590);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 0.44444444775581359863);
}

TEST(FloatxOperationsTest, ADD_6_24_simple)
{
    using T = flx::floatx<6, 24>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333332836627960205);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 0.44444444775581359863);
}

TEST(FloatxOperationsTest, ADD_6_25_simple)
{
    using T = flx::floatx<6, 25>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333581686019897);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 0.44444444775581359863);
}

TEST(FloatxOperationsTest, ADD_6_26_simple)
{
    using T = flx::floatx<6, 26>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333209156990051);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 0.44444444403052330017);
}

TEST(FloatxOperationsTest, ADD_6_27_simple)
{
    using T = flx::floatx<6, 27>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333395421504974);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 0.44444444403052330017);
}

TEST(FloatxOperationsTest, ADD_6_28_simple)
{
    using T = flx::floatx<6, 28>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333302289247513);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 0.44444444403052330017);
}

TEST(FloatxOperationsTest, ADD_6_29_simple)
{
    using T = flx::floatx<6, 29>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333348855376244);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 0.44444444449618458748);
}

TEST(FloatxOperationsTest, ADD_6_30_simple)
{
    using T = flx::floatx<6, 30>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333325572311878);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 0.44444444449618458748);
}

TEST(FloatxOperationsTest, ADD_6_31_simple)
{
    using T = flx::floatx<6, 31>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333337213844061);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 0.44444444449618458748);
}

TEST(FloatxOperationsTest, ADD_6_32_simple)
{
    using T = flx::floatx<6, 32>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333331393077970);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 0.44444444443797692657);
}

TEST(FloatxOperationsTest, ADD_6_33_simple)
{
    using T = flx::floatx<6, 33>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333334303461015);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 0.44444444443797692657);
}

TEST(FloatxOperationsTest, ADD_6_34_simple)
{
    using T = flx::floatx<6, 34>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333332848269492);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 0.44444444443797692657);
}

TEST(FloatxOperationsTest, ADD_6_35_simple)
{
    using T = flx::floatx<6, 35>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333575865254);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 0.44444444444525288418);
}

TEST(FloatxOperationsTest, ADD_6_36_simple)
{
    using T = flx::floatx<6, 36>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333212067373);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 0.44444444444525288418);
}

TEST(FloatxOperationsTest, ADD_6_37_simple)
{
    using T = flx::floatx<6, 37>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333393966313);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 0.44444444444525288418);
}

TEST(FloatxOperationsTest, ADD_6_38_simple)
{
    using T = flx::floatx<6, 38>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333303016843);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 0.44444444444434338948);
}

TEST(FloatxOperationsTest, ADD_6_39_simple)
{
    using T = flx::floatx<6, 39>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333348491578);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 0.44444444444434338948);
}

TEST(FloatxOperationsTest, ADD_6_40_simple)
{
    using T = flx::floatx<6, 40>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333325754211);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 0.44444444444434338948);
}

TEST(FloatxOperationsTest, ADD_6_41_simple)
{
    using T = flx::floatx<6, 41>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333337122895);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 0.44444444444445707632);
}

TEST(FloatxOperationsTest, ADD_6_42_simple)
{
    using T = flx::floatx<6, 42>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333331438553);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 0.44444444444445707632);
}

TEST(FloatxOperationsTest, ADD_6_43_simple)
{
    using T = flx::floatx<6, 43>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333334280724);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 0.44444444444445707632);
}

TEST(FloatxOperationsTest, ADD_6_44_simple)
{
    using T = flx::floatx<6, 44>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333332859638);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 0.44444444444444286546);
}

TEST(FloatxOperationsTest, ADD_6_45_simple)
{
    using T = flx::floatx<6, 45>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333570181);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 0.44444444444444286546);
}

TEST(FloatxOperationsTest, ADD_6_46_simple)
{
    using T = flx::floatx<6, 46>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333214910);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 0.44444444444444286546);
}

TEST(FloatxOperationsTest, ADD_6_47_simple)
{
    using T = flx::floatx<6, 47>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333392545);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 0.44444444444444464182);
}

TEST(FloatxOperationsTest, ADD_6_48_simple)
{
    using T = flx::floatx<6, 48>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333303727);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 0.44444444444444464182);
}

TEST(FloatxOperationsTest, ADD_6_49_simple)
{
    using T = flx::floatx<6, 49>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333348136);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 0.44444444444444464182);
}

TEST(FloatxOperationsTest, ADD_6_50_simple)
{
    using T = flx::floatx<6, 50>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333325932);
    EXPECT_EQ(double(b), 0.11111111111111110494);
    EXPECT_EQ(double(c), 0.44444444444444441977);
}

TEST(FloatxOperationsTest, ADD_6_51_simple)
{
    using T = flx::floatx<6, 51>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333325932);
    EXPECT_EQ(double(b), 0.11111111111111110494);
    EXPECT_EQ(double(c), 0.44444444444444441977);
}

TEST(FloatxOperationsTest, ADD_6_52_simple)
{
    using T = flx::floatx<6, 52>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333331483);
    EXPECT_EQ(double(b), 0.11111111111111110494);
    EXPECT_EQ(double(c), 0.44444444444444441977);
}

TEST(FloatxOperationsTest, ADD_7_1_simple)
{
    using T = flx::floatx<7, 1>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.37500000000000000000);
    EXPECT_EQ(double(b), 0.12500000000000000000);
    EXPECT_EQ(double(c), 0.50000000000000000000);
}

TEST(FloatxOperationsTest, ADD_7_2_simple)
{
    using T = flx::floatx<7, 2>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.31250000000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 0.43750000000000000000);
}

TEST(FloatxOperationsTest, ADD_7_3_simple)
{
    using T = flx::floatx<7, 3>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.34375000000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 0.43750000000000000000);
}

TEST(FloatxOperationsTest, ADD_7_4_simple)
{
    using T = flx::floatx<7, 4>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.32812500000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 0.43750000000000000000);
}

TEST(FloatxOperationsTest, ADD_7_5_simple)
{
    using T = flx::floatx<7, 5>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33593750000000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 0.44531250000000000000);
}

TEST(FloatxOperationsTest, ADD_7_6_simple)
{
    using T = flx::floatx<7, 6>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33203125000000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 0.44531250000000000000);
}

TEST(FloatxOperationsTest, ADD_7_7_simple)
{
    using T = flx::floatx<7, 7>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33398437500000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 0.44531250000000000000);
}

TEST(FloatxOperationsTest, ADD_7_8_simple)
{
    using T = flx::floatx<7, 8>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33300781250000000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 0.44433593750000000000);
}

TEST(FloatxOperationsTest, ADD_7_9_simple)
{
    using T = flx::floatx<7, 9>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33349609375000000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 0.44433593750000000000);
}

TEST(FloatxOperationsTest, ADD_7_10_simple)
{
    using T = flx::floatx<7, 10>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33325195312500000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 0.44433593750000000000);
}

TEST(FloatxOperationsTest, ADD_7_11_simple)
{
    using T = flx::floatx<7, 11>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33337402343750000000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 0.44445800781250000000);
}

TEST(FloatxOperationsTest, ADD_7_12_simple)
{
    using T = flx::floatx<7, 12>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33331298828125000000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 0.44445800781250000000);
}

TEST(FloatxOperationsTest, ADD_7_13_simple)
{
    using T = flx::floatx<7, 13>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33334350585937500000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 0.44445800781250000000);
}

TEST(FloatxOperationsTest, ADD_7_14_simple)
{
    using T = flx::floatx<7, 14>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33332824707031250000);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 0.44444274902343750000);
}

TEST(FloatxOperationsTest, ADD_7_15_simple)
{
    using T = flx::floatx<7, 15>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333587646484375000);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 0.44444274902343750000);
}

TEST(FloatxOperationsTest, ADD_7_16_simple)
{
    using T = flx::floatx<7, 16>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333206176757812500);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 0.44444274902343750000);
}

TEST(FloatxOperationsTest, ADD_7_17_simple)
{
    using T = flx::floatx<7, 17>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333396911621093750);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 0.44444465637207031250);
}

TEST(FloatxOperationsTest, ADD_7_18_simple)
{
    using T = flx::floatx<7, 18>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333301544189453125);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 0.44444465637207031250);
}

TEST(FloatxOperationsTest, ADD_7_19_simple)
{
    using T = flx::floatx<7, 19>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333349227905273438);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 0.44444465637207031250);
}

TEST(FloatxOperationsTest, ADD_7_20_simple)
{
    using T = flx::floatx<7, 20>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333325386047363281);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 0.44444441795349121094);
}

TEST(FloatxOperationsTest, ADD_7_21_simple)
{
    using T = flx::floatx<7, 21>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333337306976318359);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 0.44444441795349121094);
}

TEST(FloatxOperationsTest, ADD_7_22_simple)
{
    using T = flx::floatx<7, 22>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333331346511840820);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 0.44444441795349121094);
}

TEST(FloatxOperationsTest, ADD_7_23_simple)
{
    using T = flx::floatx<7, 23>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333334326744079590);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 0.44444444775581359863);
}

TEST(FloatxOperationsTest, ADD_7_24_simple)
{
    using T = flx::floatx<7, 24>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333332836627960205);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 0.44444444775581359863);
}

TEST(FloatxOperationsTest, ADD_7_25_simple)
{
    using T = flx::floatx<7, 25>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333581686019897);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 0.44444444775581359863);
}

TEST(FloatxOperationsTest, ADD_7_26_simple)
{
    using T = flx::floatx<7, 26>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333209156990051);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 0.44444444403052330017);
}

TEST(FloatxOperationsTest, ADD_7_27_simple)
{
    using T = flx::floatx<7, 27>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333395421504974);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 0.44444444403052330017);
}

TEST(FloatxOperationsTest, ADD_7_28_simple)
{
    using T = flx::floatx<7, 28>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333302289247513);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 0.44444444403052330017);
}

TEST(FloatxOperationsTest, ADD_7_29_simple)
{
    using T = flx::floatx<7, 29>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333348855376244);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 0.44444444449618458748);
}

TEST(FloatxOperationsTest, ADD_7_30_simple)
{
    using T = flx::floatx<7, 30>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333325572311878);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 0.44444444449618458748);
}

TEST(FloatxOperationsTest, ADD_7_31_simple)
{
    using T = flx::floatx<7, 31>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333337213844061);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 0.44444444449618458748);
}

TEST(FloatxOperationsTest, ADD_7_32_simple)
{
    using T = flx::floatx<7, 32>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333331393077970);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 0.44444444443797692657);
}

TEST(FloatxOperationsTest, ADD_7_33_simple)
{
    using T = flx::floatx<7, 33>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333334303461015);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 0.44444444443797692657);
}

TEST(FloatxOperationsTest, ADD_7_34_simple)
{
    using T = flx::floatx<7, 34>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333332848269492);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 0.44444444443797692657);
}

TEST(FloatxOperationsTest, ADD_7_35_simple)
{
    using T = flx::floatx<7, 35>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333575865254);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 0.44444444444525288418);
}

TEST(FloatxOperationsTest, ADD_7_36_simple)
{
    using T = flx::floatx<7, 36>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333212067373);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 0.44444444444525288418);
}

TEST(FloatxOperationsTest, ADD_7_37_simple)
{
    using T = flx::floatx<7, 37>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333393966313);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 0.44444444444525288418);
}

TEST(FloatxOperationsTest, ADD_7_38_simple)
{
    using T = flx::floatx<7, 38>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333303016843);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 0.44444444444434338948);
}

TEST(FloatxOperationsTest, ADD_7_39_simple)
{
    using T = flx::floatx<7, 39>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333348491578);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 0.44444444444434338948);
}

TEST(FloatxOperationsTest, ADD_7_40_simple)
{
    using T = flx::floatx<7, 40>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333325754211);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 0.44444444444434338948);
}

TEST(FloatxOperationsTest, ADD_7_41_simple)
{
    using T = flx::floatx<7, 41>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333337122895);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 0.44444444444445707632);
}

TEST(FloatxOperationsTest, ADD_7_42_simple)
{
    using T = flx::floatx<7, 42>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333331438553);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 0.44444444444445707632);
}

TEST(FloatxOperationsTest, ADD_7_43_simple)
{
    using T = flx::floatx<7, 43>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333334280724);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 0.44444444444445707632);
}

TEST(FloatxOperationsTest, ADD_7_44_simple)
{
    using T = flx::floatx<7, 44>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333332859638);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 0.44444444444444286546);
}

TEST(FloatxOperationsTest, ADD_7_45_simple)
{
    using T = flx::floatx<7, 45>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333570181);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 0.44444444444444286546);
}

TEST(FloatxOperationsTest, ADD_7_46_simple)
{
    using T = flx::floatx<7, 46>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333214910);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 0.44444444444444286546);
}

TEST(FloatxOperationsTest, ADD_7_47_simple)
{
    using T = flx::floatx<7, 47>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333392545);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 0.44444444444444464182);
}

TEST(FloatxOperationsTest, ADD_7_48_simple)
{
    using T = flx::floatx<7, 48>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333303727);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 0.44444444444444464182);
}

TEST(FloatxOperationsTest, ADD_7_49_simple)
{
    using T = flx::floatx<7, 49>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333348136);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 0.44444444444444464182);
}

TEST(FloatxOperationsTest, ADD_7_50_simple)
{
    using T = flx::floatx<7, 50>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333325932);
    EXPECT_EQ(double(b), 0.11111111111111110494);
    EXPECT_EQ(double(c), 0.44444444444444441977);
}

TEST(FloatxOperationsTest, ADD_7_51_simple)
{
    using T = flx::floatx<7, 51>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333325932);
    EXPECT_EQ(double(b), 0.11111111111111110494);
    EXPECT_EQ(double(c), 0.44444444444444441977);
}

TEST(FloatxOperationsTest, ADD_7_52_simple)
{
    using T = flx::floatx<7, 52>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333331483);
    EXPECT_EQ(double(b), 0.11111111111111110494);
    EXPECT_EQ(double(c), 0.44444444444444441977);
}

TEST(FloatxOperationsTest, ADD_8_1_simple)
{
    using T = flx::floatx<8, 1>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.37500000000000000000);
    EXPECT_EQ(double(b), 0.12500000000000000000);
    EXPECT_EQ(double(c), 0.50000000000000000000);
}

TEST(FloatxOperationsTest, ADD_8_2_simple)
{
    using T = flx::floatx<8, 2>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.31250000000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 0.43750000000000000000);
}

TEST(FloatxOperationsTest, ADD_8_3_simple)
{
    using T = flx::floatx<8, 3>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.34375000000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 0.43750000000000000000);
}

TEST(FloatxOperationsTest, ADD_8_4_simple)
{
    using T = flx::floatx<8, 4>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.32812500000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 0.43750000000000000000);
}

TEST(FloatxOperationsTest, ADD_8_5_simple)
{
    using T = flx::floatx<8, 5>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33593750000000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 0.44531250000000000000);
}

TEST(FloatxOperationsTest, ADD_8_6_simple)
{
    using T = flx::floatx<8, 6>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33203125000000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 0.44531250000000000000);
}

TEST(FloatxOperationsTest, ADD_8_7_simple)
{
    using T = flx::floatx<8, 7>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33398437500000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 0.44531250000000000000);
}

TEST(FloatxOperationsTest, ADD_8_8_simple)
{
    using T = flx::floatx<8, 8>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33300781250000000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 0.44433593750000000000);
}

TEST(FloatxOperationsTest, ADD_8_9_simple)
{
    using T = flx::floatx<8, 9>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33349609375000000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 0.44433593750000000000);
}

TEST(FloatxOperationsTest, ADD_8_10_simple)
{
    using T = flx::floatx<8, 10>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33325195312500000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 0.44433593750000000000);
}

TEST(FloatxOperationsTest, ADD_8_11_simple)
{
    using T = flx::floatx<8, 11>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33337402343750000000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 0.44445800781250000000);
}

TEST(FloatxOperationsTest, ADD_8_12_simple)
{
    using T = flx::floatx<8, 12>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33331298828125000000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 0.44445800781250000000);
}

TEST(FloatxOperationsTest, ADD_8_13_simple)
{
    using T = flx::floatx<8, 13>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33334350585937500000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 0.44445800781250000000);
}

TEST(FloatxOperationsTest, ADD_8_14_simple)
{
    using T = flx::floatx<8, 14>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33332824707031250000);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 0.44444274902343750000);
}

TEST(FloatxOperationsTest, ADD_8_15_simple)
{
    using T = flx::floatx<8, 15>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333587646484375000);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 0.44444274902343750000);
}

TEST(FloatxOperationsTest, ADD_8_16_simple)
{
    using T = flx::floatx<8, 16>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333206176757812500);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 0.44444274902343750000);
}

TEST(FloatxOperationsTest, ADD_8_17_simple)
{
    using T = flx::floatx<8, 17>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333396911621093750);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 0.44444465637207031250);
}

TEST(FloatxOperationsTest, ADD_8_18_simple)
{
    using T = flx::floatx<8, 18>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333301544189453125);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 0.44444465637207031250);
}

TEST(FloatxOperationsTest, ADD_8_19_simple)
{
    using T = flx::floatx<8, 19>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333349227905273438);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 0.44444465637207031250);
}

TEST(FloatxOperationsTest, ADD_8_20_simple)
{
    using T = flx::floatx<8, 20>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333325386047363281);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 0.44444441795349121094);
}

TEST(FloatxOperationsTest, ADD_8_21_simple)
{
    using T = flx::floatx<8, 21>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333337306976318359);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 0.44444441795349121094);
}

TEST(FloatxOperationsTest, ADD_8_22_simple)
{
    using T = flx::floatx<8, 22>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333331346511840820);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 0.44444441795349121094);
}

TEST(FloatxOperationsTest, ADD_8_23_simple)
{
    using T = flx::floatx<8, 23>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333334326744079590);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 0.44444444775581359863);
}

TEST(FloatxOperationsTest, ADD_8_24_simple)
{
    using T = flx::floatx<8, 24>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333332836627960205);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 0.44444444775581359863);
}

TEST(FloatxOperationsTest, ADD_8_25_simple)
{
    using T = flx::floatx<8, 25>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333581686019897);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 0.44444444775581359863);
}

TEST(FloatxOperationsTest, ADD_8_26_simple)
{
    using T = flx::floatx<8, 26>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333209156990051);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 0.44444444403052330017);
}

TEST(FloatxOperationsTest, ADD_8_27_simple)
{
    using T = flx::floatx<8, 27>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333395421504974);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 0.44444444403052330017);
}

TEST(FloatxOperationsTest, ADD_8_28_simple)
{
    using T = flx::floatx<8, 28>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333302289247513);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 0.44444444403052330017);
}

TEST(FloatxOperationsTest, ADD_8_29_simple)
{
    using T = flx::floatx<8, 29>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333348855376244);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 0.44444444449618458748);
}

TEST(FloatxOperationsTest, ADD_8_30_simple)
{
    using T = flx::floatx<8, 30>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333325572311878);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 0.44444444449618458748);
}

TEST(FloatxOperationsTest, ADD_8_31_simple)
{
    using T = flx::floatx<8, 31>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333337213844061);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 0.44444444449618458748);
}

TEST(FloatxOperationsTest, ADD_8_32_simple)
{
    using T = flx::floatx<8, 32>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333331393077970);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 0.44444444443797692657);
}

TEST(FloatxOperationsTest, ADD_8_33_simple)
{
    using T = flx::floatx<8, 33>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333334303461015);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 0.44444444443797692657);
}

TEST(FloatxOperationsTest, ADD_8_34_simple)
{
    using T = flx::floatx<8, 34>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333332848269492);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 0.44444444443797692657);
}

TEST(FloatxOperationsTest, ADD_8_35_simple)
{
    using T = flx::floatx<8, 35>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333575865254);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 0.44444444444525288418);
}

TEST(FloatxOperationsTest, ADD_8_36_simple)
{
    using T = flx::floatx<8, 36>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333212067373);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 0.44444444444525288418);
}

TEST(FloatxOperationsTest, ADD_8_37_simple)
{
    using T = flx::floatx<8, 37>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333393966313);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 0.44444444444525288418);
}

TEST(FloatxOperationsTest, ADD_8_38_simple)
{
    using T = flx::floatx<8, 38>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333303016843);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 0.44444444444434338948);
}

TEST(FloatxOperationsTest, ADD_8_39_simple)
{
    using T = flx::floatx<8, 39>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333348491578);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 0.44444444444434338948);
}

TEST(FloatxOperationsTest, ADD_8_40_simple)
{
    using T = flx::floatx<8, 40>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333325754211);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 0.44444444444434338948);
}

TEST(FloatxOperationsTest, ADD_8_41_simple)
{
    using T = flx::floatx<8, 41>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333337122895);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 0.44444444444445707632);
}

TEST(FloatxOperationsTest, ADD_8_42_simple)
{
    using T = flx::floatx<8, 42>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333331438553);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 0.44444444444445707632);
}

TEST(FloatxOperationsTest, ADD_8_43_simple)
{
    using T = flx::floatx<8, 43>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333334280724);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 0.44444444444445707632);
}

TEST(FloatxOperationsTest, ADD_8_44_simple)
{
    using T = flx::floatx<8, 44>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333332859638);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 0.44444444444444286546);
}

TEST(FloatxOperationsTest, ADD_8_45_simple)
{
    using T = flx::floatx<8, 45>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333570181);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 0.44444444444444286546);
}

TEST(FloatxOperationsTest, ADD_8_46_simple)
{
    using T = flx::floatx<8, 46>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333214910);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 0.44444444444444286546);
}

TEST(FloatxOperationsTest, ADD_8_47_simple)
{
    using T = flx::floatx<8, 47>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333392545);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 0.44444444444444464182);
}

TEST(FloatxOperationsTest, ADD_8_48_simple)
{
    using T = flx::floatx<8, 48>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333303727);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 0.44444444444444464182);
}

TEST(FloatxOperationsTest, ADD_8_49_simple)
{
    using T = flx::floatx<8, 49>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333348136);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 0.44444444444444464182);
}

TEST(FloatxOperationsTest, ADD_8_50_simple)
{
    using T = flx::floatx<8, 50>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333325932);
    EXPECT_EQ(double(b), 0.11111111111111110494);
    EXPECT_EQ(double(c), 0.44444444444444441977);
}

TEST(FloatxOperationsTest, ADD_8_51_simple)
{
    using T = flx::floatx<8, 51>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333325932);
    EXPECT_EQ(double(b), 0.11111111111111110494);
    EXPECT_EQ(double(c), 0.44444444444444441977);
}

TEST(FloatxOperationsTest, ADD_8_52_simple)
{
    using T = flx::floatx<8, 52>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333331483);
    EXPECT_EQ(double(b), 0.11111111111111110494);
    EXPECT_EQ(double(c), 0.44444444444444441977);
}

TEST(FloatxOperationsTest, ADD_9_1_simple)
{
    using T = flx::floatx<9, 1>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.37500000000000000000);
    EXPECT_EQ(double(b), 0.12500000000000000000);
    EXPECT_EQ(double(c), 0.50000000000000000000);
}

TEST(FloatxOperationsTest, ADD_9_2_simple)
{
    using T = flx::floatx<9, 2>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.31250000000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 0.43750000000000000000);
}

TEST(FloatxOperationsTest, ADD_9_3_simple)
{
    using T = flx::floatx<9, 3>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.34375000000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 0.43750000000000000000);
}

TEST(FloatxOperationsTest, ADD_9_4_simple)
{
    using T = flx::floatx<9, 4>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.32812500000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 0.43750000000000000000);
}

TEST(FloatxOperationsTest, ADD_9_5_simple)
{
    using T = flx::floatx<9, 5>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33593750000000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 0.44531250000000000000);
}

TEST(FloatxOperationsTest, ADD_9_6_simple)
{
    using T = flx::floatx<9, 6>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33203125000000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 0.44531250000000000000);
}

TEST(FloatxOperationsTest, ADD_9_7_simple)
{
    using T = flx::floatx<9, 7>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33398437500000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 0.44531250000000000000);
}

TEST(FloatxOperationsTest, ADD_9_8_simple)
{
    using T = flx::floatx<9, 8>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33300781250000000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 0.44433593750000000000);
}

TEST(FloatxOperationsTest, ADD_9_9_simple)
{
    using T = flx::floatx<9, 9>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33349609375000000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 0.44433593750000000000);
}

TEST(FloatxOperationsTest, ADD_9_10_simple)
{
    using T = flx::floatx<9, 10>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33325195312500000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 0.44433593750000000000);
}

TEST(FloatxOperationsTest, ADD_9_11_simple)
{
    using T = flx::floatx<9, 11>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33337402343750000000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 0.44445800781250000000);
}

TEST(FloatxOperationsTest, ADD_9_12_simple)
{
    using T = flx::floatx<9, 12>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33331298828125000000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 0.44445800781250000000);
}

TEST(FloatxOperationsTest, ADD_9_13_simple)
{
    using T = flx::floatx<9, 13>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33334350585937500000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 0.44445800781250000000);
}

TEST(FloatxOperationsTest, ADD_9_14_simple)
{
    using T = flx::floatx<9, 14>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33332824707031250000);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 0.44444274902343750000);
}

TEST(FloatxOperationsTest, ADD_9_15_simple)
{
    using T = flx::floatx<9, 15>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333587646484375000);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 0.44444274902343750000);
}

TEST(FloatxOperationsTest, ADD_9_16_simple)
{
    using T = flx::floatx<9, 16>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333206176757812500);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 0.44444274902343750000);
}

TEST(FloatxOperationsTest, ADD_9_17_simple)
{
    using T = flx::floatx<9, 17>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333396911621093750);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 0.44444465637207031250);
}

TEST(FloatxOperationsTest, ADD_9_18_simple)
{
    using T = flx::floatx<9, 18>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333301544189453125);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 0.44444465637207031250);
}

TEST(FloatxOperationsTest, ADD_9_19_simple)
{
    using T = flx::floatx<9, 19>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333349227905273438);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 0.44444465637207031250);
}

TEST(FloatxOperationsTest, ADD_9_20_simple)
{
    using T = flx::floatx<9, 20>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333325386047363281);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 0.44444441795349121094);
}

TEST(FloatxOperationsTest, ADD_9_21_simple)
{
    using T = flx::floatx<9, 21>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333337306976318359);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 0.44444441795349121094);
}

TEST(FloatxOperationsTest, ADD_9_22_simple)
{
    using T = flx::floatx<9, 22>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333331346511840820);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 0.44444441795349121094);
}

TEST(FloatxOperationsTest, ADD_9_23_simple)
{
    using T = flx::floatx<9, 23>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333334326744079590);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 0.44444444775581359863);
}

TEST(FloatxOperationsTest, ADD_9_24_simple)
{
    using T = flx::floatx<9, 24>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333332836627960205);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 0.44444444775581359863);
}

TEST(FloatxOperationsTest, ADD_9_25_simple)
{
    using T = flx::floatx<9, 25>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333581686019897);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 0.44444444775581359863);
}

TEST(FloatxOperationsTest, ADD_9_26_simple)
{
    using T = flx::floatx<9, 26>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333209156990051);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 0.44444444403052330017);
}

TEST(FloatxOperationsTest, ADD_9_27_simple)
{
    using T = flx::floatx<9, 27>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333395421504974);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 0.44444444403052330017);
}

TEST(FloatxOperationsTest, ADD_9_28_simple)
{
    using T = flx::floatx<9, 28>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333302289247513);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 0.44444444403052330017);
}

TEST(FloatxOperationsTest, ADD_9_29_simple)
{
    using T = flx::floatx<9, 29>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333348855376244);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 0.44444444449618458748);
}

TEST(FloatxOperationsTest, ADD_9_30_simple)
{
    using T = flx::floatx<9, 30>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333325572311878);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 0.44444444449618458748);
}

TEST(FloatxOperationsTest, ADD_9_31_simple)
{
    using T = flx::floatx<9, 31>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333337213844061);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 0.44444444449618458748);
}

TEST(FloatxOperationsTest, ADD_9_32_simple)
{
    using T = flx::floatx<9, 32>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333331393077970);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 0.44444444443797692657);
}

TEST(FloatxOperationsTest, ADD_9_33_simple)
{
    using T = flx::floatx<9, 33>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333334303461015);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 0.44444444443797692657);
}

TEST(FloatxOperationsTest, ADD_9_34_simple)
{
    using T = flx::floatx<9, 34>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333332848269492);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 0.44444444443797692657);
}

TEST(FloatxOperationsTest, ADD_9_35_simple)
{
    using T = flx::floatx<9, 35>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333575865254);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 0.44444444444525288418);
}

TEST(FloatxOperationsTest, ADD_9_36_simple)
{
    using T = flx::floatx<9, 36>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333212067373);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 0.44444444444525288418);
}

TEST(FloatxOperationsTest, ADD_9_37_simple)
{
    using T = flx::floatx<9, 37>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333393966313);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 0.44444444444525288418);
}

TEST(FloatxOperationsTest, ADD_9_38_simple)
{
    using T = flx::floatx<9, 38>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333303016843);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 0.44444444444434338948);
}

TEST(FloatxOperationsTest, ADD_9_39_simple)
{
    using T = flx::floatx<9, 39>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333348491578);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 0.44444444444434338948);
}

TEST(FloatxOperationsTest, ADD_9_40_simple)
{
    using T = flx::floatx<9, 40>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333325754211);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 0.44444444444434338948);
}

TEST(FloatxOperationsTest, ADD_9_41_simple)
{
    using T = flx::floatx<9, 41>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333337122895);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 0.44444444444445707632);
}

TEST(FloatxOperationsTest, ADD_9_42_simple)
{
    using T = flx::floatx<9, 42>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333331438553);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 0.44444444444445707632);
}

TEST(FloatxOperationsTest, ADD_9_43_simple)
{
    using T = flx::floatx<9, 43>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333334280724);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 0.44444444444445707632);
}

TEST(FloatxOperationsTest, ADD_9_44_simple)
{
    using T = flx::floatx<9, 44>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333332859638);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 0.44444444444444286546);
}

TEST(FloatxOperationsTest, ADD_9_45_simple)
{
    using T = flx::floatx<9, 45>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333570181);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 0.44444444444444286546);
}

TEST(FloatxOperationsTest, ADD_9_46_simple)
{
    using T = flx::floatx<9, 46>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333214910);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 0.44444444444444286546);
}

TEST(FloatxOperationsTest, ADD_9_47_simple)
{
    using T = flx::floatx<9, 47>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333392545);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 0.44444444444444464182);
}

TEST(FloatxOperationsTest, ADD_9_48_simple)
{
    using T = flx::floatx<9, 48>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333303727);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 0.44444444444444464182);
}

TEST(FloatxOperationsTest, ADD_9_49_simple)
{
    using T = flx::floatx<9, 49>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333348136);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 0.44444444444444464182);
}

TEST(FloatxOperationsTest, ADD_9_50_simple)
{
    using T = flx::floatx<9, 50>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333325932);
    EXPECT_EQ(double(b), 0.11111111111111110494);
    EXPECT_EQ(double(c), 0.44444444444444441977);
}

TEST(FloatxOperationsTest, ADD_9_51_simple)
{
    using T = flx::floatx<9, 51>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333325932);
    EXPECT_EQ(double(b), 0.11111111111111110494);
    EXPECT_EQ(double(c), 0.44444444444444441977);
}

TEST(FloatxOperationsTest, ADD_9_52_simple)
{
    using T = flx::floatx<9, 52>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333331483);
    EXPECT_EQ(double(b), 0.11111111111111110494);
    EXPECT_EQ(double(c), 0.44444444444444441977);
}

TEST(FloatxOperationsTest, ADD_10_1_simple)
{
    using T = flx::floatx<10, 1>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.37500000000000000000);
    EXPECT_EQ(double(b), 0.12500000000000000000);
    EXPECT_EQ(double(c), 0.50000000000000000000);
}

TEST(FloatxOperationsTest, ADD_10_2_simple)
{
    using T = flx::floatx<10, 2>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.31250000000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 0.43750000000000000000);
}

TEST(FloatxOperationsTest, ADD_10_3_simple)
{
    using T = flx::floatx<10, 3>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.34375000000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 0.43750000000000000000);
}

TEST(FloatxOperationsTest, ADD_10_4_simple)
{
    using T = flx::floatx<10, 4>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.32812500000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 0.43750000000000000000);
}

TEST(FloatxOperationsTest, ADD_10_5_simple)
{
    using T = flx::floatx<10, 5>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33593750000000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 0.44531250000000000000);
}

TEST(FloatxOperationsTest, ADD_10_6_simple)
{
    using T = flx::floatx<10, 6>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33203125000000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 0.44531250000000000000);
}

TEST(FloatxOperationsTest, ADD_10_7_simple)
{
    using T = flx::floatx<10, 7>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33398437500000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 0.44531250000000000000);
}

TEST(FloatxOperationsTest, ADD_10_8_simple)
{
    using T = flx::floatx<10, 8>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33300781250000000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 0.44433593750000000000);
}

TEST(FloatxOperationsTest, ADD_10_9_simple)
{
    using T = flx::floatx<10, 9>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33349609375000000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 0.44433593750000000000);
}

TEST(FloatxOperationsTest, ADD_10_10_simple)
{
    using T = flx::floatx<10, 10>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33325195312500000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 0.44433593750000000000);
}

TEST(FloatxOperationsTest, ADD_10_11_simple)
{
    using T = flx::floatx<10, 11>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33337402343750000000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 0.44445800781250000000);
}

TEST(FloatxOperationsTest, ADD_10_12_simple)
{
    using T = flx::floatx<10, 12>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33331298828125000000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 0.44445800781250000000);
}

TEST(FloatxOperationsTest, ADD_10_13_simple)
{
    using T = flx::floatx<10, 13>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33334350585937500000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 0.44445800781250000000);
}

TEST(FloatxOperationsTest, ADD_10_14_simple)
{
    using T = flx::floatx<10, 14>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33332824707031250000);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 0.44444274902343750000);
}

TEST(FloatxOperationsTest, ADD_10_15_simple)
{
    using T = flx::floatx<10, 15>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333587646484375000);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 0.44444274902343750000);
}

TEST(FloatxOperationsTest, ADD_10_16_simple)
{
    using T = flx::floatx<10, 16>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333206176757812500);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 0.44444274902343750000);
}

TEST(FloatxOperationsTest, ADD_10_17_simple)
{
    using T = flx::floatx<10, 17>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333396911621093750);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 0.44444465637207031250);
}

TEST(FloatxOperationsTest, ADD_10_18_simple)
{
    using T = flx::floatx<10, 18>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333301544189453125);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 0.44444465637207031250);
}

TEST(FloatxOperationsTest, ADD_10_19_simple)
{
    using T = flx::floatx<10, 19>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333349227905273438);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 0.44444465637207031250);
}

TEST(FloatxOperationsTest, ADD_10_20_simple)
{
    using T = flx::floatx<10, 20>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333325386047363281);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 0.44444441795349121094);
}

TEST(FloatxOperationsTest, ADD_10_21_simple)
{
    using T = flx::floatx<10, 21>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333337306976318359);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 0.44444441795349121094);
}

TEST(FloatxOperationsTest, ADD_10_22_simple)
{
    using T = flx::floatx<10, 22>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333331346511840820);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 0.44444441795349121094);
}

TEST(FloatxOperationsTest, ADD_10_23_simple)
{
    using T = flx::floatx<10, 23>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333334326744079590);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 0.44444444775581359863);
}

TEST(FloatxOperationsTest, ADD_10_24_simple)
{
    using T = flx::floatx<10, 24>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333332836627960205);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 0.44444444775581359863);
}

TEST(FloatxOperationsTest, ADD_10_25_simple)
{
    using T = flx::floatx<10, 25>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333581686019897);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 0.44444444775581359863);
}

TEST(FloatxOperationsTest, ADD_10_26_simple)
{
    using T = flx::floatx<10, 26>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333209156990051);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 0.44444444403052330017);
}

TEST(FloatxOperationsTest, ADD_10_27_simple)
{
    using T = flx::floatx<10, 27>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333395421504974);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 0.44444444403052330017);
}

TEST(FloatxOperationsTest, ADD_10_28_simple)
{
    using T = flx::floatx<10, 28>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333302289247513);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 0.44444444403052330017);
}

TEST(FloatxOperationsTest, ADD_10_29_simple)
{
    using T = flx::floatx<10, 29>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333348855376244);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 0.44444444449618458748);
}

TEST(FloatxOperationsTest, ADD_10_30_simple)
{
    using T = flx::floatx<10, 30>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333325572311878);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 0.44444444449618458748);
}

TEST(FloatxOperationsTest, ADD_10_31_simple)
{
    using T = flx::floatx<10, 31>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333337213844061);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 0.44444444449618458748);
}

TEST(FloatxOperationsTest, ADD_10_32_simple)
{
    using T = flx::floatx<10, 32>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333331393077970);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 0.44444444443797692657);
}

TEST(FloatxOperationsTest, ADD_10_33_simple)
{
    using T = flx::floatx<10, 33>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333334303461015);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 0.44444444443797692657);
}

TEST(FloatxOperationsTest, ADD_10_34_simple)
{
    using T = flx::floatx<10, 34>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333332848269492);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 0.44444444443797692657);
}

TEST(FloatxOperationsTest, ADD_10_35_simple)
{
    using T = flx::floatx<10, 35>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333575865254);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 0.44444444444525288418);
}

TEST(FloatxOperationsTest, ADD_10_36_simple)
{
    using T = flx::floatx<10, 36>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333212067373);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 0.44444444444525288418);
}

TEST(FloatxOperationsTest, ADD_10_37_simple)
{
    using T = flx::floatx<10, 37>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333393966313);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 0.44444444444525288418);
}

TEST(FloatxOperationsTest, ADD_10_38_simple)
{
    using T = flx::floatx<10, 38>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333303016843);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 0.44444444444434338948);
}

TEST(FloatxOperationsTest, ADD_10_39_simple)
{
    using T = flx::floatx<10, 39>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333348491578);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 0.44444444444434338948);
}

TEST(FloatxOperationsTest, ADD_10_40_simple)
{
    using T = flx::floatx<10, 40>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333325754211);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 0.44444444444434338948);
}

TEST(FloatxOperationsTest, ADD_10_41_simple)
{
    using T = flx::floatx<10, 41>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333337122895);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 0.44444444444445707632);
}

TEST(FloatxOperationsTest, ADD_10_42_simple)
{
    using T = flx::floatx<10, 42>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333331438553);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 0.44444444444445707632);
}

TEST(FloatxOperationsTest, ADD_10_43_simple)
{
    using T = flx::floatx<10, 43>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333334280724);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 0.44444444444445707632);
}

TEST(FloatxOperationsTest, ADD_10_44_simple)
{
    using T = flx::floatx<10, 44>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333332859638);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 0.44444444444444286546);
}

TEST(FloatxOperationsTest, ADD_10_45_simple)
{
    using T = flx::floatx<10, 45>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333570181);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 0.44444444444444286546);
}

TEST(FloatxOperationsTest, ADD_10_46_simple)
{
    using T = flx::floatx<10, 46>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333214910);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 0.44444444444444286546);
}

TEST(FloatxOperationsTest, ADD_10_47_simple)
{
    using T = flx::floatx<10, 47>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333392545);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 0.44444444444444464182);
}

TEST(FloatxOperationsTest, ADD_10_48_simple)
{
    using T = flx::floatx<10, 48>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333303727);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 0.44444444444444464182);
}

TEST(FloatxOperationsTest, ADD_10_49_simple)
{
    using T = flx::floatx<10, 49>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333348136);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 0.44444444444444464182);
}

TEST(FloatxOperationsTest, ADD_10_50_simple)
{
    using T = flx::floatx<10, 50>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333325932);
    EXPECT_EQ(double(b), 0.11111111111111110494);
    EXPECT_EQ(double(c), 0.44444444444444441977);
}

TEST(FloatxOperationsTest, ADD_10_51_simple)
{
    using T = flx::floatx<10, 51>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333325932);
    EXPECT_EQ(double(b), 0.11111111111111110494);
    EXPECT_EQ(double(c), 0.44444444444444441977);
}

TEST(FloatxOperationsTest, ADD_10_52_simple)
{
    using T = flx::floatx<10, 52>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333331483);
    EXPECT_EQ(double(b), 0.11111111111111110494);
    EXPECT_EQ(double(c), 0.44444444444444441977);
}

TEST(FloatxOperationsTest, ADD_11_1_simple)
{
    using T = flx::floatx<11, 1>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.37500000000000000000);
    EXPECT_EQ(double(b), 0.12500000000000000000);
    EXPECT_EQ(double(c), 0.50000000000000000000);
}

TEST(FloatxOperationsTest, ADD_11_2_simple)
{
    using T = flx::floatx<11, 2>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.31250000000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 0.43750000000000000000);
}

TEST(FloatxOperationsTest, ADD_11_3_simple)
{
    using T = flx::floatx<11, 3>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.34375000000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 0.43750000000000000000);
}

TEST(FloatxOperationsTest, ADD_11_4_simple)
{
    using T = flx::floatx<11, 4>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.32812500000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 0.43750000000000000000);
}

TEST(FloatxOperationsTest, ADD_11_5_simple)
{
    using T = flx::floatx<11, 5>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33593750000000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 0.44531250000000000000);
}

TEST(FloatxOperationsTest, ADD_11_6_simple)
{
    using T = flx::floatx<11, 6>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33203125000000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 0.44531250000000000000);
}

TEST(FloatxOperationsTest, ADD_11_7_simple)
{
    using T = flx::floatx<11, 7>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33398437500000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 0.44531250000000000000);
}

TEST(FloatxOperationsTest, ADD_11_8_simple)
{
    using T = flx::floatx<11, 8>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33300781250000000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 0.44433593750000000000);
}

TEST(FloatxOperationsTest, ADD_11_9_simple)
{
    using T = flx::floatx<11, 9>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33349609375000000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 0.44433593750000000000);
}

TEST(FloatxOperationsTest, ADD_11_10_simple)
{
    using T = flx::floatx<11, 10>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33325195312500000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 0.44433593750000000000);
}

TEST(FloatxOperationsTest, ADD_11_11_simple)
{
    using T = flx::floatx<11, 11>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33337402343750000000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 0.44445800781250000000);
}

TEST(FloatxOperationsTest, ADD_11_12_simple)
{
    using T = flx::floatx<11, 12>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33331298828125000000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 0.44445800781250000000);
}

TEST(FloatxOperationsTest, ADD_11_13_simple)
{
    using T = flx::floatx<11, 13>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33334350585937500000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 0.44445800781250000000);
}

TEST(FloatxOperationsTest, ADD_11_14_simple)
{
    using T = flx::floatx<11, 14>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33332824707031250000);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 0.44444274902343750000);
}

TEST(FloatxOperationsTest, ADD_11_15_simple)
{
    using T = flx::floatx<11, 15>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333587646484375000);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 0.44444274902343750000);
}

TEST(FloatxOperationsTest, ADD_11_16_simple)
{
    using T = flx::floatx<11, 16>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333206176757812500);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 0.44444274902343750000);
}

TEST(FloatxOperationsTest, ADD_11_17_simple)
{
    using T = flx::floatx<11, 17>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333396911621093750);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 0.44444465637207031250);
}

TEST(FloatxOperationsTest, ADD_11_18_simple)
{
    using T = flx::floatx<11, 18>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333301544189453125);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 0.44444465637207031250);
}

TEST(FloatxOperationsTest, ADD_11_19_simple)
{
    using T = flx::floatx<11, 19>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333349227905273438);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 0.44444465637207031250);
}

TEST(FloatxOperationsTest, ADD_11_20_simple)
{
    using T = flx::floatx<11, 20>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333325386047363281);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 0.44444441795349121094);
}

TEST(FloatxOperationsTest, ADD_11_21_simple)
{
    using T = flx::floatx<11, 21>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333337306976318359);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 0.44444441795349121094);
}

TEST(FloatxOperationsTest, ADD_11_22_simple)
{
    using T = flx::floatx<11, 22>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333331346511840820);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 0.44444441795349121094);
}

TEST(FloatxOperationsTest, ADD_11_23_simple)
{
    using T = flx::floatx<11, 23>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333334326744079590);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 0.44444444775581359863);
}

TEST(FloatxOperationsTest, ADD_11_24_simple)
{
    using T = flx::floatx<11, 24>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333332836627960205);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 0.44444444775581359863);
}

TEST(FloatxOperationsTest, ADD_11_25_simple)
{
    using T = flx::floatx<11, 25>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333581686019897);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 0.44444444775581359863);
}

TEST(FloatxOperationsTest, ADD_11_26_simple)
{
    using T = flx::floatx<11, 26>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333209156990051);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 0.44444444403052330017);
}

TEST(FloatxOperationsTest, ADD_11_27_simple)
{
    using T = flx::floatx<11, 27>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333395421504974);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 0.44444444403052330017);
}

TEST(FloatxOperationsTest, ADD_11_28_simple)
{
    using T = flx::floatx<11, 28>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333302289247513);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 0.44444444403052330017);
}

TEST(FloatxOperationsTest, ADD_11_29_simple)
{
    using T = flx::floatx<11, 29>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333348855376244);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 0.44444444449618458748);
}

TEST(FloatxOperationsTest, ADD_11_30_simple)
{
    using T = flx::floatx<11, 30>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333325572311878);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 0.44444444449618458748);
}

TEST(FloatxOperationsTest, ADD_11_31_simple)
{
    using T = flx::floatx<11, 31>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333337213844061);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 0.44444444449618458748);
}

TEST(FloatxOperationsTest, ADD_11_32_simple)
{
    using T = flx::floatx<11, 32>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333331393077970);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 0.44444444443797692657);
}

TEST(FloatxOperationsTest, ADD_11_33_simple)
{
    using T = flx::floatx<11, 33>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333334303461015);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 0.44444444443797692657);
}

TEST(FloatxOperationsTest, ADD_11_34_simple)
{
    using T = flx::floatx<11, 34>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333332848269492);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 0.44444444443797692657);
}

TEST(FloatxOperationsTest, ADD_11_35_simple)
{
    using T = flx::floatx<11, 35>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333575865254);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 0.44444444444525288418);
}

TEST(FloatxOperationsTest, ADD_11_36_simple)
{
    using T = flx::floatx<11, 36>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333212067373);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 0.44444444444525288418);
}

TEST(FloatxOperationsTest, ADD_11_37_simple)
{
    using T = flx::floatx<11, 37>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333393966313);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 0.44444444444525288418);
}

TEST(FloatxOperationsTest, ADD_11_38_simple)
{
    using T = flx::floatx<11, 38>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333303016843);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 0.44444444444434338948);
}

TEST(FloatxOperationsTest, ADD_11_39_simple)
{
    using T = flx::floatx<11, 39>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333348491578);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 0.44444444444434338948);
}

TEST(FloatxOperationsTest, ADD_11_40_simple)
{
    using T = flx::floatx<11, 40>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333325754211);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 0.44444444444434338948);
}

TEST(FloatxOperationsTest, ADD_11_41_simple)
{
    using T = flx::floatx<11, 41>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333337122895);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 0.44444444444445707632);
}

TEST(FloatxOperationsTest, ADD_11_42_simple)
{
    using T = flx::floatx<11, 42>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333331438553);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 0.44444444444445707632);
}

TEST(FloatxOperationsTest, ADD_11_43_simple)
{
    using T = flx::floatx<11, 43>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333334280724);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 0.44444444444445707632);
}

TEST(FloatxOperationsTest, ADD_11_44_simple)
{
    using T = flx::floatx<11, 44>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333332859638);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 0.44444444444444286546);
}

TEST(FloatxOperationsTest, ADD_11_45_simple)
{
    using T = flx::floatx<11, 45>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333570181);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 0.44444444444444286546);
}

TEST(FloatxOperationsTest, ADD_11_46_simple)
{
    using T = flx::floatx<11, 46>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333214910);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 0.44444444444444286546);
}

TEST(FloatxOperationsTest, ADD_11_47_simple)
{
    using T = flx::floatx<11, 47>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333392545);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 0.44444444444444464182);
}

TEST(FloatxOperationsTest, ADD_11_48_simple)
{
    using T = flx::floatx<11, 48>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333303727);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 0.44444444444444464182);
}

TEST(FloatxOperationsTest, ADD_11_49_simple)
{
    using T = flx::floatx<11, 49>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333348136);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 0.44444444444444464182);
}

TEST(FloatxOperationsTest, ADD_11_50_simple)
{
    using T = flx::floatx<11, 50>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333325932);
    EXPECT_EQ(double(b), 0.11111111111111110494);
    EXPECT_EQ(double(c), 0.44444444444444441977);
}

TEST(FloatxOperationsTest, ADD_11_51_simple)
{
    using T = flx::floatx<11, 51>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333325932);
    EXPECT_EQ(double(b), 0.11111111111111110494);
    EXPECT_EQ(double(c), 0.44444444444444441977);
}

TEST(FloatxOperationsTest, ADD_11_52_simple)
{
    using T = flx::floatx<11, 52>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a + b;
    EXPECT_EQ(double(a), 0.33333333333333331483);
    EXPECT_EQ(double(b), 0.11111111111111110494);
    EXPECT_EQ(double(c), 0.44444444444444441977);
}


}  // namespace
