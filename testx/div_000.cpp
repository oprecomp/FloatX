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

TEST(FloatxOperationsTest, DIV_1_1_simple)
{
    using T = flx::floatx<1, 1>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.00000000000000000000);
    EXPECT_EQ(double(b), 0.00000000000000000000);
    EXPECT_NE(c, c);  // holds only for NANs
}

TEST(FloatxOperationsTest, DIV_1_2_simple)
{
    using T = flx::floatx<1, 2>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.50000000000000000000);
    EXPECT_EQ(double(b), 0.00000000000000000000);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_1_3_simple)
{
    using T = flx::floatx<1, 3>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.25000000000000000000);
    EXPECT_EQ(double(b), 0.00000000000000000000);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_1_4_simple)
{
    using T = flx::floatx<1, 4>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.37500000000000000000);
    EXPECT_EQ(double(b), 0.12500000000000000000);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_1_5_simple)
{
    using T = flx::floatx<1, 5>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.31250000000000000000);
    EXPECT_EQ(double(b), 0.12500000000000000000);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_1_6_simple)
{
    using T = flx::floatx<1, 6>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.34375000000000000000);
    EXPECT_EQ(double(b), 0.12500000000000000000);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_1_7_simple)
{
    using T = flx::floatx<1, 7>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.32812500000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_1_8_simple)
{
    using T = flx::floatx<1, 8>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33593750000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_1_9_simple)
{
    using T = flx::floatx<1, 9>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33203125000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_1_10_simple)
{
    using T = flx::floatx<1, 10>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33398437500000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_1_11_simple)
{
    using T = flx::floatx<1, 11>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33300781250000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_1_12_simple)
{
    using T = flx::floatx<1, 12>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33349609375000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_1_13_simple)
{
    using T = flx::floatx<1, 13>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33325195312500000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_1_14_simple)
{
    using T = flx::floatx<1, 14>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33337402343750000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_1_15_simple)
{
    using T = flx::floatx<1, 15>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33331298828125000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_1_16_simple)
{
    using T = flx::floatx<1, 16>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33334350585937500000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_1_17_simple)
{
    using T = flx::floatx<1, 17>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33332824707031250000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_1_18_simple)
{
    using T = flx::floatx<1, 18>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333587646484375000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_1_19_simple)
{
    using T = flx::floatx<1, 19>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333206176757812500);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_1_20_simple)
{
    using T = flx::floatx<1, 20>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333396911621093750);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_1_21_simple)
{
    using T = flx::floatx<1, 21>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333301544189453125);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_1_22_simple)
{
    using T = flx::floatx<1, 22>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333349227905273438);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_1_23_simple)
{
    using T = flx::floatx<1, 23>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333325386047363281);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_1_24_simple)
{
    using T = flx::floatx<1, 24>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333337306976318359);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_1_25_simple)
{
    using T = flx::floatx<1, 25>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333331346511840820);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_1_26_simple)
{
    using T = flx::floatx<1, 26>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333334326744079590);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_1_27_simple)
{
    using T = flx::floatx<1, 27>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333332836627960205);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_1_28_simple)
{
    using T = flx::floatx<1, 28>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333581686019897);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_1_29_simple)
{
    using T = flx::floatx<1, 29>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333209156990051);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_1_30_simple)
{
    using T = flx::floatx<1, 30>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333395421504974);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_1_31_simple)
{
    using T = flx::floatx<1, 31>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333302289247513);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_1_32_simple)
{
    using T = flx::floatx<1, 32>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333348855376244);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_1_33_simple)
{
    using T = flx::floatx<1, 33>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333325572311878);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_1_34_simple)
{
    using T = flx::floatx<1, 34>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333337213844061);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_1_35_simple)
{
    using T = flx::floatx<1, 35>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333331393077970);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_1_36_simple)
{
    using T = flx::floatx<1, 36>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333334303461015);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_1_37_simple)
{
    using T = flx::floatx<1, 37>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333332848269492);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_1_38_simple)
{
    using T = flx::floatx<1, 38>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333575865254);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_1_39_simple)
{
    using T = flx::floatx<1, 39>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333212067373);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_1_40_simple)
{
    using T = flx::floatx<1, 40>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333393966313);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_1_41_simple)
{
    using T = flx::floatx<1, 41>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333303016843);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_1_42_simple)
{
    using T = flx::floatx<1, 42>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333348491578);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_1_43_simple)
{
    using T = flx::floatx<1, 43>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333325754211);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_1_44_simple)
{
    using T = flx::floatx<1, 44>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333337122895);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_1_45_simple)
{
    using T = flx::floatx<1, 45>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333331438553);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_1_46_simple)
{
    using T = flx::floatx<1, 46>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333334280724);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_1_47_simple)
{
    using T = flx::floatx<1, 47>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333332859638);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_1_48_simple)
{
    using T = flx::floatx<1, 48>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333570181);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_1_49_simple)
{
    using T = flx::floatx<1, 49>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333214910);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_1_50_simple)
{
    using T = flx::floatx<1, 50>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333392545);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_1_51_simple)
{
    using T = flx::floatx<1, 51>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333303727);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_1_52_simple)
{
    using T = flx::floatx<1, 52>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333348136);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_2_1_simple)
{
    using T = flx::floatx<2, 1>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.50000000000000000000);
    EXPECT_EQ(double(b), 0.00000000000000000000);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_2_2_simple)
{
    using T = flx::floatx<2, 2>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.25000000000000000000);
    EXPECT_EQ(double(b), 0.00000000000000000000);
    EXPECT_EQ(double(c), inf);
}

TEST(FloatxOperationsTest, DIV_2_3_simple)
{
    using T = flx::floatx<2, 3>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.37500000000000000000);
    EXPECT_EQ(double(b), 0.12500000000000000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_2_4_simple)
{
    using T = flx::floatx<2, 4>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.31250000000000000000);
    EXPECT_EQ(double(b), 0.12500000000000000000);
    EXPECT_EQ(double(c), 2.50000000000000000000);
}

TEST(FloatxOperationsTest, DIV_2_5_simple)
{
    using T = flx::floatx<2, 5>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.34375000000000000000);
    EXPECT_EQ(double(b), 0.12500000000000000000);
    EXPECT_EQ(double(c), 2.75000000000000000000);
}

TEST(FloatxOperationsTest, DIV_2_6_simple)
{
    using T = flx::floatx<2, 6>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.32812500000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_2_7_simple)
{
    using T = flx::floatx<2, 7>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33593750000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 3.07812500000000000000);
}

TEST(FloatxOperationsTest, DIV_2_8_simple)
{
    using T = flx::floatx<2, 8>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33203125000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 3.03906250000000000000);
}

TEST(FloatxOperationsTest, DIV_2_9_simple)
{
    using T = flx::floatx<2, 9>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33398437500000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_2_10_simple)
{
    using T = flx::floatx<2, 10>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33300781250000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 2.99218750000000000000);
}

TEST(FloatxOperationsTest, DIV_2_11_simple)
{
    using T = flx::floatx<2, 11>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33349609375000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 2.99609375000000000000);
}

TEST(FloatxOperationsTest, DIV_2_12_simple)
{
    using T = flx::floatx<2, 12>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33325195312500000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_2_13_simple)
{
    using T = flx::floatx<2, 13>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33337402343750000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 3.00122070312500000000);
}

TEST(FloatxOperationsTest, DIV_2_14_simple)
{
    using T = flx::floatx<2, 14>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33331298828125000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 3.00061035156250000000);
}

TEST(FloatxOperationsTest, DIV_2_15_simple)
{
    using T = flx::floatx<2, 15>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33334350585937500000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_2_16_simple)
{
    using T = flx::floatx<2, 16>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33332824707031250000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 2.99987792968750000000);
}

TEST(FloatxOperationsTest, DIV_2_17_simple)
{
    using T = flx::floatx<2, 17>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333587646484375000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 2.99993896484375000000);
}

TEST(FloatxOperationsTest, DIV_2_18_simple)
{
    using T = flx::floatx<2, 18>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333206176757812500);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_2_19_simple)
{
    using T = flx::floatx<2, 19>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333396911621093750);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 3.00001907348632812500);
}

TEST(FloatxOperationsTest, DIV_2_20_simple)
{
    using T = flx::floatx<2, 20>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333301544189453125);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 3.00000953674316406250);
}

TEST(FloatxOperationsTest, DIV_2_21_simple)
{
    using T = flx::floatx<2, 21>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333349227905273438);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_2_22_simple)
{
    using T = flx::floatx<2, 22>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333325386047363281);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 2.99999809265136718750);
}

TEST(FloatxOperationsTest, DIV_2_23_simple)
{
    using T = flx::floatx<2, 23>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333337306976318359);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 2.99999904632568359375);
}

TEST(FloatxOperationsTest, DIV_2_24_simple)
{
    using T = flx::floatx<2, 24>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333331346511840820);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_2_25_simple)
{
    using T = flx::floatx<2, 25>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333334326744079590);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 3.00000029802322387695);
}

TEST(FloatxOperationsTest, DIV_2_26_simple)
{
    using T = flx::floatx<2, 26>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333332836627960205);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 3.00000014901161193848);
}

TEST(FloatxOperationsTest, DIV_2_27_simple)
{
    using T = flx::floatx<2, 27>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333581686019897);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_2_28_simple)
{
    using T = flx::floatx<2, 28>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333209156990051);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 2.99999997019767761230);
}

TEST(FloatxOperationsTest, DIV_2_29_simple)
{
    using T = flx::floatx<2, 29>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333395421504974);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 2.99999998509883880615);
}

TEST(FloatxOperationsTest, DIV_2_30_simple)
{
    using T = flx::floatx<2, 30>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333302289247513);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_2_31_simple)
{
    using T = flx::floatx<2, 31>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333348855376244);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 3.00000000372529029846);
}

TEST(FloatxOperationsTest, DIV_2_32_simple)
{
    using T = flx::floatx<2, 32>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333325572311878);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 3.00000000186264514923);
}

TEST(FloatxOperationsTest, DIV_2_33_simple)
{
    using T = flx::floatx<2, 33>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333337213844061);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_2_34_simple)
{
    using T = flx::floatx<2, 34>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333331393077970);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 2.99999999953433871269);
}

TEST(FloatxOperationsTest, DIV_2_35_simple)
{
    using T = flx::floatx<2, 35>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333334303461015);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 2.99999999976716935635);
}

TEST(FloatxOperationsTest, DIV_2_36_simple)
{
    using T = flx::floatx<2, 36>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333332848269492);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_2_37_simple)
{
    using T = flx::floatx<2, 37>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333575865254);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 3.00000000005820766091);
}

TEST(FloatxOperationsTest, DIV_2_38_simple)
{
    using T = flx::floatx<2, 38>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333212067373);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 3.00000000002910383046);
}

TEST(FloatxOperationsTest, DIV_2_39_simple)
{
    using T = flx::floatx<2, 39>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333393966313);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_2_40_simple)
{
    using T = flx::floatx<2, 40>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333303016843);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 2.99999999999272404239);
}

TEST(FloatxOperationsTest, DIV_2_41_simple)
{
    using T = flx::floatx<2, 41>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333348491578);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 2.99999999999636202119);
}

TEST(FloatxOperationsTest, DIV_2_42_simple)
{
    using T = flx::floatx<2, 42>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333325754211);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_2_43_simple)
{
    using T = flx::floatx<2, 43>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333337122895);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 3.00000000000090949470);
}

TEST(FloatxOperationsTest, DIV_2_44_simple)
{
    using T = flx::floatx<2, 44>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333331438553);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 3.00000000000045474735);
}

TEST(FloatxOperationsTest, DIV_2_45_simple)
{
    using T = flx::floatx<2, 45>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333334280724);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_2_46_simple)
{
    using T = flx::floatx<2, 46>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333332859638);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 2.99999999999988631316);
}

TEST(FloatxOperationsTest, DIV_2_47_simple)
{
    using T = flx::floatx<2, 47>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333570181);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 2.99999999999994315658);
}

TEST(FloatxOperationsTest, DIV_2_48_simple)
{
    using T = flx::floatx<2, 48>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333214910);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_2_49_simple)
{
    using T = flx::floatx<2, 49>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333392545);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 3.00000000000001421085);
}

TEST(FloatxOperationsTest, DIV_2_50_simple)
{
    using T = flx::floatx<2, 50>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333303727);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 3.00000000000000710543);
}

TEST(FloatxOperationsTest, DIV_2_51_simple)
{
    using T = flx::floatx<2, 51>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333348136);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_2_52_simple)
{
    using T = flx::floatx<2, 52>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333325932);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 2.99999999999999822364);
}

TEST(FloatxOperationsTest, DIV_3_1_simple)
{
    using T = flx::floatx<3, 1>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.37500000000000000000);
    EXPECT_EQ(double(b), 0.12500000000000000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_3_2_simple)
{
    using T = flx::floatx<3, 2>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.31250000000000000000);
    EXPECT_EQ(double(b), 0.12500000000000000000);
    EXPECT_EQ(double(c), 2.50000000000000000000);
}

TEST(FloatxOperationsTest, DIV_3_3_simple)
{
    using T = flx::floatx<3, 3>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.34375000000000000000);
    EXPECT_EQ(double(b), 0.12500000000000000000);
    EXPECT_EQ(double(c), 2.75000000000000000000);
}

TEST(FloatxOperationsTest, DIV_3_4_simple)
{
    using T = flx::floatx<3, 4>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.32812500000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_3_5_simple)
{
    using T = flx::floatx<3, 5>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33593750000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 3.06250000000000000000);
}

TEST(FloatxOperationsTest, DIV_3_6_simple)
{
    using T = flx::floatx<3, 6>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33203125000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 3.03125000000000000000);
}

TEST(FloatxOperationsTest, DIV_3_7_simple)
{
    using T = flx::floatx<3, 7>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33398437500000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_3_8_simple)
{
    using T = flx::floatx<3, 8>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33300781250000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 2.99218750000000000000);
}

TEST(FloatxOperationsTest, DIV_3_9_simple)
{
    using T = flx::floatx<3, 9>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33349609375000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 2.99609375000000000000);
}

TEST(FloatxOperationsTest, DIV_3_10_simple)
{
    using T = flx::floatx<3, 10>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33325195312500000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_3_11_simple)
{
    using T = flx::floatx<3, 11>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33337402343750000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 3.00097656250000000000);
}

TEST(FloatxOperationsTest, DIV_3_12_simple)
{
    using T = flx::floatx<3, 12>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33331298828125000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 3.00048828125000000000);
}

TEST(FloatxOperationsTest, DIV_3_13_simple)
{
    using T = flx::floatx<3, 13>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33334350585937500000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_3_14_simple)
{
    using T = flx::floatx<3, 14>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33332824707031250000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 2.99987792968750000000);
}

TEST(FloatxOperationsTest, DIV_3_15_simple)
{
    using T = flx::floatx<3, 15>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333587646484375000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 2.99993896484375000000);
}

TEST(FloatxOperationsTest, DIV_3_16_simple)
{
    using T = flx::floatx<3, 16>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333206176757812500);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_3_17_simple)
{
    using T = flx::floatx<3, 17>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333396911621093750);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 3.00001525878906250000);
}

TEST(FloatxOperationsTest, DIV_3_18_simple)
{
    using T = flx::floatx<3, 18>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333301544189453125);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 3.00000762939453125000);
}

TEST(FloatxOperationsTest, DIV_3_19_simple)
{
    using T = flx::floatx<3, 19>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333349227905273438);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_3_20_simple)
{
    using T = flx::floatx<3, 20>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333325386047363281);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 2.99999809265136718750);
}

TEST(FloatxOperationsTest, DIV_3_21_simple)
{
    using T = flx::floatx<3, 21>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333337306976318359);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 2.99999904632568359375);
}

TEST(FloatxOperationsTest, DIV_3_22_simple)
{
    using T = flx::floatx<3, 22>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333331346511840820);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_3_23_simple)
{
    using T = flx::floatx<3, 23>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333334326744079590);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 3.00000023841857910156);
}

TEST(FloatxOperationsTest, DIV_3_24_simple)
{
    using T = flx::floatx<3, 24>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333332836627960205);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 3.00000011920928955078);
}

TEST(FloatxOperationsTest, DIV_3_25_simple)
{
    using T = flx::floatx<3, 25>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333581686019897);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_3_26_simple)
{
    using T = flx::floatx<3, 26>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333209156990051);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 2.99999997019767761230);
}

TEST(FloatxOperationsTest, DIV_3_27_simple)
{
    using T = flx::floatx<3, 27>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333395421504974);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 2.99999998509883880615);
}

TEST(FloatxOperationsTest, DIV_3_28_simple)
{
    using T = flx::floatx<3, 28>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333302289247513);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_3_29_simple)
{
    using T = flx::floatx<3, 29>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333348855376244);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 3.00000000372529029846);
}

TEST(FloatxOperationsTest, DIV_3_30_simple)
{
    using T = flx::floatx<3, 30>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333325572311878);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 3.00000000186264514923);
}

TEST(FloatxOperationsTest, DIV_3_31_simple)
{
    using T = flx::floatx<3, 31>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333337213844061);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_3_32_simple)
{
    using T = flx::floatx<3, 32>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333331393077970);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 2.99999999953433871269);
}

TEST(FloatxOperationsTest, DIV_3_33_simple)
{
    using T = flx::floatx<3, 33>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333334303461015);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 2.99999999976716935635);
}

TEST(FloatxOperationsTest, DIV_3_34_simple)
{
    using T = flx::floatx<3, 34>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333332848269492);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_3_35_simple)
{
    using T = flx::floatx<3, 35>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333575865254);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 3.00000000005820766091);
}

TEST(FloatxOperationsTest, DIV_3_36_simple)
{
    using T = flx::floatx<3, 36>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333212067373);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 3.00000000002910383046);
}

TEST(FloatxOperationsTest, DIV_3_37_simple)
{
    using T = flx::floatx<3, 37>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333393966313);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_3_38_simple)
{
    using T = flx::floatx<3, 38>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333303016843);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 2.99999999999272404239);
}

TEST(FloatxOperationsTest, DIV_3_39_simple)
{
    using T = flx::floatx<3, 39>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333348491578);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 2.99999999999636202119);
}

TEST(FloatxOperationsTest, DIV_3_40_simple)
{
    using T = flx::floatx<3, 40>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333325754211);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_3_41_simple)
{
    using T = flx::floatx<3, 41>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333337122895);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 3.00000000000090949470);
}

TEST(FloatxOperationsTest, DIV_3_42_simple)
{
    using T = flx::floatx<3, 42>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333331438553);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 3.00000000000045474735);
}

TEST(FloatxOperationsTest, DIV_3_43_simple)
{
    using T = flx::floatx<3, 43>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333334280724);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_3_44_simple)
{
    using T = flx::floatx<3, 44>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333332859638);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 2.99999999999988631316);
}

TEST(FloatxOperationsTest, DIV_3_45_simple)
{
    using T = flx::floatx<3, 45>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333570181);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 2.99999999999994315658);
}

TEST(FloatxOperationsTest, DIV_3_46_simple)
{
    using T = flx::floatx<3, 46>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333214910);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_3_47_simple)
{
    using T = flx::floatx<3, 47>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333392545);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 3.00000000000001421085);
}

TEST(FloatxOperationsTest, DIV_3_48_simple)
{
    using T = flx::floatx<3, 48>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333303727);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 3.00000000000000710543);
}

TEST(FloatxOperationsTest, DIV_3_49_simple)
{
    using T = flx::floatx<3, 49>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333348136);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_3_50_simple)
{
    using T = flx::floatx<3, 50>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333325932);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 2.99999999999999822364);
}

TEST(FloatxOperationsTest, DIV_3_51_simple)
{
    using T = flx::floatx<3, 51>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333325932);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 2.99999999999999822364);
}

TEST(FloatxOperationsTest, DIV_3_52_simple)
{
    using T = flx::floatx<3, 52>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333331483);
    EXPECT_EQ(double(b), 0.11111111111111110494);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_4_1_simple)
{
    using T = flx::floatx<4, 1>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.37500000000000000000);
    EXPECT_EQ(double(b), 0.12500000000000000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_4_2_simple)
{
    using T = flx::floatx<4, 2>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.31250000000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_4_3_simple)
{
    using T = flx::floatx<4, 3>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.34375000000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 3.25000000000000000000);
}

TEST(FloatxOperationsTest, DIV_4_4_simple)
{
    using T = flx::floatx<4, 4>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.32812500000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_4_5_simple)
{
    using T = flx::floatx<4, 5>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33593750000000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_4_6_simple)
{
    using T = flx::floatx<4, 6>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33203125000000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 2.96875000000000000000);
}

TEST(FloatxOperationsTest, DIV_4_7_simple)
{
    using T = flx::floatx<4, 7>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33398437500000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_4_8_simple)
{
    using T = flx::floatx<4, 8>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33300781250000000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_4_9_simple)
{
    using T = flx::floatx<4, 9>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33349609375000000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 3.00390625000000000000);
}

TEST(FloatxOperationsTest, DIV_4_10_simple)
{
    using T = flx::floatx<4, 10>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33325195312500000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_4_11_simple)
{
    using T = flx::floatx<4, 11>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33337402343750000000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_4_12_simple)
{
    using T = flx::floatx<4, 12>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33331298828125000000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 2.99951171875000000000);
}

TEST(FloatxOperationsTest, DIV_4_13_simple)
{
    using T = flx::floatx<4, 13>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33334350585937500000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_4_14_simple)
{
    using T = flx::floatx<4, 14>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33332824707031250000);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_4_15_simple)
{
    using T = flx::floatx<4, 15>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333587646484375000);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 3.00006103515625000000);
}

TEST(FloatxOperationsTest, DIV_4_16_simple)
{
    using T = flx::floatx<4, 16>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333206176757812500);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_4_17_simple)
{
    using T = flx::floatx<4, 17>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333396911621093750);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_4_18_simple)
{
    using T = flx::floatx<4, 18>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333301544189453125);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 2.99999237060546875000);
}

TEST(FloatxOperationsTest, DIV_4_19_simple)
{
    using T = flx::floatx<4, 19>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333349227905273438);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_4_20_simple)
{
    using T = flx::floatx<4, 20>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333325386047363281);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_4_21_simple)
{
    using T = flx::floatx<4, 21>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333337306976318359);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 3.00000095367431640625);
}

TEST(FloatxOperationsTest, DIV_4_22_simple)
{
    using T = flx::floatx<4, 22>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333331346511840820);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_4_23_simple)
{
    using T = flx::floatx<4, 23>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333334326744079590);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_4_24_simple)
{
    using T = flx::floatx<4, 24>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333332836627960205);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 2.99999988079071044922);
}

TEST(FloatxOperationsTest, DIV_4_25_simple)
{
    using T = flx::floatx<4, 25>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333581686019897);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_4_26_simple)
{
    using T = flx::floatx<4, 26>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333209156990051);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_4_27_simple)
{
    using T = flx::floatx<4, 27>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333395421504974);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 3.00000001490116119385);
}

TEST(FloatxOperationsTest, DIV_4_28_simple)
{
    using T = flx::floatx<4, 28>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333302289247513);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_4_29_simple)
{
    using T = flx::floatx<4, 29>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333348855376244);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_4_30_simple)
{
    using T = flx::floatx<4, 30>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333325572311878);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 2.99999999813735485077);
}

TEST(FloatxOperationsTest, DIV_4_31_simple)
{
    using T = flx::floatx<4, 31>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333337213844061);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_4_32_simple)
{
    using T = flx::floatx<4, 32>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333331393077970);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_4_33_simple)
{
    using T = flx::floatx<4, 33>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333334303461015);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 3.00000000023283064365);
}

TEST(FloatxOperationsTest, DIV_4_34_simple)
{
    using T = flx::floatx<4, 34>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333332848269492);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_4_35_simple)
{
    using T = flx::floatx<4, 35>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333575865254);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_4_36_simple)
{
    using T = flx::floatx<4, 36>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333212067373);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 2.99999999997089616954);
}

TEST(FloatxOperationsTest, DIV_4_37_simple)
{
    using T = flx::floatx<4, 37>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333393966313);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_4_38_simple)
{
    using T = flx::floatx<4, 38>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333303016843);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_4_39_simple)
{
    using T = flx::floatx<4, 39>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333348491578);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 3.00000000000363797881);
}

TEST(FloatxOperationsTest, DIV_4_40_simple)
{
    using T = flx::floatx<4, 40>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333325754211);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_4_41_simple)
{
    using T = flx::floatx<4, 41>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333337122895);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_4_42_simple)
{
    using T = flx::floatx<4, 42>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333331438553);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 2.99999999999954525265);
}

TEST(FloatxOperationsTest, DIV_4_43_simple)
{
    using T = flx::floatx<4, 43>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333334280724);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_4_44_simple)
{
    using T = flx::floatx<4, 44>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333332859638);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_4_45_simple)
{
    using T = flx::floatx<4, 45>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333570181);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 3.00000000000005684342);
}

TEST(FloatxOperationsTest, DIV_4_46_simple)
{
    using T = flx::floatx<4, 46>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333214910);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_4_47_simple)
{
    using T = flx::floatx<4, 47>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333392545);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_4_48_simple)
{
    using T = flx::floatx<4, 48>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333303727);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 2.99999999999999289457);
}

TEST(FloatxOperationsTest, DIV_4_49_simple)
{
    using T = flx::floatx<4, 49>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333348136);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_4_50_simple)
{
    using T = flx::floatx<4, 50>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333325932);
    EXPECT_EQ(double(b), 0.11111111111111110494);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_4_51_simple)
{
    using T = flx::floatx<4, 51>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333325932);
    EXPECT_EQ(double(b), 0.11111111111111110494);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_4_52_simple)
{
    using T = flx::floatx<4, 52>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333331483);
    EXPECT_EQ(double(b), 0.11111111111111110494);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_5_1_simple)
{
    using T = flx::floatx<5, 1>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.37500000000000000000);
    EXPECT_EQ(double(b), 0.12500000000000000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_5_2_simple)
{
    using T = flx::floatx<5, 2>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.31250000000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_5_3_simple)
{
    using T = flx::floatx<5, 3>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.34375000000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 3.25000000000000000000);
}

TEST(FloatxOperationsTest, DIV_5_4_simple)
{
    using T = flx::floatx<5, 4>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.32812500000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_5_5_simple)
{
    using T = flx::floatx<5, 5>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33593750000000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_5_6_simple)
{
    using T = flx::floatx<5, 6>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33203125000000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 2.96875000000000000000);
}

TEST(FloatxOperationsTest, DIV_5_7_simple)
{
    using T = flx::floatx<5, 7>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33398437500000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_5_8_simple)
{
    using T = flx::floatx<5, 8>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33300781250000000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_5_9_simple)
{
    using T = flx::floatx<5, 9>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33349609375000000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 3.00390625000000000000);
}

TEST(FloatxOperationsTest, DIV_5_10_simple)
{
    using T = flx::floatx<5, 10>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33325195312500000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_5_11_simple)
{
    using T = flx::floatx<5, 11>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33337402343750000000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_5_12_simple)
{
    using T = flx::floatx<5, 12>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33331298828125000000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 2.99951171875000000000);
}

TEST(FloatxOperationsTest, DIV_5_13_simple)
{
    using T = flx::floatx<5, 13>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33334350585937500000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_5_14_simple)
{
    using T = flx::floatx<5, 14>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33332824707031250000);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_5_15_simple)
{
    using T = flx::floatx<5, 15>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333587646484375000);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 3.00006103515625000000);
}

TEST(FloatxOperationsTest, DIV_5_16_simple)
{
    using T = flx::floatx<5, 16>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333206176757812500);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_5_17_simple)
{
    using T = flx::floatx<5, 17>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333396911621093750);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_5_18_simple)
{
    using T = flx::floatx<5, 18>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333301544189453125);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 2.99999237060546875000);
}

TEST(FloatxOperationsTest, DIV_5_19_simple)
{
    using T = flx::floatx<5, 19>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333349227905273438);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_5_20_simple)
{
    using T = flx::floatx<5, 20>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333325386047363281);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_5_21_simple)
{
    using T = flx::floatx<5, 21>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333337306976318359);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 3.00000095367431640625);
}

TEST(FloatxOperationsTest, DIV_5_22_simple)
{
    using T = flx::floatx<5, 22>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333331346511840820);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_5_23_simple)
{
    using T = flx::floatx<5, 23>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333334326744079590);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_5_24_simple)
{
    using T = flx::floatx<5, 24>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333332836627960205);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 2.99999988079071044922);
}

TEST(FloatxOperationsTest, DIV_5_25_simple)
{
    using T = flx::floatx<5, 25>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333581686019897);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_5_26_simple)
{
    using T = flx::floatx<5, 26>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333209156990051);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_5_27_simple)
{
    using T = flx::floatx<5, 27>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333395421504974);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 3.00000001490116119385);
}

TEST(FloatxOperationsTest, DIV_5_28_simple)
{
    using T = flx::floatx<5, 28>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333302289247513);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_5_29_simple)
{
    using T = flx::floatx<5, 29>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333348855376244);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_5_30_simple)
{
    using T = flx::floatx<5, 30>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333325572311878);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 2.99999999813735485077);
}

TEST(FloatxOperationsTest, DIV_5_31_simple)
{
    using T = flx::floatx<5, 31>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333337213844061);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_5_32_simple)
{
    using T = flx::floatx<5, 32>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333331393077970);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_5_33_simple)
{
    using T = flx::floatx<5, 33>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333334303461015);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 3.00000000023283064365);
}

TEST(FloatxOperationsTest, DIV_5_34_simple)
{
    using T = flx::floatx<5, 34>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333332848269492);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_5_35_simple)
{
    using T = flx::floatx<5, 35>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333575865254);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_5_36_simple)
{
    using T = flx::floatx<5, 36>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333212067373);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 2.99999999997089616954);
}

TEST(FloatxOperationsTest, DIV_5_37_simple)
{
    using T = flx::floatx<5, 37>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333393966313);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_5_38_simple)
{
    using T = flx::floatx<5, 38>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333303016843);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_5_39_simple)
{
    using T = flx::floatx<5, 39>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333348491578);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 3.00000000000363797881);
}

TEST(FloatxOperationsTest, DIV_5_40_simple)
{
    using T = flx::floatx<5, 40>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333325754211);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_5_41_simple)
{
    using T = flx::floatx<5, 41>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333337122895);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_5_42_simple)
{
    using T = flx::floatx<5, 42>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333331438553);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 2.99999999999954525265);
}

TEST(FloatxOperationsTest, DIV_5_43_simple)
{
    using T = flx::floatx<5, 43>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333334280724);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_5_44_simple)
{
    using T = flx::floatx<5, 44>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333332859638);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_5_45_simple)
{
    using T = flx::floatx<5, 45>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333570181);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 3.00000000000005684342);
}

TEST(FloatxOperationsTest, DIV_5_46_simple)
{
    using T = flx::floatx<5, 46>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333214910);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_5_47_simple)
{
    using T = flx::floatx<5, 47>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333392545);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_5_48_simple)
{
    using T = flx::floatx<5, 48>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333303727);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 2.99999999999999289457);
}

TEST(FloatxOperationsTest, DIV_5_49_simple)
{
    using T = flx::floatx<5, 49>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333348136);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_5_50_simple)
{
    using T = flx::floatx<5, 50>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333325932);
    EXPECT_EQ(double(b), 0.11111111111111110494);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_5_51_simple)
{
    using T = flx::floatx<5, 51>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333325932);
    EXPECT_EQ(double(b), 0.11111111111111110494);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_5_52_simple)
{
    using T = flx::floatx<5, 52>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333331483);
    EXPECT_EQ(double(b), 0.11111111111111110494);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_6_1_simple)
{
    using T = flx::floatx<6, 1>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.37500000000000000000);
    EXPECT_EQ(double(b), 0.12500000000000000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_6_2_simple)
{
    using T = flx::floatx<6, 2>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.31250000000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_6_3_simple)
{
    using T = flx::floatx<6, 3>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.34375000000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 3.25000000000000000000);
}

TEST(FloatxOperationsTest, DIV_6_4_simple)
{
    using T = flx::floatx<6, 4>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.32812500000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_6_5_simple)
{
    using T = flx::floatx<6, 5>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33593750000000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_6_6_simple)
{
    using T = flx::floatx<6, 6>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33203125000000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 2.96875000000000000000);
}

TEST(FloatxOperationsTest, DIV_6_7_simple)
{
    using T = flx::floatx<6, 7>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33398437500000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_6_8_simple)
{
    using T = flx::floatx<6, 8>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33300781250000000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_6_9_simple)
{
    using T = flx::floatx<6, 9>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33349609375000000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 3.00390625000000000000);
}

TEST(FloatxOperationsTest, DIV_6_10_simple)
{
    using T = flx::floatx<6, 10>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33325195312500000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_6_11_simple)
{
    using T = flx::floatx<6, 11>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33337402343750000000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_6_12_simple)
{
    using T = flx::floatx<6, 12>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33331298828125000000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 2.99951171875000000000);
}

TEST(FloatxOperationsTest, DIV_6_13_simple)
{
    using T = flx::floatx<6, 13>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33334350585937500000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_6_14_simple)
{
    using T = flx::floatx<6, 14>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33332824707031250000);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_6_15_simple)
{
    using T = flx::floatx<6, 15>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333587646484375000);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 3.00006103515625000000);
}

TEST(FloatxOperationsTest, DIV_6_16_simple)
{
    using T = flx::floatx<6, 16>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333206176757812500);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_6_17_simple)
{
    using T = flx::floatx<6, 17>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333396911621093750);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_6_18_simple)
{
    using T = flx::floatx<6, 18>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333301544189453125);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 2.99999237060546875000);
}

TEST(FloatxOperationsTest, DIV_6_19_simple)
{
    using T = flx::floatx<6, 19>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333349227905273438);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_6_20_simple)
{
    using T = flx::floatx<6, 20>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333325386047363281);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_6_21_simple)
{
    using T = flx::floatx<6, 21>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333337306976318359);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 3.00000095367431640625);
}

TEST(FloatxOperationsTest, DIV_6_22_simple)
{
    using T = flx::floatx<6, 22>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333331346511840820);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_6_23_simple)
{
    using T = flx::floatx<6, 23>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333334326744079590);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_6_24_simple)
{
    using T = flx::floatx<6, 24>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333332836627960205);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 2.99999988079071044922);
}

TEST(FloatxOperationsTest, DIV_6_25_simple)
{
    using T = flx::floatx<6, 25>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333581686019897);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_6_26_simple)
{
    using T = flx::floatx<6, 26>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333209156990051);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_6_27_simple)
{
    using T = flx::floatx<6, 27>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333395421504974);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 3.00000001490116119385);
}

TEST(FloatxOperationsTest, DIV_6_28_simple)
{
    using T = flx::floatx<6, 28>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333302289247513);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_6_29_simple)
{
    using T = flx::floatx<6, 29>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333348855376244);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_6_30_simple)
{
    using T = flx::floatx<6, 30>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333325572311878);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 2.99999999813735485077);
}

TEST(FloatxOperationsTest, DIV_6_31_simple)
{
    using T = flx::floatx<6, 31>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333337213844061);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_6_32_simple)
{
    using T = flx::floatx<6, 32>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333331393077970);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_6_33_simple)
{
    using T = flx::floatx<6, 33>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333334303461015);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 3.00000000023283064365);
}

TEST(FloatxOperationsTest, DIV_6_34_simple)
{
    using T = flx::floatx<6, 34>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333332848269492);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_6_35_simple)
{
    using T = flx::floatx<6, 35>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333575865254);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_6_36_simple)
{
    using T = flx::floatx<6, 36>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333212067373);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 2.99999999997089616954);
}

TEST(FloatxOperationsTest, DIV_6_37_simple)
{
    using T = flx::floatx<6, 37>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333393966313);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_6_38_simple)
{
    using T = flx::floatx<6, 38>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333303016843);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_6_39_simple)
{
    using T = flx::floatx<6, 39>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333348491578);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 3.00000000000363797881);
}

TEST(FloatxOperationsTest, DIV_6_40_simple)
{
    using T = flx::floatx<6, 40>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333325754211);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_6_41_simple)
{
    using T = flx::floatx<6, 41>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333337122895);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_6_42_simple)
{
    using T = flx::floatx<6, 42>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333331438553);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 2.99999999999954525265);
}

TEST(FloatxOperationsTest, DIV_6_43_simple)
{
    using T = flx::floatx<6, 43>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333334280724);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_6_44_simple)
{
    using T = flx::floatx<6, 44>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333332859638);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_6_45_simple)
{
    using T = flx::floatx<6, 45>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333570181);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 3.00000000000005684342);
}

TEST(FloatxOperationsTest, DIV_6_46_simple)
{
    using T = flx::floatx<6, 46>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333214910);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_6_47_simple)
{
    using T = flx::floatx<6, 47>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333392545);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_6_48_simple)
{
    using T = flx::floatx<6, 48>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333303727);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 2.99999999999999289457);
}

TEST(FloatxOperationsTest, DIV_6_49_simple)
{
    using T = flx::floatx<6, 49>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333348136);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_6_50_simple)
{
    using T = flx::floatx<6, 50>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333325932);
    EXPECT_EQ(double(b), 0.11111111111111110494);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_6_51_simple)
{
    using T = flx::floatx<6, 51>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333325932);
    EXPECT_EQ(double(b), 0.11111111111111110494);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_6_52_simple)
{
    using T = flx::floatx<6, 52>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333331483);
    EXPECT_EQ(double(b), 0.11111111111111110494);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_7_1_simple)
{
    using T = flx::floatx<7, 1>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.37500000000000000000);
    EXPECT_EQ(double(b), 0.12500000000000000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_7_2_simple)
{
    using T = flx::floatx<7, 2>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.31250000000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_7_3_simple)
{
    using T = flx::floatx<7, 3>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.34375000000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 3.25000000000000000000);
}

TEST(FloatxOperationsTest, DIV_7_4_simple)
{
    using T = flx::floatx<7, 4>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.32812500000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_7_5_simple)
{
    using T = flx::floatx<7, 5>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33593750000000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_7_6_simple)
{
    using T = flx::floatx<7, 6>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33203125000000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 2.96875000000000000000);
}

TEST(FloatxOperationsTest, DIV_7_7_simple)
{
    using T = flx::floatx<7, 7>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33398437500000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_7_8_simple)
{
    using T = flx::floatx<7, 8>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33300781250000000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_7_9_simple)
{
    using T = flx::floatx<7, 9>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33349609375000000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 3.00390625000000000000);
}

TEST(FloatxOperationsTest, DIV_7_10_simple)
{
    using T = flx::floatx<7, 10>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33325195312500000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_7_11_simple)
{
    using T = flx::floatx<7, 11>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33337402343750000000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_7_12_simple)
{
    using T = flx::floatx<7, 12>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33331298828125000000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 2.99951171875000000000);
}

TEST(FloatxOperationsTest, DIV_7_13_simple)
{
    using T = flx::floatx<7, 13>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33334350585937500000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_7_14_simple)
{
    using T = flx::floatx<7, 14>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33332824707031250000);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_7_15_simple)
{
    using T = flx::floatx<7, 15>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333587646484375000);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 3.00006103515625000000);
}

TEST(FloatxOperationsTest, DIV_7_16_simple)
{
    using T = flx::floatx<7, 16>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333206176757812500);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_7_17_simple)
{
    using T = flx::floatx<7, 17>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333396911621093750);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_7_18_simple)
{
    using T = flx::floatx<7, 18>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333301544189453125);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 2.99999237060546875000);
}

TEST(FloatxOperationsTest, DIV_7_19_simple)
{
    using T = flx::floatx<7, 19>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333349227905273438);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_7_20_simple)
{
    using T = flx::floatx<7, 20>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333325386047363281);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_7_21_simple)
{
    using T = flx::floatx<7, 21>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333337306976318359);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 3.00000095367431640625);
}

TEST(FloatxOperationsTest, DIV_7_22_simple)
{
    using T = flx::floatx<7, 22>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333331346511840820);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_7_23_simple)
{
    using T = flx::floatx<7, 23>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333334326744079590);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_7_24_simple)
{
    using T = flx::floatx<7, 24>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333332836627960205);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 2.99999988079071044922);
}

TEST(FloatxOperationsTest, DIV_7_25_simple)
{
    using T = flx::floatx<7, 25>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333581686019897);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_7_26_simple)
{
    using T = flx::floatx<7, 26>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333209156990051);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_7_27_simple)
{
    using T = flx::floatx<7, 27>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333395421504974);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 3.00000001490116119385);
}

TEST(FloatxOperationsTest, DIV_7_28_simple)
{
    using T = flx::floatx<7, 28>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333302289247513);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_7_29_simple)
{
    using T = flx::floatx<7, 29>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333348855376244);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_7_30_simple)
{
    using T = flx::floatx<7, 30>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333325572311878);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 2.99999999813735485077);
}

TEST(FloatxOperationsTest, DIV_7_31_simple)
{
    using T = flx::floatx<7, 31>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333337213844061);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_7_32_simple)
{
    using T = flx::floatx<7, 32>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333331393077970);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_7_33_simple)
{
    using T = flx::floatx<7, 33>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333334303461015);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 3.00000000023283064365);
}

TEST(FloatxOperationsTest, DIV_7_34_simple)
{
    using T = flx::floatx<7, 34>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333332848269492);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_7_35_simple)
{
    using T = flx::floatx<7, 35>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333575865254);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_7_36_simple)
{
    using T = flx::floatx<7, 36>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333212067373);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 2.99999999997089616954);
}

TEST(FloatxOperationsTest, DIV_7_37_simple)
{
    using T = flx::floatx<7, 37>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333393966313);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_7_38_simple)
{
    using T = flx::floatx<7, 38>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333303016843);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_7_39_simple)
{
    using T = flx::floatx<7, 39>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333348491578);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 3.00000000000363797881);
}

TEST(FloatxOperationsTest, DIV_7_40_simple)
{
    using T = flx::floatx<7, 40>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333325754211);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_7_41_simple)
{
    using T = flx::floatx<7, 41>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333337122895);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_7_42_simple)
{
    using T = flx::floatx<7, 42>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333331438553);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 2.99999999999954525265);
}

TEST(FloatxOperationsTest, DIV_7_43_simple)
{
    using T = flx::floatx<7, 43>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333334280724);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_7_44_simple)
{
    using T = flx::floatx<7, 44>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333332859638);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_7_45_simple)
{
    using T = flx::floatx<7, 45>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333570181);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 3.00000000000005684342);
}

TEST(FloatxOperationsTest, DIV_7_46_simple)
{
    using T = flx::floatx<7, 46>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333214910);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_7_47_simple)
{
    using T = flx::floatx<7, 47>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333392545);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_7_48_simple)
{
    using T = flx::floatx<7, 48>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333303727);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 2.99999999999999289457);
}

TEST(FloatxOperationsTest, DIV_7_49_simple)
{
    using T = flx::floatx<7, 49>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333348136);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_7_50_simple)
{
    using T = flx::floatx<7, 50>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333325932);
    EXPECT_EQ(double(b), 0.11111111111111110494);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_7_51_simple)
{
    using T = flx::floatx<7, 51>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333325932);
    EXPECT_EQ(double(b), 0.11111111111111110494);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_7_52_simple)
{
    using T = flx::floatx<7, 52>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333331483);
    EXPECT_EQ(double(b), 0.11111111111111110494);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_8_1_simple)
{
    using T = flx::floatx<8, 1>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.37500000000000000000);
    EXPECT_EQ(double(b), 0.12500000000000000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_8_2_simple)
{
    using T = flx::floatx<8, 2>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.31250000000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_8_3_simple)
{
    using T = flx::floatx<8, 3>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.34375000000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 3.25000000000000000000);
}

TEST(FloatxOperationsTest, DIV_8_4_simple)
{
    using T = flx::floatx<8, 4>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.32812500000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_8_5_simple)
{
    using T = flx::floatx<8, 5>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33593750000000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_8_6_simple)
{
    using T = flx::floatx<8, 6>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33203125000000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 2.96875000000000000000);
}

TEST(FloatxOperationsTest, DIV_8_7_simple)
{
    using T = flx::floatx<8, 7>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33398437500000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_8_8_simple)
{
    using T = flx::floatx<8, 8>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33300781250000000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_8_9_simple)
{
    using T = flx::floatx<8, 9>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33349609375000000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 3.00390625000000000000);
}

TEST(FloatxOperationsTest, DIV_8_10_simple)
{
    using T = flx::floatx<8, 10>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33325195312500000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_8_11_simple)
{
    using T = flx::floatx<8, 11>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33337402343750000000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_8_12_simple)
{
    using T = flx::floatx<8, 12>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33331298828125000000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 2.99951171875000000000);
}

TEST(FloatxOperationsTest, DIV_8_13_simple)
{
    using T = flx::floatx<8, 13>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33334350585937500000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_8_14_simple)
{
    using T = flx::floatx<8, 14>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33332824707031250000);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_8_15_simple)
{
    using T = flx::floatx<8, 15>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333587646484375000);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 3.00006103515625000000);
}

TEST(FloatxOperationsTest, DIV_8_16_simple)
{
    using T = flx::floatx<8, 16>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333206176757812500);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_8_17_simple)
{
    using T = flx::floatx<8, 17>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333396911621093750);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_8_18_simple)
{
    using T = flx::floatx<8, 18>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333301544189453125);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 2.99999237060546875000);
}

TEST(FloatxOperationsTest, DIV_8_19_simple)
{
    using T = flx::floatx<8, 19>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333349227905273438);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_8_20_simple)
{
    using T = flx::floatx<8, 20>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333325386047363281);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_8_21_simple)
{
    using T = flx::floatx<8, 21>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333337306976318359);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 3.00000095367431640625);
}

TEST(FloatxOperationsTest, DIV_8_22_simple)
{
    using T = flx::floatx<8, 22>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333331346511840820);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_8_23_simple)
{
    using T = flx::floatx<8, 23>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333334326744079590);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_8_24_simple)
{
    using T = flx::floatx<8, 24>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333332836627960205);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 2.99999988079071044922);
}

TEST(FloatxOperationsTest, DIV_8_25_simple)
{
    using T = flx::floatx<8, 25>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333581686019897);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_8_26_simple)
{
    using T = flx::floatx<8, 26>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333209156990051);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_8_27_simple)
{
    using T = flx::floatx<8, 27>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333395421504974);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 3.00000001490116119385);
}

TEST(FloatxOperationsTest, DIV_8_28_simple)
{
    using T = flx::floatx<8, 28>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333302289247513);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_8_29_simple)
{
    using T = flx::floatx<8, 29>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333348855376244);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_8_30_simple)
{
    using T = flx::floatx<8, 30>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333325572311878);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 2.99999999813735485077);
}

TEST(FloatxOperationsTest, DIV_8_31_simple)
{
    using T = flx::floatx<8, 31>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333337213844061);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_8_32_simple)
{
    using T = flx::floatx<8, 32>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333331393077970);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_8_33_simple)
{
    using T = flx::floatx<8, 33>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333334303461015);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 3.00000000023283064365);
}

TEST(FloatxOperationsTest, DIV_8_34_simple)
{
    using T = flx::floatx<8, 34>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333332848269492);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_8_35_simple)
{
    using T = flx::floatx<8, 35>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333575865254);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_8_36_simple)
{
    using T = flx::floatx<8, 36>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333212067373);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 2.99999999997089616954);
}

TEST(FloatxOperationsTest, DIV_8_37_simple)
{
    using T = flx::floatx<8, 37>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333393966313);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_8_38_simple)
{
    using T = flx::floatx<8, 38>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333303016843);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_8_39_simple)
{
    using T = flx::floatx<8, 39>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333348491578);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 3.00000000000363797881);
}

TEST(FloatxOperationsTest, DIV_8_40_simple)
{
    using T = flx::floatx<8, 40>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333325754211);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_8_41_simple)
{
    using T = flx::floatx<8, 41>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333337122895);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_8_42_simple)
{
    using T = flx::floatx<8, 42>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333331438553);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 2.99999999999954525265);
}

TEST(FloatxOperationsTest, DIV_8_43_simple)
{
    using T = flx::floatx<8, 43>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333334280724);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_8_44_simple)
{
    using T = flx::floatx<8, 44>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333332859638);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_8_45_simple)
{
    using T = flx::floatx<8, 45>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333570181);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 3.00000000000005684342);
}

TEST(FloatxOperationsTest, DIV_8_46_simple)
{
    using T = flx::floatx<8, 46>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333214910);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_8_47_simple)
{
    using T = flx::floatx<8, 47>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333392545);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_8_48_simple)
{
    using T = flx::floatx<8, 48>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333303727);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 2.99999999999999289457);
}

TEST(FloatxOperationsTest, DIV_8_49_simple)
{
    using T = flx::floatx<8, 49>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333348136);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_8_50_simple)
{
    using T = flx::floatx<8, 50>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333325932);
    EXPECT_EQ(double(b), 0.11111111111111110494);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_8_51_simple)
{
    using T = flx::floatx<8, 51>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333325932);
    EXPECT_EQ(double(b), 0.11111111111111110494);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_8_52_simple)
{
    using T = flx::floatx<8, 52>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333331483);
    EXPECT_EQ(double(b), 0.11111111111111110494);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_9_1_simple)
{
    using T = flx::floatx<9, 1>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.37500000000000000000);
    EXPECT_EQ(double(b), 0.12500000000000000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_9_2_simple)
{
    using T = flx::floatx<9, 2>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.31250000000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_9_3_simple)
{
    using T = flx::floatx<9, 3>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.34375000000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 3.25000000000000000000);
}

TEST(FloatxOperationsTest, DIV_9_4_simple)
{
    using T = flx::floatx<9, 4>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.32812500000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_9_5_simple)
{
    using T = flx::floatx<9, 5>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33593750000000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_9_6_simple)
{
    using T = flx::floatx<9, 6>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33203125000000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 2.96875000000000000000);
}

TEST(FloatxOperationsTest, DIV_9_7_simple)
{
    using T = flx::floatx<9, 7>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33398437500000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_9_8_simple)
{
    using T = flx::floatx<9, 8>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33300781250000000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_9_9_simple)
{
    using T = flx::floatx<9, 9>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33349609375000000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 3.00390625000000000000);
}

TEST(FloatxOperationsTest, DIV_9_10_simple)
{
    using T = flx::floatx<9, 10>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33325195312500000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_9_11_simple)
{
    using T = flx::floatx<9, 11>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33337402343750000000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_9_12_simple)
{
    using T = flx::floatx<9, 12>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33331298828125000000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 2.99951171875000000000);
}

TEST(FloatxOperationsTest, DIV_9_13_simple)
{
    using T = flx::floatx<9, 13>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33334350585937500000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_9_14_simple)
{
    using T = flx::floatx<9, 14>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33332824707031250000);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_9_15_simple)
{
    using T = flx::floatx<9, 15>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333587646484375000);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 3.00006103515625000000);
}

TEST(FloatxOperationsTest, DIV_9_16_simple)
{
    using T = flx::floatx<9, 16>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333206176757812500);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_9_17_simple)
{
    using T = flx::floatx<9, 17>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333396911621093750);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_9_18_simple)
{
    using T = flx::floatx<9, 18>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333301544189453125);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 2.99999237060546875000);
}

TEST(FloatxOperationsTest, DIV_9_19_simple)
{
    using T = flx::floatx<9, 19>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333349227905273438);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_9_20_simple)
{
    using T = flx::floatx<9, 20>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333325386047363281);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_9_21_simple)
{
    using T = flx::floatx<9, 21>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333337306976318359);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 3.00000095367431640625);
}

TEST(FloatxOperationsTest, DIV_9_22_simple)
{
    using T = flx::floatx<9, 22>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333331346511840820);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_9_23_simple)
{
    using T = flx::floatx<9, 23>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333334326744079590);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_9_24_simple)
{
    using T = flx::floatx<9, 24>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333332836627960205);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 2.99999988079071044922);
}

TEST(FloatxOperationsTest, DIV_9_25_simple)
{
    using T = flx::floatx<9, 25>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333581686019897);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_9_26_simple)
{
    using T = flx::floatx<9, 26>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333209156990051);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_9_27_simple)
{
    using T = flx::floatx<9, 27>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333395421504974);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 3.00000001490116119385);
}

TEST(FloatxOperationsTest, DIV_9_28_simple)
{
    using T = flx::floatx<9, 28>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333302289247513);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_9_29_simple)
{
    using T = flx::floatx<9, 29>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333348855376244);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_9_30_simple)
{
    using T = flx::floatx<9, 30>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333325572311878);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 2.99999999813735485077);
}

TEST(FloatxOperationsTest, DIV_9_31_simple)
{
    using T = flx::floatx<9, 31>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333337213844061);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_9_32_simple)
{
    using T = flx::floatx<9, 32>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333331393077970);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_9_33_simple)
{
    using T = flx::floatx<9, 33>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333334303461015);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 3.00000000023283064365);
}

TEST(FloatxOperationsTest, DIV_9_34_simple)
{
    using T = flx::floatx<9, 34>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333332848269492);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_9_35_simple)
{
    using T = flx::floatx<9, 35>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333575865254);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_9_36_simple)
{
    using T = flx::floatx<9, 36>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333212067373);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 2.99999999997089616954);
}

TEST(FloatxOperationsTest, DIV_9_37_simple)
{
    using T = flx::floatx<9, 37>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333393966313);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_9_38_simple)
{
    using T = flx::floatx<9, 38>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333303016843);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_9_39_simple)
{
    using T = flx::floatx<9, 39>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333348491578);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 3.00000000000363797881);
}

TEST(FloatxOperationsTest, DIV_9_40_simple)
{
    using T = flx::floatx<9, 40>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333325754211);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_9_41_simple)
{
    using T = flx::floatx<9, 41>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333337122895);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_9_42_simple)
{
    using T = flx::floatx<9, 42>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333331438553);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 2.99999999999954525265);
}

TEST(FloatxOperationsTest, DIV_9_43_simple)
{
    using T = flx::floatx<9, 43>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333334280724);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_9_44_simple)
{
    using T = flx::floatx<9, 44>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333332859638);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_9_45_simple)
{
    using T = flx::floatx<9, 45>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333570181);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 3.00000000000005684342);
}

TEST(FloatxOperationsTest, DIV_9_46_simple)
{
    using T = flx::floatx<9, 46>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333214910);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_9_47_simple)
{
    using T = flx::floatx<9, 47>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333392545);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_9_48_simple)
{
    using T = flx::floatx<9, 48>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333303727);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 2.99999999999999289457);
}

TEST(FloatxOperationsTest, DIV_9_49_simple)
{
    using T = flx::floatx<9, 49>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333348136);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_9_50_simple)
{
    using T = flx::floatx<9, 50>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333325932);
    EXPECT_EQ(double(b), 0.11111111111111110494);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_9_51_simple)
{
    using T = flx::floatx<9, 51>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333325932);
    EXPECT_EQ(double(b), 0.11111111111111110494);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_9_52_simple)
{
    using T = flx::floatx<9, 52>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333331483);
    EXPECT_EQ(double(b), 0.11111111111111110494);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_10_1_simple)
{
    using T = flx::floatx<10, 1>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.37500000000000000000);
    EXPECT_EQ(double(b), 0.12500000000000000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_10_2_simple)
{
    using T = flx::floatx<10, 2>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.31250000000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_10_3_simple)
{
    using T = flx::floatx<10, 3>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.34375000000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 3.25000000000000000000);
}

TEST(FloatxOperationsTest, DIV_10_4_simple)
{
    using T = flx::floatx<10, 4>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.32812500000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_10_5_simple)
{
    using T = flx::floatx<10, 5>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33593750000000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_10_6_simple)
{
    using T = flx::floatx<10, 6>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33203125000000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 2.96875000000000000000);
}

TEST(FloatxOperationsTest, DIV_10_7_simple)
{
    using T = flx::floatx<10, 7>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33398437500000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_10_8_simple)
{
    using T = flx::floatx<10, 8>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33300781250000000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_10_9_simple)
{
    using T = flx::floatx<10, 9>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33349609375000000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 3.00390625000000000000);
}

TEST(FloatxOperationsTest, DIV_10_10_simple)
{
    using T = flx::floatx<10, 10>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33325195312500000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_10_11_simple)
{
    using T = flx::floatx<10, 11>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33337402343750000000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_10_12_simple)
{
    using T = flx::floatx<10, 12>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33331298828125000000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 2.99951171875000000000);
}

TEST(FloatxOperationsTest, DIV_10_13_simple)
{
    using T = flx::floatx<10, 13>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33334350585937500000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_10_14_simple)
{
    using T = flx::floatx<10, 14>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33332824707031250000);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_10_15_simple)
{
    using T = flx::floatx<10, 15>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333587646484375000);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 3.00006103515625000000);
}

TEST(FloatxOperationsTest, DIV_10_16_simple)
{
    using T = flx::floatx<10, 16>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333206176757812500);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_10_17_simple)
{
    using T = flx::floatx<10, 17>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333396911621093750);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_10_18_simple)
{
    using T = flx::floatx<10, 18>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333301544189453125);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 2.99999237060546875000);
}

TEST(FloatxOperationsTest, DIV_10_19_simple)
{
    using T = flx::floatx<10, 19>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333349227905273438);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_10_20_simple)
{
    using T = flx::floatx<10, 20>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333325386047363281);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_10_21_simple)
{
    using T = flx::floatx<10, 21>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333337306976318359);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 3.00000095367431640625);
}

TEST(FloatxOperationsTest, DIV_10_22_simple)
{
    using T = flx::floatx<10, 22>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333331346511840820);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_10_23_simple)
{
    using T = flx::floatx<10, 23>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333334326744079590);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_10_24_simple)
{
    using T = flx::floatx<10, 24>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333332836627960205);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 2.99999988079071044922);
}

TEST(FloatxOperationsTest, DIV_10_25_simple)
{
    using T = flx::floatx<10, 25>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333581686019897);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_10_26_simple)
{
    using T = flx::floatx<10, 26>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333209156990051);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_10_27_simple)
{
    using T = flx::floatx<10, 27>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333395421504974);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 3.00000001490116119385);
}

TEST(FloatxOperationsTest, DIV_10_28_simple)
{
    using T = flx::floatx<10, 28>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333302289247513);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_10_29_simple)
{
    using T = flx::floatx<10, 29>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333348855376244);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_10_30_simple)
{
    using T = flx::floatx<10, 30>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333325572311878);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 2.99999999813735485077);
}

TEST(FloatxOperationsTest, DIV_10_31_simple)
{
    using T = flx::floatx<10, 31>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333337213844061);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_10_32_simple)
{
    using T = flx::floatx<10, 32>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333331393077970);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_10_33_simple)
{
    using T = flx::floatx<10, 33>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333334303461015);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 3.00000000023283064365);
}

TEST(FloatxOperationsTest, DIV_10_34_simple)
{
    using T = flx::floatx<10, 34>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333332848269492);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_10_35_simple)
{
    using T = flx::floatx<10, 35>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333575865254);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_10_36_simple)
{
    using T = flx::floatx<10, 36>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333212067373);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 2.99999999997089616954);
}

TEST(FloatxOperationsTest, DIV_10_37_simple)
{
    using T = flx::floatx<10, 37>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333393966313);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_10_38_simple)
{
    using T = flx::floatx<10, 38>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333303016843);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_10_39_simple)
{
    using T = flx::floatx<10, 39>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333348491578);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 3.00000000000363797881);
}

TEST(FloatxOperationsTest, DIV_10_40_simple)
{
    using T = flx::floatx<10, 40>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333325754211);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_10_41_simple)
{
    using T = flx::floatx<10, 41>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333337122895);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_10_42_simple)
{
    using T = flx::floatx<10, 42>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333331438553);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 2.99999999999954525265);
}

TEST(FloatxOperationsTest, DIV_10_43_simple)
{
    using T = flx::floatx<10, 43>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333334280724);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_10_44_simple)
{
    using T = flx::floatx<10, 44>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333332859638);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_10_45_simple)
{
    using T = flx::floatx<10, 45>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333570181);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 3.00000000000005684342);
}

TEST(FloatxOperationsTest, DIV_10_46_simple)
{
    using T = flx::floatx<10, 46>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333214910);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_10_47_simple)
{
    using T = flx::floatx<10, 47>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333392545);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_10_48_simple)
{
    using T = flx::floatx<10, 48>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333303727);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 2.99999999999999289457);
}

TEST(FloatxOperationsTest, DIV_10_49_simple)
{
    using T = flx::floatx<10, 49>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333348136);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_10_50_simple)
{
    using T = flx::floatx<10, 50>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333325932);
    EXPECT_EQ(double(b), 0.11111111111111110494);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_10_51_simple)
{
    using T = flx::floatx<10, 51>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333325932);
    EXPECT_EQ(double(b), 0.11111111111111110494);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_10_52_simple)
{
    using T = flx::floatx<10, 52>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333331483);
    EXPECT_EQ(double(b), 0.11111111111111110494);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_11_1_simple)
{
    using T = flx::floatx<11, 1>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.37500000000000000000);
    EXPECT_EQ(double(b), 0.12500000000000000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_11_2_simple)
{
    using T = flx::floatx<11, 2>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.31250000000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_11_3_simple)
{
    using T = flx::floatx<11, 3>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.34375000000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 3.25000000000000000000);
}

TEST(FloatxOperationsTest, DIV_11_4_simple)
{
    using T = flx::floatx<11, 4>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.32812500000000000000);
    EXPECT_EQ(double(b), 0.10937500000000000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_11_5_simple)
{
    using T = flx::floatx<11, 5>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33593750000000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_11_6_simple)
{
    using T = flx::floatx<11, 6>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33203125000000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 2.96875000000000000000);
}

TEST(FloatxOperationsTest, DIV_11_7_simple)
{
    using T = flx::floatx<11, 7>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33398437500000000000);
    EXPECT_EQ(double(b), 0.11132812500000000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_11_8_simple)
{
    using T = flx::floatx<11, 8>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33300781250000000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_11_9_simple)
{
    using T = flx::floatx<11, 9>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33349609375000000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 3.00390625000000000000);
}

TEST(FloatxOperationsTest, DIV_11_10_simple)
{
    using T = flx::floatx<11, 10>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33325195312500000000);
    EXPECT_EQ(double(b), 0.11108398437500000000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_11_11_simple)
{
    using T = flx::floatx<11, 11>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33337402343750000000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_11_12_simple)
{
    using T = flx::floatx<11, 12>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33331298828125000000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 2.99951171875000000000);
}

TEST(FloatxOperationsTest, DIV_11_13_simple)
{
    using T = flx::floatx<11, 13>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33334350585937500000);
    EXPECT_EQ(double(b), 0.11111450195312500000);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_11_14_simple)
{
    using T = flx::floatx<11, 14>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33332824707031250000);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_11_15_simple)
{
    using T = flx::floatx<11, 15>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333587646484375000);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 3.00006103515625000000);
}

TEST(FloatxOperationsTest, DIV_11_16_simple)
{
    using T = flx::floatx<11, 16>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333206176757812500);
    EXPECT_EQ(double(b), 0.11111068725585937500);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_11_17_simple)
{
    using T = flx::floatx<11, 17>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333396911621093750);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_11_18_simple)
{
    using T = flx::floatx<11, 18>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333301544189453125);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 2.99999237060546875000);
}

TEST(FloatxOperationsTest, DIV_11_19_simple)
{
    using T = flx::floatx<11, 19>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333349227905273438);
    EXPECT_EQ(double(b), 0.11111116409301757812);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_11_20_simple)
{
    using T = flx::floatx<11, 20>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333325386047363281);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_11_21_simple)
{
    using T = flx::floatx<11, 21>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333337306976318359);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 3.00000095367431640625);
}

TEST(FloatxOperationsTest, DIV_11_22_simple)
{
    using T = flx::floatx<11, 22>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333331346511840820);
    EXPECT_EQ(double(b), 0.11111110448837280273);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_11_23_simple)
{
    using T = flx::floatx<11, 23>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333334326744079590);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_11_24_simple)
{
    using T = flx::floatx<11, 24>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333332836627960205);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 2.99999988079071044922);
}

TEST(FloatxOperationsTest, DIV_11_25_simple)
{
    using T = flx::floatx<11, 25>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333581686019897);
    EXPECT_EQ(double(b), 0.11111111193895339966);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_11_26_simple)
{
    using T = flx::floatx<11, 26>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333209156990051);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_11_27_simple)
{
    using T = flx::floatx<11, 27>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333395421504974);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 3.00000001490116119385);
}

TEST(FloatxOperationsTest, DIV_11_28_simple)
{
    using T = flx::floatx<11, 28>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333302289247513);
    EXPECT_EQ(double(b), 0.11111111100763082504);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_11_29_simple)
{
    using T = flx::floatx<11, 29>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333348855376244);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_11_30_simple)
{
    using T = flx::floatx<11, 30>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333325572311878);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 2.99999999813735485077);
}

TEST(FloatxOperationsTest, DIV_11_31_simple)
{
    using T = flx::floatx<11, 31>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333337213844061);
    EXPECT_EQ(double(b), 0.11111111112404614687);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_11_32_simple)
{
    using T = flx::floatx<11, 32>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333331393077970);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_11_33_simple)
{
    using T = flx::floatx<11, 33>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333334303461015);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 3.00000000023283064365);
}

TEST(FloatxOperationsTest, DIV_11_34_simple)
{
    using T = flx::floatx<11, 34>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333332848269492);
    EXPECT_EQ(double(b), 0.11111111110949423164);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_11_35_simple)
{
    using T = flx::floatx<11, 35>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333575865254);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_11_36_simple)
{
    using T = flx::floatx<11, 36>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333212067373);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 2.99999999997089616954);
}

TEST(FloatxOperationsTest, DIV_11_37_simple)
{
    using T = flx::floatx<11, 37>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333393966313);
    EXPECT_EQ(double(b), 0.11111111111131322104);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_11_38_simple)
{
    using T = flx::floatx<11, 38>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333303016843);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_11_39_simple)
{
    using T = flx::floatx<11, 39>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333348491578);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 3.00000000000363797881);
}

TEST(FloatxOperationsTest, DIV_11_40_simple)
{
    using T = flx::floatx<11, 40>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333325754211);
    EXPECT_EQ(double(b), 0.11111111111108584737);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_11_41_simple)
{
    using T = flx::floatx<11, 41>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333337122895);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_11_42_simple)
{
    using T = flx::floatx<11, 42>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333331438553);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 2.99999999999954525265);
}

TEST(FloatxOperationsTest, DIV_11_43_simple)
{
    using T = flx::floatx<11, 43>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333334280724);
    EXPECT_EQ(double(b), 0.11111111111111426908);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_11_44_simple)
{
    using T = flx::floatx<11, 44>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333332859638);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_11_45_simple)
{
    using T = flx::floatx<11, 45>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333570181);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 3.00000000000005684342);
}

TEST(FloatxOperationsTest, DIV_11_46_simple)
{
    using T = flx::floatx<11, 46>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333214910);
    EXPECT_EQ(double(b), 0.11111111111111071637);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_11_47_simple)
{
    using T = flx::floatx<11, 47>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333392545);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_11_48_simple)
{
    using T = flx::floatx<11, 48>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333303727);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 2.99999999999999289457);
}

TEST(FloatxOperationsTest, DIV_11_49_simple)
{
    using T = flx::floatx<11, 49>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333348136);
    EXPECT_EQ(double(b), 0.11111111111111116045);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_11_50_simple)
{
    using T = flx::floatx<11, 50>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333325932);
    EXPECT_EQ(double(b), 0.11111111111111110494);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_11_51_simple)
{
    using T = flx::floatx<11, 51>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333325932);
    EXPECT_EQ(double(b), 0.11111111111111110494);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}

TEST(FloatxOperationsTest, DIV_11_52_simple)
{
    using T = flx::floatx<11, 52>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a / b;
    EXPECT_EQ(double(a), 0.33333333333333331483);
    EXPECT_EQ(double(b), 0.11111111111111110494);
    EXPECT_EQ(double(c), 3.00000000000000000000);
}


}  // namespace
