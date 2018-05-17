#include <gtest/gtest.h>
#include <floatx.hpp>

#include "cuda_assert_op.cuh"
#include "cuda_test.cuh"

#define nan double(0.0 / 0.0)
#define inf double(1.0 / 0.0)

namespace {

CUDA_TEST(FloatxOperationsTest, MUL_1_1_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 1>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.00000000000000000000);
    CUDA_ASSERT_EQ(double(b), 0.00000000000000000000);
    CUDA_ASSERT_EQ(double(c), 0.00000000000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_1_2_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 2>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.50000000000000000000);
    CUDA_ASSERT_EQ(double(b), 0.00000000000000000000);
    CUDA_ASSERT_EQ(double(c), 0.00000000000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_1_3_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 3>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.25000000000000000000);
    CUDA_ASSERT_EQ(double(b), 0.00000000000000000000);
    CUDA_ASSERT_EQ(double(c), 0.00000000000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_1_4_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 4>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.37500000000000000000);
    CUDA_ASSERT_EQ(double(b), 0.12500000000000000000);
    CUDA_ASSERT_EQ(double(c), 0.00000000000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_1_5_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 5>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.31250000000000000000);
    CUDA_ASSERT_EQ(double(b), 0.12500000000000000000);
    CUDA_ASSERT_EQ(double(c), 0.06250000000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_1_6_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 6>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.34375000000000000000);
    CUDA_ASSERT_EQ(double(b), 0.12500000000000000000);
    CUDA_ASSERT_EQ(double(c), 0.03125000000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_1_7_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 7>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.32812500000000000000);
    CUDA_ASSERT_EQ(double(b), 0.10937500000000000000);
    CUDA_ASSERT_EQ(double(c), 0.03125000000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_1_8_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 8>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33593750000000000000);
    CUDA_ASSERT_EQ(double(b), 0.10937500000000000000);
    CUDA_ASSERT_EQ(double(c), 0.03906250000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_1_9_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 9>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33203125000000000000);
    CUDA_ASSERT_EQ(double(b), 0.10937500000000000000);
    CUDA_ASSERT_EQ(double(c), 0.03515625000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_1_10_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 10>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33398437500000000000);
    CUDA_ASSERT_EQ(double(b), 0.11132812500000000000);
    CUDA_ASSERT_EQ(double(c), 0.03710937500000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_1_11_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 11>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33300781250000000000);
    CUDA_ASSERT_EQ(double(b), 0.11132812500000000000);
    CUDA_ASSERT_EQ(double(c), 0.03710937500000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_1_12_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 12>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33349609375000000000);
    CUDA_ASSERT_EQ(double(b), 0.11132812500000000000);
    CUDA_ASSERT_EQ(double(c), 0.03710937500000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_1_13_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 13>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33325195312500000000);
    CUDA_ASSERT_EQ(double(b), 0.11108398437500000000);
    CUDA_ASSERT_EQ(double(c), 0.03710937500000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_1_14_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 14>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33337402343750000000);
    CUDA_ASSERT_EQ(double(b), 0.11108398437500000000);
    CUDA_ASSERT_EQ(double(c), 0.03698730468750000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_1_15_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 15>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33331298828125000000);
    CUDA_ASSERT_EQ(double(b), 0.11108398437500000000);
    CUDA_ASSERT_EQ(double(c), 0.03704833984375000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_1_16_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 16>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33334350585937500000);
    CUDA_ASSERT_EQ(double(b), 0.11111450195312500000);
    CUDA_ASSERT_EQ(double(c), 0.03704833984375000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_1_17_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 17>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33332824707031250000);
    CUDA_ASSERT_EQ(double(b), 0.11111450195312500000);
    CUDA_ASSERT_EQ(double(c), 0.03703308105468750000);
}

CUDA_TEST(FloatxOperationsTest, MUL_1_18_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 18>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333587646484375000);
    CUDA_ASSERT_EQ(double(b), 0.11111450195312500000);
    CUDA_ASSERT_EQ(double(c), 0.03704071044921875000);
}

CUDA_TEST(FloatxOperationsTest, MUL_1_19_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 19>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333206176757812500);
    CUDA_ASSERT_EQ(double(b), 0.11111068725585937500);
    CUDA_ASSERT_EQ(double(c), 0.03703689575195312500);
}

CUDA_TEST(FloatxOperationsTest, MUL_1_20_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 20>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333396911621093750);
    CUDA_ASSERT_EQ(double(b), 0.11111068725585937500);
    CUDA_ASSERT_EQ(double(c), 0.03703689575195312500);
}

CUDA_TEST(FloatxOperationsTest, MUL_1_21_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 21>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333301544189453125);
    CUDA_ASSERT_EQ(double(b), 0.11111068725585937500);
    CUDA_ASSERT_EQ(double(c), 0.03703689575195312500);
}

CUDA_TEST(FloatxOperationsTest, MUL_1_22_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 22>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333349227905273438);
    CUDA_ASSERT_EQ(double(b), 0.11111116409301757812);
    CUDA_ASSERT_EQ(double(c), 0.03703689575195312500);
}

CUDA_TEST(FloatxOperationsTest, MUL_1_23_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 23>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333325386047363281);
    CUDA_ASSERT_EQ(double(b), 0.11111116409301757812);
    CUDA_ASSERT_EQ(double(c), 0.03703713417053222656);
}

CUDA_TEST(FloatxOperationsTest, MUL_1_24_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 24>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333337306976318359);
    CUDA_ASSERT_EQ(double(b), 0.11111116409301757812);
    CUDA_ASSERT_EQ(double(c), 0.03703701496124267578);
}

CUDA_TEST(FloatxOperationsTest, MUL_1_25_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 25>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333331346511840820);
    CUDA_ASSERT_EQ(double(b), 0.11111110448837280273);
    CUDA_ASSERT_EQ(double(c), 0.03703701496124267578);
}

CUDA_TEST(FloatxOperationsTest, MUL_1_26_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 26>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333334326744079590);
    CUDA_ASSERT_EQ(double(b), 0.11111110448837280273);
    CUDA_ASSERT_EQ(double(c), 0.03703704476356506348);
}

CUDA_TEST(FloatxOperationsTest, MUL_1_27_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 27>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333332836627960205);
    CUDA_ASSERT_EQ(double(b), 0.11111110448837280273);
    CUDA_ASSERT_EQ(double(c), 0.03703702986240386963);
}

CUDA_TEST(FloatxOperationsTest, MUL_1_28_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 28>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333581686019897);
    CUDA_ASSERT_EQ(double(b), 0.11111111193895339966);
    CUDA_ASSERT_EQ(double(c), 0.03703703731298446655);
}

CUDA_TEST(FloatxOperationsTest, MUL_1_29_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 29>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333209156990051);
    CUDA_ASSERT_EQ(double(b), 0.11111111193895339966);
    CUDA_ASSERT_EQ(double(c), 0.03703703731298446655);
}

CUDA_TEST(FloatxOperationsTest, MUL_1_30_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 30>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333395421504974);
    CUDA_ASSERT_EQ(double(b), 0.11111111193895339966);
    CUDA_ASSERT_EQ(double(c), 0.03703703731298446655);
}

CUDA_TEST(FloatxOperationsTest, MUL_1_31_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 31>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333302289247513);
    CUDA_ASSERT_EQ(double(b), 0.11111111100763082504);
    CUDA_ASSERT_EQ(double(c), 0.03703703731298446655);
}

CUDA_TEST(FloatxOperationsTest, MUL_1_32_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 32>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333348855376244);
    CUDA_ASSERT_EQ(double(b), 0.11111111100763082504);
    CUDA_ASSERT_EQ(double(c), 0.03703703684732317924);
}

CUDA_TEST(FloatxOperationsTest, MUL_1_33_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 33>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333325572311878);
    CUDA_ASSERT_EQ(double(b), 0.11111111100763082504);
    CUDA_ASSERT_EQ(double(c), 0.03703703708015382290);
}

CUDA_TEST(FloatxOperationsTest, MUL_1_34_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 34>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333337213844061);
    CUDA_ASSERT_EQ(double(b), 0.11111111112404614687);
    CUDA_ASSERT_EQ(double(c), 0.03703703708015382290);
}

CUDA_TEST(FloatxOperationsTest, MUL_1_35_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 35>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333331393077970);
    CUDA_ASSERT_EQ(double(b), 0.11111111112404614687);
    CUDA_ASSERT_EQ(double(c), 0.03703703702194616199);
}

CUDA_TEST(FloatxOperationsTest, MUL_1_36_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 36>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333334303461015);
    CUDA_ASSERT_EQ(double(b), 0.11111111112404614687);
    CUDA_ASSERT_EQ(double(c), 0.03703703705104999244);
}

CUDA_TEST(FloatxOperationsTest, MUL_1_37_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 37>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333332848269492);
    CUDA_ASSERT_EQ(double(b), 0.11111111110949423164);
    CUDA_ASSERT_EQ(double(c), 0.03703703703649807721);
}

CUDA_TEST(FloatxOperationsTest, MUL_1_38_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 38>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333575865254);
    CUDA_ASSERT_EQ(double(b), 0.11111111110949423164);
    CUDA_ASSERT_EQ(double(c), 0.03703703703649807721);
}

CUDA_TEST(FloatxOperationsTest, MUL_1_39_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 39>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333212067373);
    CUDA_ASSERT_EQ(double(b), 0.11111111110949423164);
    CUDA_ASSERT_EQ(double(c), 0.03703703703649807721);
}

CUDA_TEST(FloatxOperationsTest, MUL_1_40_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 40>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333393966313);
    CUDA_ASSERT_EQ(double(b), 0.11111111111131322104);
    CUDA_ASSERT_EQ(double(c), 0.03703703703649807721);
}

CUDA_TEST(FloatxOperationsTest, MUL_1_41_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 41>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333303016843);
    CUDA_ASSERT_EQ(double(b), 0.11111111111131322104);
    CUDA_ASSERT_EQ(double(c), 0.03703703703740757192);
}

CUDA_TEST(FloatxOperationsTest, MUL_1_42_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 42>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333348491578);
    CUDA_ASSERT_EQ(double(b), 0.11111111111131322104);
    CUDA_ASSERT_EQ(double(c), 0.03703703703695282456);
}

CUDA_TEST(FloatxOperationsTest, MUL_1_43_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 43>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333325754211);
    CUDA_ASSERT_EQ(double(b), 0.11111111111108584737);
    CUDA_ASSERT_EQ(double(c), 0.03703703703695282456);
}

CUDA_TEST(FloatxOperationsTest, MUL_1_44_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 44>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333337122895);
    CUDA_ASSERT_EQ(double(b), 0.11111111111108584737);
    CUDA_ASSERT_EQ(double(c), 0.03703703703706651140);
}

CUDA_TEST(FloatxOperationsTest, MUL_1_45_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 45>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333331438553);
    CUDA_ASSERT_EQ(double(b), 0.11111111111108584737);
    CUDA_ASSERT_EQ(double(c), 0.03703703703700966798);
}

CUDA_TEST(FloatxOperationsTest, MUL_1_46_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 46>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333334280724);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111426908);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703808969);
}

CUDA_TEST(FloatxOperationsTest, MUL_1_47_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 47>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333332859638);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111426908);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703808969);
}

CUDA_TEST(FloatxOperationsTest, MUL_1_48_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 48>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333570181);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111426908);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703808969);
}

CUDA_TEST(FloatxOperationsTest, MUL_1_49_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 49>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333214910);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111071637);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703808969);
}

CUDA_TEST(FloatxOperationsTest, MUL_1_50_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 50>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333392545);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111071637);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703631334);
}

CUDA_TEST(FloatxOperationsTest, MUL_1_51_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 51>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333303727);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111071637);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703720151);
}

CUDA_TEST(FloatxOperationsTest, MUL_1_52_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 52>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333348136);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111116045);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703720151);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_1_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 1>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.50000000000000000000);
    CUDA_ASSERT_EQ(double(b), 0.00000000000000000000);
    CUDA_ASSERT_EQ(double(c), 0.00000000000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_2_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 2>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.25000000000000000000);
    CUDA_ASSERT_EQ(double(b), 0.00000000000000000000);
    CUDA_ASSERT_EQ(double(c), 0.00000000000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_3_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 3>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.37500000000000000000);
    CUDA_ASSERT_EQ(double(b), 0.12500000000000000000);
    CUDA_ASSERT_EQ(double(c), 0.00000000000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_4_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 4>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.31250000000000000000);
    CUDA_ASSERT_EQ(double(b), 0.12500000000000000000);
    CUDA_ASSERT_EQ(double(c), 0.06250000000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_5_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 5>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.34375000000000000000);
    CUDA_ASSERT_EQ(double(b), 0.12500000000000000000);
    CUDA_ASSERT_EQ(double(c), 0.03125000000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_6_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 6>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.32812500000000000000);
    CUDA_ASSERT_EQ(double(b), 0.10937500000000000000);
    CUDA_ASSERT_EQ(double(c), 0.03125000000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_7_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 7>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33593750000000000000);
    CUDA_ASSERT_EQ(double(b), 0.10937500000000000000);
    CUDA_ASSERT_EQ(double(c), 0.03906250000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_8_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 8>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33203125000000000000);
    CUDA_ASSERT_EQ(double(b), 0.10937500000000000000);
    CUDA_ASSERT_EQ(double(c), 0.03515625000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_9_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 9>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33398437500000000000);
    CUDA_ASSERT_EQ(double(b), 0.11132812500000000000);
    CUDA_ASSERT_EQ(double(c), 0.03710937500000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_10_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 10>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33300781250000000000);
    CUDA_ASSERT_EQ(double(b), 0.11132812500000000000);
    CUDA_ASSERT_EQ(double(c), 0.03710937500000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_11_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 11>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33349609375000000000);
    CUDA_ASSERT_EQ(double(b), 0.11132812500000000000);
    CUDA_ASSERT_EQ(double(c), 0.03710937500000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_12_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 12>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33325195312500000000);
    CUDA_ASSERT_EQ(double(b), 0.11108398437500000000);
    CUDA_ASSERT_EQ(double(c), 0.03710937500000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_13_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 13>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33337402343750000000);
    CUDA_ASSERT_EQ(double(b), 0.11108398437500000000);
    CUDA_ASSERT_EQ(double(c), 0.03698730468750000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_14_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 14>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33331298828125000000);
    CUDA_ASSERT_EQ(double(b), 0.11108398437500000000);
    CUDA_ASSERT_EQ(double(c), 0.03704833984375000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_15_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 15>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33334350585937500000);
    CUDA_ASSERT_EQ(double(b), 0.11111450195312500000);
    CUDA_ASSERT_EQ(double(c), 0.03704833984375000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_16_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 16>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33332824707031250000);
    CUDA_ASSERT_EQ(double(b), 0.11111450195312500000);
    CUDA_ASSERT_EQ(double(c), 0.03703308105468750000);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_17_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 17>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333587646484375000);
    CUDA_ASSERT_EQ(double(b), 0.11111450195312500000);
    CUDA_ASSERT_EQ(double(c), 0.03704071044921875000);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_18_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 18>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333206176757812500);
    CUDA_ASSERT_EQ(double(b), 0.11111068725585937500);
    CUDA_ASSERT_EQ(double(c), 0.03703689575195312500);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_19_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 19>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333396911621093750);
    CUDA_ASSERT_EQ(double(b), 0.11111068725585937500);
    CUDA_ASSERT_EQ(double(c), 0.03703689575195312500);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_20_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 20>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333301544189453125);
    CUDA_ASSERT_EQ(double(b), 0.11111068725585937500);
    CUDA_ASSERT_EQ(double(c), 0.03703689575195312500);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_21_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 21>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333349227905273438);
    CUDA_ASSERT_EQ(double(b), 0.11111116409301757812);
    CUDA_ASSERT_EQ(double(c), 0.03703689575195312500);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_22_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 22>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333325386047363281);
    CUDA_ASSERT_EQ(double(b), 0.11111116409301757812);
    CUDA_ASSERT_EQ(double(c), 0.03703713417053222656);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_23_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 23>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333337306976318359);
    CUDA_ASSERT_EQ(double(b), 0.11111116409301757812);
    CUDA_ASSERT_EQ(double(c), 0.03703701496124267578);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_24_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 24>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333331346511840820);
    CUDA_ASSERT_EQ(double(b), 0.11111110448837280273);
    CUDA_ASSERT_EQ(double(c), 0.03703701496124267578);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_25_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 25>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333334326744079590);
    CUDA_ASSERT_EQ(double(b), 0.11111110448837280273);
    CUDA_ASSERT_EQ(double(c), 0.03703704476356506348);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_26_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 26>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333332836627960205);
    CUDA_ASSERT_EQ(double(b), 0.11111110448837280273);
    CUDA_ASSERT_EQ(double(c), 0.03703702986240386963);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_27_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 27>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333581686019897);
    CUDA_ASSERT_EQ(double(b), 0.11111111193895339966);
    CUDA_ASSERT_EQ(double(c), 0.03703703731298446655);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_28_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 28>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333209156990051);
    CUDA_ASSERT_EQ(double(b), 0.11111111193895339966);
    CUDA_ASSERT_EQ(double(c), 0.03703703731298446655);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_29_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 29>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333395421504974);
    CUDA_ASSERT_EQ(double(b), 0.11111111193895339966);
    CUDA_ASSERT_EQ(double(c), 0.03703703731298446655);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_30_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 30>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333302289247513);
    CUDA_ASSERT_EQ(double(b), 0.11111111100763082504);
    CUDA_ASSERT_EQ(double(c), 0.03703703731298446655);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_31_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 31>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333348855376244);
    CUDA_ASSERT_EQ(double(b), 0.11111111100763082504);
    CUDA_ASSERT_EQ(double(c), 0.03703703684732317924);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_32_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 32>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333325572311878);
    CUDA_ASSERT_EQ(double(b), 0.11111111100763082504);
    CUDA_ASSERT_EQ(double(c), 0.03703703708015382290);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_33_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 33>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333337213844061);
    CUDA_ASSERT_EQ(double(b), 0.11111111112404614687);
    CUDA_ASSERT_EQ(double(c), 0.03703703708015382290);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_34_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 34>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333331393077970);
    CUDA_ASSERT_EQ(double(b), 0.11111111112404614687);
    CUDA_ASSERT_EQ(double(c), 0.03703703702194616199);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_35_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 35>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333334303461015);
    CUDA_ASSERT_EQ(double(b), 0.11111111112404614687);
    CUDA_ASSERT_EQ(double(c), 0.03703703705104999244);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_36_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 36>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333332848269492);
    CUDA_ASSERT_EQ(double(b), 0.11111111110949423164);
    CUDA_ASSERT_EQ(double(c), 0.03703703703649807721);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_37_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 37>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333575865254);
    CUDA_ASSERT_EQ(double(b), 0.11111111110949423164);
    CUDA_ASSERT_EQ(double(c), 0.03703703703649807721);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_38_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 38>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333212067373);
    CUDA_ASSERT_EQ(double(b), 0.11111111110949423164);
    CUDA_ASSERT_EQ(double(c), 0.03703703703649807721);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_39_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 39>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333393966313);
    CUDA_ASSERT_EQ(double(b), 0.11111111111131322104);
    CUDA_ASSERT_EQ(double(c), 0.03703703703649807721);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_40_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 40>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333303016843);
    CUDA_ASSERT_EQ(double(b), 0.11111111111131322104);
    CUDA_ASSERT_EQ(double(c), 0.03703703703740757192);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_41_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 41>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333348491578);
    CUDA_ASSERT_EQ(double(b), 0.11111111111131322104);
    CUDA_ASSERT_EQ(double(c), 0.03703703703695282456);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_42_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 42>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333325754211);
    CUDA_ASSERT_EQ(double(b), 0.11111111111108584737);
    CUDA_ASSERT_EQ(double(c), 0.03703703703695282456);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_43_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 43>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333337122895);
    CUDA_ASSERT_EQ(double(b), 0.11111111111108584737);
    CUDA_ASSERT_EQ(double(c), 0.03703703703706651140);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_44_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 44>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333331438553);
    CUDA_ASSERT_EQ(double(b), 0.11111111111108584737);
    CUDA_ASSERT_EQ(double(c), 0.03703703703700966798);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_45_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 45>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333334280724);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111426908);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703808969);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_46_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 46>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333332859638);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111426908);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703808969);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_47_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 47>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333570181);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111426908);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703808969);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_48_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 48>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333214910);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111071637);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703808969);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_49_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 49>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333392545);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111071637);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703631334);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_50_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 50>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333303727);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111071637);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703720151);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_51_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 51>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333348136);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111116045);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703720151);
}

CUDA_TEST(FloatxOperationsTest, MUL_2_52_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<2, 52>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333325932);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111116045);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703697947);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_1_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 1>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.37500000000000000000);
    CUDA_ASSERT_EQ(double(b), 0.12500000000000000000);
    CUDA_ASSERT_EQ(double(c), 0.00000000000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_2_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 2>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.31250000000000000000);
    CUDA_ASSERT_EQ(double(b), 0.12500000000000000000);
    CUDA_ASSERT_EQ(double(c), 0.06250000000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_3_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 3>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.34375000000000000000);
    CUDA_ASSERT_EQ(double(b), 0.12500000000000000000);
    CUDA_ASSERT_EQ(double(c), 0.03125000000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_4_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 4>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.32812500000000000000);
    CUDA_ASSERT_EQ(double(b), 0.10937500000000000000);
    CUDA_ASSERT_EQ(double(c), 0.03125000000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_5_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 5>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33593750000000000000);
    CUDA_ASSERT_EQ(double(b), 0.10937500000000000000);
    CUDA_ASSERT_EQ(double(c), 0.03906250000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_6_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 6>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33203125000000000000);
    CUDA_ASSERT_EQ(double(b), 0.10937500000000000000);
    CUDA_ASSERT_EQ(double(c), 0.03515625000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_7_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 7>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33398437500000000000);
    CUDA_ASSERT_EQ(double(b), 0.11132812500000000000);
    CUDA_ASSERT_EQ(double(c), 0.03710937500000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_8_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 8>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33300781250000000000);
    CUDA_ASSERT_EQ(double(b), 0.11132812500000000000);
    CUDA_ASSERT_EQ(double(c), 0.03710937500000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_9_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 9>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33349609375000000000);
    CUDA_ASSERT_EQ(double(b), 0.11132812500000000000);
    CUDA_ASSERT_EQ(double(c), 0.03710937500000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_10_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 10>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33325195312500000000);
    CUDA_ASSERT_EQ(double(b), 0.11108398437500000000);
    CUDA_ASSERT_EQ(double(c), 0.03710937500000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_11_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 11>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33337402343750000000);
    CUDA_ASSERT_EQ(double(b), 0.11108398437500000000);
    CUDA_ASSERT_EQ(double(c), 0.03698730468750000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_12_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 12>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33331298828125000000);
    CUDA_ASSERT_EQ(double(b), 0.11108398437500000000);
    CUDA_ASSERT_EQ(double(c), 0.03704833984375000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_13_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 13>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33334350585937500000);
    CUDA_ASSERT_EQ(double(b), 0.11111450195312500000);
    CUDA_ASSERT_EQ(double(c), 0.03704833984375000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_14_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 14>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33332824707031250000);
    CUDA_ASSERT_EQ(double(b), 0.11111450195312500000);
    CUDA_ASSERT_EQ(double(c), 0.03703308105468750000);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_15_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 15>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333587646484375000);
    CUDA_ASSERT_EQ(double(b), 0.11111450195312500000);
    CUDA_ASSERT_EQ(double(c), 0.03704071044921875000);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_16_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 16>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333206176757812500);
    CUDA_ASSERT_EQ(double(b), 0.11111068725585937500);
    CUDA_ASSERT_EQ(double(c), 0.03703689575195312500);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_17_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 17>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333396911621093750);
    CUDA_ASSERT_EQ(double(b), 0.11111068725585937500);
    CUDA_ASSERT_EQ(double(c), 0.03703689575195312500);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_18_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 18>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333301544189453125);
    CUDA_ASSERT_EQ(double(b), 0.11111068725585937500);
    CUDA_ASSERT_EQ(double(c), 0.03703689575195312500);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_19_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 19>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333349227905273438);
    CUDA_ASSERT_EQ(double(b), 0.11111116409301757812);
    CUDA_ASSERT_EQ(double(c), 0.03703689575195312500);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_20_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 20>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333325386047363281);
    CUDA_ASSERT_EQ(double(b), 0.11111116409301757812);
    CUDA_ASSERT_EQ(double(c), 0.03703713417053222656);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_21_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 21>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333337306976318359);
    CUDA_ASSERT_EQ(double(b), 0.11111116409301757812);
    CUDA_ASSERT_EQ(double(c), 0.03703701496124267578);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_22_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 22>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333331346511840820);
    CUDA_ASSERT_EQ(double(b), 0.11111110448837280273);
    CUDA_ASSERT_EQ(double(c), 0.03703701496124267578);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_23_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 23>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333334326744079590);
    CUDA_ASSERT_EQ(double(b), 0.11111110448837280273);
    CUDA_ASSERT_EQ(double(c), 0.03703704476356506348);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_24_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 24>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333332836627960205);
    CUDA_ASSERT_EQ(double(b), 0.11111110448837280273);
    CUDA_ASSERT_EQ(double(c), 0.03703702986240386963);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_25_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 25>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333581686019897);
    CUDA_ASSERT_EQ(double(b), 0.11111111193895339966);
    CUDA_ASSERT_EQ(double(c), 0.03703703731298446655);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_26_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 26>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333209156990051);
    CUDA_ASSERT_EQ(double(b), 0.11111111193895339966);
    CUDA_ASSERT_EQ(double(c), 0.03703703731298446655);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_27_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 27>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333395421504974);
    CUDA_ASSERT_EQ(double(b), 0.11111111193895339966);
    CUDA_ASSERT_EQ(double(c), 0.03703703731298446655);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_28_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 28>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333302289247513);
    CUDA_ASSERT_EQ(double(b), 0.11111111100763082504);
    CUDA_ASSERT_EQ(double(c), 0.03703703731298446655);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_29_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 29>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333348855376244);
    CUDA_ASSERT_EQ(double(b), 0.11111111100763082504);
    CUDA_ASSERT_EQ(double(c), 0.03703703684732317924);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_30_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 30>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333325572311878);
    CUDA_ASSERT_EQ(double(b), 0.11111111100763082504);
    CUDA_ASSERT_EQ(double(c), 0.03703703708015382290);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_31_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 31>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333337213844061);
    CUDA_ASSERT_EQ(double(b), 0.11111111112404614687);
    CUDA_ASSERT_EQ(double(c), 0.03703703708015382290);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_32_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 32>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333331393077970);
    CUDA_ASSERT_EQ(double(b), 0.11111111112404614687);
    CUDA_ASSERT_EQ(double(c), 0.03703703702194616199);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_33_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 33>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333334303461015);
    CUDA_ASSERT_EQ(double(b), 0.11111111112404614687);
    CUDA_ASSERT_EQ(double(c), 0.03703703705104999244);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_34_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 34>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333332848269492);
    CUDA_ASSERT_EQ(double(b), 0.11111111110949423164);
    CUDA_ASSERT_EQ(double(c), 0.03703703703649807721);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_35_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 35>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333575865254);
    CUDA_ASSERT_EQ(double(b), 0.11111111110949423164);
    CUDA_ASSERT_EQ(double(c), 0.03703703703649807721);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_36_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 36>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333212067373);
    CUDA_ASSERT_EQ(double(b), 0.11111111110949423164);
    CUDA_ASSERT_EQ(double(c), 0.03703703703649807721);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_37_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 37>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333393966313);
    CUDA_ASSERT_EQ(double(b), 0.11111111111131322104);
    CUDA_ASSERT_EQ(double(c), 0.03703703703649807721);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_38_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 38>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333303016843);
    CUDA_ASSERT_EQ(double(b), 0.11111111111131322104);
    CUDA_ASSERT_EQ(double(c), 0.03703703703740757192);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_39_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 39>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333348491578);
    CUDA_ASSERT_EQ(double(b), 0.11111111111131322104);
    CUDA_ASSERT_EQ(double(c), 0.03703703703695282456);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_40_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 40>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333325754211);
    CUDA_ASSERT_EQ(double(b), 0.11111111111108584737);
    CUDA_ASSERT_EQ(double(c), 0.03703703703695282456);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_41_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 41>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333337122895);
    CUDA_ASSERT_EQ(double(b), 0.11111111111108584737);
    CUDA_ASSERT_EQ(double(c), 0.03703703703706651140);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_42_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 42>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333331438553);
    CUDA_ASSERT_EQ(double(b), 0.11111111111108584737);
    CUDA_ASSERT_EQ(double(c), 0.03703703703700966798);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_43_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 43>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333334280724);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111426908);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703808969);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_44_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 44>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333332859638);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111426908);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703808969);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_45_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 45>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333570181);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111426908);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703808969);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_46_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 46>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333214910);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111071637);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703808969);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_47_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 47>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333392545);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111071637);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703631334);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_48_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 48>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333303727);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111071637);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703720151);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_49_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 49>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333348136);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111116045);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703720151);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_50_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 50>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333325932);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111116045);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703697947);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_51_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 51>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333325932);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111116045);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703709049);
}

CUDA_TEST(FloatxOperationsTest, MUL_3_52_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<3, 52>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333331483);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111110494);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703703498);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_1_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 1>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.37500000000000000000);
    CUDA_ASSERT_EQ(double(b), 0.12500000000000000000);
    CUDA_ASSERT_EQ(double(c), 0.04687500000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_2_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 2>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.31250000000000000000);
    CUDA_ASSERT_EQ(double(b), 0.10937500000000000000);
    CUDA_ASSERT_EQ(double(c), 0.03125000000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_3_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 3>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.34375000000000000000);
    CUDA_ASSERT_EQ(double(b), 0.10937500000000000000);
    CUDA_ASSERT_EQ(double(c), 0.03906250000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_4_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 4>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.32812500000000000000);
    CUDA_ASSERT_EQ(double(b), 0.10937500000000000000);
    CUDA_ASSERT_EQ(double(c), 0.03515625000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_5_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 5>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33593750000000000000);
    CUDA_ASSERT_EQ(double(b), 0.11132812500000000000);
    CUDA_ASSERT_EQ(double(c), 0.03710937500000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_6_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 6>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33203125000000000000);
    CUDA_ASSERT_EQ(double(b), 0.11132812500000000000);
    CUDA_ASSERT_EQ(double(c), 0.03710937500000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_7_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 7>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33398437500000000000);
    CUDA_ASSERT_EQ(double(b), 0.11132812500000000000);
    CUDA_ASSERT_EQ(double(c), 0.03710937500000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_8_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 8>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33300781250000000000);
    CUDA_ASSERT_EQ(double(b), 0.11108398437500000000);
    CUDA_ASSERT_EQ(double(c), 0.03698730468750000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_9_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 9>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33349609375000000000);
    CUDA_ASSERT_EQ(double(b), 0.11108398437500000000);
    CUDA_ASSERT_EQ(double(c), 0.03704833984375000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_10_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 10>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33325195312500000000);
    CUDA_ASSERT_EQ(double(b), 0.11108398437500000000);
    CUDA_ASSERT_EQ(double(c), 0.03701782226562500000);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_11_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 11>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33337402343750000000);
    CUDA_ASSERT_EQ(double(b), 0.11111450195312500000);
    CUDA_ASSERT_EQ(double(c), 0.03704833984375000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_12_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 12>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33331298828125000000);
    CUDA_ASSERT_EQ(double(b), 0.11111450195312500000);
    CUDA_ASSERT_EQ(double(c), 0.03703308105468750000);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_13_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 13>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33334350585937500000);
    CUDA_ASSERT_EQ(double(b), 0.11111450195312500000);
    CUDA_ASSERT_EQ(double(c), 0.03704071044921875000);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_14_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 14>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33332824707031250000);
    CUDA_ASSERT_EQ(double(b), 0.11111068725585937500);
    CUDA_ASSERT_EQ(double(c), 0.03703689575195312500);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_15_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 15>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333587646484375000);
    CUDA_ASSERT_EQ(double(b), 0.11111068725585937500);
    CUDA_ASSERT_EQ(double(c), 0.03703689575195312500);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_16_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 16>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333206176757812500);
    CUDA_ASSERT_EQ(double(b), 0.11111068725585937500);
    CUDA_ASSERT_EQ(double(c), 0.03703689575195312500);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_17_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 17>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333396911621093750);
    CUDA_ASSERT_EQ(double(b), 0.11111116409301757812);
    CUDA_ASSERT_EQ(double(c), 0.03703713417053222656);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_18_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 18>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333301544189453125);
    CUDA_ASSERT_EQ(double(b), 0.11111116409301757812);
    CUDA_ASSERT_EQ(double(c), 0.03703701496124267578);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_19_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 19>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333349227905273438);
    CUDA_ASSERT_EQ(double(b), 0.11111116409301757812);
    CUDA_ASSERT_EQ(double(c), 0.03703707456588745117);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_20_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 20>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333325386047363281);
    CUDA_ASSERT_EQ(double(b), 0.11111110448837280273);
    CUDA_ASSERT_EQ(double(c), 0.03703701496124267578);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_21_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 21>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333337306976318359);
    CUDA_ASSERT_EQ(double(b), 0.11111110448837280273);
    CUDA_ASSERT_EQ(double(c), 0.03703704476356506348);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_22_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 22>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333331346511840820);
    CUDA_ASSERT_EQ(double(b), 0.11111110448837280273);
    CUDA_ASSERT_EQ(double(c), 0.03703702986240386963);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_23_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 23>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333334326744079590);
    CUDA_ASSERT_EQ(double(b), 0.11111111193895339966);
    CUDA_ASSERT_EQ(double(c), 0.03703703731298446655);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_24_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 24>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333332836627960205);
    CUDA_ASSERT_EQ(double(b), 0.11111111193895339966);
    CUDA_ASSERT_EQ(double(c), 0.03703703731298446655);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_25_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 25>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333581686019897);
    CUDA_ASSERT_EQ(double(b), 0.11111111193895339966);
    CUDA_ASSERT_EQ(double(c), 0.03703703731298446655);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_26_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 26>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333209156990051);
    CUDA_ASSERT_EQ(double(b), 0.11111111100763082504);
    CUDA_ASSERT_EQ(double(c), 0.03703703684732317924);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_27_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 27>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333395421504974);
    CUDA_ASSERT_EQ(double(b), 0.11111111100763082504);
    CUDA_ASSERT_EQ(double(c), 0.03703703708015382290);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_28_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 28>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333302289247513);
    CUDA_ASSERT_EQ(double(b), 0.11111111100763082504);
    CUDA_ASSERT_EQ(double(c), 0.03703703696373850107);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_29_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 29>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333348855376244);
    CUDA_ASSERT_EQ(double(b), 0.11111111112404614687);
    CUDA_ASSERT_EQ(double(c), 0.03703703708015382290);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_30_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 30>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333325572311878);
    CUDA_ASSERT_EQ(double(b), 0.11111111112404614687);
    CUDA_ASSERT_EQ(double(c), 0.03703703702194616199);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_31_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 31>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333337213844061);
    CUDA_ASSERT_EQ(double(b), 0.11111111112404614687);
    CUDA_ASSERT_EQ(double(c), 0.03703703705104999244);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_32_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 32>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333331393077970);
    CUDA_ASSERT_EQ(double(b), 0.11111111110949423164);
    CUDA_ASSERT_EQ(double(c), 0.03703703703649807721);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_33_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 33>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333334303461015);
    CUDA_ASSERT_EQ(double(b), 0.11111111110949423164);
    CUDA_ASSERT_EQ(double(c), 0.03703703703649807721);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_34_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 34>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333332848269492);
    CUDA_ASSERT_EQ(double(b), 0.11111111110949423164);
    CUDA_ASSERT_EQ(double(c), 0.03703703703649807721);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_35_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 35>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333575865254);
    CUDA_ASSERT_EQ(double(b), 0.11111111111131322104);
    CUDA_ASSERT_EQ(double(c), 0.03703703703740757192);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_36_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 36>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333212067373);
    CUDA_ASSERT_EQ(double(b), 0.11111111111131322104);
    CUDA_ASSERT_EQ(double(c), 0.03703703703695282456);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_37_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 37>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333393966313);
    CUDA_ASSERT_EQ(double(b), 0.11111111111131322104);
    CUDA_ASSERT_EQ(double(c), 0.03703703703718019824);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_38_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 38>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333303016843);
    CUDA_ASSERT_EQ(double(b), 0.11111111111108584737);
    CUDA_ASSERT_EQ(double(c), 0.03703703703695282456);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_39_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 39>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333348491578);
    CUDA_ASSERT_EQ(double(b), 0.11111111111108584737);
    CUDA_ASSERT_EQ(double(c), 0.03703703703706651140);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_40_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 40>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333325754211);
    CUDA_ASSERT_EQ(double(b), 0.11111111111108584737);
    CUDA_ASSERT_EQ(double(c), 0.03703703703700966798);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_41_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 41>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333337122895);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111426908);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703808969);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_42_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 42>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333331438553);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111426908);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703808969);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_43_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 43>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333334280724);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111426908);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703808969);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_44_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 44>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333332859638);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111071637);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703631334);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_45_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 45>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333570181);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111071637);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703720151);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_46_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 46>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333214910);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111071637);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703675743);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_47_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 47>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333392545);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111116045);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703720151);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_48_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 48>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333303727);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111116045);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703697947);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_49_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 49>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333348136);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111116045);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703709049);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_50_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 50>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333325932);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111110494);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703703498);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_51_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 51>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333325932);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111110494);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703703498);
}

CUDA_TEST(FloatxOperationsTest, MUL_4_52_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<4, 52>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333331483);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111110494);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703703498);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_1_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 1>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.37500000000000000000);
    CUDA_ASSERT_EQ(double(b), 0.12500000000000000000);
    CUDA_ASSERT_EQ(double(c), 0.04687500000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_2_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 2>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.31250000000000000000);
    CUDA_ASSERT_EQ(double(b), 0.10937500000000000000);
    CUDA_ASSERT_EQ(double(c), 0.03125000000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_3_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 3>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.34375000000000000000);
    CUDA_ASSERT_EQ(double(b), 0.10937500000000000000);
    CUDA_ASSERT_EQ(double(c), 0.03906250000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_4_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 4>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.32812500000000000000);
    CUDA_ASSERT_EQ(double(b), 0.10937500000000000000);
    CUDA_ASSERT_EQ(double(c), 0.03515625000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_5_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 5>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33593750000000000000);
    CUDA_ASSERT_EQ(double(b), 0.11132812500000000000);
    CUDA_ASSERT_EQ(double(c), 0.03710937500000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_6_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 6>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33203125000000000000);
    CUDA_ASSERT_EQ(double(b), 0.11132812500000000000);
    CUDA_ASSERT_EQ(double(c), 0.03710937500000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_7_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 7>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33398437500000000000);
    CUDA_ASSERT_EQ(double(b), 0.11132812500000000000);
    CUDA_ASSERT_EQ(double(c), 0.03710937500000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_8_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 8>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33300781250000000000);
    CUDA_ASSERT_EQ(double(b), 0.11108398437500000000);
    CUDA_ASSERT_EQ(double(c), 0.03698730468750000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_9_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 9>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33349609375000000000);
    CUDA_ASSERT_EQ(double(b), 0.11108398437500000000);
    CUDA_ASSERT_EQ(double(c), 0.03704833984375000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_10_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 10>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33325195312500000000);
    CUDA_ASSERT_EQ(double(b), 0.11108398437500000000);
    CUDA_ASSERT_EQ(double(c), 0.03701782226562500000);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_11_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 11>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33337402343750000000);
    CUDA_ASSERT_EQ(double(b), 0.11111450195312500000);
    CUDA_ASSERT_EQ(double(c), 0.03704833984375000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_12_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 12>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33331298828125000000);
    CUDA_ASSERT_EQ(double(b), 0.11111450195312500000);
    CUDA_ASSERT_EQ(double(c), 0.03703308105468750000);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_13_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 13>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33334350585937500000);
    CUDA_ASSERT_EQ(double(b), 0.11111450195312500000);
    CUDA_ASSERT_EQ(double(c), 0.03704071044921875000);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_14_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 14>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33332824707031250000);
    CUDA_ASSERT_EQ(double(b), 0.11111068725585937500);
    CUDA_ASSERT_EQ(double(c), 0.03703689575195312500);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_15_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 15>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333587646484375000);
    CUDA_ASSERT_EQ(double(b), 0.11111068725585937500);
    CUDA_ASSERT_EQ(double(c), 0.03703689575195312500);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_16_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 16>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333206176757812500);
    CUDA_ASSERT_EQ(double(b), 0.11111068725585937500);
    CUDA_ASSERT_EQ(double(c), 0.03703689575195312500);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_17_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 17>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333396911621093750);
    CUDA_ASSERT_EQ(double(b), 0.11111116409301757812);
    CUDA_ASSERT_EQ(double(c), 0.03703713417053222656);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_18_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 18>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333301544189453125);
    CUDA_ASSERT_EQ(double(b), 0.11111116409301757812);
    CUDA_ASSERT_EQ(double(c), 0.03703701496124267578);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_19_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 19>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333349227905273438);
    CUDA_ASSERT_EQ(double(b), 0.11111116409301757812);
    CUDA_ASSERT_EQ(double(c), 0.03703707456588745117);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_20_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 20>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333325386047363281);
    CUDA_ASSERT_EQ(double(b), 0.11111110448837280273);
    CUDA_ASSERT_EQ(double(c), 0.03703701496124267578);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_21_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 21>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333337306976318359);
    CUDA_ASSERT_EQ(double(b), 0.11111110448837280273);
    CUDA_ASSERT_EQ(double(c), 0.03703704476356506348);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_22_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 22>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333331346511840820);
    CUDA_ASSERT_EQ(double(b), 0.11111110448837280273);
    CUDA_ASSERT_EQ(double(c), 0.03703702986240386963);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_23_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 23>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333334326744079590);
    CUDA_ASSERT_EQ(double(b), 0.11111111193895339966);
    CUDA_ASSERT_EQ(double(c), 0.03703703731298446655);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_24_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 24>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333332836627960205);
    CUDA_ASSERT_EQ(double(b), 0.11111111193895339966);
    CUDA_ASSERT_EQ(double(c), 0.03703703731298446655);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_25_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 25>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333581686019897);
    CUDA_ASSERT_EQ(double(b), 0.11111111193895339966);
    CUDA_ASSERT_EQ(double(c), 0.03703703731298446655);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_26_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 26>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333209156990051);
    CUDA_ASSERT_EQ(double(b), 0.11111111100763082504);
    CUDA_ASSERT_EQ(double(c), 0.03703703684732317924);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_27_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 27>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333395421504974);
    CUDA_ASSERT_EQ(double(b), 0.11111111100763082504);
    CUDA_ASSERT_EQ(double(c), 0.03703703708015382290);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_28_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 28>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333302289247513);
    CUDA_ASSERT_EQ(double(b), 0.11111111100763082504);
    CUDA_ASSERT_EQ(double(c), 0.03703703696373850107);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_29_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 29>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333348855376244);
    CUDA_ASSERT_EQ(double(b), 0.11111111112404614687);
    CUDA_ASSERT_EQ(double(c), 0.03703703708015382290);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_30_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 30>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333325572311878);
    CUDA_ASSERT_EQ(double(b), 0.11111111112404614687);
    CUDA_ASSERT_EQ(double(c), 0.03703703702194616199);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_31_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 31>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333337213844061);
    CUDA_ASSERT_EQ(double(b), 0.11111111112404614687);
    CUDA_ASSERT_EQ(double(c), 0.03703703705104999244);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_32_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 32>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333331393077970);
    CUDA_ASSERT_EQ(double(b), 0.11111111110949423164);
    CUDA_ASSERT_EQ(double(c), 0.03703703703649807721);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_33_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 33>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333334303461015);
    CUDA_ASSERT_EQ(double(b), 0.11111111110949423164);
    CUDA_ASSERT_EQ(double(c), 0.03703703703649807721);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_34_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 34>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333332848269492);
    CUDA_ASSERT_EQ(double(b), 0.11111111110949423164);
    CUDA_ASSERT_EQ(double(c), 0.03703703703649807721);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_35_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 35>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333575865254);
    CUDA_ASSERT_EQ(double(b), 0.11111111111131322104);
    CUDA_ASSERT_EQ(double(c), 0.03703703703740757192);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_36_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 36>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333212067373);
    CUDA_ASSERT_EQ(double(b), 0.11111111111131322104);
    CUDA_ASSERT_EQ(double(c), 0.03703703703695282456);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_37_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 37>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333393966313);
    CUDA_ASSERT_EQ(double(b), 0.11111111111131322104);
    CUDA_ASSERT_EQ(double(c), 0.03703703703718019824);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_38_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 38>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333303016843);
    CUDA_ASSERT_EQ(double(b), 0.11111111111108584737);
    CUDA_ASSERT_EQ(double(c), 0.03703703703695282456);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_39_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 39>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333348491578);
    CUDA_ASSERT_EQ(double(b), 0.11111111111108584737);
    CUDA_ASSERT_EQ(double(c), 0.03703703703706651140);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_40_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 40>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333325754211);
    CUDA_ASSERT_EQ(double(b), 0.11111111111108584737);
    CUDA_ASSERT_EQ(double(c), 0.03703703703700966798);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_41_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 41>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333337122895);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111426908);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703808969);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_42_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 42>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333331438553);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111426908);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703808969);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_43_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 43>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333334280724);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111426908);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703808969);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_44_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 44>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333332859638);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111071637);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703631334);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_45_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 45>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333570181);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111071637);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703720151);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_46_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 46>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333214910);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111071637);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703675743);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_47_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 47>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333392545);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111116045);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703720151);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_48_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 48>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333303727);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111116045);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703697947);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_49_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 49>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333348136);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111116045);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703709049);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_50_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 50>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333325932);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111110494);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703703498);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_51_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 51>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333325932);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111110494);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703703498);
}

CUDA_TEST(FloatxOperationsTest, MUL_5_52_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<5, 52>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333331483);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111110494);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703703498);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_1_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 1>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.37500000000000000000);
    CUDA_ASSERT_EQ(double(b), 0.12500000000000000000);
    CUDA_ASSERT_EQ(double(c), 0.04687500000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_2_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 2>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.31250000000000000000);
    CUDA_ASSERT_EQ(double(b), 0.10937500000000000000);
    CUDA_ASSERT_EQ(double(c), 0.03125000000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_3_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 3>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.34375000000000000000);
    CUDA_ASSERT_EQ(double(b), 0.10937500000000000000);
    CUDA_ASSERT_EQ(double(c), 0.03906250000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_4_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 4>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.32812500000000000000);
    CUDA_ASSERT_EQ(double(b), 0.10937500000000000000);
    CUDA_ASSERT_EQ(double(c), 0.03515625000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_5_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 5>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33593750000000000000);
    CUDA_ASSERT_EQ(double(b), 0.11132812500000000000);
    CUDA_ASSERT_EQ(double(c), 0.03710937500000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_6_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 6>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33203125000000000000);
    CUDA_ASSERT_EQ(double(b), 0.11132812500000000000);
    CUDA_ASSERT_EQ(double(c), 0.03710937500000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_7_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 7>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33398437500000000000);
    CUDA_ASSERT_EQ(double(b), 0.11132812500000000000);
    CUDA_ASSERT_EQ(double(c), 0.03710937500000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_8_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 8>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33300781250000000000);
    CUDA_ASSERT_EQ(double(b), 0.11108398437500000000);
    CUDA_ASSERT_EQ(double(c), 0.03698730468750000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_9_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 9>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33349609375000000000);
    CUDA_ASSERT_EQ(double(b), 0.11108398437500000000);
    CUDA_ASSERT_EQ(double(c), 0.03704833984375000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_10_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 10>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33325195312500000000);
    CUDA_ASSERT_EQ(double(b), 0.11108398437500000000);
    CUDA_ASSERT_EQ(double(c), 0.03701782226562500000);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_11_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 11>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33337402343750000000);
    CUDA_ASSERT_EQ(double(b), 0.11111450195312500000);
    CUDA_ASSERT_EQ(double(c), 0.03704833984375000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_12_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 12>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33331298828125000000);
    CUDA_ASSERT_EQ(double(b), 0.11111450195312500000);
    CUDA_ASSERT_EQ(double(c), 0.03703308105468750000);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_13_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 13>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33334350585937500000);
    CUDA_ASSERT_EQ(double(b), 0.11111450195312500000);
    CUDA_ASSERT_EQ(double(c), 0.03704071044921875000);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_14_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 14>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33332824707031250000);
    CUDA_ASSERT_EQ(double(b), 0.11111068725585937500);
    CUDA_ASSERT_EQ(double(c), 0.03703689575195312500);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_15_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 15>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333587646484375000);
    CUDA_ASSERT_EQ(double(b), 0.11111068725585937500);
    CUDA_ASSERT_EQ(double(c), 0.03703689575195312500);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_16_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 16>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333206176757812500);
    CUDA_ASSERT_EQ(double(b), 0.11111068725585937500);
    CUDA_ASSERT_EQ(double(c), 0.03703689575195312500);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_17_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 17>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333396911621093750);
    CUDA_ASSERT_EQ(double(b), 0.11111116409301757812);
    CUDA_ASSERT_EQ(double(c), 0.03703713417053222656);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_18_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 18>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333301544189453125);
    CUDA_ASSERT_EQ(double(b), 0.11111116409301757812);
    CUDA_ASSERT_EQ(double(c), 0.03703701496124267578);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_19_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 19>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333349227905273438);
    CUDA_ASSERT_EQ(double(b), 0.11111116409301757812);
    CUDA_ASSERT_EQ(double(c), 0.03703707456588745117);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_20_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 20>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333325386047363281);
    CUDA_ASSERT_EQ(double(b), 0.11111110448837280273);
    CUDA_ASSERT_EQ(double(c), 0.03703701496124267578);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_21_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 21>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333337306976318359);
    CUDA_ASSERT_EQ(double(b), 0.11111110448837280273);
    CUDA_ASSERT_EQ(double(c), 0.03703704476356506348);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_22_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 22>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333331346511840820);
    CUDA_ASSERT_EQ(double(b), 0.11111110448837280273);
    CUDA_ASSERT_EQ(double(c), 0.03703702986240386963);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_23_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 23>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333334326744079590);
    CUDA_ASSERT_EQ(double(b), 0.11111111193895339966);
    CUDA_ASSERT_EQ(double(c), 0.03703703731298446655);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_24_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 24>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333332836627960205);
    CUDA_ASSERT_EQ(double(b), 0.11111111193895339966);
    CUDA_ASSERT_EQ(double(c), 0.03703703731298446655);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_25_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 25>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333581686019897);
    CUDA_ASSERT_EQ(double(b), 0.11111111193895339966);
    CUDA_ASSERT_EQ(double(c), 0.03703703731298446655);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_26_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 26>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333209156990051);
    CUDA_ASSERT_EQ(double(b), 0.11111111100763082504);
    CUDA_ASSERT_EQ(double(c), 0.03703703684732317924);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_27_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 27>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333395421504974);
    CUDA_ASSERT_EQ(double(b), 0.11111111100763082504);
    CUDA_ASSERT_EQ(double(c), 0.03703703708015382290);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_28_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 28>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333302289247513);
    CUDA_ASSERT_EQ(double(b), 0.11111111100763082504);
    CUDA_ASSERT_EQ(double(c), 0.03703703696373850107);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_29_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 29>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333348855376244);
    CUDA_ASSERT_EQ(double(b), 0.11111111112404614687);
    CUDA_ASSERT_EQ(double(c), 0.03703703708015382290);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_30_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 30>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333325572311878);
    CUDA_ASSERT_EQ(double(b), 0.11111111112404614687);
    CUDA_ASSERT_EQ(double(c), 0.03703703702194616199);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_31_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 31>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333337213844061);
    CUDA_ASSERT_EQ(double(b), 0.11111111112404614687);
    CUDA_ASSERT_EQ(double(c), 0.03703703705104999244);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_32_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 32>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333331393077970);
    CUDA_ASSERT_EQ(double(b), 0.11111111110949423164);
    CUDA_ASSERT_EQ(double(c), 0.03703703703649807721);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_33_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 33>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333334303461015);
    CUDA_ASSERT_EQ(double(b), 0.11111111110949423164);
    CUDA_ASSERT_EQ(double(c), 0.03703703703649807721);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_34_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 34>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333332848269492);
    CUDA_ASSERT_EQ(double(b), 0.11111111110949423164);
    CUDA_ASSERT_EQ(double(c), 0.03703703703649807721);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_35_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 35>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333575865254);
    CUDA_ASSERT_EQ(double(b), 0.11111111111131322104);
    CUDA_ASSERT_EQ(double(c), 0.03703703703740757192);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_36_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 36>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333212067373);
    CUDA_ASSERT_EQ(double(b), 0.11111111111131322104);
    CUDA_ASSERT_EQ(double(c), 0.03703703703695282456);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_37_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 37>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333393966313);
    CUDA_ASSERT_EQ(double(b), 0.11111111111131322104);
    CUDA_ASSERT_EQ(double(c), 0.03703703703718019824);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_38_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 38>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333303016843);
    CUDA_ASSERT_EQ(double(b), 0.11111111111108584737);
    CUDA_ASSERT_EQ(double(c), 0.03703703703695282456);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_39_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 39>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333348491578);
    CUDA_ASSERT_EQ(double(b), 0.11111111111108584737);
    CUDA_ASSERT_EQ(double(c), 0.03703703703706651140);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_40_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 40>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333325754211);
    CUDA_ASSERT_EQ(double(b), 0.11111111111108584737);
    CUDA_ASSERT_EQ(double(c), 0.03703703703700966798);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_41_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 41>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333337122895);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111426908);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703808969);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_42_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 42>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333331438553);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111426908);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703808969);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_43_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 43>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333334280724);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111426908);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703808969);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_44_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 44>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333332859638);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111071637);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703631334);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_45_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 45>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333570181);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111071637);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703720151);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_46_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 46>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333214910);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111071637);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703675743);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_47_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 47>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333392545);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111116045);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703720151);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_48_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 48>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333303727);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111116045);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703697947);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_49_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 49>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333348136);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111116045);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703709049);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_50_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 50>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333325932);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111110494);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703703498);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_51_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 51>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333325932);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111110494);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703703498);
}

CUDA_TEST(FloatxOperationsTest, MUL_6_52_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<6, 52>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333331483);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111110494);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703703498);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_1_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 1>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.37500000000000000000);
    CUDA_ASSERT_EQ(double(b), 0.12500000000000000000);
    CUDA_ASSERT_EQ(double(c), 0.04687500000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_2_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 2>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.31250000000000000000);
    CUDA_ASSERT_EQ(double(b), 0.10937500000000000000);
    CUDA_ASSERT_EQ(double(c), 0.03125000000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_3_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 3>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.34375000000000000000);
    CUDA_ASSERT_EQ(double(b), 0.10937500000000000000);
    CUDA_ASSERT_EQ(double(c), 0.03906250000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_4_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 4>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.32812500000000000000);
    CUDA_ASSERT_EQ(double(b), 0.10937500000000000000);
    CUDA_ASSERT_EQ(double(c), 0.03515625000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_5_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 5>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33593750000000000000);
    CUDA_ASSERT_EQ(double(b), 0.11132812500000000000);
    CUDA_ASSERT_EQ(double(c), 0.03710937500000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_6_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 6>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33203125000000000000);
    CUDA_ASSERT_EQ(double(b), 0.11132812500000000000);
    CUDA_ASSERT_EQ(double(c), 0.03710937500000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_7_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 7>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33398437500000000000);
    CUDA_ASSERT_EQ(double(b), 0.11132812500000000000);
    CUDA_ASSERT_EQ(double(c), 0.03710937500000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_8_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 8>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33300781250000000000);
    CUDA_ASSERT_EQ(double(b), 0.11108398437500000000);
    CUDA_ASSERT_EQ(double(c), 0.03698730468750000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_9_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 9>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33349609375000000000);
    CUDA_ASSERT_EQ(double(b), 0.11108398437500000000);
    CUDA_ASSERT_EQ(double(c), 0.03704833984375000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_10_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 10>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33325195312500000000);
    CUDA_ASSERT_EQ(double(b), 0.11108398437500000000);
    CUDA_ASSERT_EQ(double(c), 0.03701782226562500000);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_11_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 11>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33337402343750000000);
    CUDA_ASSERT_EQ(double(b), 0.11111450195312500000);
    CUDA_ASSERT_EQ(double(c), 0.03704833984375000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_12_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 12>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33331298828125000000);
    CUDA_ASSERT_EQ(double(b), 0.11111450195312500000);
    CUDA_ASSERT_EQ(double(c), 0.03703308105468750000);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_13_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 13>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33334350585937500000);
    CUDA_ASSERT_EQ(double(b), 0.11111450195312500000);
    CUDA_ASSERT_EQ(double(c), 0.03704071044921875000);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_14_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 14>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33332824707031250000);
    CUDA_ASSERT_EQ(double(b), 0.11111068725585937500);
    CUDA_ASSERT_EQ(double(c), 0.03703689575195312500);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_15_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 15>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333587646484375000);
    CUDA_ASSERT_EQ(double(b), 0.11111068725585937500);
    CUDA_ASSERT_EQ(double(c), 0.03703689575195312500);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_16_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 16>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333206176757812500);
    CUDA_ASSERT_EQ(double(b), 0.11111068725585937500);
    CUDA_ASSERT_EQ(double(c), 0.03703689575195312500);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_17_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 17>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333396911621093750);
    CUDA_ASSERT_EQ(double(b), 0.11111116409301757812);
    CUDA_ASSERT_EQ(double(c), 0.03703713417053222656);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_18_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 18>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333301544189453125);
    CUDA_ASSERT_EQ(double(b), 0.11111116409301757812);
    CUDA_ASSERT_EQ(double(c), 0.03703701496124267578);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_19_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 19>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333349227905273438);
    CUDA_ASSERT_EQ(double(b), 0.11111116409301757812);
    CUDA_ASSERT_EQ(double(c), 0.03703707456588745117);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_20_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 20>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333325386047363281);
    CUDA_ASSERT_EQ(double(b), 0.11111110448837280273);
    CUDA_ASSERT_EQ(double(c), 0.03703701496124267578);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_21_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 21>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333337306976318359);
    CUDA_ASSERT_EQ(double(b), 0.11111110448837280273);
    CUDA_ASSERT_EQ(double(c), 0.03703704476356506348);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_22_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 22>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333331346511840820);
    CUDA_ASSERT_EQ(double(b), 0.11111110448837280273);
    CUDA_ASSERT_EQ(double(c), 0.03703702986240386963);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_23_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 23>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333334326744079590);
    CUDA_ASSERT_EQ(double(b), 0.11111111193895339966);
    CUDA_ASSERT_EQ(double(c), 0.03703703731298446655);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_24_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 24>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333332836627960205);
    CUDA_ASSERT_EQ(double(b), 0.11111111193895339966);
    CUDA_ASSERT_EQ(double(c), 0.03703703731298446655);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_25_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 25>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333581686019897);
    CUDA_ASSERT_EQ(double(b), 0.11111111193895339966);
    CUDA_ASSERT_EQ(double(c), 0.03703703731298446655);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_26_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 26>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333209156990051);
    CUDA_ASSERT_EQ(double(b), 0.11111111100763082504);
    CUDA_ASSERT_EQ(double(c), 0.03703703684732317924);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_27_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 27>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333395421504974);
    CUDA_ASSERT_EQ(double(b), 0.11111111100763082504);
    CUDA_ASSERT_EQ(double(c), 0.03703703708015382290);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_28_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 28>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333302289247513);
    CUDA_ASSERT_EQ(double(b), 0.11111111100763082504);
    CUDA_ASSERT_EQ(double(c), 0.03703703696373850107);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_29_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 29>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333348855376244);
    CUDA_ASSERT_EQ(double(b), 0.11111111112404614687);
    CUDA_ASSERT_EQ(double(c), 0.03703703708015382290);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_30_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 30>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333325572311878);
    CUDA_ASSERT_EQ(double(b), 0.11111111112404614687);
    CUDA_ASSERT_EQ(double(c), 0.03703703702194616199);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_31_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 31>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333337213844061);
    CUDA_ASSERT_EQ(double(b), 0.11111111112404614687);
    CUDA_ASSERT_EQ(double(c), 0.03703703705104999244);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_32_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 32>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333331393077970);
    CUDA_ASSERT_EQ(double(b), 0.11111111110949423164);
    CUDA_ASSERT_EQ(double(c), 0.03703703703649807721);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_33_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 33>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333334303461015);
    CUDA_ASSERT_EQ(double(b), 0.11111111110949423164);
    CUDA_ASSERT_EQ(double(c), 0.03703703703649807721);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_34_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 34>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333332848269492);
    CUDA_ASSERT_EQ(double(b), 0.11111111110949423164);
    CUDA_ASSERT_EQ(double(c), 0.03703703703649807721);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_35_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 35>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333575865254);
    CUDA_ASSERT_EQ(double(b), 0.11111111111131322104);
    CUDA_ASSERT_EQ(double(c), 0.03703703703740757192);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_36_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 36>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333212067373);
    CUDA_ASSERT_EQ(double(b), 0.11111111111131322104);
    CUDA_ASSERT_EQ(double(c), 0.03703703703695282456);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_37_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 37>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333393966313);
    CUDA_ASSERT_EQ(double(b), 0.11111111111131322104);
    CUDA_ASSERT_EQ(double(c), 0.03703703703718019824);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_38_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 38>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333303016843);
    CUDA_ASSERT_EQ(double(b), 0.11111111111108584737);
    CUDA_ASSERT_EQ(double(c), 0.03703703703695282456);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_39_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 39>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333348491578);
    CUDA_ASSERT_EQ(double(b), 0.11111111111108584737);
    CUDA_ASSERT_EQ(double(c), 0.03703703703706651140);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_40_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 40>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333325754211);
    CUDA_ASSERT_EQ(double(b), 0.11111111111108584737);
    CUDA_ASSERT_EQ(double(c), 0.03703703703700966798);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_41_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 41>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333337122895);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111426908);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703808969);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_42_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 42>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333331438553);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111426908);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703808969);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_43_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 43>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333334280724);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111426908);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703808969);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_44_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 44>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333332859638);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111071637);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703631334);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_45_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 45>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333570181);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111071637);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703720151);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_46_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 46>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333214910);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111071637);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703675743);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_47_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 47>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333392545);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111116045);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703720151);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_48_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 48>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333303727);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111116045);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703697947);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_49_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 49>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333348136);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111116045);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703709049);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_50_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 50>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333325932);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111110494);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703703498);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_51_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 51>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333325932);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111110494);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703703498);
}

CUDA_TEST(FloatxOperationsTest, MUL_7_52_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<7, 52>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333331483);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111110494);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703703498);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_1_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 1>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.37500000000000000000);
    CUDA_ASSERT_EQ(double(b), 0.12500000000000000000);
    CUDA_ASSERT_EQ(double(c), 0.04687500000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_2_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 2>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.31250000000000000000);
    CUDA_ASSERT_EQ(double(b), 0.10937500000000000000);
    CUDA_ASSERT_EQ(double(c), 0.03125000000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_3_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 3>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.34375000000000000000);
    CUDA_ASSERT_EQ(double(b), 0.10937500000000000000);
    CUDA_ASSERT_EQ(double(c), 0.03906250000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_4_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 4>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.32812500000000000000);
    CUDA_ASSERT_EQ(double(b), 0.10937500000000000000);
    CUDA_ASSERT_EQ(double(c), 0.03515625000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_5_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 5>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33593750000000000000);
    CUDA_ASSERT_EQ(double(b), 0.11132812500000000000);
    CUDA_ASSERT_EQ(double(c), 0.03710937500000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_6_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 6>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33203125000000000000);
    CUDA_ASSERT_EQ(double(b), 0.11132812500000000000);
    CUDA_ASSERT_EQ(double(c), 0.03710937500000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_7_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 7>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33398437500000000000);
    CUDA_ASSERT_EQ(double(b), 0.11132812500000000000);
    CUDA_ASSERT_EQ(double(c), 0.03710937500000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_8_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 8>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33300781250000000000);
    CUDA_ASSERT_EQ(double(b), 0.11108398437500000000);
    CUDA_ASSERT_EQ(double(c), 0.03698730468750000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_9_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 9>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33349609375000000000);
    CUDA_ASSERT_EQ(double(b), 0.11108398437500000000);
    CUDA_ASSERT_EQ(double(c), 0.03704833984375000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_10_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 10>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33325195312500000000);
    CUDA_ASSERT_EQ(double(b), 0.11108398437500000000);
    CUDA_ASSERT_EQ(double(c), 0.03701782226562500000);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_11_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 11>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33337402343750000000);
    CUDA_ASSERT_EQ(double(b), 0.11111450195312500000);
    CUDA_ASSERT_EQ(double(c), 0.03704833984375000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_12_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 12>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33331298828125000000);
    CUDA_ASSERT_EQ(double(b), 0.11111450195312500000);
    CUDA_ASSERT_EQ(double(c), 0.03703308105468750000);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_13_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 13>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33334350585937500000);
    CUDA_ASSERT_EQ(double(b), 0.11111450195312500000);
    CUDA_ASSERT_EQ(double(c), 0.03704071044921875000);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_14_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 14>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33332824707031250000);
    CUDA_ASSERT_EQ(double(b), 0.11111068725585937500);
    CUDA_ASSERT_EQ(double(c), 0.03703689575195312500);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_15_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 15>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333587646484375000);
    CUDA_ASSERT_EQ(double(b), 0.11111068725585937500);
    CUDA_ASSERT_EQ(double(c), 0.03703689575195312500);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_16_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 16>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333206176757812500);
    CUDA_ASSERT_EQ(double(b), 0.11111068725585937500);
    CUDA_ASSERT_EQ(double(c), 0.03703689575195312500);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_17_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 17>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333396911621093750);
    CUDA_ASSERT_EQ(double(b), 0.11111116409301757812);
    CUDA_ASSERT_EQ(double(c), 0.03703713417053222656);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_18_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 18>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333301544189453125);
    CUDA_ASSERT_EQ(double(b), 0.11111116409301757812);
    CUDA_ASSERT_EQ(double(c), 0.03703701496124267578);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_19_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 19>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333349227905273438);
    CUDA_ASSERT_EQ(double(b), 0.11111116409301757812);
    CUDA_ASSERT_EQ(double(c), 0.03703707456588745117);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_20_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 20>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333325386047363281);
    CUDA_ASSERT_EQ(double(b), 0.11111110448837280273);
    CUDA_ASSERT_EQ(double(c), 0.03703701496124267578);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_21_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 21>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333337306976318359);
    CUDA_ASSERT_EQ(double(b), 0.11111110448837280273);
    CUDA_ASSERT_EQ(double(c), 0.03703704476356506348);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_22_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 22>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333331346511840820);
    CUDA_ASSERT_EQ(double(b), 0.11111110448837280273);
    CUDA_ASSERT_EQ(double(c), 0.03703702986240386963);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_23_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 23>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333334326744079590);
    CUDA_ASSERT_EQ(double(b), 0.11111111193895339966);
    CUDA_ASSERT_EQ(double(c), 0.03703703731298446655);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_24_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 24>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333332836627960205);
    CUDA_ASSERT_EQ(double(b), 0.11111111193895339966);
    CUDA_ASSERT_EQ(double(c), 0.03703703731298446655);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_25_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 25>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333581686019897);
    CUDA_ASSERT_EQ(double(b), 0.11111111193895339966);
    CUDA_ASSERT_EQ(double(c), 0.03703703731298446655);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_26_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 26>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333209156990051);
    CUDA_ASSERT_EQ(double(b), 0.11111111100763082504);
    CUDA_ASSERT_EQ(double(c), 0.03703703684732317924);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_27_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 27>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333395421504974);
    CUDA_ASSERT_EQ(double(b), 0.11111111100763082504);
    CUDA_ASSERT_EQ(double(c), 0.03703703708015382290);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_28_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 28>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333302289247513);
    CUDA_ASSERT_EQ(double(b), 0.11111111100763082504);
    CUDA_ASSERT_EQ(double(c), 0.03703703696373850107);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_29_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 29>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333348855376244);
    CUDA_ASSERT_EQ(double(b), 0.11111111112404614687);
    CUDA_ASSERT_EQ(double(c), 0.03703703708015382290);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_30_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 30>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333325572311878);
    CUDA_ASSERT_EQ(double(b), 0.11111111112404614687);
    CUDA_ASSERT_EQ(double(c), 0.03703703702194616199);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_31_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 31>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333337213844061);
    CUDA_ASSERT_EQ(double(b), 0.11111111112404614687);
    CUDA_ASSERT_EQ(double(c), 0.03703703705104999244);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_32_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 32>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333331393077970);
    CUDA_ASSERT_EQ(double(b), 0.11111111110949423164);
    CUDA_ASSERT_EQ(double(c), 0.03703703703649807721);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_33_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 33>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333334303461015);
    CUDA_ASSERT_EQ(double(b), 0.11111111110949423164);
    CUDA_ASSERT_EQ(double(c), 0.03703703703649807721);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_34_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 34>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333332848269492);
    CUDA_ASSERT_EQ(double(b), 0.11111111110949423164);
    CUDA_ASSERT_EQ(double(c), 0.03703703703649807721);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_35_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 35>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333575865254);
    CUDA_ASSERT_EQ(double(b), 0.11111111111131322104);
    CUDA_ASSERT_EQ(double(c), 0.03703703703740757192);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_36_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 36>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333212067373);
    CUDA_ASSERT_EQ(double(b), 0.11111111111131322104);
    CUDA_ASSERT_EQ(double(c), 0.03703703703695282456);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_37_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 37>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333393966313);
    CUDA_ASSERT_EQ(double(b), 0.11111111111131322104);
    CUDA_ASSERT_EQ(double(c), 0.03703703703718019824);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_38_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 38>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333303016843);
    CUDA_ASSERT_EQ(double(b), 0.11111111111108584737);
    CUDA_ASSERT_EQ(double(c), 0.03703703703695282456);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_39_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 39>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333348491578);
    CUDA_ASSERT_EQ(double(b), 0.11111111111108584737);
    CUDA_ASSERT_EQ(double(c), 0.03703703703706651140);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_40_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 40>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333325754211);
    CUDA_ASSERT_EQ(double(b), 0.11111111111108584737);
    CUDA_ASSERT_EQ(double(c), 0.03703703703700966798);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_41_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 41>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333337122895);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111426908);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703808969);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_42_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 42>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333331438553);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111426908);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703808969);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_43_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 43>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333334280724);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111426908);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703808969);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_44_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 44>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333332859638);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111071637);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703631334);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_45_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 45>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333570181);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111071637);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703720151);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_46_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 46>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333214910);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111071637);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703675743);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_47_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 47>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333392545);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111116045);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703720151);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_48_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 48>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333303727);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111116045);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703697947);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_49_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 49>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333348136);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111116045);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703709049);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_50_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 50>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333325932);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111110494);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703703498);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_51_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 51>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333325932);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111110494);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703703498);
}

CUDA_TEST(FloatxOperationsTest, MUL_8_52_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<8, 52>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333331483);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111110494);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703703498);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_1_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 1>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.37500000000000000000);
    CUDA_ASSERT_EQ(double(b), 0.12500000000000000000);
    CUDA_ASSERT_EQ(double(c), 0.04687500000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_2_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 2>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.31250000000000000000);
    CUDA_ASSERT_EQ(double(b), 0.10937500000000000000);
    CUDA_ASSERT_EQ(double(c), 0.03125000000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_3_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 3>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.34375000000000000000);
    CUDA_ASSERT_EQ(double(b), 0.10937500000000000000);
    CUDA_ASSERT_EQ(double(c), 0.03906250000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_4_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 4>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.32812500000000000000);
    CUDA_ASSERT_EQ(double(b), 0.10937500000000000000);
    CUDA_ASSERT_EQ(double(c), 0.03515625000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_5_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 5>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33593750000000000000);
    CUDA_ASSERT_EQ(double(b), 0.11132812500000000000);
    CUDA_ASSERT_EQ(double(c), 0.03710937500000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_6_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 6>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33203125000000000000);
    CUDA_ASSERT_EQ(double(b), 0.11132812500000000000);
    CUDA_ASSERT_EQ(double(c), 0.03710937500000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_7_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 7>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33398437500000000000);
    CUDA_ASSERT_EQ(double(b), 0.11132812500000000000);
    CUDA_ASSERT_EQ(double(c), 0.03710937500000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_8_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 8>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33300781250000000000);
    CUDA_ASSERT_EQ(double(b), 0.11108398437500000000);
    CUDA_ASSERT_EQ(double(c), 0.03698730468750000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_9_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 9>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33349609375000000000);
    CUDA_ASSERT_EQ(double(b), 0.11108398437500000000);
    CUDA_ASSERT_EQ(double(c), 0.03704833984375000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_10_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 10>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33325195312500000000);
    CUDA_ASSERT_EQ(double(b), 0.11108398437500000000);
    CUDA_ASSERT_EQ(double(c), 0.03701782226562500000);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_11_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 11>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33337402343750000000);
    CUDA_ASSERT_EQ(double(b), 0.11111450195312500000);
    CUDA_ASSERT_EQ(double(c), 0.03704833984375000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_12_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 12>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33331298828125000000);
    CUDA_ASSERT_EQ(double(b), 0.11111450195312500000);
    CUDA_ASSERT_EQ(double(c), 0.03703308105468750000);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_13_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 13>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33334350585937500000);
    CUDA_ASSERT_EQ(double(b), 0.11111450195312500000);
    CUDA_ASSERT_EQ(double(c), 0.03704071044921875000);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_14_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 14>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33332824707031250000);
    CUDA_ASSERT_EQ(double(b), 0.11111068725585937500);
    CUDA_ASSERT_EQ(double(c), 0.03703689575195312500);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_15_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 15>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333587646484375000);
    CUDA_ASSERT_EQ(double(b), 0.11111068725585937500);
    CUDA_ASSERT_EQ(double(c), 0.03703689575195312500);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_16_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 16>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333206176757812500);
    CUDA_ASSERT_EQ(double(b), 0.11111068725585937500);
    CUDA_ASSERT_EQ(double(c), 0.03703689575195312500);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_17_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 17>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333396911621093750);
    CUDA_ASSERT_EQ(double(b), 0.11111116409301757812);
    CUDA_ASSERT_EQ(double(c), 0.03703713417053222656);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_18_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 18>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333301544189453125);
    CUDA_ASSERT_EQ(double(b), 0.11111116409301757812);
    CUDA_ASSERT_EQ(double(c), 0.03703701496124267578);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_19_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 19>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333349227905273438);
    CUDA_ASSERT_EQ(double(b), 0.11111116409301757812);
    CUDA_ASSERT_EQ(double(c), 0.03703707456588745117);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_20_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 20>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333325386047363281);
    CUDA_ASSERT_EQ(double(b), 0.11111110448837280273);
    CUDA_ASSERT_EQ(double(c), 0.03703701496124267578);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_21_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 21>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333337306976318359);
    CUDA_ASSERT_EQ(double(b), 0.11111110448837280273);
    CUDA_ASSERT_EQ(double(c), 0.03703704476356506348);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_22_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 22>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333331346511840820);
    CUDA_ASSERT_EQ(double(b), 0.11111110448837280273);
    CUDA_ASSERT_EQ(double(c), 0.03703702986240386963);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_23_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 23>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333334326744079590);
    CUDA_ASSERT_EQ(double(b), 0.11111111193895339966);
    CUDA_ASSERT_EQ(double(c), 0.03703703731298446655);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_24_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 24>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333332836627960205);
    CUDA_ASSERT_EQ(double(b), 0.11111111193895339966);
    CUDA_ASSERT_EQ(double(c), 0.03703703731298446655);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_25_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 25>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333581686019897);
    CUDA_ASSERT_EQ(double(b), 0.11111111193895339966);
    CUDA_ASSERT_EQ(double(c), 0.03703703731298446655);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_26_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 26>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333209156990051);
    CUDA_ASSERT_EQ(double(b), 0.11111111100763082504);
    CUDA_ASSERT_EQ(double(c), 0.03703703684732317924);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_27_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 27>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333395421504974);
    CUDA_ASSERT_EQ(double(b), 0.11111111100763082504);
    CUDA_ASSERT_EQ(double(c), 0.03703703708015382290);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_28_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 28>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333302289247513);
    CUDA_ASSERT_EQ(double(b), 0.11111111100763082504);
    CUDA_ASSERT_EQ(double(c), 0.03703703696373850107);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_29_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 29>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333348855376244);
    CUDA_ASSERT_EQ(double(b), 0.11111111112404614687);
    CUDA_ASSERT_EQ(double(c), 0.03703703708015382290);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_30_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 30>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333325572311878);
    CUDA_ASSERT_EQ(double(b), 0.11111111112404614687);
    CUDA_ASSERT_EQ(double(c), 0.03703703702194616199);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_31_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 31>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333337213844061);
    CUDA_ASSERT_EQ(double(b), 0.11111111112404614687);
    CUDA_ASSERT_EQ(double(c), 0.03703703705104999244);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_32_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 32>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333331393077970);
    CUDA_ASSERT_EQ(double(b), 0.11111111110949423164);
    CUDA_ASSERT_EQ(double(c), 0.03703703703649807721);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_33_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 33>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333334303461015);
    CUDA_ASSERT_EQ(double(b), 0.11111111110949423164);
    CUDA_ASSERT_EQ(double(c), 0.03703703703649807721);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_34_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 34>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333332848269492);
    CUDA_ASSERT_EQ(double(b), 0.11111111110949423164);
    CUDA_ASSERT_EQ(double(c), 0.03703703703649807721);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_35_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 35>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333575865254);
    CUDA_ASSERT_EQ(double(b), 0.11111111111131322104);
    CUDA_ASSERT_EQ(double(c), 0.03703703703740757192);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_36_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 36>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333212067373);
    CUDA_ASSERT_EQ(double(b), 0.11111111111131322104);
    CUDA_ASSERT_EQ(double(c), 0.03703703703695282456);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_37_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 37>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333393966313);
    CUDA_ASSERT_EQ(double(b), 0.11111111111131322104);
    CUDA_ASSERT_EQ(double(c), 0.03703703703718019824);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_38_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 38>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333303016843);
    CUDA_ASSERT_EQ(double(b), 0.11111111111108584737);
    CUDA_ASSERT_EQ(double(c), 0.03703703703695282456);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_39_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 39>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333348491578);
    CUDA_ASSERT_EQ(double(b), 0.11111111111108584737);
    CUDA_ASSERT_EQ(double(c), 0.03703703703706651140);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_40_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 40>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333325754211);
    CUDA_ASSERT_EQ(double(b), 0.11111111111108584737);
    CUDA_ASSERT_EQ(double(c), 0.03703703703700966798);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_41_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 41>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333337122895);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111426908);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703808969);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_42_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 42>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333331438553);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111426908);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703808969);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_43_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 43>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333334280724);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111426908);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703808969);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_44_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 44>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333332859638);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111071637);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703631334);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_45_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 45>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333570181);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111071637);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703720151);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_46_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 46>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333214910);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111071637);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703675743);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_47_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 47>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333392545);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111116045);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703720151);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_48_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 48>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333303727);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111116045);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703697947);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_49_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 49>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333348136);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111116045);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703709049);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_50_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 50>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333325932);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111110494);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703703498);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_51_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 51>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333325932);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111110494);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703703498);
}

CUDA_TEST(FloatxOperationsTest, MUL_9_52_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<9, 52>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333331483);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111110494);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703703498);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_1_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 1>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.37500000000000000000);
    CUDA_ASSERT_EQ(double(b), 0.12500000000000000000);
    CUDA_ASSERT_EQ(double(c), 0.04687500000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_2_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 2>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.31250000000000000000);
    CUDA_ASSERT_EQ(double(b), 0.10937500000000000000);
    CUDA_ASSERT_EQ(double(c), 0.03125000000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_3_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 3>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.34375000000000000000);
    CUDA_ASSERT_EQ(double(b), 0.10937500000000000000);
    CUDA_ASSERT_EQ(double(c), 0.03906250000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_4_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 4>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.32812500000000000000);
    CUDA_ASSERT_EQ(double(b), 0.10937500000000000000);
    CUDA_ASSERT_EQ(double(c), 0.03515625000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_5_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 5>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33593750000000000000);
    CUDA_ASSERT_EQ(double(b), 0.11132812500000000000);
    CUDA_ASSERT_EQ(double(c), 0.03710937500000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_6_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 6>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33203125000000000000);
    CUDA_ASSERT_EQ(double(b), 0.11132812500000000000);
    CUDA_ASSERT_EQ(double(c), 0.03710937500000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_7_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 7>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33398437500000000000);
    CUDA_ASSERT_EQ(double(b), 0.11132812500000000000);
    CUDA_ASSERT_EQ(double(c), 0.03710937500000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_8_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 8>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33300781250000000000);
    CUDA_ASSERT_EQ(double(b), 0.11108398437500000000);
    CUDA_ASSERT_EQ(double(c), 0.03698730468750000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_9_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 9>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33349609375000000000);
    CUDA_ASSERT_EQ(double(b), 0.11108398437500000000);
    CUDA_ASSERT_EQ(double(c), 0.03704833984375000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_10_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 10>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33325195312500000000);
    CUDA_ASSERT_EQ(double(b), 0.11108398437500000000);
    CUDA_ASSERT_EQ(double(c), 0.03701782226562500000);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_11_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 11>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33337402343750000000);
    CUDA_ASSERT_EQ(double(b), 0.11111450195312500000);
    CUDA_ASSERT_EQ(double(c), 0.03704833984375000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_12_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 12>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33331298828125000000);
    CUDA_ASSERT_EQ(double(b), 0.11111450195312500000);
    CUDA_ASSERT_EQ(double(c), 0.03703308105468750000);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_13_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 13>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33334350585937500000);
    CUDA_ASSERT_EQ(double(b), 0.11111450195312500000);
    CUDA_ASSERT_EQ(double(c), 0.03704071044921875000);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_14_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 14>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33332824707031250000);
    CUDA_ASSERT_EQ(double(b), 0.11111068725585937500);
    CUDA_ASSERT_EQ(double(c), 0.03703689575195312500);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_15_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 15>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333587646484375000);
    CUDA_ASSERT_EQ(double(b), 0.11111068725585937500);
    CUDA_ASSERT_EQ(double(c), 0.03703689575195312500);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_16_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 16>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333206176757812500);
    CUDA_ASSERT_EQ(double(b), 0.11111068725585937500);
    CUDA_ASSERT_EQ(double(c), 0.03703689575195312500);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_17_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 17>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333396911621093750);
    CUDA_ASSERT_EQ(double(b), 0.11111116409301757812);
    CUDA_ASSERT_EQ(double(c), 0.03703713417053222656);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_18_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 18>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333301544189453125);
    CUDA_ASSERT_EQ(double(b), 0.11111116409301757812);
    CUDA_ASSERT_EQ(double(c), 0.03703701496124267578);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_19_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 19>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333349227905273438);
    CUDA_ASSERT_EQ(double(b), 0.11111116409301757812);
    CUDA_ASSERT_EQ(double(c), 0.03703707456588745117);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_20_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 20>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333325386047363281);
    CUDA_ASSERT_EQ(double(b), 0.11111110448837280273);
    CUDA_ASSERT_EQ(double(c), 0.03703701496124267578);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_21_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 21>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333337306976318359);
    CUDA_ASSERT_EQ(double(b), 0.11111110448837280273);
    CUDA_ASSERT_EQ(double(c), 0.03703704476356506348);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_22_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 22>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333331346511840820);
    CUDA_ASSERT_EQ(double(b), 0.11111110448837280273);
    CUDA_ASSERT_EQ(double(c), 0.03703702986240386963);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_23_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 23>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333334326744079590);
    CUDA_ASSERT_EQ(double(b), 0.11111111193895339966);
    CUDA_ASSERT_EQ(double(c), 0.03703703731298446655);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_24_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 24>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333332836627960205);
    CUDA_ASSERT_EQ(double(b), 0.11111111193895339966);
    CUDA_ASSERT_EQ(double(c), 0.03703703731298446655);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_25_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 25>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333581686019897);
    CUDA_ASSERT_EQ(double(b), 0.11111111193895339966);
    CUDA_ASSERT_EQ(double(c), 0.03703703731298446655);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_26_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 26>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333209156990051);
    CUDA_ASSERT_EQ(double(b), 0.11111111100763082504);
    CUDA_ASSERT_EQ(double(c), 0.03703703684732317924);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_27_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 27>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333395421504974);
    CUDA_ASSERT_EQ(double(b), 0.11111111100763082504);
    CUDA_ASSERT_EQ(double(c), 0.03703703708015382290);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_28_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 28>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333302289247513);
    CUDA_ASSERT_EQ(double(b), 0.11111111100763082504);
    CUDA_ASSERT_EQ(double(c), 0.03703703696373850107);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_29_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 29>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333348855376244);
    CUDA_ASSERT_EQ(double(b), 0.11111111112404614687);
    CUDA_ASSERT_EQ(double(c), 0.03703703708015382290);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_30_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 30>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333325572311878);
    CUDA_ASSERT_EQ(double(b), 0.11111111112404614687);
    CUDA_ASSERT_EQ(double(c), 0.03703703702194616199);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_31_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 31>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333337213844061);
    CUDA_ASSERT_EQ(double(b), 0.11111111112404614687);
    CUDA_ASSERT_EQ(double(c), 0.03703703705104999244);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_32_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 32>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333331393077970);
    CUDA_ASSERT_EQ(double(b), 0.11111111110949423164);
    CUDA_ASSERT_EQ(double(c), 0.03703703703649807721);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_33_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 33>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333334303461015);
    CUDA_ASSERT_EQ(double(b), 0.11111111110949423164);
    CUDA_ASSERT_EQ(double(c), 0.03703703703649807721);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_34_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 34>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333332848269492);
    CUDA_ASSERT_EQ(double(b), 0.11111111110949423164);
    CUDA_ASSERT_EQ(double(c), 0.03703703703649807721);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_35_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 35>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333575865254);
    CUDA_ASSERT_EQ(double(b), 0.11111111111131322104);
    CUDA_ASSERT_EQ(double(c), 0.03703703703740757192);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_36_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 36>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333212067373);
    CUDA_ASSERT_EQ(double(b), 0.11111111111131322104);
    CUDA_ASSERT_EQ(double(c), 0.03703703703695282456);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_37_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 37>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333393966313);
    CUDA_ASSERT_EQ(double(b), 0.11111111111131322104);
    CUDA_ASSERT_EQ(double(c), 0.03703703703718019824);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_38_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 38>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333303016843);
    CUDA_ASSERT_EQ(double(b), 0.11111111111108584737);
    CUDA_ASSERT_EQ(double(c), 0.03703703703695282456);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_39_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 39>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333348491578);
    CUDA_ASSERT_EQ(double(b), 0.11111111111108584737);
    CUDA_ASSERT_EQ(double(c), 0.03703703703706651140);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_40_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 40>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333325754211);
    CUDA_ASSERT_EQ(double(b), 0.11111111111108584737);
    CUDA_ASSERT_EQ(double(c), 0.03703703703700966798);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_41_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 41>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333337122895);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111426908);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703808969);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_42_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 42>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333331438553);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111426908);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703808969);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_43_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 43>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333334280724);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111426908);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703808969);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_44_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 44>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333332859638);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111071637);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703631334);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_45_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 45>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333570181);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111071637);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703720151);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_46_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 46>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333214910);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111071637);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703675743);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_47_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 47>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333392545);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111116045);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703720151);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_48_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 48>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333303727);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111116045);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703697947);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_49_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 49>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333348136);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111116045);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703709049);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_50_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 50>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333325932);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111110494);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703703498);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_51_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 51>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333325932);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111110494);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703703498);
}

CUDA_TEST(FloatxOperationsTest, MUL_10_52_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<10, 52>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333331483);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111110494);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703703498);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_1_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 1>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.37500000000000000000);
    CUDA_ASSERT_EQ(double(b), 0.12500000000000000000);
    CUDA_ASSERT_EQ(double(c), 0.04687500000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_2_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 2>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.31250000000000000000);
    CUDA_ASSERT_EQ(double(b), 0.10937500000000000000);
    CUDA_ASSERT_EQ(double(c), 0.03125000000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_3_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 3>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.34375000000000000000);
    CUDA_ASSERT_EQ(double(b), 0.10937500000000000000);
    CUDA_ASSERT_EQ(double(c), 0.03906250000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_4_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 4>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.32812500000000000000);
    CUDA_ASSERT_EQ(double(b), 0.10937500000000000000);
    CUDA_ASSERT_EQ(double(c), 0.03515625000000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_5_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 5>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33593750000000000000);
    CUDA_ASSERT_EQ(double(b), 0.11132812500000000000);
    CUDA_ASSERT_EQ(double(c), 0.03710937500000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_6_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 6>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33203125000000000000);
    CUDA_ASSERT_EQ(double(b), 0.11132812500000000000);
    CUDA_ASSERT_EQ(double(c), 0.03710937500000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_7_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 7>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33398437500000000000);
    CUDA_ASSERT_EQ(double(b), 0.11132812500000000000);
    CUDA_ASSERT_EQ(double(c), 0.03710937500000000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_8_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 8>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33300781250000000000);
    CUDA_ASSERT_EQ(double(b), 0.11108398437500000000);
    CUDA_ASSERT_EQ(double(c), 0.03698730468750000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_9_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 9>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33349609375000000000);
    CUDA_ASSERT_EQ(double(b), 0.11108398437500000000);
    CUDA_ASSERT_EQ(double(c), 0.03704833984375000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_10_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 10>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33325195312500000000);
    CUDA_ASSERT_EQ(double(b), 0.11108398437500000000);
    CUDA_ASSERT_EQ(double(c), 0.03701782226562500000);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_11_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 11>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33337402343750000000);
    CUDA_ASSERT_EQ(double(b), 0.11111450195312500000);
    CUDA_ASSERT_EQ(double(c), 0.03704833984375000000);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_12_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 12>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33331298828125000000);
    CUDA_ASSERT_EQ(double(b), 0.11111450195312500000);
    CUDA_ASSERT_EQ(double(c), 0.03703308105468750000);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_13_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 13>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33334350585937500000);
    CUDA_ASSERT_EQ(double(b), 0.11111450195312500000);
    CUDA_ASSERT_EQ(double(c), 0.03704071044921875000);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_14_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 14>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33332824707031250000);
    CUDA_ASSERT_EQ(double(b), 0.11111068725585937500);
    CUDA_ASSERT_EQ(double(c), 0.03703689575195312500);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_15_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 15>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333587646484375000);
    CUDA_ASSERT_EQ(double(b), 0.11111068725585937500);
    CUDA_ASSERT_EQ(double(c), 0.03703689575195312500);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_16_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 16>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333206176757812500);
    CUDA_ASSERT_EQ(double(b), 0.11111068725585937500);
    CUDA_ASSERT_EQ(double(c), 0.03703689575195312500);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_17_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 17>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333396911621093750);
    CUDA_ASSERT_EQ(double(b), 0.11111116409301757812);
    CUDA_ASSERT_EQ(double(c), 0.03703713417053222656);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_18_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 18>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333301544189453125);
    CUDA_ASSERT_EQ(double(b), 0.11111116409301757812);
    CUDA_ASSERT_EQ(double(c), 0.03703701496124267578);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_19_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 19>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333349227905273438);
    CUDA_ASSERT_EQ(double(b), 0.11111116409301757812);
    CUDA_ASSERT_EQ(double(c), 0.03703707456588745117);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_20_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 20>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333325386047363281);
    CUDA_ASSERT_EQ(double(b), 0.11111110448837280273);
    CUDA_ASSERT_EQ(double(c), 0.03703701496124267578);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_21_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 21>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333337306976318359);
    CUDA_ASSERT_EQ(double(b), 0.11111110448837280273);
    CUDA_ASSERT_EQ(double(c), 0.03703704476356506348);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_22_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 22>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333331346511840820);
    CUDA_ASSERT_EQ(double(b), 0.11111110448837280273);
    CUDA_ASSERT_EQ(double(c), 0.03703702986240386963);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_23_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 23>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333334326744079590);
    CUDA_ASSERT_EQ(double(b), 0.11111111193895339966);
    CUDA_ASSERT_EQ(double(c), 0.03703703731298446655);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_24_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 24>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333332836627960205);
    CUDA_ASSERT_EQ(double(b), 0.11111111193895339966);
    CUDA_ASSERT_EQ(double(c), 0.03703703731298446655);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_25_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 25>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333581686019897);
    CUDA_ASSERT_EQ(double(b), 0.11111111193895339966);
    CUDA_ASSERT_EQ(double(c), 0.03703703731298446655);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_26_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 26>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333209156990051);
    CUDA_ASSERT_EQ(double(b), 0.11111111100763082504);
    CUDA_ASSERT_EQ(double(c), 0.03703703684732317924);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_27_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 27>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333395421504974);
    CUDA_ASSERT_EQ(double(b), 0.11111111100763082504);
    CUDA_ASSERT_EQ(double(c), 0.03703703708015382290);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_28_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 28>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333302289247513);
    CUDA_ASSERT_EQ(double(b), 0.11111111100763082504);
    CUDA_ASSERT_EQ(double(c), 0.03703703696373850107);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_29_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 29>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333348855376244);
    CUDA_ASSERT_EQ(double(b), 0.11111111112404614687);
    CUDA_ASSERT_EQ(double(c), 0.03703703708015382290);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_30_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 30>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333325572311878);
    CUDA_ASSERT_EQ(double(b), 0.11111111112404614687);
    CUDA_ASSERT_EQ(double(c), 0.03703703702194616199);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_31_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 31>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333337213844061);
    CUDA_ASSERT_EQ(double(b), 0.11111111112404614687);
    CUDA_ASSERT_EQ(double(c), 0.03703703705104999244);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_32_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 32>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333331393077970);
    CUDA_ASSERT_EQ(double(b), 0.11111111110949423164);
    CUDA_ASSERT_EQ(double(c), 0.03703703703649807721);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_33_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 33>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333334303461015);
    CUDA_ASSERT_EQ(double(b), 0.11111111110949423164);
    CUDA_ASSERT_EQ(double(c), 0.03703703703649807721);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_34_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 34>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333332848269492);
    CUDA_ASSERT_EQ(double(b), 0.11111111110949423164);
    CUDA_ASSERT_EQ(double(c), 0.03703703703649807721);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_35_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 35>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333575865254);
    CUDA_ASSERT_EQ(double(b), 0.11111111111131322104);
    CUDA_ASSERT_EQ(double(c), 0.03703703703740757192);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_36_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 36>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333212067373);
    CUDA_ASSERT_EQ(double(b), 0.11111111111131322104);
    CUDA_ASSERT_EQ(double(c), 0.03703703703695282456);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_37_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 37>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333393966313);
    CUDA_ASSERT_EQ(double(b), 0.11111111111131322104);
    CUDA_ASSERT_EQ(double(c), 0.03703703703718019824);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_38_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 38>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333303016843);
    CUDA_ASSERT_EQ(double(b), 0.11111111111108584737);
    CUDA_ASSERT_EQ(double(c), 0.03703703703695282456);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_39_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 39>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333348491578);
    CUDA_ASSERT_EQ(double(b), 0.11111111111108584737);
    CUDA_ASSERT_EQ(double(c), 0.03703703703706651140);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_40_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 40>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333325754211);
    CUDA_ASSERT_EQ(double(b), 0.11111111111108584737);
    CUDA_ASSERT_EQ(double(c), 0.03703703703700966798);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_41_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 41>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333337122895);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111426908);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703808969);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_42_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 42>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333331438553);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111426908);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703808969);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_43_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 43>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333334280724);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111426908);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703808969);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_44_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 44>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333332859638);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111071637);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703631334);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_45_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 45>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333570181);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111071637);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703720151);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_46_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 46>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333214910);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111071637);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703675743);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_47_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 47>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333392545);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111116045);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703720151);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_48_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 48>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333303727);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111116045);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703697947);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_49_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 49>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333348136);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111116045);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703709049);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_50_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 50>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333325932);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111110494);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703703498);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_51_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 51>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333325932);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111110494);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703703498);
}

CUDA_TEST(FloatxOperationsTest, MUL_11_52_simple, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<11, 52>;
    T a = 0.33333333333333331483;
    T b = 0.11111111111111110494;
    T c = 0;
    c = a * b;
    CUDA_ASSERT_EQ(double(a), 0.33333333333333331483);
    CUDA_ASSERT_EQ(double(b), 0.11111111111111110494);
    CUDA_ASSERT_EQ(double(c), 0.03703703703703703498);
}

}  // namespace
