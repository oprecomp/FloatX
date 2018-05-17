#include <gtest/gtest.h>
#include <floatx.hpp>

#include "cuda_assert_op.cuh"
#include "cuda_test.cuh"

#define nan double(0.0 / 0.0)
#define inf double(1.0 / 0.0)

namespace {


CUDA_TEST(FloatxOperations, Sqrt, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 3>;
    const T a = 0.25;
    const T b = sqrt(a);
    CUDA_ASSERT_EQ(double(a), 0.25);
    CUDA_ASSERT_EQ(double(b), 0.50);
}


CUDA_TEST(FloatxOperations, Abs, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 3>;
    const T a = -0.25;
    const T b = abs(a);
    CUDA_ASSERT_EQ(double(a), -0.25);
    CUDA_ASSERT_EQ(double(b), 0.25);
}


CUDA_TEST(FloatxOperations, Real, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 3>;
    const T a = 0.25;
    const T b = real(a);
    CUDA_ASSERT_EQ(double(a), 0.25);
    CUDA_ASSERT_EQ(double(b), 0.25);
}


CUDA_TEST(FloatxOperations, UnaryPlus, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 3>;
    const T a = 0.25;
    const T b = +a;
    CUDA_ASSERT_EQ(double(a), 0.25);
    CUDA_ASSERT_EQ(double(b), 0.25);
}


CUDA_TEST(FloatxOperations, UnaryMinus, 0, 1, 1, 0, 0)
{
    using T = flx::floatx<1, 3>;
    const T a = 0.25;
    const T b = -a;
    CUDA_ASSERT_EQ(double(a), 0.25);
    CUDA_ASSERT_EQ(double(b), -0.25);
}


CUDA_TEST(FloatxrOperations, Sqrt, 0, 1, 1, 0, 0)
{
    using T = flx::floatxr<>;
    const T a = {1, 3, 0.25};
    const T b = sqrt(a);
    CUDA_ASSERT_EQ(double(a), 0.25);
    CUDA_ASSERT_EQ(double(b), 0.50);
}


CUDA_TEST(FloatxrOperations, Abs, 0, 1, 1, 0, 0)
{
    using T = flx::floatxr<>;
    const T a = {1, 3, -0.25};
    const T b = abs(a);
    CUDA_ASSERT_EQ(double(a), -0.25);
    CUDA_ASSERT_EQ(double(b), 0.25);
}


CUDA_TEST(FloatxrOperations, Real, 0, 1, 1, 0, 0)
{
    using T = flx::floatxr<>;
    const T a = {1, 3, 0.25};
    const T b = real(a);
    CUDA_ASSERT_EQ(double(a), 0.25);
    CUDA_ASSERT_EQ(double(b), 0.25);
}


}  // namespace
