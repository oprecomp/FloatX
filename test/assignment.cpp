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


TEST(FloatxAssignmentTest, PreservesPrecision)
{
    const double val = 1.0 + 1e-15;
    flx::floatx<11, 52> fx_val;
    fx_val = val;
    EXPECT_EQ(val, double(fx_val));
}

TEST(FloatxAssignmentTest, LowersPrecision)
{
    const double val = 1.0 + 1e-15;
    flx::floatx<8, 23> fx_val;
    fx_val = val;
    EXPECT_NE(val, double(fx_val));  // round to float
}


TEST(FloatxAssignmentTest, AssignsBetweenFormats)
{
    const double val = 1.0 + 1e-15;
    flx::floatx<11, 52> d_val(val);
    flx::floatx<8, 23> s_val;
    s_val = d_val;
    EXPECT_NE(val, double(s_val));
    EXPECT_EQ(float(val), float(s_val));
}


TEST(FloatxrAssignmentTest, PreservesPrecision)
{
    const double val = 1.0 + 1e-15;
    flx::floatxr<> fx_val(11, 52);
    fx_val = val;
    EXPECT_EQ(val, double(fx_val));
}

TEST(FloatxrAssignmentTest, LowersPrecision)
{
    const double val = 1.0 + 1e-15;
    flx::floatxr<> fx_val(8, 23);
    fx_val = val;
    EXPECT_NE(val, double(fx_val));  // round to float
}


TEST(FloatxrAssignmentTest, AssignsBetweenFormats)
{
    const double val = 1.0 + 1e-15;
    flx::floatx<11, 52> d_val(val);
    flx::floatxr<> s_val(8, 23);
    s_val = d_val;
    EXPECT_NE(val, double(s_val));
    EXPECT_EQ(float(val), float(s_val));
}


}  // namespace
