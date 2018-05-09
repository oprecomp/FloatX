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

#include <sstream>


#include <gtest/gtest.h>
#include <floatx.hpp>


namespace {


TEST(FloatxStreamTest, WritesToOutputStream)
{
    flx::floatx<8, 23> val = 1.0 + 1e-15;
    std::stringstream os;
    os << val;
    EXPECT_EQ("1", os.str());
}


TEST(FloatxStreamTest, ReadsFromOutputStream)
{
    flx::floatx<8, 23> val;
    std::stringstream is("1.00000000000001");
    is >> val;
    EXPECT_EQ(1.0, val);
}


TEST(FloatxrStreamTest, WritesToOutputStream)
{
    flx::floatxr<> val = 1.0 + 1e-15;
    val.set_precision(8, 23);
    std::stringstream os;
    os << val;
    EXPECT_EQ("1", os.str());
}


TEST(FloatxrStreamTest, ReadsFromOutputStream)
{
    flx::floatxr<> val(8, 23);
    std::stringstream is("1.00000000000001");
    is >> val;
    EXPECT_EQ(1.0, val);
}


}  // namespace
