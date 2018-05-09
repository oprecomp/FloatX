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

TEST(MyTestBF, BF_3_5)
{
    const uint8_t E = 3;
    const uint8_t M = 3;

    uint64_t mx = ((uint64_t)0x1) << (1 + E + M);
    for (uint64_t cnt = 0x0; cnt < mx; ++cnt) {
        // define input pattern
        std::bitset<1 + E + M> pattern(cnt);

        // get a backend number
        double bd = flx::detail::construct_number<E, M>(pattern);

        // check that the backend number is valid.
        // e.g. cast to fx and back to double (that should not change its value)
        flx::floatx<E, M> fx = bd;
        double r = double(fx);
        EXPECT_EQ(*reinterpret_cast<uint64_t*>(&r),
                  *reinterpret_cast<uint64_t*>(&bd));

        // get the reverse functionallity
        std::bitset<1 + E + M> out =
            flx::detail::get_fullbit_representation_BS<E, M>(r);

        // printf("value: %.20e\n", r );
        // std::cout << "IN:  " << pattern << std::endl;
        // std::cout << "OUT: " << out << std::endl;

        EXPECT_EQ(pattern, out);
    }
}

TEST(MyTestBF, BF_5_2)
{
    const uint8_t E = 5;
    const uint8_t M = 2;

    uint64_t mx = ((uint64_t)0x1) << (1 + E + M);
    for (uint64_t cnt = 0x0; cnt < mx; ++cnt) {
        // define input pattern
        std::bitset<1 + E + M> pattern(cnt);

        // get a backend number
        double bd = flx::detail::construct_number<E, M>(pattern);

        // check that the backend number is valid.
        // e.g. cast to fx and back to double (that should not change its value)
        flx::floatx<E, M> fx = bd;
        double r = double(fx);
        EXPECT_EQ(*reinterpret_cast<uint64_t*>(&r),
                  *reinterpret_cast<uint64_t*>(&bd));

        // get the reverse functionallity
        std::bitset<1 + E + M> out =
            flx::detail::get_fullbit_representation_BS<E, M>(r);

        // printf("value: %.20e\n", r );
        // std::cout << "IN:  " << pattern << std::endl;
        // std::cout << "OUT: " << out << std::endl;

        EXPECT_EQ(pattern, out);
    }
}

TEST(MyTestBF, BF_5_10)
{
    const uint8_t E = 5;
    const uint8_t M = 10;

    uint64_t mx = ((uint64_t)0x1) << (1 + E + M);
    for (uint64_t cnt = 0x0; cnt < mx; ++cnt) {
        // define input pattern
        std::bitset<1 + E + M> pattern(cnt);

        // get a backend number
        double bd = flx::detail::construct_number<E, M>(pattern);

        // check that the backend number is valid.
        // e.g. cast to fx and back to double (that should not change its value)
        flx::floatx<E, M> fx = bd;
        double r = double(fx);
        EXPECT_EQ(*reinterpret_cast<uint64_t*>(&r),
                  *reinterpret_cast<uint64_t*>(&bd));

        // get the reverse functionallity
        std::bitset<1 + E + M> out =
            flx::detail::get_fullbit_representation_BS<E, M>(r);

        // printf("value: %.20e\n", r );
        // std::cout << "IN:  " << pattern << std::endl;
        // std::cout << "OUT: " << out << std::endl;

        EXPECT_EQ(pattern, out);
    }
}