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

// Check internal functions.
// Rounding of a uint64_t type value.
// Checks the following routine:
// inline uint64_t SHIFT_RIGHT_ROUND_NEAREST(uint64_t mant, uint16_t SHIFT)
// IMPLEMENTS rounding according the IEEE 754 standard with a NEAREST policy and
// ties are resolved to even.

TEST(RoundNearest, down)
{
    //																						  RRRRRRRRRR
    // number:    	0x3e6999999999999a
    // 0011111001101001100110011001100110011001100110011001100110011010
    // >> 10 :    	0x000f9a6666666666
    // 0000000000001111100110100110011001100110011001100110011001100110 RND 10:
    // 0x000f9a6666666666
    // 0000000000001111100110100110011001100110011001100110011001100110
    uint64_t number = 0x3e6999999999999a;
    int shift_amount = 10;
    uint64_t expected = 0x000f9a6666666666 << shift_amount;

    EXPECT_EQ(expected, flx::detail::round_nearest(number, shift_amount));
}

TEST(RoundNearest, up)
{
    //																						RRRRRRRRRRRR
    // number:    	0x3e6999999999999a
    // 0011111001101001100110011001100110011001100110011001100110011010
    // >> 12 :    	0x0003e69999999999
    // 0000000000000011111001101001100110011001100110011001100110011001 RND 12:
    // 0x0003e6999999999a
    // 0000000000000011111001101001100110011001100110011001100110011010
    uint64_t number = 0x3e6999999999999a;
    int shift_amount = 12;
    uint64_t expected = 0x0003e6999999999a << shift_amount;

    EXPECT_EQ(expected, flx::detail::round_nearest(number, shift_amount));
}

TEST(RoundNearest, RoundNearestWithTiesToEvenRoundsUp)
{
    //																								RRRR
    // number:    	0x0ffffffff00000f8
    // 0000111111111111111111111111111111110000000000000000000011111000
    // >>  4 :    	0x00ffffffff00000f
    // 0000000011111111111111111111111111111111000000000000000000001111 RND  4:
    // 0x00ffffffff000010
    // 0000000011111111111111111111111111111111000000000000000000010000
    uint64_t number = 0x0ffffffff00000f8;
    int shift_amount = 4;
    uint64_t expected = 0x00ffffffff000010 << shift_amount;

    EXPECT_EQ(expected, flx::detail::round_nearest(number, shift_amount));
}

TEST(RoundNearest, RoundNearestWithTiesToEvenRoundsDown)
{
    //																								RRRR
    // number:    	0x0ffffffff00000e8
    // 0000111111111111111111111111111111110000000000000000000011101000
    // >>  4 :    	0x00ffffffff00000e
    // 0000000011111111111111111111111111111111000000000000000000001110 RND  4:
    // 0x00ffffffff00000e
    // 0000000011111111111111111111111111111111000000000000000000001110
    uint64_t number = 0x0ffffffff00000e8;
    int shift_amount = 4;
    uint64_t expected = 0x00ffffffff00000e << shift_amount;

    EXPECT_EQ(expected, flx::detail::round_nearest(number, shift_amount));
}
