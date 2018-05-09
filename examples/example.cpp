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

/*
 * Example of using the FloatX library. Compile with
 * g++ -std=c++11 -I ../src test.cpp
 */
#include <iostream>


#include <floatx.hpp>


int main()
{
    std::cout << "sizeof(floatx)  = " << sizeof(flx::floatx<11, 52>)
              << "\nsizeof(floatxr) = " << sizeof(flx::floatxr<>) << std::endl;
    // compile-time types
    flx::floatx<11, 52> f;      // double
    flx::floatx<7, 22> g(5.3);  // float with 7 exp and 22 significand bits

    // runtime types
    flx::floatxr<> fr(11, 52);
    flx::floatxr<> gr(7, 22, 5.3);

    std::cout << std::scientific;

    // conversion to double
    std::cout << double(f) << std::endl
              << double(g) << std::endl
              << double(fr) << std::endl
              << double(gr) << std::endl;

    // conversion to flexfloat
    flx::floatx<3, 2> lg(g);
    flx::floatx<3, 2> lgr(gr);

    std::cout << double(lg) << ", precision = "
              << "(" << get_exp_bits(lg) << ", " << get_sig_bits(lg) << ")\n"
              << double(lgr) << ", precision = "
              << "(" << get_exp_bits(lgr) << ", " << get_sig_bits(lgr) << ")"
              << std::endl;
    return 0;
}
