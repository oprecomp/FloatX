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

#include <iostream>


// Uncoment to disable common type resolution.
// #define FLOATX_NO_TYPE_RESOLUTION
#include <floatx.hpp>


int main()
{
    using float1 = flx::floatx<5, 7>;
    using float2 = flx::floatx<4, 8>;
    std::cout << float1(2.6) + float1(6.2) << std::endl;  // always works
    std::cout << float1(2.6) + float2(6.2) << std::endl;  // fails with flag
}
