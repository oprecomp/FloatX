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
 * compile that file
 * g++ -std=c++11 -Wall -o demo_newton -I../src demo_newton.cpp
 */

#include <iostream>
#include "floatx.hpp"

// Babylonian method:
// Derived from Netwon
// See: https://en.wikipedia.org/wiki/Methods_of_computing_square_roots#Example
// based on float type
float myroot(float a, float a0, float tol)
{
    float x = a0;
    float xnext;
    float err;

    int k = 0;
    do {
        xnext = 0.5 * (x + a / x);
        err = fabs(x - xnext);
        printf("[k=%i]: %f %e\n", k++, xnext, err);
        x = xnext;
    } while (err > tol);
    return xnext;
}

// general routine based on type T
// note, if T is float that routine is the same as above.
template <typename T>
T myroot_general(T a, T a0, T tol)
{
    T x = a0;
    T xnext;
    T err;

    int k = 0;
    do {
        xnext = 0.5 * (x + a / x);
        // for example fabs(...) is not defined for the floatx type
        // hence, we use a cast to double and back to our type
        err = (T)fabs(double(x - xnext));
        printf("[k=%i]: %f %e\n", k++, double(xnext), double(err));
        x = xnext;
    } while (err > tol);
    return xnext;
}

int main(int argc, char** argv)
{
    printf(
        "floatx working "
        "example\n==============================================\n");
    printf("Iteratively compute the square root of a\n");

    if (argc != 3) {
        printf("Usage: %s <a> <a0>\n computes root(a) by Newton Iterations.\n",
               argv[0]);
        printf("Example: \n %s 2 1\n", argv[0]);
        exit(-1);
    }

    float a = atof(argv[1]);
    float a0 = atof(argv[2]);

    float res = myroot(a, a0, 1e-6);

    printf("\n\nBaseline version (float)\n\n");

    printf("==============================================\n");
    printf("Result Computed (float):      %.20f\n", res);
    float ref = sqrt(a);
    printf("Reference:                    %.20f\n", ref);
    printf("==============================================\n");
    printf("Error:                        %e\n", ref - res);
    printf("==============================================\n");


    printf("\n\nFloatx Version IEEE 16bit, e.g., floatx<5,10>\n\n");
    res = myroot_general<flx::floatx<5, 10>>(a, a0, 1e-6);
    printf("==============================================\n");
    printf("Result Computed (floatx<E,M>: %.20f\n", res);
    printf("Reference:                    %.20f\n", ref);
    printf("==============================================\n");
    printf("Error:                        %e\n", ref - res);
    printf("==============================================\n");

    return 0;
}
