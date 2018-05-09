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

template <typename T>
void foo(T* a, int n)
{
    printf("HI FOO ROUTINE\n");
    for (unsigned i = 0; i < n; ++i) {
        if (i == 0 || i == 1) {
            a[i] = 1;
        } else {
            a[i] = a[i - 1] + a[i - 2];
        }
    }
}

void compileExample()
{
    typedef flx::floatx<11, 48> T;
    T res = 0;

    double a = 3.1;
    double b = 5.2;

    res = (T)a * b;  // ERROR

    std::cout << "[cout] res = " << res << std::endl;
}

void compileExample2()
{
    typedef flx::floatx<11, 48> T;
    T res = 33.45;
    int i = 4;

    std::cout << "[cout] res = " << res << " and i = " << i << std::endl;

    // TODO (withouth double does not yet work)
    if (res == (double)i) {
        std::cout << "TRUE\n";
    } else {
        std::cout << "FALSE\n";
    }

    if (res == 3.0) {
        std::cout << "TRUE 2\n";
    } else {
        std::cout << "FALSE 2\n";
    }
}

int main()
{
    compileExample();
    compileExample2();
    printf("--------------------------\n");
    // Double-precision variables

    // simple use case
    // flx::floatx<11, 52> ff_a, ff_b, ff_c;
    // flx::floatx<5, 30> ff_a, ff_b, ff_c;

    // other use case
    flx::floatx<11, 48> ff_a;
    flx::floatx<5, 5> ff_b;
    flx::floatx<11, 30> ff_c;

    // Assigment with cast (from double literal)
    ff_a = 10.4;
    ff_b = 11.5;
    // Overloaded operators
    // ff_b += 2;   // DOES NOT WORK (cast from int not defined)
    ff_b += 2.0;  // WORKS.

    // ff_b = ff_b + 2;  // DOES NOT WORK (cast from int not defined)
    ff_b = ff_b + 2.0;  // DOES NOT WORK (except flex is as double)
    // ff_b = double( ff_b  + flx::floatx<11, 32>(2)); //WORKS

    ff_c = ff_a + ff_b;

    // C++ output stream
    // Explicit output as double.
    std::cout << "output after double cast:\n";
    std::cout << "[cout] ff_a = " << double(ff_a) << std::endl;
    std::cout << "[cout] ff_b = " << double(ff_b) << std::endl;
    std::cout << "[cout] ff_c = " << double(ff_c) << std::endl;

    // Implicit output works as well.
    std::cout << "Output:\n";
    std::cout << "[cout] ff_a = " << ff_a << std::endl;
    std::cout << "[cout] ff_b = " << ff_b << std::endl;
    std::cout << "[cout] ff_c = " << ff_c << std::endl;

    std::cout << "Get information about type:\n";
    std::cout << "[cout] ff_a = " << ff_a << " <" << get_exp_bits(ff_a) << ","
              << get_sig_bits(ff_a) << ">" << std::endl;
    std::cout << "[cout] ff_b = " << ff_b << " <" << get_exp_bits(ff_b) << ","
              << get_sig_bits(ff_b) << ">" << std::endl;
    std::cout << "[cout] ff_c = " << ff_c << " <" << get_exp_bits(ff_c) << ","
              << get_sig_bits(ff_c) << ">" << std::endl;

    std::cout << "Sizeof Results (it is the static case):\n";
    std::cout << "sizeof( ff_a ) = " << sizeof(ff_a) << "\n";
    std::cout << "sizeof( ff_b ) = " << sizeof(ff_b) << "\n";
    std::cout << "sizeof( ff_c ) = " << sizeof(ff_c) << "\n";

    // get_exp_bits()
    // Binary output.
    // std::cout << "[cout] ff_a = " << ff_a << " (" << flexfloat_as_bits <<
    // ff_a << flexfloat_as_double << ")" << std::endl; std::cout << "[cout]
    // ff_b = " << ff_b << " (" << flexfloat_as_bits << ff_b <<
    // flexfloat_as_double << ")" << std::endl; std::cout << "[cout] ff_c = " <<
    // ff_c << " (" << flexfloat_as_bits << ff_c << flexfloat_as_double << ")"
    // << std::endl;

    // generate arrays of data
    // flexfloat<11, 52> ff_a
    int n = 100;
    // double* a = new double[n];
    flx::floatx<5, 12>* a = new flx::floatx<5, 12>[n];

    for (unsigned i = 0; i < n; ++i) {
        a[i] = i;
    }

    // foo< flx::floatx<5, 12> >( a, n);         // OK
    // foo< flx::floatx<5, 52> >( a, n);         //wrong type
    foo(a, n);  // infers type, ok

    for (unsigned i = 0; i < n; ++i) {
        // std::cout << i << ":\t" << a[i] << " (" << flexfloat_as_bits << a[i]
        // << flexfloat_as_double << ")" <<  std::endl;
        std::cout << i << ":\t" << double(a[i]) << std::endl;
    }
    delete[] a;

    // std::cout << "sizeof(floatx)  = " << sizeof(flx::floatx<11, 52>)
    //           << "\nsizeof(floatxr) = " << sizeof(flx::floatxr<>)
    //           << std::endl;
    // // compile-time types
    // flx::floatx<11, 52> f;  // double
    // flx::floatx<7, 22> g(5.3);  // float with 7 exp and 22 significand bits

    // // runtime types
    // flx::floatxr<> fr(11, 52);
    // flx::floatxr<> gr(7, 22, 5.3);

    // std::cout << std::scientific;

    // // conversion to double
    // std::cout << double(f) << std::endl
    //           << double(g) << std::endl
    //           << double(fr) << std::endl
    //           << double(gr) << std::endl;

    // // conversion to flexfloat
    // flx::floatx<3, 2> lg(g);
    // flx::floatx<3, 2> lgr(gr);

    // std::cout << double(lg) << ", precision = "
    //           << "(" << get_exp_bits(lg) << ", " << get_sig_bits(lg) << ")\n"
    //           << double(lgr) << ", precision = "
    //           << "(" << get_exp_bits(lgr) << ", " << get_sig_bits(lgr) << ")"
    //           << std::endl;
    // return 0;
}
