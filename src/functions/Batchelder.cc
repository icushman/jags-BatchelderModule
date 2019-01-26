#include <config.h>
#include "Batchelder.h" // class header file
#include <util/nainf.h> // provides na and inf functions

#include <cmath> // basic math operations
#include <iostream>
#include <typeinfo>

#include <algorithm>

#include <util/dim.h>
#include <util/logical.h>

using std::vector; // vector is used in the code
using std::string; // string is used in the code


// a, b, r, v, t, L1, L2

// potential error source: pointers vs references... arg#3 had a pointer for some reason.

#define a (*args[0]) //unlearned -> intermediate
#define b (*args[1]) //immediate retrival -> learned
#define r (*args[2]) //unlearned -> learned
#define v (*args[3]) //intermediate -> learned
#define t (*args[4]) //itermediate -> immediate retrieval
#define L1 (*args[5]) //learned -> immediate retrieval
#define L2 (*args[6]) //learned -> delayed retrieval

namespace jags {
namespace batchelder {

    BATCHELDER::BATCHELDER() :VectorFunction ("Batchelder", 7)
    {}

    void BATCHELDER::evaluate (double *value, vector <double const *> const &args,
                          vector<unsigned int> const &lengths) const
    {

    //0000

    value[0] = 
             r*(1.0-L1)*(1.0-L1)*(1.0-L1)*(1.0-L2) +
             (1-r)*a*(1-t)*v*(1-L1)*(1-L1)*(1-L2) +
             (1-r)*(1-a)*r*(1-L1)*(1-L1)*(1-L2) +
             (1-r)*a*(1-t)*(1-v)*(1-t)*v*(1-L1)*(1-L2) +
             (1-r)*(1-a)*(1-r)*a*(1-t)*v*(1-L1)*(1-L2) +
             (1-r)*(1-a)*(1-r)*(1-a)*r*(1-L1)*(1-L2) +
             (1-r)*a*(1-t)*(1-v)*(1-t)*(1-v)*(1-t) +
             (1-r)*(1-a)*(1-r)*a*(1-t)*(1-v)*(1-t) +
             (1-r)*(1-a)*(1-r)*(1-a)*(1-r)*a*(1-t) +
             (1-r)*(1-a)*(1-r)*(1-a)*(1-r)*(1-a);

    // 0001

    value[1] =
             r*(1-L1)*(1-L1)*(1-L1)*L2 +
             (1-r)*a*(1-t)*v*(1-L1)*(1-L1)*L2 +
             (1-r)*(1-a)*r*(1-L1)*(1-L1)*L2 +
             (1-r)*a*(1-t)*(1-v)*(1-t)*v*(1-L1)*L2 +
             (1-r)*(1-a)*(1-r)*a*(1-t)*v*(1-L1)*L2 +
             (1-r)*(1-a)*(1-r)*(1-a)*r*(1-L1)*L2;

    // 0010

    value[2] =
             r*(1-L1)*(1-L1)*L1*(1-L2) +
             (1-r)*a*(1-t)*v*(1-L1)*L1*(1-L2) +
             (1-r)*(1-a)*r*(1-L1)*L1*(1-L2) +
             (1-r)*a*(1-t)*(1-v)*(1-t)*(v*L1+(1-v)*t*b)*(1-L2) + 
             (1-r)*(1-a)*(1-r)*a*(1-t)*(v*L1+(1-v)*t*b)*(1-L2) +
             (1-r)*(1-a)*(1-r)*(1-a)*(r*L1+(1-r)*a*t*b)*(1-L2) +
             (1-r)*a*(1-t)*(1-v)*(1-t)*(1-v)*t*(1-b) +
             (1-r)*(1-a)*(1-r)*a*(1-t)*(1-v)*t*(1-b) +
             (1-r)*(1-a)*(1-r)*(1-a)*(1-r)*a*t*(1-b);

    // 0011

    value[3] =
             r*(1-L1)*(1-L1)*L1*L2 +
             (1-r)*a*(1-t)*v*(1-L1)*L1*L2 +
             (1-r)*(1-a)*r*(1-L1)*L1*L2 +
             (1-r)*a*(1-t)*(1-v)*(1-t)*(v*L1+(1-v)*t*b)*L2 +
             (1-r)*(1-a)*(1-r)*a*(1-t)*(v*L1+(1-v)*t*b)*L2 +
             (1-r)*(1-a)*(1-r)*(1-a)*(r*L1+(1-r)*a*t*b)*L2;

    // 0100

    value[4] =
             r*(1-L1)*L1*(1-L1)*(1-L2) +
             (1-r)*a*(1-t)*(v*L1+(1-v)*t*b)*(1-L1)*(1-L2) +
             (1-r)*(1-a)*(r*L1+(1-r)*a*t*b)*(1-L1)*(1-L2) +
             (1-r)*a*(1-t)*(1-v)*t*(1-b)*v*(1-L1)*(1-L2) +
             (1-r)*(1-a)*(1-r)*a*t*(1-b)*v*(1-L1)*(1-L2) +
             (1-r)*a*(1-t)*(1-v)*t*(1-b)*(1-v)*(1-t) +
             (1-r)*(1-a)*(1-r)*a*t*(1-b)*(1-v)*(1-t);

    // 0101

    value[5] =
             r*(1-L1)*L1*(1-L1)*L2 +
             (1-r)*a*(1-t)*(v*L1+(1-v)*t*b)*(1-L1)*L2 +
             (1-r)*(1-a)*(r*L1+(1-r)*a*t*b)*(1-L1)*L2 +
             (1-r)*a*(1-t)*(1-v)*t*(1-b)*(v)*(1-L1)*L2 +
             (1-r)*(1-a)*(1-r)*a*t*(1-b)*(v)*(1-L1)*L2;

    // 0110

    value[6] =
             r*(1-L1)*L1*L1*(1-L2) +
             (1-r)*a*(1-t)*(v*L1+(1-v)*t*b)*L1*(1-L2) +
             (1-r)*(1-a)*(r*L1+(1-r)*a*t*b)*L1*(1-L2) +
             (1-r)*a*(1-t)*(1-v)*t*(1-b)*(v*L1+(1-v)*t*b)*(1-L2) +
             (1-r)*(1-a)*(1-r)*a*t*(1-b)*(v*L1+(1-v)*t*b)*(1-L2) +
             (1-r)*a*(1-t)*(1-v)*t*(1-b)*(1-v)*t*(1-b) +
             (1-r)*(1-a)*(1-r)*a*t*(1-b)*(1-v)*t*(1-b);

    // 0111

    value[7] =
             r*(1-L1)*L1*L1*L2 +
             (1-r)*a*(1-t)*(v*L1+(1-v)*t*b)*L1*L2 +
             (1-r)*(1-a)*(r*L1+(1-r)*a*t*b)*L1*L2 +
             (1-r)*a*(1-t)*(1-v)*t*(1-b)*(v*L1+(1-v)*t*b)*L2 +
             (1-r)*(1-a)*(1-r)*a*t*(1-b)*(v*L1+(1-v)*t*b)*L2;

    // 1000

    value[8] =
                (r*L1+(1-r)*a*t*b)*(1-L1)*(1-L1)*(1-L2) +
                (1-r)*a*t*(1-b)*v*(1-L1)*(1-L1)*(1-L2) +        
                (1-r)*a*t*(1-b)*(1-v)*(1-t)*v*(1-L1)*(1-L2) +
                (1-r)*a*t*(1-b)*(1-v)*(1-t)*(1-v)*(1-t);

    // 1001

    value[9] =
                (r*L1+(1-r)*a*t*b)*(1-L1)*(1-L1)*L2 +
                (1-r)*a*t*(1-b)*v*(1-L1)*(1-L1)*L2 +
                (1-r)*a*t*(1-b)*(1-v)*(1-t)*v*(1-L1)*L2;

    // 1010

    value[10] =

                (r*L1+(1-r)*a*t*b)*(1-L1)*L1*(1-L2) +
                (1-r)*a*t*(1-b)*v*(1-L1)*L1*(1-L2) +
                (1-r)*a*t*(1-b)*(1-v)*(1-t)*(v*L1+(1-v)*t*b)*(1-L2) +
                (1-r)*a*t*(1-b)*(1-v)*(1-t)*(1-v)*t*(1-b);

    // 1011

    value[11] =
                (r*L1+(1-r)*a*t*b)*(1-L1)*L1*L2 +
                (1-r)*a*t*(1-b)*v*(1-L1)*L1*L2 +
                (1-r)*a*t*(1-b)*(1-v)*(1-t)*(v*L1+(1-v)*t*b)*L2;

    // 1100

    value[12] =
                (r*L1+(1-r)*a*t*b)*L1*(1-L1)*(1-L2) +
                (1-r)*a*t*(1-b)*(v*L1+(1-v)*t*b)*(1-L1)*(1-L2) +
                (1-r)*a*t*(1-b)*(1-v)*t*(1-b)*v*(1-L1)*(1-L2) +
                (1-r)*a*t*(1-b)*(1-v)*t*(1-b)*(1-v)*(1-t);

    // 1101

    value[13] =
                (r*L1+(1-r)*a*t*b)*L1*(1-L1)*L2 +
                (1-r)*a*t*(1-b)*(v*L1+(1-v)*t*b)*(1-L1)*L2 +
                (1-r)*a*t*(1-b)*(1-v)*t*(1-b)*v*(1-L1)*L2;

    // 1110

    value[14] =
                (r*L1+(1-r)*a*t*b)*L1*L1*(1-L2) +
                (1-r)*a*t*(1-b)*(v*L1+(1-v)*t*b)*L1*(1-L2) +
                (1-r)*a*t*(1-b)*(1-v)*t*(1-b)*(v*L1+(1-v)*t*b)*(1-L2) +
                (1-r)*a*t*(1-b)*(1-v)*t*(1-b)*(1-v)*t*(1-b);

    // 1111

    value[15] =
                (r*L1+(1-r)*a*t*b)*L1*L1*L2 +
                (1-r)*a*t*(1-b)*(v*L1+(1-v)*t*b)*L1*L2 +
                (1-r)*a*t*(1-b)*(1-v)*t*(1-b)*(v*L1+(1-v)*t*b)*L2;

        // int N = lengths[0];
        // std::vector<int> index (N, 0);
        // for (unsigned int i = 0; i < N; i++) {
        //     index[i] = (int)s[i]-1;
        // }

        // int evA = 0;
        // int evB = 0;
        // for (unsigned int i = 0; i < k; i++) {
        //     if (stimA[index[i]] == 1) {
        //         evA++;
        //     }

        //     if (stimB[index[i]] == 1) {
        //         evB++;
        //     }
        // }

        // if (evA > evB) {
        //     value[0] = 1;
        // } else if (evB > evA) {
        //     value[0] = 0;
        // } else {
        //     value[0] = 0.5;
        // }

        // value[1] = evA - evB;
    }

    unsigned int BATCHELDER::length (vector<unsigned int> const &parlengths,
                                vector<double const *> const &parvalues) const
    {
        return 16;
    }

    // bool TALLYk::isDiscreteValued(vector<bool> const &mask) const
    // {
    //     return allTrue(mask);
    // }
}}
