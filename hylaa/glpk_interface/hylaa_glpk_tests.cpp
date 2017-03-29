// Stanley Bak
// Hylaa GLPK unit test implementation

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "hylaa_glpk.h"

namespace hylaa_glpk_tests
{
    
LpData* initLp(int numStandardVars, int numBasisVars)
{
    LpData* data = new LpData(numStandardVars, numBasisVars);

    if (data == nullptr)
    {
        printf("Fatal Error: LpData memory allocation failed in %s.\n", __FILE__);
        exit(1);
    }

    return data;
}

void delLp(LpData* ptr)
{
    delete ptr;
}

void test1d()
{
    LpData* lpd = initLp(1, 1);

    double basis[] = {1};
    lpd->updateBasisMatrix(basis, 1, 1);

    double a1[] = {1};
    double b1 = 2;
    lpd->addBasisConstraint(a1, 1, b1);

    double a2[] = {-1};
    double b2 = -1;
    lpd->addBasisConstraint(a2, 1, b2);

    double result[1] = {0};
    double direction[] = {-1};

    if (lpd->minimize(direction, 1, result, 1))
    {
        printf("call to minimize failed in 1-d self test\n");
        exit(1);
    }

    // printf("1-d result = %f\n", result[0]);

    if (fabs(result[0] - (2)) > 1e-6)
    {
        printf("lp self-test-1d failed result: %f; expected: 2\n", result[0]);
        exit(1);
    }

    delLp(lpd);
}

void test2d()
{
    LpData* lpd = initLp(2, 2);

    double basis[] = {1, 0, /* */ 0, 1};
    lpd->updateBasisMatrix(basis, 2, 2);

    double a1[] = {1, 0};
    double b1 = -4;
    lpd->addBasisConstraint(a1, 2, b1);

    double a2[] = {-1, 0};
    double b2 = 5;
    lpd->addBasisConstraint(a2, 2, b2);

    double a3[] = {0, 1};
    double b3 = 1;
    lpd->addBasisConstraint(a3, 2, b3);

    double a4[] = {0, -1};
    double b4 = 0;
    lpd->addBasisConstraint(a4, 2, b4);

    double result[2] = {0, 0};
    double direction[] = {-1, -1};

    if (lpd->minimize(direction, 2, result, 2))
    {
        printf("first call to minimize failed in 2-d self test\n");
        exit(1);
    }

    if (fabs(result[0] - (-4)) > 1e-6 || fabs(result[1] - (1)) > 1e-6)
    {
        printf("lp self-test-2d failed #1 result = (%f, %f); expected (-4, 1)\n", result[0],
               result[1]);
        exit(1);
    }

    int before = global.iterations;

    // make minor change to basis matrix
    basis[0] = 1.1;
    lpd->updateBasisMatrix(basis, 2, 2);

    if (lpd->minimize(direction, 2, result, 2))
    {
        printf("second call to minimize failed in 2-d self test\n");
        exit(1);
    }

    int after = global.iterations;

    if (before != after)
    {
        printf("lp-self-test failed: re-solving lp required more iterations.\n");
        exit(1);
    }

    // revert basis back to orthonormal
    basis[0] = 1.0;
    lpd->updateBasisMatrix(basis, 2, 2);

    direction[0] = 1;
    direction[1] = 1;

    if (lpd->minimize(direction, 2, result, 2))
    {
        printf("third call to minimize failed in 2-d self test\n");
        exit(1);
    }

    // printf("2-d result = %f, %f\n", result[0], result[1]);

    if (fabs(result[0] - (-5)) > 1e-6 || fabs(result[1] - (0)) > 1e-6)
    {
        printf("lp self-test-2d failed #2 result = (%f, %f); expected (-5, 0)\n", result[0],
               result[1]);
        exit(1);
    }

    delLp(lpd);
}

void test2dInputs()
{
    LpData* lpd = initLp(2, 2);

    double basis[] = {1.1, 0.1, /* */ 0.1, 1.1};
    lpd->updateBasisMatrix(basis, 2, 2);

    double a1[] = {1, 0};
    double b1 = -4;
    lpd->addBasisConstraint(a1, 2, b1);

    double a2[] = {-1, 0};
    double b2 = 5;
    lpd->addBasisConstraint(a2, 2, b2);

    double a3[] = {0, 1};
    double b3 = 1;
    lpd->addBasisConstraint(a3, 2, b3);

    double a4[] = {0, -1};
    double b4 = 0;
    lpd->addBasisConstraint(a4, 2, b4);

    double result[2] = {0, 0};
    double direction[] = {-1, -1};

    if (lpd->minimize(direction, 2, result, 2))
    {
        printf("first call to minimize failed in 2-d-input self test\n");
        exit(1);
    }

    double maxX = result[0];
    double maxY = result[1];

    // also do other direction
    direction[0] = direction[1] = 1;

    if (lpd->minimize(direction, 2, result, 2))
    {
        printf("second call to minimize failed in 2-d-input self test\n");
        exit(1);
    }

    double minX = result[0];
    double minY = result[1];

    // add one input with range [-1, 1] for both variables
    double inputBasis[] = {1.0, 0.0, /* */ 0.0, 1.0};
    double inputConstraintATranspose[] = {1.0, -1.0, 0.0, 0.0, /* */ 0.0, 0.0, 1.0, -1.0};
    double inputConstraintB[] = {1.0, 1.0, 1.0, 1.0};

    int bLen = sizeof(inputConstraintB) / sizeof(inputConstraintB[0]);
    lpd->addInputStar(inputConstraintATranspose, 4, 2, inputConstraintB, bLen, inputBasis, 2, 2);

    direction[0] = direction[1] = -1;

    if (lpd->minimize(direction, 2, result, 2))
    {
        printf("third call to minimize failed in 2-d-input self test\n");
        exit(1);
    }

    double newMaxX = result[0];
    double newMaxY = result[1];

    // minimize negative = add
    if (fabs(newMaxX - (maxX + 1.0)) > 1e-6 || fabs(newMaxY - (maxY + 1.0)) > 1e-6)
    {
        printf("lp self-test-2d-inputs failed #3 result = (%f, %f); expected (%f, %f)\n", newMaxX,
               newMaxY, maxX - 1, maxY - 1);
        exit(1);
    }

    // also do other direction
    direction[0] = direction[1] = 1;

    if (lpd->minimize(direction, 2, result, 2))
    {
        printf("fourth call to minimize failed in 2-d-input self test\n");
        exit(1);
    }

    double newMinX = result[0];
    double newMinY = result[1];

    if (fabs(newMinX - (minX - 1.0)) > 1e-6 || fabs(newMinY - (minY - 1.0)) > 1e-6)
    {
        printf("lp self-test-2d-inputs failed #4 result = (%f, %f); expected (%f, %f)\n", newMinX,
               newMinY, minX - 1, minY - 1);
        exit(1);
    }

    // ok, finally, try adding a second identical input and doing an identical optimization
    // this should take zero iterations, since we want the result (column/row status)
    // from the first input to be copied

    int startIter = global.iterations;
    lpd->addInputStar(inputConstraintATranspose, 4, 2, inputConstraintB, bLen, inputBasis, 2, 2);

    if (lpd->minimize(direction, 2, result, 2))
    {
        printf("fifth call to minimize failed in 2-d-input self test\n");
        exit(1);
    }

    int diffIter = global.iterations - startIter;

    if (diffIter != 0)
    {
        printf(
            "lp self-test-2d-inputs failed. Expected zero LP iterations after adding identical "
            "input, instead got: %d\n",
            diffIter);
        exit(1);
    }

    // make sure if we started from scratch we would have lots of iterations
    lpd->resetLp();
    startIter = global.iterations;

    if (lpd->minimize(direction, 2, result, 2))
    {
        printf("sixth call to minimize failed in 2-d-input self test\n");
        exit(1);
    }

    diffIter = global.iterations - startIter;

    if (diffIter == 0)
    {
        printf("lp self-test-2d-inputs failed. Expected many LP iterations after lp reset\n");
        exit(1);
    }

    delLp(lpd);
}

void test2dUnsat()
{
    LpData* lpd = initLp(2, 2);

    double basis[] = {1.01, 0.01, /* */ 0.01, 1.01};
    lpd->updateBasisMatrix(basis, 2, 2);

    double a1[] = {1, 0};
    double b1 = -4;
    lpd->addBasisConstraint(a1, 2, b1);

    double a2[] = {-1, 0};
    double b2 = 5;
    lpd->addBasisConstraint(a2, 2, b2);

    double a3[] = {0, 1};
    double b3 = 1;
    lpd->addBasisConstraint(a3, 2, b3);

    double a4[] = {0, -1};
    double b4 = 0;
    lpd->addBasisConstraint(a4, 2, b4);

    double result[2] = {0, 0};
    double direction[] = {0, 0};

    // add standard constraint x <= -6.5
    double aStandard[] = {1, 0};
    double bStandard = -6.5;

    lpd->addStandardConstraint(aStandard, 2, bStandard);

    if (lpd->minimize(direction, 2, result, 2) != 1)
    {
        printf("first call to minimize was not unsat in 2-d-unsat self test\n");
        exit(1);
    }

    // add one input with range [-1, 1] for both variables
    double inputBasis[] = {1.0, 0.0, /* */ 0.0, 1.0};
    double inputConstraintATranspose[] = {1.0, -1.0, 0.0, 0.0, /* */ 0.0, 0.0, 1.0, -1.0};
    double inputConstraintB[] = {1.0, 1.0, 1.0, 1.0};

    int bLen = sizeof(inputConstraintB) / sizeof(inputConstraintB[0]);
    lpd->addInputStar(inputConstraintATranspose, 4, 2, inputConstraintB, bLen, inputBasis, 2, 2);

    if (lpd->minimize(direction, 2, result, 2) != 1)
    {
        printf("second call to minimize was not unsat in 2-d-unsat self test\n");
        printf("(result = %f %f)\n", result[0], result[1]);
        exit(1);
    }

    // ok, finally, try adding a second identical input and doing an identical optimization
    // this should take zero iterations, since we want the result (column/row status)
    // from the first input to be copied

    int startIter = global.iterations;
    lpd->addInputStar(inputConstraintATranspose, 4, 2, inputConstraintB, bLen, inputBasis, 2, 2);

    if (lpd->minimize(direction, 2, result, 2))
    {
        printf("minimize-2d with two inputs failed in 2d-sat self test\n");
        exit(1);
    }

    int diffIter = global.iterations - startIter;

    if (diffIter != 0)
    {
        printf(
            "lp self-test-2d-unsat failed. Expected zero LP iterations after adding identical "
            "second input, instead got: %d\n",
            diffIter);
        exit(1);
    }

    delLp(lpd);
}

} //  end namespace hylaa_glpk_tests


void hylaa_glpk_unit_test()
{
    hylaa_glpk_tests::test1d();
    hylaa_glpk_tests::test2d();
    hylaa_glpk_tests::test2dInputs();
    hylaa_glpk_tests::test2dUnsat();
}
