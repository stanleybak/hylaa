// Stanley Bak
// Hylaa GLPK interface
// Nov 2016

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "hylaa_glpk.h"

GlobalLpData global;

namespace hylaa
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

int updateBasisMatrix(LpData* lpd, double* matrix, int w, int h)
{
    return lpd->updateBasisMatrix(matrix, w, h);
}

void addInputStar(LpData* lpd, double* aMatrixT, int aWidth, int aHeight, double* bVec, int bLen,
                  double* basisMatrix, int bmWidth, int bmHeight)
{
    lpd->addInputStar(aMatrixT, aWidth, aHeight, bVec, bLen, basisMatrix, bmWidth, bmHeight);
}

void addBasisConstraint(LpData* lpd, double* aVec, int aVecLen, double bVal)
{
    lpd->addBasisConstraint(aVec, aVecLen, bVal);
}

void addStandardConstraint(LpData* lpd, double* aVec, int aVecLen, double bVal)
{
    lpd->addStandardConstraint(aVec, aVecLen, bVal);
}

int minimize(LpData* lpd, double* direction, int dirLen, double* result, int resLen)
{
    return lpd->minimize(direction, dirLen, result, resLen);
}

void printLp(LpData* lpd)
{
    lpd->printLp();
}

int getColStatuses(LpData* lpd, char* store, int storeLen)
{
    return lpd->getColStatuses(store, storeLen);
}

int getRowStatuses(LpData* lpd, char* store, int storeLen)
{
    return lpd->getRowStatuses(store, storeLen);
}

void setLastInputStatuses(LpData* lpd, char* rowStats, int rLen, char* colStats, int cLen)
{
    lpd->setLastInputStatuses(rowStats, rLen, colStats, cLen);
}

void setStandardBasisStatuses(LpData* lpd, char* rowStats, int rLen, char* colStats, int cLen)
{
    lpd->setStandardBasisStatuses(rowStats, rLen, colStats, cLen);
}

int totalIterations()
{
    return global.iterations;
}

int totalOptimizations()
{
    return global.optimizations;
}

void test1d()
{
    LpData* lpd = initLp(1, 1);

    double basis[] = {1};
    updateBasisMatrix(lpd, basis, 1, 1);

    double a1[] = {1};
    double b1 = 2;
    addBasisConstraint(lpd, a1, 1, b1);

    double a2[] = {-1};
    double b2 = -1;
    addBasisConstraint(lpd, a2, 1, b2);

    double result[1] = {0};
    double direction[] = {-1};

    if (minimize(lpd, direction, 1, result, 1))
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
    updateBasisMatrix(lpd, basis, 2, 2);

    double a1[] = {1, 0};
    double b1 = -4;
    addBasisConstraint(lpd, a1, 2, b1);

    double a2[] = {-1, 0};
    double b2 = 5;
    addBasisConstraint(lpd, a2, 2, b2);

    double a3[] = {0, 1};
    double b3 = 1;
    addBasisConstraint(lpd, a3, 2, b3);

    double a4[] = {0, -1};
    double b4 = 0;
    addBasisConstraint(lpd, a4, 2, b4);

    double result[2] = {0, 0};
    double direction[] = {-1, -1};

    if (minimize(lpd, direction, 2, result, 2))
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
    updateBasisMatrix(lpd, basis, 2, 2);

    if (minimize(lpd, direction, 2, result, 2))
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
    updateBasisMatrix(lpd, basis, 2, 2);

    direction[0] = 1;
    direction[1] = 1;

    if (minimize(lpd, direction, 2, result, 2))
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
    updateBasisMatrix(lpd, basis, 2, 2);

    double a1[] = {1, 0};
    double b1 = -4;
    addBasisConstraint(lpd, a1, 2, b1);

    double a2[] = {-1, 0};
    double b2 = 5;
    addBasisConstraint(lpd, a2, 2, b2);

    double a3[] = {0, 1};
    double b3 = 1;
    addBasisConstraint(lpd, a3, 2, b3);

    double a4[] = {0, -1};
    double b4 = 0;
    addBasisConstraint(lpd, a4, 2, b4);

    double result[2] = {0, 0};
    double direction[] = {-1, -1};

    if (minimize(lpd, direction, 2, result, 2))
    {
        printf("first call to minimize failed in 2-d-input self test\n");
        exit(1);
    }

    double maxX = result[0];
    double maxY = result[1];

    // also do other direction
    direction[0] = direction[1] = 1;

    if (minimize(lpd, direction, 2, result, 2))
    {
        printf("second call to minimize failed in 2-d-input self test\n");
        exit(1);
    }

    double minX = result[0];
    double minY = result[1];

    // add one input with range [-1, 1] for both variables
    double inputBasis[] = {1.0, 0.0, /* */ 0.0, 1.0};
    double inputConstraintA[] = {1.0, 0.0, /* */ -1.0, 0.0, /* */ 0.0, 1.0, /* */ 0.0, -1.0};
    double inputConstraintB[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

    lpd->addInputStar(inputConstraintA, 2, 4, inputConstraintB, 4, inputBasis, 2, 2);

    direction[0] = direction[1] = -1;

    if (minimize(lpd, direction, 2, result, 2))
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

    if (minimize(lpd, direction, 2, result, 2))
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

    int startIter = totalIterations();
    lpd->addInputStar(inputConstraintA, 2, 4, inputConstraintB, 4, inputBasis, 2, 2);

    if (minimize(lpd, direction, 2, result, 2))
    {
        printf("fifth call to minimize failed in 2-d-input self test\n");
        exit(1);
    }

    int diffIter = totalIterations() - startIter;

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
    startIter = totalIterations();

    if (minimize(lpd, direction, 2, result, 2))
    {
        printf("sixth call to minimize failed in 2-d-input self test\n");
        exit(1);
    }

    diffIter = totalIterations() - startIter;

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
    updateBasisMatrix(lpd, basis, 2, 2);

    double a1[] = {1, 0};
    double b1 = -4;
    addBasisConstraint(lpd, a1, 2, b1);

    double a2[] = {-1, 0};
    double b2 = 5;
    addBasisConstraint(lpd, a2, 2, b2);

    double a3[] = {0, 1};
    double b3 = 1;
    addBasisConstraint(lpd, a3, 2, b3);

    double a4[] = {0, -1};
    double b4 = 0;
    addBasisConstraint(lpd, a4, 2, b4);

    double result[2] = {0, 0};
    double direction[] = {0, 0};

    // add standard constraint x <= -6.5
    double aStandard[] = {1, 0};
    double bStandard = -6.5;

    addStandardConstraint(lpd, aStandard, 2, bStandard);

    if (minimize(lpd, direction, 2, result, 2) != 1)
    {
        printf("first call to minimize was not unsat in 2-d-unsat self test\n");
        exit(1);
    }

    // add one input with range [-1, 1] for both variables
    double inputBasis[] = {1.0, 0.0, /* */ 0.0, 1.0};
    double inputConstraintA[] = {1.0, 0.0, /* */ -1.0, 0.0, /* */ 0.0, 1.0, /* */ 0.0, -1.0};
    double inputConstraintB[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

    lpd->addInputStar(inputConstraintA, 2, 4, inputConstraintB, 4, inputBasis, 2, 2);

    if (minimize(lpd, direction, 2, result, 2) != 1)
    {
        printf("second call to minimize was not unsat in 2-d-unsat self test\n");
        printf("(result = %f %f)\n", result[0], result[1]);
        exit(1);
    }

    // ok, finally, try adding a second identical input and doing an identical optimization
    // this should take zero iterations, since we want the result (column/row status)
    // from the first input to be copied

    int startIter = totalIterations();
    lpd->addInputStar(inputConstraintA, 2, 4, inputConstraintB, 4, inputBasis, 2, 2);

    if (minimize(lpd, direction, 2, result, 2))
    {
        printf("minimize-2d with two inputs failed in 2d-sat self test\n");
        exit(1);
    }

    int diffIter = totalIterations() - startIter;

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

void setStandardConstraintValues(LpData* lpd, double* vals, int len)
{
    lpd->setStandardConstraintValues(vals, len);
}

void test()
{
    test1d();
    test2d();
    test2dInputs();
    test2dUnsat();
}
}  // namespace "hylaa"

/////////////////////////
// Interface functions //
/////////////////////////
extern "C" {
// returns a LpData* instance
void* initLp(int numStandardVars, int numBasisVars)
{
    return (void*)hylaa::initLp(numStandardVars, numBasisVars);
}

// frees a LpData* instance
void delLp(void* lpdata)
{
    hylaa::delLp((LpData*)lpdata);
}

int updateBasisMatrix(void* lpdata, double* matrix, int w, int h)
{
    return hylaa::updateBasisMatrix((LpData*)lpdata, matrix, w, h);
}

void addInputStar(void* lpdata, double* aMatrixT, int aWidth, int aHeight, double* bVec, int bLen,
                  double* basisMatrix, int bmWidth, int bmHeight)
{
    hylaa::addInputStar((LpData*)lpdata, aMatrixT, aWidth, aHeight, bVec, bLen, basisMatrix,
                        bmWidth, bmHeight);
}

void addBasisConstraint(void* lpdata, double* aVec, int aVecLen, double bVal)
{
    hylaa::addBasisConstraint((LpData*)lpdata, aVec, aVecLen, bVal);
}

void addStandardConstraint(void* lpdata, double* aVec, int aVecLen, double bVal)
{
    hylaa::addStandardConstraint((LpData*)lpdata, aVec, aVecLen, bVal);
}

int minimize(void* lpdata, double* direction, int dirLen, double* result, int resLen)
{
    return hylaa::minimize((LpData*)lpdata, direction, dirLen, result, resLen);
}

int totalIterations()
{
    return hylaa::totalIterations();
}

int totalOptimizations()
{
    return hylaa::totalOptimizations();
}

void printLp(void* lpdata)
{
    hylaa::printLp((LpData*)lpdata);
}

int getColStatuses(void* lpdata, char* store, int storeLen)
{
    return hylaa::getColStatuses((LpData*)lpdata, store, storeLen);
}

int getRowStatuses(void* lpdata, char* store, int storeLen)
{
    return hylaa::getRowStatuses((LpData*)lpdata, store, storeLen);
}

void setLastInputStatuses(void* lpdata, char* rowStats, int rLen, char* colStats, int cLen)
{
    hylaa::setLastInputStatuses((LpData*)lpdata, rowStats, rLen, colStats, cLen);
}

void setStandardBasisStatuses(void* lpdata, char* rowStats, int rLen, char* colStats, int cLen)
{
    hylaa::setStandardBasisStatuses((LpData*)lpdata, rowStats, rLen, colStats, cLen);
}

void setStandardConstraintValues(void* lpdata, double* vals, int len)
{
    hylaa::setStandardConstraintValues((LpData*)lpdata, vals, len);
}

void test()
{
    hylaa::test();
}
}  // extern "C"

/*int main()
{
    test();

    return 0;
}*/
