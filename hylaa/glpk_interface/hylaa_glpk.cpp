// Stanley Bak
// Hylaa GLPK interface
// Nov 2016

#include <stdio.h>
#include <stdlib.h>
#include "hylaa_glpk.h"
#include "hylaa_glpk_tests.h"

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

int updateTimeElapseMatrix(LpData* lpd, double* matrix, int w, int h)
{
    return lpd->updateTimeElapseMatrix(matrix, w, h);
}

void addInitConstraint(LpData* lpd, double* aVec, int aVecLen, double bVal)
{
    lpd->addInitConstraint(aVec, aVecLen, bVal);
}

void addCurTimeConstraint(LpData* lpd, double* aVec, int aVecLen, double bVal)
{
    lpd->addCurTimeConstraint(aVec, aVecLen, bVal);
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

void test()
{
    hylaa_glpk_unit_test();
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

int updateTimeElapseMatrix(void* lpdata, double* matrix, int w, int h)
{
    return hylaa::updateTimeElapseMatrix((LpData*)lpdata, matrix, w, h);
}

void addInitConstraint(void* lpdata, double* aVec, int aVecLen, double bVal)
{
    hylaa::addInitConstraint((LpData*)lpdata, aVec, aVecLen, bVal);
}

void addCurTimeConstraint(void* lpdata, double* aVec, int aVecLen, double bVal)
{
    hylaa::addCurTimeConstraint((LpData*)lpdata, aVec, aVecLen, bVal);
}

int minimize(void* lpdata, double* direction, int dirLen, double* result, int resLen)
{
    return hylaa::minimize((LpData*)lpdata, direction, dirLen, result, resLen);
}

int totalIterations()
{
    return global.iterations;
}

int totalOptimizations()
{
    return global.optimizations;
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

void test()
{
    hylaa::test();
}
}  // extern "C"

int main()
{
    test();

    return 0;
}
