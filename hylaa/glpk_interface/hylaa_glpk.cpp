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

void setStandardConstraintValues(LpData* lpd, double* vals, int len)
{
    lpd->setStandardConstraintValues(vals, len);
}

void setBasisConstraintValues(LpData* lpd, double* vals, int len)
{
    lpd->setBasisConstraintValues(vals, len);
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

void setBasisConstraintValues(void* lpdata, double* vals, int len)
{
    hylaa::setBasisConstraintValues((LpData*)lpdata, vals, len);
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
