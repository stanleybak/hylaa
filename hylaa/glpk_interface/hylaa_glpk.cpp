// Stanley Bak
// Hylaa GLPK interface
// Nov 2016

#include <stdio.h>
#include <stdlib.h>
#include "hylaa_glpk.h"
#include "hylaa_glpk_tests.h"

GlobalLpData global;

namespace hylaa_glpk
{
LpData* initLp(int numOutputVars, int numInitVars, int numInputs)
{
    LpData* data = new (std::nothrow) LpData(numOutputVars, numInitVars, numInputs);

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

void updateBasisMatrix(LpData* lpd, double* matrix, int w, int h)
{
    return lpd->updateBasisMatrix(matrix, w, h);
}

void addInputEffectsMatrix(LpData* lpd, double* matrix, int w, int h)
{
    return lpd->addInputEffectsMatrix(matrix, w, h);
}

void setInitConstraints(LpData* lpd, double* matrix, int w, int h, double* rhs, int rhsLen)
{
    lpd->setInitConstraints(matrix, w, h, rhs, rhsLen);
}

void setInputConstraintsCsr(LpData* lpd, double* data, int dataLen, int* indices, int indicesLen,
                            int* indptr, int indptrLen, double* rhs, int rhsLen)
{
    lpd->setInputConstraintsCsr(data, dataLen, indices, indicesLen, indptr, indptrLen, rhs, rhsLen);
}

void setCurTimeConstraintBounds(LpData* lpd, double* rhs, int rhsLen)
{
    lpd->setCurTimeConstraintBounds(rhs, rhsLen);
}

void commitCurTimeRows(LpData* lpd)
{
    lpd->commitCurTimeRows();
}

int minimize(LpData* lpd, double* direction, int dirLen, double* result, int resLen)
{
    return lpd->minimize(direction, dirLen, result, resLen);
}

void printLp(LpData* lpd)
{
    lpd->printLp();
}

void test()
{
    run_hylaa_glpk_tests();
}

}  // namespace "hylaa"

/////////////////////////
// Interface functions //
/////////////////////////
extern "C" {
// returns a LpData* instance
void* initLp(int numOutputVars, int numInitVars, int numInputs)
{
    return (void*)hylaa_glpk::initLp(numOutputVars, numInitVars, numInputs);
}

// frees a LpData* instance
void delLp(void* lpdata)
{
    hylaa_glpk::delLp((LpData*)lpdata);
}

void updateBasisMatrix(void* lpdata, double* matrix, int w, int h)
{
    hylaa_glpk::updateBasisMatrix((LpData*)lpdata, matrix, w, h);
}

void addInputEffectsMatrix(void* lpdata, double* matrix, int w, int h)
{
    hylaa_glpk::addInputEffectsMatrix((LpData*)lpdata, matrix, w, h);
}

void setInitConstraints(void* lpdata, double* matrix, int w, int h, double* rhs, int rhsLen)
{
    hylaa_glpk::setInitConstraints((LpData*)lpdata, matrix, w, h, rhs, rhsLen);
}

void setInputConstraintsCsr(void* lpdata, double* data, int dataLen, int* indices, int indicesLen,
                            int* indptr, int indptrLen, double* rhs, int rhsLen)
{
    hylaa_glpk::setInputConstraintsCsr((LpData*)lpdata, data, dataLen, indices, indicesLen, indptr,
                                       indptrLen, rhs, rhsLen);
}

void setCurTimeConstraintBounds(void* lpdata, double* rhs, int rhsLen)
{
    hylaa_glpk::setCurTimeConstraintBounds((LpData*)lpdata, rhs, rhsLen);
}

void commitCurTimeRows(void* lpdata)
{
    hylaa_glpk::commitCurTimeRows((LpData*)lpdata);
}

int minimize(void* lpdata, double* direction, int dirLen, double* result, int resLen)
{
    return hylaa_glpk::minimize((LpData*)lpdata, direction, dirLen, result, resLen);
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
    hylaa_glpk::printLp((LpData*)lpdata);
}

void test()
{
    hylaa_glpk::test();
}
}  // extern "C"

int main()
{
    test();
    printf("hylaa_glpk Tests Passed!\n");

    return 0;
}
