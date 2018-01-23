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

void setInitConstraintsCsr(LpData* lpd, int w, int h, double* data, int dataLen, int* inds,
                           int indsLen, int* indptr, int indptrLen, double* rhs, int rhsLen)
{
    lpd->setInitConstraintsCsr(w, h, data, dataLen, inds, indsLen, indptr, indptrLen, rhs, rhsLen);
}

void setOutputConstraintsCsr(LpData* lpd, int w, int h, double* data, int dataLen, int* inds,
                             int indsLen, int* indptr, int indptrLen, double* rhs, int rhsLen)
{
    lpd->setOutputConstraintsCsr(w, h, data, dataLen, inds, indsLen, indptr, indptrLen, rhs,
                                 rhsLen);
}

void setNoOutputConstraints(LpData* lpd)
{
    lpd->setNoOutputConstraints();
}

int minimize(LpData* lpd, double* direction, int dirLen, double* result, int resLen)
{
    return lpd->minimize(direction, dirLen, result, resLen);
}

void printLp(LpData* lpd)
{
    lpd->printLp();
}

void resetLp(LpData* lpd)
{
    lpd->resetLp();
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

void setInitConstraintsCsr(void* lpdata, int w, int h, double* data, int dataLen, int* inds,
                           int indsLen, int* indptr, int indptrLen, double* rhs, int rhsLen)
{
    hylaa_glpk::setInitConstraintsCsr((LpData*)lpdata, w, h, data, dataLen, inds, indsLen, indptr,
                                      indptrLen, rhs, rhsLen);
}

void setOutputConstraintsCsr(void* lpdata, int w, int h, double* data, int dataLen, int* inds,
                             int indsLen, int* indptr, int indptrLen, double* rhs, int rhsLen)
{
    hylaa_glpk::setOutputConstraintsCsr((LpData*)lpdata, w, h, data, dataLen, inds, indsLen, indptr,
                                        indptrLen, rhs, rhsLen);
}

void setNoOutputConstraints(void* lpdata)
{
    hylaa_glpk::setNoOutputConstraints((LpData*)lpdata);
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

void resetLp(void* lpdata)
{
    hylaa_glpk::resetLp((LpData*)lpdata);
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
