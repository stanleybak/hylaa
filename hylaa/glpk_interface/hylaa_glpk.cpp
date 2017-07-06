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
LpData* initLp(int numCurTimeVars, int numInitVars, int numInputs)
{
    LpData* data = new (std::nothrow) LpData(numCurTimeVars, numInitVars, numInputs);

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

void updateTimeElapseMatrix(LpData* lpd, double* matrix, int w, int h)
{
    return lpd->updateTimeElapseMatrix(matrix, w, h);
}

void addInputEffectsMatrix(LpData* lpd, double* matrix, int w, int h)
{
    return lpd->addInputEffectsMatrix(matrix, w, h);
}

void setInitConstraintsCsr(LpData* lpd, double* data, int dataLen, int* indices, int indicesLen,
                           int* indptr, int indptrLen, double* rhs, int rhsLen)
{
    lpd->setInitConstraintsCsr(data, dataLen, indices, indicesLen, indptr, indptrLen, rhs, rhsLen);
}

void setInputConstraintsCsc(LpData* lpd, double* data, int dataLen, int* indices, int indicesLen,
                            int* indptr, int indptrLen, double* rhs, int rhsLen)
{
    lpd->setInputConstraintsCsc(data, dataLen, indices, indicesLen, indptr, indptrLen, rhs, rhsLen);
}

void setCurTimeConstraints(LpData* lpd, double* data, int dataLen, int* indices, int indicesLen,
                           int* indptr, int indptrLen, double* rhs, int rhsLen)
{
    lpd->setCurTimeConstraints(data, dataLen, indices, indicesLen, indptr, indptrLen, rhs, rhsLen);
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
    hylaa_glpk_unit_test();
}

}  // namespace "hylaa"

/////////////////////////
// Interface functions //
/////////////////////////
extern "C" {
// returns a LpData* instance
void* initLp(int numCurTimeVars, int numInitVars, int numInputs)
{
    return (void*)hylaa::initLp(numCurTimeVars, numInitVars, numInputs);
}

// frees a LpData* instance
void delLp(void* lpdata)
{
    hylaa::delLp((LpData*)lpdata);
}

void updateTimeElapseMatrix(void* lpdata, double* matrix, int w, int h)
{
    return hylaa::updateTimeElapseMatrix((LpData*)lpdata, matrix, w, h);
}

void addInputEffectsMatrix(void* lpdata, double* matrix, int w, int h)
{
    return hylaa::addInputEffectsMatrix((LpData*)lpdata, matrix, w, h);
}

void setInitConstraintsCsr(void* lpdata, double* data, int dataLen, int* indices, int indicesLen,
                           int* indptr, int indptrLen, double* rhs, int rhsLen)
{
    hylaa::setInitConstraintsCsr((LpData*)lpdata, data, dataLen, indices, indicesLen, indptr,
                                 indptrLen, rhs, rhsLen);
}

void setInputConstraintsCsc(void* lpdata, double* data, int dataLen, int* indices, int indicesLen,
                            int* indptr, int indptrLen, double* rhs, int rhsLen)
{
    hylaa::setInputConstraintsCsc((LpData*)lpdata, data, dataLen, indices, indicesLen, indptr,
                                  indptrLen, rhs, rhsLen);
}

void setCurTimeConstraints(void* lpdata, double* data, int dataLen, int* indices, int indicesLen,
                           int* indptr, int indptrLen, double* rhs, int rhsLen)
{
    hylaa::setCurTimeConstraints((LpData*)lpdata, data, dataLen, indices, indicesLen, indptr,
                                 indptrLen, rhs, rhsLen);
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

void test()
{
    hylaa::test();
}
}  // extern "C"

int main()
{
    test();
    printf("Tests Passed!\n");

    return 0;
}
