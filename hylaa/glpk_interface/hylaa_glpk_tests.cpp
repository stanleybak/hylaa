// Stanley Bak
// Hylaa GLPK unit test implementation

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "hylaa_glpk.h"

namespace hylaa_glpk_tests
{
LpData* initLp(int numCurTimeVars, int numInitVars)
{
    LpData* data = new (std::nothrow) LpData(numCurTimeVars, numInitVars);

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

    double data[] = {1, -1};
    int indices[] = {0, 0};
    int indptr[] = {0, 1, 2};
    double rhs[] = {2, -1};

    lpd->setInitConstraints(data, 2, indices, 2, indptr, 3, rhs, 2);

    double basis[] = {0.1};
    lpd->updateTimeElapseMatrix(basis, 1, 1);

    double result[2] = {0, 0};
    double direction1[] = {1};

    if (lpd->minimize(direction1, 1, result, 2))
    {
        printf("1st call to minimize failed in 1-d self test\n");
        exit(1);
    }

    double expected1[] = {1.0, 0.1};

    if (fabs(result[0] - (expected1[0])) > 1e-6 || fabs(result[1] - (expected1[1])) > 1e-6)
    {
        printf("lp self-test-1d failed 1st result: (%f, %f); expected: (%f, %f)\n", result[0],
               result[1], expected1[0], expected1[1]);
        exit(1);
    }

    ////////////////////

    double direction2[] = {-1};

    if (lpd->minimize(direction2, 1, result, 2))
    {
        printf("2nd call to minimize failed in 1-d self test\n");
        exit(1);
    }

    double expected2[] = {2.0, 0.2};

    if (fabs(result[0] - (expected2[0])) > 1e-6 || fabs(result[1] - (expected2[1])) > 1e-6)
    {
        printf("lp self-test-1d failed 2nd result: (%f, %f); expected: (%f, %f)\n", result[0],
               result[1], expected2[0], expected2[1]);
        exit(1);
    }

    delLp(lpd);
}

// 1-d example with a cur-time constraint
void test1d_constraint()
{
    LpData* lpd = initLp(1, 1);

    double initData[] = {1, -1};
    int initIndices[] = {0, 0};
    int initIndPtr[] = {0, 1, 2};
    double initRhs[] = {2, -1};

    lpd->setInitConstraints(initData, 2, initIndices, 2, initIndPtr, 3, initRhs, 2);

    double curTimeData[] = {-1};
    int curTimeIndices[] = {0};
    int curTimeIndPtr[] = {0, 1};
    double curTimeRhs[] = {-1.5};

    lpd->setCurTimeConstraints(curTimeData, 1, curTimeIndices, 1, curTimeIndPtr, 2, curTimeRhs, 1);

    double basis[] = {1};
    lpd->updateTimeElapseMatrix(basis, 1, 1);

    double result[1] = {0};
    double direction[] = {1};

    if (lpd->minimize(direction, 1, result, 1))
    {
        printf("call to minimize failed in 1-d-constraint self test\n");
        exit(1);
    }

    if (fabs(result[0] - (1.5)) > 1e-6)
    {
        printf("lp self-test-1d-constraint failed result: %f; expected: 1.5\n", result[0]);
        exit(1);
    }

    delLp(lpd);
}

}  //  end namespace hylaa_glpk_tests

void hylaa_glpk_unit_test()
{
    hylaa_glpk_tests::test1d();
    hylaa_glpk_tests::test1d_constraint();
}
