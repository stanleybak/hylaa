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

    double a1[] = {1};
    double b1 = 2;
    lpd->addInitConstraint(a1, 1, b1);

    double a2[] = {-1};
    double b2 = -1;
    lpd->addInitConstraint(a2, 1, b2);

    double basis[] = {1};
    lpd->updateTimeElapseMatrix(basis, 1, 1);

    double result[1] = {0};
    double direction[] = {-1};

    if (lpd->minimize(direction, 1, result, 1))
    {
        printf("call to minimize failed in 1-d self test\n");
        exit(1);
    }

    if (fabs(result[0] - (2)) > 1e-6)
    {
        printf("lp self-test-1d failed result: %f; expected: 2\n", result[0]);
        exit(1);
    }

    delLp(lpd);
}

// 1-d example with a cur-time constraint
void test1d_constraint()
{
    LpData* lpd = initLp(1, 1);

    double a1[] = {1};
    double b1 = 2;
    lpd->addInitConstraint(a1, 1, b1);

    double a2[] = {-1};
    double b2 = -1;
    lpd->addInitConstraint(a2, 1, b2);

    double a3[] = {1};
    double b3 = 1.5;
    lpd->addCurTimeConstraint(a3, 1, b3);

    double basis[] = {1};
    lpd->updateTimeElapseMatrix(basis, 1, 1);

    double result[1] = {0};
    double direction[] = {-1};

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
