// Stanley Bak
// Hylaa GLPK unit test implementation

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "hylaa_glpk.h"

namespace hylaa_glpk_tests
{
void test1d()
{
    LpData lpd(1, 1, 0);

    // 1 <= x <= 2
    double data[] = {1, -1};
    int inds[] = {0, 0};
    int indptr[] = {0, 1, 2};
    double rhs[] = {2, -1};

    lpd.setInitConstraintsCsr(1, 2, data, 2, inds, 2, indptr, 3, rhs, 2);

    // simple maximization problem (no constraints)
    lpd.setNoOutputConstraints();

    double basis[] = {0.1};
    lpd.updateBasisMatrix(basis, 1, 1);

    double result[2] = {0, 0};
    double direction1[] = {1};

    if (lpd.minimize(direction1, 1, result, 2))
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

    if (lpd.minimize(direction2, 1, result, 2))
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
}

// 1-d example with a cur-time constraint
void test1d_constraint()
{
    LpData lpd(1, 1, 0);

    // 1 <= x <= 2
    double data[] = {1, -1};
    int inds[] = {0, 0};
    int indptr[] = {0, 1, 2};
    double rhs[] = {2, -1};

    lpd.setInitConstraintsCsr(1, 2, data, 2, inds, 2, indptr, 3, rhs, 2);

    // x_now <= 1.5
    double outputData[] = {1};
    int outputInds[] = {0};
    int outputIndptr[] = {0, 1};
    double outputRhs[] = {1.5};

    lpd.setOutputConstraintsCsr(1, 1, outputData, 1, outputInds, 1, outputIndptr, 2, outputRhs, 1);

    double basis[] = {1};
    lpd.updateBasisMatrix(basis, 1, 1);

    double result[1] = {0};
    double direction[] = {-1};

    if (lpd.minimize(direction, 1, result, 1))
    {
        printf("call to minimize failed in 1-d-constraint self test\n");
        exit(1);
    }

    if (fabs(result[0] - (1.5)) > 1e-6)
    {
        printf("lp self-test-1d-constraint failed result: %f; expected: 1.5\n", result[0]);
        exit(1);
    }
}

/*void test1d_inputs()
{
    LpData lpd(1, 1, 1);

    double data[] = {1, -1};
    int indices[] = {0, 0};
    int indptr[] = {0, 1, 2};
    double rhs[] = {2, -1};

    lpd.setInitConstraintsCsr(data, 2, indices, 2, indptr, 3, rhs, 2);
    // 1 <= x <= 2
    // 1st step input is bounded to [-0.1, 0.2]
    // 2nd step inputs is bounded to [-0.2, 0.4] (scaled by 2 using input-effects matrix)
    double inputData[] = {1, -1};
    int inputIndices[] = {0, 0};
    int inputIndptr[] = {0, 1, 2};
    double inputRhs[] = {0.2, 0.1};
    lpd.setInputConstraintsCsr(inputData, 2, inputIndices, 2, inputIndptr, 3, inputRhs, 2);

    double basis[] = {1.0};
    lpd.updateTimeElapseMatrix(basis, 1, 1);

    double inputEffects[] = {1.0};
    lpd.addInputEffectsMatrix(inputEffects, 1, 1);

    lpd.commitCurTimeRows();

    double result[4] = {0, 0, 0, 0};
    double direction1[] = {1};

    if (lpd.minimize(direction1, 1, result, 3))
    {
        printf("1st call to minimize failed in 1-d-input self test\n");
        exit(1);
    }

    double expected1[] = {1.0, 0.9, -0.1};

    if (fabs(result[0] - (expected1[0])) > 1e-6 || fabs(result[1] - (expected1[1])) > 1e-6 ||
        fabs(result[2] - (expected1[2])) > 1e-6)
    {
        printf("lp self-test-1d-input failed 1st result: (%f, %f, %f); expected: (%f, %f, %f)\n",
               result[0], result[1], result[3], expected1[0], expected1[1], expected1[2]);
        exit(1);
    }

    ////////////////////////////

    double inputEffects2[] = {2.0};
    lpd.addInputEffectsMatrix(inputEffects2, 1, 1);

    lpd.commitCurTimeRows();

    double direction2[] = {-1};

    if (lpd.minimize(direction2, 1, result, 4))
    {
        printf("2nd call to minimize failed in 1-d-input self test\n");
        exit(1);
    }

    double expected2[] = {2.0, 2.6, 0.2, 0.2};

    if (fabs(result[0] - (expected2[0])) > 1e-6 || fabs(result[1] - (expected2[1])) > 1e-6 ||
        fabs(result[2] - (expected2[2])) > 1e-6 || fabs(result[3] - (expected2[3])) > 1e-6)
    {
        printf(
            "lp self-test-1d-input failed 2nd result: (%f, %f, %f, %f); expected: (%f, %f, %f, "
            "%f)\n",
            result[0], result[1], result[2], result[2], expected2[0], expected2[1], expected2[2],
            expected2[3]);
        exit(1);
    }
    }*/

}  //  end namespace hylaa_glpk_tests

void run_hylaa_glpk_tests()
{
    hylaa_glpk_tests::test1d();
    hylaa_glpk_tests::test1d_constraint();
    // hylaa_glpk_tests::test1d_inputs();
}
