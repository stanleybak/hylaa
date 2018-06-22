// Stanley Bak
// Lightweight GLPK interface for python
// May 2018

#include <glpk.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
using namespace std;

extern "C"
{
    glp_prob* initLp()
    {
        glp_prob* lp = glp_create_prob();

        // setup lp
        glp_set_obj_dir(lp, GLP_MIN);

        return lp;
    }

    void delLp(glp_prob* lp)
    {
        glp_delete_prob(lp);
    }

    void printLp(glp_prob* lp)
    {
        int rows = glp_get_num_rows(lp);
        int cols = glp_get_num_cols(lp);

        printf("Lp has %d columns (variables) and %d rows (constraints)\n", cols, rows);

        const char* stat_labels[] = {"?(0)?", "BS", "NL", "NU", "NF", "NS", "?(6)?"};
        // const char* stat_labels[] = {"?(0)?", "Basic (1=BS)", "Non-Basic on Lower Bound
        // (2=NL)",
        //                                     "Non-Basic on Upper Bound (3=NU)",
        //                                    "Non-Basic Free Variable (4=NF)",
        //                                   "Non-Basic Fixed Variable (5=NS)", "?(6)?"};

        int inds[cols + 1];
        double vals[cols + 1];
        char buf[16];

        // first print all the column statuses
        printf("   ");
        for (int col = 1; col <= cols; ++col)
            printf("%6s ", stat_labels[glp_get_col_stat(lp, col)]);

        printf("\n");

        for (int row = 1; row <= rows; ++row)
        {
            printf("%2s ", stat_labels[glp_get_row_stat(lp, row)]);

            int len = glp_get_mat_row(lp, row, inds, vals);

            for (int col = 1; col <= cols; ++col)
            {
                double val = 0;

                for (int index = 1; index <= len; ++index)
                {
                    if (inds[index] == col)
                    {
                        val = vals[index];
                        break;
                    }
                }

                buf[6] = 0;
                snprintf(buf, sizeof(buf), "%5.3g", val);
                buf[6] = 0;  //////////////////

                if (buf[6] == 0)
                    printf("%6s ", buf);
                else
                    printf("%6.3g ", val);
            }

            // check if the row is equality or lesseq
            int type = glp_get_row_type(lp, row);
            double val = glp_get_row_ub(lp, row);

            if (type == GLP_FX)
                printf(" == %g", val);
            else if (type == GLP_UP)
                printf(" <= %g", val);
            else
                printf(" <?> (unknown bounds)");

            printf("\n");
        }
    }

    void addCols(glp_prob* lp, int num)
    {
        int numCols = glp_get_num_cols(lp);

        glp_add_cols(lp, num);

        for (int i = 0; i < num; ++i)
            glp_set_col_bnds(lp, numCols + i + 1, GLP_FR, 0,
                             0);  // free variable (bounds -inf to inf)
    }

    // add '<=' constraints
    void addRowsLessEqual(glp_prob* lp, double* rhs, int rhsLen)
    {
        int numRows = glp_get_num_rows(lp);

        // create new row for each constraint
        glp_add_rows(lp, rhsLen);

        for (int i = 0; i < rhsLen; ++i)
            glp_set_row_bnds(lp, numRows + i + 1, GLP_UP, 0,
                             rhs[i]);  // '<=' constraint
    }

    // add rows with '== 0 ' constraints
    void addRowsEqualZero(glp_prob* lp, int num)
    {
        int numRows = glp_get_num_rows(lp);

        // create new row for each constraint
        glp_add_rows(lp, num);

        for (int i = 0; i < num; ++i)
            glp_set_row_bnds(lp, numRows + i + 1, GLP_FX, 0, 0);  // '== 0' constraints
    }

    // check if csr matrix is valid, returns 0 if valud, 1 otherwise (also prints error message)
    int checkCsr(int w, int h, double* data, int dataLen, int* indices, int indicesLen, int* indptr,
                 int indptrLen)
    {
        int rv = 0;

        if (dataLen != indicesLen)
        {
            printf("Error: csr sparse matrix should have dataLen == indicesLen.\n");
            rv = 1;
        }
        else if (indptrLen != h + 1)
        {
            printf("Error: csr sparse matrix should have indptrLen (%d) == h + 1 (%d).\n",
                   indptrLen, h + 1);
            rv = 1;
        }
        else
        {
            // make sure each index is less than width
            for (int i = 0; i < indicesLen; ++i)
            {
                if (indices[i] >= w)
                {
                    printf(
                        "Error: csr sparse matrix has indices[%d]=%d, which is "
                        ">= matrix "
                        "width (%d)\n",
                        i, indices[i], w);
                    rv = 1;
                    break;
                }
            }
        }

        return rv;
    }

    // check if csc matrix is valid, returns 0 if valud, 1 otherwise (also prints error message)
    int checkCsc(int w, int h, double* data, int dataLen, int* indices, int indicesLen, int* indptr,
                 int indptrLen)
    {
        int rv = 0;

        if (dataLen != indicesLen)
        {
            printf("Error: csc sparse matrix should have dataLen == indicesLen.\n");
            rv = 1;
        }
        else if (indptrLen != w + 1)
        {
            printf("Error: csc sparse matrix should have indptrLen (%d) == w + 1 (%d).\n",
                   indptrLen, w + 1);
            rv = 1;
        }
        else
        {
            // make sure each index is less than width
            for (int i = 0; i < indicesLen; ++i)
            {
                if (indices[i] >= h)
                {
                    printf(
                        "Error: csc sparse matrix has indices[%d]=%d, which is "
                        ">= matrix "
                        "height (%d)\n",
                        i, indices[i], h);
                    rv = 1;
                    break;
                }
            }
        }

        return rv;
    }

    // sets the constraints from a csr matrix, offset by some x/y
    // returns 0 on success, nonzero on error
    int setConstraintsCsr(glp_prob* lp, int rowOffset, int colOffset, double* data, int dataLen,
                          int* indices, int indicesLen, int* indptr, int indptrLen, int numRows,
                          int numCols)
    {
        int rv = checkCsr(numCols, numRows, data, dataLen, indices, indicesLen, indptr, indptrLen);

        if (rv == 0)
        {
            // check that the matrix is in bounds
            int lpRows = glp_get_num_rows(lp);
            int lpCols = glp_get_num_cols(lp);

            if (rowOffset < 0 || colOffset < 0 || rowOffset + numRows > lpRows ||
                colOffset + numCols > lpCols)
            {
                printf(
                    "Error: set constraints matrix out of bounds (offset was "
                    "row=%d, col=%d, "
                    "matrix "
                    "size was %d, %d), but lp size was (%d, %d)\n",
                    rowOffset, colOffset, numRows, numCols, lpRows, lpCols);
                rv = 1;
            }
            else
            {
                // actually set the constraints row by row
                for (int row = 0; row < numRows; ++row)
                {
                    // we must copy the indices since glpk is offset by 1 :(
                    int count = indptr[row + 1] - indptr[row];
                    vector<int> indicesVec(count + 1);
                    int vIndex = 1;

                    for (int index = indptr[row]; index < indptr[row + 1]; ++index)
                        indicesVec[vIndex++] = 1 + colOffset + indices[index];

                    double* dataPtr = &(data[indptr[row] - 1]);

                    glp_set_mat_row(lp, rowOffset + row + 1, count, &(indicesVec[0]), dataPtr);
                }
            }
        }

        return rv;
    }

    // sets the constraints from a csc matrix, offset by some x/y
    // returns 0 on success, nonzero on error
    int setConstraintsCsc(glp_prob* lp, int rowOffset, int colOffset, double* data, int dataLen,
                          int* indices, int indicesLen, int* indptr, int indptrLen, int numRows,
                          int numCols)
    {
        int rv = checkCsc(numCols, numRows, data, dataLen, indices, indicesLen, indptr, indptrLen);

        if (rv == 0)
        {
            // check that the matrix is in bounds
            int lpRows = glp_get_num_rows(lp);
            int lpCols = glp_get_num_cols(lp);

            if (rowOffset < 0 || colOffset < 0 || rowOffset + numRows > lpRows ||
                colOffset + numCols > lpCols)
            {
                printf(
                    "Error: set constraints matrix out of bounds (offset was "
                    "row=%d, col=%d, "
                    "matrix "
                    "size was %d, %d), but lp size was (%d, %d)\n",
                    rowOffset, colOffset, numRows, numCols, lpRows, lpCols);
                rv = 1;
            }
            else
            {
                // actually set the constraints row by row
                for (int col = 0; col < numCols; ++col)
                {
                    // we must copy the indices since glpk is offset by 1 :(
                    int count = indptr[col + 1] - indptr[col];
                    vector<int> indicesVec(count + 1);
                    int vIndex = 1;

                    for (int index = indptr[col]; index < indptr[col + 1]; ++index)
                        indicesVec[vIndex++] = 1 + rowOffset + indices[index];

                    double* dataPtr = &(data[indptr[col] - 1]);

                    glp_set_mat_col(lp, colOffset + col + 1, count, &(indicesVec[0]), dataPtr);
                }
            }
        }

        return rv;
    }

    void resetLp(glp_prob* lp)
    {
        // set the status of all columns to GLP_NF and all rows are GLP_BS
        int rows = glp_get_num_rows(lp);
        int cols = glp_get_num_cols(lp);

        for (int r = 0; r < rows; ++r)
            glp_set_row_stat(lp, r + 1, GLP_BS);

        for (int c = 0; c < cols; ++c)
            glp_set_col_stat(lp, c + 1, GLP_NF);
    }

    // function used for getting the result of simplex
    // returns 0 on success
    // returns 1 on unsat
    // returns -1 on error
    int processSimplexResult(glp_prob* lp, int simplexRes, int* columns, double* result, int resLen)
    {
        int rv = 0;

        if (simplexRes == GLP_ENOPFS)  // no primal feasible w/ presolver
        {
            rv = 1;
        }
        else if (simplexRes != 0)
        {
            const char* msg = "Unknown error";

            int codes[] = {GLP_EBADB,  GLP_ESING,  GLP_ECOND,  GLP_EBOUND, GLP_EFAIL, GLP_EOBJLL,
                           GLP_EOBJUL, GLP_EITLIM, GLP_ETMLIM, GLP_ENOPFS, GLP_ENODFS};

            const char* msgs[] = {
                "Unable to start the search, because the initial basis specified "
                "in the problem object is invalid—the number of basic (auxiliary "
                "and structural) variables is not the same as the number of rows "
                "in the problem object.",

                "Unable to start the search, because the basis matrix corresponding "
                "to the initial basis is singular within the working "
                "precision.",

                "Unable to start the search, because the basis matrix corresponding "
                "to the initial basis is ill-conditioned, i.e. its "
                "condition number is too large.",

                "Unable to start the search, because some double-bounded "
                "(auxiliary or structural) variables have incorrect bounds.",

                "The search was prematurely terminated due to the solver "
                "failure.",

                "The search was prematurely terminated, because the objective "
                "function being maximized has reached its lower "
                "limit and continues decreasing (the dual simplex only).",

                "The search was prematurely terminated, because the objective "
                "function being minimized has reached its upper "
                "limit and continues increasing (the dual simplex only).",

                "The search was prematurely terminated, because the simplex "
                "iteration limit has been exceeded.",

                "The search was prematurely terminated, because the time "
                "limit has been exceeded.",

                "The LP problem instance has no primal feasible solution "
                "(only if the LP presolver is used).",

                "The LP problem instance has no dual feasible solution "
                "(only if the LP presolver is used).",
            };

            const int numCodes = sizeof(codes) / sizeof(codes[0]);
            const int numMsgs = sizeof(msgs) / sizeof(msgs[0]);

            if (numCodes != numMsgs)
            {
                printf(
                    "Error: num simplex error codes(%d) is not equal to num "
                    "messages (%d).\n",
                    numCodes, numMsgs);
            }
            else
            {
                for (unsigned int i = 0; i < sizeof(codes) / sizeof(codes[0]); ++i)
                {
                    if (simplexRes == codes[i])
                    {
                        msg = msgs[i];
                        break;
                    }
                }

                printf(
                    "Error: glp_simplex returned nonzero status (%s) in "
                    "minimize(): %d\n",
                    msg, simplexRes);
            }

            rv = -1;
        }
        else
        {
            int status = glp_get_status(lp);

            if (status == GLP_OPT)
            {
                int numCols = glp_get_num_cols(lp);

                // copy the output vars
                for (int resIndex = 0; resIndex < resLen; ++resIndex)
                {
                    int col = columns[resIndex];

                    if (col < 0 || col >= numCols)
                    {
                        printf(
                            "Error: out of bounds column requested in LP "
                            "result: %d (numCols = "
                            "%d)\n",
                            col, numCols);
                        rv = -1;
                        break;
                    }

                    result[resIndex] = glp_get_col_prim(lp, col + 1);
                }
            }
            else if (status == GLP_NOFEAS)
            {
                // infeasible LP... not an error
                rv = 1;
            }
            else
            {
                int codes[] = {GLP_OPT, GLP_FEAS, GLP_INFEAS, GLP_NOFEAS, GLP_UNBND, GLP_UNDEF};
                const char* msgs[] = {"solution is optimal",
                                      "solution is feasible",
                                      "solution is infeasible",
                                      "problem has no feasible solution",
                                      "problem has unbounded solution",
                                      "solution is undefined"};

                const char* message = "Unknown Error";

                for (unsigned int i = 0; i < sizeof(codes) / sizeof(codes[0]); ++i)
                {
                    if (status == codes[i])
                    {
                        message = msgs[i];
                        break;
                    }
                }

                printf("Error: LP Status after solving in minimize() was '%s': %d\n", message,
                       status);
                rv = -1;
            }
        }

        return rv;
    }

    // returns 0 on success, -1 on error
    // this sets the first 'dirLen' variables
    int setMinimizeDirection(glp_prob* lp, double* direction, int dirLen)
    {
        int numCols = glp_get_num_cols(lp);
        int rv = 0;

        if (dirLen > numCols)
        {
            printf("Error: dirLen(%d) > numCols(%d) in call to minimize()\n", dirLen, numCols);
            rv = -1;
        }
        else
        {
            for (int i = 0; i < dirLen; ++i)
                glp_set_obj_coef(lp, 1 + i, direction[i]);
        }

        return rv;
    }

    // it is assumed columns and result are the same length
    // columns is a list of columns to assign upon SAT
    // they get assigned to the result array
    // returns 0 on success (sat)
    // returns 1 on unsat
    // returns -1 on error
    int minimize(glp_prob* lp, int* columns, double* result, int resLen)
    {
        int rv = 0;

        // setup lp params
        glp_smcp params;
        glp_init_smcp(&params);
        params.msg_lev = GLP_MSG_OFF;

        int simplexRes = glp_simplex(lp, &params);

        if (simplexRes != 0)
        {
            // sometimes the previous solution is singular wrt. current constraints...
            // need to reset
            //
            //    "Warning: hylaa_glpk.h - simplexRes was nonzero (%d). Resetting
            //    statuses and " "retrying.\n", simplexRes);
            resetLp(lp);

            simplexRes = glp_simplex(lp, &params);
        }

        rv = processSimplexResult(lp, simplexRes, columns, result, resLen);

        return rv;
    }

    int getIterations(glp_prob* lp)
    {
        // needs a newish version of GLPK
        // download from https://ftp.gnu.org/gnu/glpk/
        // glpk-4.65 or newer should work

        return glp_get_it_cnt(lp);
    }

    int getNumRows(glp_prob* lp)
    {
        return glp_get_num_rows(lp);
    }

    int getNumCols(glp_prob* lp)
    {
        return glp_get_num_cols(lp);
    }

    // get the LP matrix. This is slow, so it's meant for testing.
    // returns 0 on success, 1 on error (mismatched dimensions)
    int getMatrix(glp_prob* lp, double* mat, double* vec, int rows, int cols)
    {
        int lpRows = glp_get_num_rows(lp);
        int lpCols = glp_get_num_cols(lp);
        int rv = 0;

        if (rows != lpRows)
        {
            printf("Error: matRows(%d) != lpRows(%d)\n", rows, lpRows);
            rv = 1;
        }
        else if (cols != lpCols)
        {
            printf("Error: matCols(%d) != lpCols(%d)\n", cols, lpCols);
            rv = 1;
        }
        else
        {
            vector<int> inds(cols + 1);
            vector<double> vals(cols + 1);

            for (int row = 1; row <= rows; ++row)
            {
                int len = glp_get_mat_row(lp, row, &inds[0], &vals[0]);

                for (int col = 1; col <= cols; ++col)
                {
                    double val = 0;

                    for (int index = 1; index <= len; ++index)
                    {
                        if (inds[index] == col)
                        {
                            val = vals[index];
                            break;
                        }
                    }

                    mat[(row - 1) * cols + (col - 1)] = val;
                }

                double rhs = glp_get_row_ub(lp, row);
                vec[row - 1] = rhs;
            }
        }

        return rv;
    }

    // simple testing function... minimize x, 1 <= x <= 2
    // return 0 on success
    int test()
    {
        int rv = 1;
        glp_prob* lp = initLp();

        addCols(lp, 1);

        double rhs[] = {-1, 2};
        addRowsLessEqual(lp, rhs, 2);

        double csrData[] = {-1, 1};
        int csrIndices[] = {0, 0};
        int csrIndPtrs[] = {0, 1, 2};

        int setRv = setConstraintsCsr(lp, 0, 0, csrData, 2, csrIndices, 2, csrIndPtrs, 3, 2, 1);

        if (setRv != 0)
            printf("Test Error: setConstraintsCsr failed\n");
        else
        {
            double result[] = {0};
            int columns[] = {0};
            double direction[] = {1};

            setMinimizeDirection(lp, direction, 1);

            rv = minimize(lp, columns, result, 1);

            if (rv != 0)
                printf("Test Error: minimize failed\n");
            else
            {
                if (result[0] < 1 - 1e-9 || result[0] > 1 + 1e-9)
                    printf("Test Error: wrong result (expected 1.0, got %f)\n", result[0]);
                else
                    rv = 0;  // test succeeded
            }
        }

        return rv;
    }
}

// intended use is through python interface
int main()
{
    printf("Using GLPK version %s\n", glp_version());

    if (test())
        printf("Test failed\n");
    else
        printf("Test passed\n");

    return 0;
}
