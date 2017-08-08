// Stanley Bak
// Hylaa GLPK interface header
// Nov 2016

/*
 * The set of linear constraints (after 2 steps) is organized as follows:
 *
 * init_constriants 0                    | 0          | 0          <= init_constraints_vec
 * 0                identity_matrix      | 0          | 0          <= cur_time_constraints_vec
 * curTimeMatrix    -1 * identity_matrix | input_ef1  | input_ef2  == 0
 * --------------------------------------------------------------
 * 0                0                    | input_cons | 0          <= input_constraints_vec
 * 0                0                    | 0          | input_cons <= input_constraints_vec
 *
 * The first columns (numInitVars) are the initial variables, full number of dimensions
 * the second set of variables (numCurTimeVars), are equal to the number of linear constraints
 * in the safety condition. The second row set is an identity matrix because we don't
 * store the variables at the current time, only the projection of the variables on to the
 * set of linear constraints.
 *
 * The third set of variables encodes each components contribution to the curTimeVars. I need
 * to experiment on optimizing this (setting sit column by column vs row by row. Currently, we
 * store this in curTimeData (offset at 1) and curTimeIndices (offset at 1). It actually gets
 * set upon calling commitCurTimeRows() (seperated out so profiling can be done).
 */

#include <glpk.h>
#include <vector>

using namespace std;

#ifndef HYLAA_GLPK_H_
#define HYLAA_GPLK_H_

struct GlobalLpData
{
    int optimizations = 0;
    int iterations = 0;
};

extern GlobalLpData global;

class LpData
{
   public:
    LpData(int numCurTimeVars, int numInitVars, int numInputs)
        : numCurTimeVars(numCurTimeVars), numInitVars(numInitVars), numInputs(numInputs)
    {
        if (numCurTimeVars <= 0 || numInitVars <= 0)
        {
            printf("Fatal Error: numCurTimeVars(%d) and numInitVars(%d) must be positive.\n",
                   numCurTimeVars, numInitVars);
            exit(1);
        }

        // setup lp
        lp = glp_create_prob();
        glp_set_obj_dir(lp, GLP_MIN);

        // setup lp params
        glp_init_smcp(&params);
        params.msg_lev = GLP_MSG_OFF;
        // params.out_frq = 1;

        // params.presolve = GLP_ON;
        // params.meth = GLP_DUALP;

        // the first n variables are the init variables
        // the first m variables are the standard variables
        glp_add_cols(lp, numCurTimeVars + numInitVars);

        for (int i = 0; i < numCurTimeVars + numInitVars; ++i)
            glp_set_col_bnds(lp, i + 1, GLP_FR, 0, 0);  // free variable (bounds -inf to inf)

        curTimeData.resize(numCurTimeVars);
        curTimeIndices.resize(numCurTimeVars);

        // rows are added to the lp instance once time-elapse matrix is updated
    };

    ~LpData()
    {
        glp_delete_prob(lp);
        lp = nullptr;
    };

    // reset the current solution in the LP
    void resetLp()
    {
        // set the status of all columns to GLP_NF and all rows are GLP_BS
        int rows = glp_get_num_rows(lp);
        int cols = glp_get_num_cols(lp);

        for (int r = 0; r < rows; ++r)
            glp_set_row_stat(lp, r + 1, GLP_BS);

        for (int c = 0; c < cols; ++c)
            glp_set_col_stat(lp, c + 1, GLP_NF);
    }

    void printLp()
    {
        if (!committed)
        {
            printf(
                "Fatal Error: printLp() called but matrix was not committed with "
                "commitCurTimeRows()\n");
            exit(1);
        }

        int rows = glp_get_num_rows(lp);
        int cols = glp_get_num_cols(lp);

        printf("Lp has %d columns (variables) and %d rows (constraints)\n", cols, rows);

        const char* stat_labels[] = {"?(0)?", "BS", "NL", "NU", "NF", "NS", "?(6)?"};
        // const char* stat_labels[] = {"?(0)?", "Basic (1=BS)", "Non-Basic on Lower Bound (2=NL)",
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
                        val = vals[index++];
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

    void addInputEffectsMatrix(double* matrix, int w, int h)
    {
        if (w != numInputs || h != numCurTimeVars)
        {
            printf(
                "Fatal Error: Matrix dimensions mismatch in addInputEffectsMatrix: "
                "w(%d) != numInputs(%d) || h(%d) != numCurTimeVars(%d)\n",
                w, numInputs, h, numCurTimeVars);
            exit(1);
        }

        if (inputRhs.size() == 0)
        {
            printf(
                "Fatal Error: input constraints should be set before calling "
                "addInputEffectsMatrix()\n");
            exit(1);
        }

        if (curTimeData[0].size() == 0)
        {
            printf(
                "Fatal Error: Time elapse matrix should be set before calling "
                "addInputEffectsMatrix()\n");
            exit(1);
        }

        int precols = glp_get_num_cols(lp);
        int prerows = glp_get_num_rows(lp);

        if (prerows == numInitConstraints + numCurTimeConstraints)
        {
            printf(
                "Fatal Error: updateTimeElpaseMatrix() should be called before "
                "addInputEffectsMatrix()\n");
            exit(1);
        }

        // add new columns (for input variables)
        glp_add_cols(lp, numInputs);

        for (int i = 0; i < numInputs; ++i)
            glp_set_col_bnds(lp, precols + i + 1, GLP_FR, 0,
                             0);  // free variable (bounds -inf to inf)

        // add new rows (for input constraints)
        int numInputConstraints = (int)inputRhs.size();

        glp_add_rows(lp, numInputConstraints);

        for (int i = 0; i < numInputConstraints; ++i)
            glp_set_row_bnds(lp, prerows + i + 1, GLP_UP, 0, inputRhs[i]);

        // worst case entries in one row is dataLen
        int rowIndices[inputCsrData.size() + 1];
        double rowData[inputCsrData.size() + 1];

        for (int row = 0; row < numInputConstraints; ++row)
        {
            int rowIndex = 1;

            for (int i = inputCsrIndptr[row]; i < inputCsrIndptr[row + 1]; ++i)
            {
                rowIndices[rowIndex] = precols + inputCsrIndices[i] + 1;
                rowData[rowIndex++] = inputCsrData[i];
            }

            glp_set_mat_row(lp, prerows + row + 1, rowIndex - 1, rowIndices, rowData);
        }

        // finally, set the values for the input effects matrix that was passed in
        // this will be in row (numInitConstraints + row#), and column (precols + col#)
        for (int r = 0; r < numCurTimeVars; ++r)
        {
            for (int c = 0; c < numInputs; ++c)
            {
                double value = matrix[r * w + c];
                curTimeData[r].push_back(value);
                curTimeIndices[r].push_back(precols + c + 1);
            }
        }

        committed = false;
    }

    /*
     * commit the cur-time rows (curTimeMatirx + input-effects matrices) to the lp instance
     */
    void commitCurTimeRows()
    {
        if (committed)
        {
            printf("Fatal Error: commitCurTimeRows() called twice without any changes.\n");
            exit(1);
        }

        if (curTimeData[0].size() == 0)
        {
            printf(
                "Fatal Error: updateTimeElapseMatrix() should be called before "
                "commitCurTimeRows().\n");
            exit(1);
        }

        // assign from curTimeData and curTimeVertices
        for (int r = 0; r < numCurTimeVars; ++r)
        {
            int lpRow = numInitConstraints + numCurTimeConstraints + r + 1;

            double* vals = &curTimeData[r][0];
            int* inds = &curTimeIndices[r][0];
            int len = curTimeData[r].size() - 1;  // these are offset by one

            glp_set_mat_row(lp, lpRow, len, inds, vals);
        }

        committed = true;
    }

    void updateTimeElapseMatrix(double* matrix, int w, int h)
    {
        if (w != numInitVars || h != numCurTimeVars)
        {
            printf(
                "Fatal Error: Matrix dimensions mismatch in updateTimeElapseMatrix: "
                "w(%d) != numInitVars(%d) || h(%d) != numCurTimeVars(%d)\n",
                w, numInitVars, h, numCurTimeVars);
            exit(1);
        }

        if (numInitConstraints == 0)
        {
            printf("Fatal Error: Init Constraints should be set before updateTimeElapseMatrix.\n");
            exit(1);
        }

        // add rows, if new instance
        if (glp_get_num_rows(lp) == numInitConstraints + numCurTimeConstraints)
        {
            // new problem instance, create one constraint row for each equality constraint
            glp_add_rows(lp, numCurTimeVars);

            // set bounds == 0
            for (int r = 0; r < numCurTimeVars; ++r)
            {
                int row = numInitConstraints + numCurTimeConstraints + r + 1;
                glp_set_row_bnds(lp, row, GLP_FX, 0, 0);
            }
        }

        // create curTimeData and curTimeIndices
        if (curTimeData[0].size() == 0)
        {
            // initialize each row
            for (int r = 0; r < numCurTimeVars; ++r)
            {
                curTimeData[r].resize(numInitVars + 2);  // 0 unused & 1 for the -1 * Identity part
                curTimeIndices[r].resize(numInitVars + 2);

                curTimeData[r][numInitVars + 1] = -1;
                curTimeIndices[r][numInitVars + 1] = numInitVars + r + 1;

                for (int i = 0; i < numInitVars; ++i)
                    curTimeIndices[r][i + 1] = i + 1;
            }
        }

        // assign to curTimeData and curTimeIndices
        for (int row = 0; row < h; ++row)
            for (int col = 0; col < w; ++col)
                curTimeData[row][1 + col] = matrix[row * w + col];

        committed = false;
    }

    // Set the input constraints (Csr matrix)
    void setInputConstraintsCsr(double* data, int dataLen, int* indices, int indicesLen,
                                int* indptr, int indptrLen, double* rhs, int rhsLen)
    {
        if (dataLen != indicesLen)
        {
            printf(
                "Fatal Error: setInputConstraintsCsr() expected sparse matrix with dataLen == "
                "indicesLen.\n");
            exit(1);
        }

        if (indptrLen != rhsLen + 1)
        {
            printf(
                "Fatal Error: setInputConstraintsCsr() matrix should have indptrLen (%d) == "
                "rhsLen(%d) + 1.\n",
                indptrLen, rhsLen);
            exit(1);
        }

        if (indptr[indptrLen - 1] != dataLen)
        {
            printf(
                "Fatal Error: setInputConstraintsCsr() sparse matrix should have indptr[-1] == "
                "dataLen.\n");
            exit(1);
        }

        inputCsrData.resize(dataLen);
        inputCsrIndices.resize(indicesLen);
        inputCsrIndptr.resize(indptrLen);
        inputRhs.resize(rhsLen);

        for (int i = 0; i < dataLen; ++i)
            inputCsrData[i] = data[i];

        for (int i = 0; i < indicesLen; ++i)
            inputCsrIndices[i] = indices[i];

        for (int i = 0; i < indptrLen; ++i)
            inputCsrIndptr[i] = indptr[i];

        for (int i = 0; i < rhsLen; ++i)
            inputRhs[i] = rhs[i];
    }

    /**
     * Set the constraints on the initial states. A CSR sparse matrix is passed in, along with
     * a right-hand side vector
     */
    void setInitConstraintsCsr(double* data, int dataLen, int* indices, int indicesLen, int* indptr,
                               int indptrLen, double* rhs, int rhsLen)
    {
        if (numInitConstraints != 0)
        {
            printf("Fatal Error: setInitConstraintsCsr() called twice.\n");
            exit(1);
        }

        numInitConstraints = rhsLen;

        if (dataLen != indicesLen)
        {
            printf(
                "Fatal Error: setInitConstraintsCsr() expected CSR matrix with dataLen == "
                "indicesLen.\n");
            exit(1);
        }

        if (indptrLen != rhsLen + 1)
        {
            printf(
                "Fatal Error: setInitConstraintsCsr() CSR matrix should have indptrLen (%d) == "
                "rhsLen(%d) + 1.\n",
                indptrLen, rhsLen);
            exit(1);
        }

        if (indptr[indptrLen - 1] != dataLen)
        {
            printf(
                "Fatal Error: setInitConstraintsCsr() CSR matrix should have indptr[-1] == "
                "dataLen.\n");
            exit(1);
        }

        if (glp_get_num_rows(lp) != 0)
        {
            printf("Fatal Error: setInitConstraintsCsr() should be called with 0 rows in the lp\n");
            exit(1);
        }

        // create new row for each constraint
        glp_add_rows(lp, rhsLen);

        for (int i = 0; i < rhsLen; ++i)
            glp_set_row_bnds(lp, i + 1, GLP_UP, 0, rhs[i]);

        // worst case entries in one row is dataLen
        vector <int> rowIndices(dataLen + 1, 0);
        vector <double> rowData(dataLen + 1, 0.0);

        for (int row = 0; row < rhsLen; ++row)
        {
            int rowIndex = 1;

            for (int i = indptr[row]; i < indptr[row + 1]; ++i)
            {
                rowIndices[rowIndex] = indices[i] + 1;
                rowData[rowIndex++] = data[i];
            }

            glp_set_mat_row(lp, row + 1, rowIndex - 1, &rowIndices[0], &rowData[0]);
        }
    }

    void setCurTimeConstraintBounds(double* rhs, int rhsLen)
    {
        if (rhsLen != numCurTimeVars)
        {
            printf(
                "Fatal Error: num constraints being set in setCurTimeConstraints(..., %d) should "
                "be equal "
                "to the number of curTime variables (%d).\n",
                rhsLen, numCurTimeVars);
            exit(1);
        }

        if (glp_get_num_rows(lp) != numInitConstraints)
        {
            printf(
                "Fatal Error: Cur-time constraints should be set before time-elapse matrix "
                "is updated in setCurTimeConstraints().\n");
            exit(1);
        }

        numCurTimeConstraints = rhsLen;

        // create new row for the constraint
        glp_add_rows(lp, rhsLen);

        for (int r = 0; r < rhsLen; ++r)
            glp_set_row_bnds(lp, numInitConstraints + r + 1, GLP_UP, 0, rhs[r]);

        // assign identity matrix
        int inds[2];
        double vals[2];

        for (int row = 0; row < rhsLen; ++row)
        {
            inds[1] = 1 + numInitVars + row;
            vals[1] = 1;

            glp_set_mat_row(lp, numInitConstraints + row + 1, 1, inds, vals);
        }
    }

    // returns 0 on success
    // returns 1 on unsat
    int minimize(double* direction, int dirLen, double* result, int resLen)
    {
        if (!committed)
        {
            printf("Fatal Error: minimize called without commitCurTimeRows()\n");
            exit(1);
        }

        if (dirLen != numCurTimeVars)
        {
            printf(
                "Fatal Error: dirLen(%d) is not equal to numCurTimeVars(%d) in call to "
                "minimize()\n",
                dirLen, numCurTimeVars);
            exit(1);
        }

        for (int i = 0; i < numCurTimeVars; ++i)
            glp_set_obj_coef(lp, 1 + numInitVars + i, direction[i]);

        int startIterations = glp_get_it_cnt(lp);

        int simplexRes = glp_simplex(lp, &params);

        int newIterations = glp_get_it_cnt(lp) - startIterations;
        global.iterations += newIterations;

        ++global.optimizations;

        return processSimplexResult(simplexRes, result, resLen);
    }

    /////////////////////////////////
   private:
    int numCurTimeVars = 0;  // number of standard variables
    int numInitVars = 0;     // number of basis variables
    int numInputs = 0;

    int numInitConstraints = 0;
    int numCurTimeConstraints = 0;

    // the curTimeMatrix & input_effect rows are assigned piece by piece, then committed all at once
    // these are offset by 1 (0th element is not used), so they can directly be used with glpk
    vector<vector<double>> curTimeData;
    vector<vector<int>> curTimeIndices;
    bool committed = false;  // tracks if commitCurTimeRows() was called after updating data

    // saved input constraints (need to be set at each step, if input is present)
    vector<double> inputCsrData;
    vector<int> inputCsrIndices;
    vector<int> inputCsrIndptr;
    vector<double> inputRhs;

    glp_prob* lp = nullptr;
    glp_smcp params;

    void addRows(int num, double* bound_vec)
    {
        int curRows = glp_get_num_rows(lp);

        glp_add_rows(lp, num);

        for (int r = 0; r < num; ++r)
            glp_set_row_bnds(lp, curRows + r + 1, GLP_UP, 0, bound_vec[r]);  // row <= constraint_b
    }

    // a debug printing function
    void printIndsVals(const char* funcName, int row, int len, int inds[], double vals[])
    {
        printf("%s(%d, {", funcName, row);

        for (int i = 1; i <= len; ++i)
            printf("%d ", inds[i]);

        printf("}, {");

        for (int i = 1; i <= len; ++i)
            printf("%f ", vals[i]);

        printf("})\n");
    }

    // internal function used for getting the result of simplex
    int processSimplexResult(int simplexRes, double* result, int resLen)
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
                    "Fatal error: num simplex error codes(%d) is not equal to num messages (%d).\n",
                    numCodes, numMsgs);
                exit(1);
            }

            for (unsigned int i = 0; i < sizeof(codes) / sizeof(codes[0]); ++i)
            {
                if (simplexRes == codes[i])
                {
                    msg = msgs[i];
                    break;
                }
            }

            printf("Fatal Error: glp_simplex returned nonzero status (%s) in minimize(): %d\n", msg,
                   simplexRes);
            exit(1);
        }
        else
        {
            int status = glp_get_status(lp);

            if (status == GLP_OPT)
            {
                int numCols = glp_get_num_cols(lp);

                for (int col = 0; col < resLen && col < numCols; ++col)
                    result[col] = glp_get_col_prim(lp, col + 1);
            }
            else if (status == GLP_NOFEAS)
            {
                // infeasible LP
                rv = 1;
            }
            else
            {
                int codes[] = {GLP_OPT, GLP_FEAS, GLP_INFEAS, GLP_NOFEAS, GLP_UNBND, GLP_UNDEF};
                const char* msgs[] = {"solution is optimal", "solution is feasible",
                                      "solution is infeasible", "problem has no feasible solution",
                                      "problem has unbounded solution", "solution is undefined"};

                const char* message = "Unknown Error";

                for (unsigned int i = 0; i < sizeof(codes) / sizeof(codes[0]); ++i)
                {
                    if (status == codes[i])
                    {
                        message = msgs[i];
                        break;
                    }
                }

                printf("Fatal Error: LP Status after solving in minimize() was '%s': %d\n", message,
                       status);
                exit(1);
            }
        }

        return rv;
    }
};

#endif
