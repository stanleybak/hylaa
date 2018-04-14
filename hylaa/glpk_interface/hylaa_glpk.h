// Stanley Bak
// Hylaa GLPK interface header
// Original: Nov 2016
// Reorganized in Jan 2018 based on Input / Output spaces

/*
 * The set of linear constraints is organized as follows:
 *
 * init_constraints | 0                    | <= init_constraints_vec
 * -----------------+----------------------+--------------------------
 * 0                | output_constraints   | <= output_constraints_vec
 * -----------------+----------------------+--------------------------
 * basisMatrix      | -1 * identity_matrix | == 0
 *
 * The first set of columns are the initial variables (count is numInitVars).
 * The second set of columns are the output variables (count is numOutputVars).
 *
 * Based on this, the width of the basis matrix is numInitVars, and the height is numOutputVars.
 *
 * When you add input effects, you probably want to add new variables for total input effects, so
 * that updating the basis matrix can be done without re-setting the init constraints or input
 * basis matrices. Something like this (after two steps):
 *
 * init_cons | 0           | 0          | 0            | 0            | <= init_cons_rhs
 * ----------+-------------+------------+--------------+--------------+-------
 * 0         | output_cons | 0          | 0            | 0            | <= output_cons_rhs
 * ----------+-------------+------------+--------------+--------------+-------
 * basis_mat | -1 * ident  | ident      | 0            |              | == 0
 * ----------+-------------+------------+-------------------------------------
 * 0         | 0           | -1 * ident | input_basis1 | input_basis2 | == 0
 * 0         | 0           | 0          | input_cons   | 0            | <= input_cons_rhs
 * 0         | 0           | 0          | 0            | input_cons   | <= input_cons_rhs
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
    LpData(int numOutputVars, int numInitVars, int numInputs)
        : numOutputVars(numOutputVars), numInitVars(numInitVars), numInputs(numInputs)
    {
        if (numOutputVars <= 0 || numInitVars <= 0)
        {
            printf("Fatal Error: numOutputVars(%d) and numInitVars(%d) must be positive.\n",
                   numOutputVars, numInitVars);
            exit(1);
        }

        if (numInputs < 0)
        {
            printf("Fatal Error: numInputs(%d) must be nonnegative.\n", numInputs);
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

        // the first i variables are the init variables
        // the next o variables are the output variables
        // the next o variables are the total-output-effects-from-inputs variables
        int numVars = numOutputVars + numInitVars + numOutputVars;
        glp_add_cols(lp, numVars);

        for (int i = 0; i < numVars; ++i)
            glp_set_col_bnds(lp, i + 1, GLP_FR, 0, 0);  // free variable (bounds -inf to inf)

        // rows are added to the lp instance once the basis matrix is updated
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

    // Set the input constraints
    void setInputConstraintsCsc(int w, int h, double* data, int dataLen, int* indices,
                                int indicesLen, int* indptr, int indptrLen, double* rhs, int rhsLen)
    {
        checkCsc("setInputConstraintsCsc", w, h, data, dataLen, indices, indicesLen, indptr,
                 indptrLen);

        inputCscData.resize(dataLen);
        inputCscIndices.resize(indicesLen);
        inputCscIndptr.resize(indptrLen);
        inputRhs.resize(rhsLen);

        for (int i = 0; i < dataLen; ++i)
            inputCscData[i] = data[i];

        for (int i = 0; i < indicesLen; ++i)
            inputCscIndices[i] = indices[i];

        for (int i = 0; i < indptrLen; ++i)
            inputCscIndptr[i] = indptr[i];

        for (int i = 0; i < rhsLen; ++i)
            inputRhs[i] = rhs[i];
    }

    /**
     * Set the constraints on the initial states as a csr matrix
     */
    void setInitConstraintsCsr(int w, int h, double* data, int dataLen, int* inds, int indsLen,
                               int* indptr, int indptrLen, double* rhs, int rhsLen)
    {
        checkCsr("setInitConstraintsCsr", w, h, data, dataLen, inds, indsLen, indptr, indptrLen);

        if (numInitConstraints != -1)
        {
            printf("Fatal Error: setInitConstraints() called twice.\n");
            exit(1);
        }

        numInitConstraints = rhsLen;

        if (h != rhsLen)
        {
            printf("Fatal Error: setInitConstraints() matrix h != rhsLen.\n");
            exit(1);
        }

        if (w != numInitVars)
        {
            printf(
                "Fatal Error: setInitConstraints() matrix w (%d) should equal numInitVars (%d)\n",
                w, numInitVars);
            exit(1);
        }

        if (glp_get_num_rows(lp) != 0)
        {
            printf("Fatal Error: setInitConstraints() should be called with 0 rows in the lp\n");
            exit(1);
        }

        // create new row for each constraint
        glp_add_rows(lp, rhsLen);

        for (int i = 0; i < rhsLen; ++i)
            glp_set_row_bnds(lp, i + 1, GLP_UP, 0, rhs[i]);  // '<=' constraint

        // use memory on the heap (stack may be too small)
        vector<int> rowIndices(w + 1, 0);
        vector<double> rowData(w + 1, 0.0);

        for (int row = 0; row < rhsLen; ++row)
        {
            int rowIndex = 1;

            for (int i = indptr[row]; i < indptr[row + 1]; ++i)
            {
                int col = inds[i];
                double d = data[i];

                rowIndices[rowIndex] = col + 1;
                rowData[rowIndex++] = d;
            }

            glp_set_mat_row(lp, row + 1, rowIndex - 1, &rowIndices[0], &rowData[0]);
        }
    }

    // indicate that there are not constraints on the output variables (for plotting)
    void setNoOutputConstraints()
    {
        if (numOutputConstraints != -1)
        {
            printf(
                "Fatal Error: setNoOutputConstraints() called, but numOutputConstraints was "
                "already set\n");
            exit(1);
        }

        int indptr[] = {0};

        setOutputConstraintsCsr(numOutputVars, 0, 0, 0, 0, 0, indptr, 1, 0, 0);
    }

    void setOutputConstraintsCsr(int w, int h, double* data, int dataLen, int* inds, int indsLen,
                                 int* indptr, int indptrLen, double* rhs, int rhsLen)
    {
        checkCsr("setOutputConstraintsCsr", w, h, data, dataLen, inds, indsLen, indptr, indptrLen);

        if (numOutputConstraints != -1)
        {
            printf("Fatal Error: setOutputConstraints() called twice\n");
            exit(1);
        }

        numOutputConstraints = rhsLen;

        if (w != numOutputVars)
        {
            printf(
                "Fatal Error: matrix width in setOutputConstraints (%d) should "
                "be equal to the number of output variables (%d).\n",
                w, numOutputVars);
            exit(1);
        }

        if (glp_get_num_rows(lp) != numInitConstraints)
        {
            printf(
                "Fatal Error: setOutputConstraints() should be called right after "
                "setInitConstraints()\n");
            exit(1);
        }

        if (numOutputConstraints > 0)
        {
            // create new rows for the output constraints
            glp_add_rows(lp, rhsLen);

            for (int r = 0; r < rhsLen; ++r)
                glp_set_row_bnds(lp, numInitConstraints + r + 1, GLP_UP, 0,
                                 rhs[r]);  // '<=' constraint

            // use memory on the heap (stack may be too small)
            vector<int> rowIndices(w + 1, 0);
            vector<double> rowData(w + 1, 0.0);

            for (int row = 0; row < rhsLen; ++row)
            {
                int rowIndex = 1;

                for (int i = indptr[row]; i < indptr[row + 1]; ++i)
                {
                    int col = inds[i];
                    double d = data[i];

                    rowIndices[rowIndex] = numInitVars + col + 1;
                    rowData[rowIndex++] = d;
                }

                glp_set_mat_row(lp, numInitConstraints + row + 1, rowIndex - 1, &rowIndices[0],
                                &rowData[0]);
            }
        }

        // at this point, we can also create new rows for the basis matrix
        glp_add_rows(lp, numOutputVars);

        // set bounds == 0
        for (int r = 0; r < numOutputVars; ++r)
        {
            int row = numInitConstraints + numOutputConstraints + r + 1;
            glp_set_row_bnds(lp, row, GLP_FX, 0, 0);  // '==' constraints
        }

        // also create rows for the total inputs effects (number = total output effects
        glp_add_rows(lp, numOutputVars);

        for (int r = 0; r < numOutputVars; ++r)
        {
            int row = numInitConstraints + numOutputConstraints + numOutputVars + r + 1;

            // set bounds == 0
            glp_set_row_bnds(lp, row, GLP_FX, 0, 0);  // '==' constraint

            // assign values to the row initially
            int indices[2] = {0, 0};
            double data[2] = {0, -1};

            indices[1] = numInitVars + numOutputVars + r + 1;

            glp_set_mat_row(lp, row, 1, indices, data);
        }
    }

    void addInputEffectsMatrix(double* mat, int w, int h)
    {
        if (w != numInputs || h != numOutputVars)
        {
            printf(
                "Fatal Error: Matrix dimensions mismatch in addInputEffectsMatrix: "
                "w(%d) != numInputs(%d) || h(%d) != numOutputVars(%d)\n",
                w, numInitVars, h, numOutputVars);
            exit(1);
        }

        if (numOutputConstraints == -1)
        {
            printf("Fatal Error: Output Constraints should be set before addInputEffectsMatrix.\n");
            exit(1);
        }

        // this is done by adding new columns, and setting the constraints
        // use memory on the heap (stack may be too small)
        int numInputConstraints = (int)inputRhs.size();
        vector<int> colIndices(1 + h + numInputConstraints, 0);
        vector<double> colData(1 + h + numInputConstraints, 0.0);

        int numColsBefore = glp_get_num_cols(lp);
        int numRowsBefore = glp_get_num_rows(lp);

        glp_add_cols(lp, numInputs);

        for (int i = 0; i < numInputs; ++i)
            glp_set_col_bnds(lp, 1 + numColsBefore + i, GLP_FR, 0,
                             0);  // free variable (bounds -inf to inf)

        addRows((int)(inputRhs.size()), &inputRhs[0]);

        for (int c = 0; c < numInputs; ++c)
        {
            int lpCol = 1 + c + numColsBefore;

            for (int i = 0; i < h; ++i)
            {
                if (c == 0)  // no sense in re-assigning the indices
                    colIndices[i + 1] =
                        1 + numInitConstraints + numOutputConstraints + numOutputVars + i;

                colData[i + 1] = mat[i * w + c];
            }

            // add the constraints as well

            int arrayIndex = 1 + h;

            for (int cscIndex = inputCscIndptr[c]; cscIndex < inputCscIndptr[c + 1]; ++cscIndex)
            {
                colIndices[arrayIndex] = 1 + numRowsBefore + inputCscIndices[cscIndex];
                colData[arrayIndex++] = inputCscData[cscIndex];
            }

            glp_set_mat_col(lp, lpCol, arrayIndex - 1, &colIndices[0], &colData[0]);
        }
    }

    void updateBasisMatrix(double* mat, int w, int h)
    {
        if (w != numInitVars || h != numOutputVars)
        {
            printf(
                "Fatal Error: Matrix dimensions mismatch in updateBasisMatrix: "
                "w(%d) != numInitVars(%d) || h(%d) != numOutputVars(%d)\n",
                w, numInitVars, h, numOutputVars);
            exit(1);
        }

        if (numOutputConstraints == -1)
        {
            printf("Fatal Error: Output Constraints should be set before updateBasisMatrix.\n");
            exit(1);
        }

        // use memory on the heap (stack may be too small)
        vector<int> rowIndices(w + 3, 0);
        vector<double> rowData(w + 3, 0.0);

        for (int r = 0; r < numOutputVars; ++r)
        {
            int lpRow = numInitConstraints + numOutputConstraints + r + 1;

            for (int i = 0; i < w; ++i)
            {
                if (r == 0)  // no sense in re-assigning the indices
                    rowIndices[i + 1] = 1 + i;

                rowData[i + 1] = mat[r * w + i];
            }

            // negative identity entry (output variables columns)
            rowIndices[w + 1] = 1 + w + r;
            rowData[w + 1] = -1;

            // positive identity entry (total input effects columns)
            rowIndices[w + 2] = 1 + w + numOutputVars + r;
            rowData[w + 2] = 1;

            glp_set_mat_row(lp, lpRow, w + 2, &rowIndices[0], &rowData[0]);
        }
    }

    // returns 0 on success
    // returns 1 on unsat
    int minimize(double* direction, int dirLen, double* result, int resLen)
    {
        if (numInitConstraints == -1 || numOutputConstraints == -1)
        {
            printf("Fatal Error: minimize() called without setting init or output constraints\n");
            exit(1);
        }

        if (dirLen != numOutputVars)
        {
            printf(
                "Fatal Error: dirLen(%d) is not equal numOutputVars(%d) in "
                "call to minimize()\n",
                dirLen, numOutputVars);
            exit(1);
        }

        if (resLen != numOutputVars && resLen != glp_get_num_cols(lp))
        {
            printf(
                "Fatal Error: resultLen(%d) is not equal to numOutputVars (%d) or numCols(%d) in "
                "call to minimize()\n",
                resLen, numOutputVars, glp_get_num_cols(lp));
            exit(1);
        }

        for (int i = 0; i < numOutputVars; ++i)
            glp_set_obj_coef(lp, 1 + numInitVars + i, direction[i]);

        int startIterations = glp_get_it_cnt(lp);

        int simplexRes = glp_simplex(lp, &params);

        if (simplexRes != 0)
        {
            // sometimes the previous solution is singular wrt. current constraints... need to reset
            printf(
                "Warning: hylaa_glpk.h - simplexRes was nonzero (%d). Resetting statuses and "
                "retrying.\n",
                simplexRes);
            resetLp();

            simplexRes = glp_simplex(lp, &params);
        }

        int newIterations = glp_get_it_cnt(lp) - startIterations;
        global.iterations += newIterations;

        ++global.optimizations;

        return processSimplexResult(simplexRes, result, resLen);
    }

    /////////////////////////////////
   private:
    int numOutputVars = 0;
    int numInitVars = 0;
    int numInputs = 0;

    int numInitConstraints = -1;
    int numOutputConstraints = -1;

    // saved input constraints (need to be set at each step, if input is present)
    vector<double> inputCscData;
    vector<int> inputCscIndices;
    vector<int> inputCscIndptr;
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

                if (resLen == numOutputVars)
                {
                    int outputIndex = 0;

                    // copy just the output vars
                    for (int col = numInitVars; col < numInitVars + numOutputVars; ++col)
                        result[outputIndex++] = glp_get_col_prim(lp, col + 1);
                }
                else
                {
                    for (int col = 0; col < resLen && col < numCols; ++col)
                        result[col] = glp_get_col_prim(lp, col + 1);
                }
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

    void checkCsr(const char* label, int w, int h, double* data, int dataLen, int* indices,
                  int indicesLen, int* indptr, int indptrLen)
    {
        if (dataLen != indicesLen)
        {
            printf("Fatal Error: %s csr sparse matrix should have dataLen == indicesLen.\n", label);
            exit(1);
        }

        if (indptrLen != h + 1)
        {
            printf("Fatal Error: %s csr sparse matrix should have indptrLen (%d) == h + 1 (%d).\n",
                   label, indptrLen, h + 1);
            exit(1);
        }

        // make sure each index is less than width
        for (int i = 0; i < indicesLen; ++i)
        {
            if (indices[i] >= w)
            {
                printf(
                    "Fatal Error: %s sparse csr matrix has indices[%d]=%d, which is >= matrix "
                    "width (%d)\n",
                    label, i, indices[i], w);
                exit(1);
            }
        }
    }

    void checkCsc(const char* label, int w, int h, double* data, int dataLen, int* indices,
                  int indicesLen, int* indptr, int indptrLen)
    {
        if (dataLen != indicesLen)
        {
            printf("Fatal Error: %s csc sparse matrix should have dataLen == indicesLen.\n", label);
            exit(1);
        }

        if (indptrLen != w + 1)
        {
            printf("Fatal Error: %s csc sparse matrix should have indptrLen (%d) == w + 1 (%d).\n",
                   label, indptrLen, w + 1);
            exit(1);
        }

        // make sure each index is less than height
        for (int i = 0; i < indicesLen; ++i)
        {
            if (indices[i] >= h)
            {
                printf(
                    "Fatal Error: %s sparse csc matrix has indices[%d]=%d, which is >= matrix "
                    "width (%d)\n",
                    label, i, indices[i], w);
                exit(1);
            }
        }
    }
};

#endif
