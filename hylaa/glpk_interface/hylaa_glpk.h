// Stanley Bak
// Hylaa GLPK interface header
// Nov 2016

#include <glpk.h>
#include <vector>

using namespace std;

#ifndef HYLAA_GLPK_H_
#define HYLAA_GLPK_H_

struct GlobalLpData
{
    int optimizations = 0;
    int iterations = 0;
};

extern GlobalLpData global;

class LpData
{
   public:
    LpData(int numStandardVars, int numBasisVars)
        : numStandardVars(numStandardVars),
          numBasisVars(numBasisVars),
          basisConstraintCols(numBasisVars),
          basisConstraintVals(numBasisVars)
    {
        // setup lp
        lp = glp_create_prob();
        glp_set_obj_dir(lp, GLP_MIN);

        // setup lp params
        glp_init_smcp(&params);
        params.msg_lev = GLP_MSG_OFF;
        // params.out_frq = 1;

        // params.presolve = GLP_ON;
        // params.meth = GLP_DUALP;

        // the first n variables are the standard basis variables
        // the second m variables are the star's basis variables
        glp_add_cols(lp, numStandardVars + numBasisVars);

        for (int i = 0; i < numStandardVars + numBasisVars; ++i)
            glp_set_col_bnds(lp, i + 1, GLP_FR, 0, 0);  // free variable (bounds -inf to inf)
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

    // returns 0 on success, 1 on error
    int getRowStatuses(char* store, int storeLen)
    {
        int rows = glp_get_num_rows(lp);

        if (storeLen != rows)
        {
            printf(
                "Error: Vector dimensions mismatch in getRowStatuses: %d (storage length) != "
                "%d (num rows)\n",
                storeLen, rows);
            return 1;
        }

        for (int row = 1; row <= rows; ++row)
            store[row - 1] = glp_get_row_stat(lp, row);

        return 0;
    }

    int getColStatuses(char* store, int storeLen)
    {
        int cols = glp_get_num_cols(lp);

        if (storeLen != cols)
        {
            printf(
                "Error: Vector dimensions mismatch in getColStatuses: %d (storage length) != "
                "%d (num cols)\n",
                storeLen, cols);
            return 1;
        }

        for (int col = 1; col <= cols; ++col)
            store[col - 1] = glp_get_col_stat(lp, col);

        return 0;
    }

    // returns 1 on error, 0 on success
    int updateBasisMatrix(double* matrix, int w, int h)
    {
        // the transpose of the matrix goes into the LP
        if (w != numStandardVars || h != numBasisVars)
        {
            printf(
                "Fatal Error: Matrix dimensions mismatch in updateBasisMatrix: "
                "w(%d) != numStandardVars(%d) || h(%d) != numBasisVars(%d)\n",
                w, numStandardVars, h, numBasisVars);
            return 1;
        }

        int numRows = glp_get_num_rows(lp);

        if (numRows == 0)
        {
            // new problem instance, create one constraint row for each equality constraint
            glp_add_rows(lp, numStandardVars);

            for (int row = 0; row < numStandardVars; ++row)
            {
                glp_set_row_bnds(lp, row + 1, GLP_FX, 0, 0);

                int inds[] = {0, row + 1};
                double vals[] = {0, -1};

                glp_set_mat_col(lp, row + 1, 1, inds, vals);
            }
        }

        // replace each column in the lp with a row in the current basis matrix
        // (this transposes the matrix)
        for (int row = 0; row < h; ++row)
        {
            int index = 1;

            int inds[numStandardVars + basisConstraintCols[row].size() + 1];
            double vals[numStandardVars + basisConstraintCols[row].size() + 1];

            for (int col = 0; col < w; ++col)
            {
                double val = matrix[row * w + col];

                if (val != 0)
                {
                    inds[index] = col + 1;
                    vals[index++] = val;
                }
            }

            // also add all the basis constraint columns
            for (unsigned int i = 0; i < basisConstraintCols[row].size(); ++i)
            {
                inds[index] = basisConstraintCols[row][i];
                vals[index++] = basisConstraintVals[row][i];
            }

            glp_set_mat_col(lp, numStandardVars + row + 1, index - 1, inds, vals);
        }

        return 0;
    }

    /**
     * Add a constraint in the basis matrix (a star constraint)
     */
    void addBasisConstraint(double* aVec, int aVecLen, double bVal)
    {
        if (aVecLen != numBasisVars)
        {
            printf("Fatal Error: aVecLen wrong in addBasisConstraint().\n");
            exit(1);
        }
        
        // added this check
        if (addedInput)
        {
            printf("Fatal Error: All basis constraints should be added before inputs.\n");
            exit(1);
        }

        // create new row for the constraint
        glp_add_rows(lp, 1);

        int row = glp_get_num_rows(lp);

        glp_set_row_bnds(lp, row, GLP_UP, 0, bVal);

        int inds[numBasisVars + 1];
        double vals[numBasisVars + 1];

        int index = 1;

        for (int i = 0; i < numBasisVars; ++i)
        {
            double val = aVec[i];

            if (val != 0)
            {
                inds[index] = numStandardVars + i + 1;  // constraint on basis variable i
                vals[index++] = val;

                basisConstraintCols[i].push_back(row);
                basisConstraintVals[i].push_back(val);
            }
        }

        glp_set_mat_row(lp, row, index - 1, inds, vals);
        basisConstraintRows.push_back(row);
    }

    // this is used to offset by the center simulation in a combined_lpi with inputs
    void setStandardConstraintValues(double* vals, int len)
    {
        if (len != (int)standardConstraintRows.size())
        {
            printf("Fatal Error: vals array length wrong in setStandardConstraintValues.\n");
            exit(1);
        }

        for (int r = 0; r < len; ++r)
        {
            int row = (int)standardConstraintRows[r];
            
            glp_set_row_bnds(lp, row, GLP_UP, 0, vals[r]);
        }
    }
    
    // this is used in the eat_star operation, to relax a star's constraints
    void setBasisConstraintValues(double* vals, int len)
    {
        if (len != (int)basisConstraintRows.size())
        {
            printf("Fatal Error: valls array length wrong in setBasisConstraintValues.\n");
            exit(1);
        }

        for (int r = 0; r < len; ++r)
        {
            int row = basisConstraintRows[r];
            
            glp_set_row_bnds(lp, row, GLP_UP, 0, vals[r]);
        }
    }

    void addStandardConstraint(double* aVec, int aVecLen, double bVal)
    {
        if (aVecLen != numStandardVars)
        {
            printf("Fatal Error: aVecLen length wrong in addStandardConstraint.\n");
            exit(1);
        }

        if (addedInput)
        {
            printf("Fatal Error: All standard constraints should be added before inputs.\n");
            exit(1);
        }

        // create new row for the constraint
        glp_add_rows(lp, 1);

        int row = glp_get_num_rows(lp);

        glp_set_row_bnds(lp, row, GLP_UP, 0, bVal);

        int inds[numStandardVars + 1];
        double vals[numStandardVars + 1];

        int index = 1;

        for (int i = 0; i < numStandardVars; ++i)
        {
            double val = aVec[i];

            if (val != 0)
            {
                inds[index] = i + 1;  // constraint on standard variable i
                vals[index++] = val;
            }
        }

        glp_set_mat_row(lp, row, index - 1, inds, vals);
        standardConstraintRows.push_back(row);

        // printf("add_constraint (<= %f): ", b_val);
        // print_inds_vals("set_mat_row", row, index - 1, inds, vals);
    }

    void addInputStar(double* aMatrixT, int aWidth, int aHeight, double* bVec, int bLen,
                      double* basisMatrix, int bmWidth, int bmHeight)
    {
        if (bmWidth != numStandardVars || aHeight != bmHeight || aWidth != bLen)
        {
            printf("Fatal Error: Matrix size error in addInputStar. One of the following conditions failed: "
                "bmWidth(%d)==numStandardVars(%d) aHeight(%d)==bmHeight(%d), aWidth(%d)==bLen(%d)\n", 
                bmWidth, numStandardVars, aHeight, bmHeight, aWidth, bLen);
            exit(1);
        }

        if (numInputs > 0 && numInputs != aHeight)
        {
            printf("Fatal Error: num inputs changed\n");
            exit(1);
        }

        if (numInputConstraints > 0 && numInputConstraints != aWidth)
        {
            printf("Fatal Error: num input constraints changed\n");
            exit(1);
        }

        int lpRows = glp_get_num_rows(lp);
        int lpCols = glp_get_num_cols(lp);

        addRows(lpRows, bLen, bVec);
        addCols(lpCols, aHeight);

        populateInputConstraints(lpRows, lpCols, aMatrixT, aWidth, aHeight, basisMatrix, bmWidth,
                                 bmHeight);

        // if it's not the first input, then copy the previous-inputs lp solution
        if (!addedInput)
        {
            addedInput = true;
            numInputs = aHeight;
            numInputConstraints = aWidth;
        }
        else
            copyLastInputSolution(lpRows, lpCols, aHeight, aWidth);
    }

    void setStandardBasisStatuses(char* rowStats, int rLen, char* colStats, int cLen)
    {
        if (cLen != numStandardVars + numBasisVars)
        {
            printf(
                "Fatal Error: num col statuses (%d) in setStandardBasisStatuses was not equal to "
                "numStandardVars + numBasisVars (%d)\n",
                cLen, numStandardVars + numBasisVars);
            exit(1);
        }

        if (rLen != numStandardVars + (int)basisConstraintRows.size())
        {
            printf(
                "Fatal Error: num row statuses (%d) in setStandardBasisStatuses was not equal to "
                "numStandardVars + basisConstraints.size() (%d)\n",
                rLen, numStandardVars + (int)basisConstraintRows.size());
            exit(1);
        }

        for (int c = 0; c < cLen; ++c)
            glp_set_col_stat(lp, c + 1, colStats[c]);
            
        // rows are split between standard rows (first few rows) and basis statuses (use basisConstraintRows)
        for (int r = 0; r < numStandardVars; ++r)
            glp_set_row_stat(lp, r + 1, rowStats[r]);
            
        for (int r = numStandardVars; r < rLen; ++r)
        {
            int row = basisConstraintRows[r - numStandardVars];
            glp_set_row_stat(lp, row, rowStats[r]);
        }
    }

    void setLastInputStatuses(char* rowStats, int rLen, char* colStats, int cLen)
    {
        if (!addedInput)
        {
            printf("Fatal Error: setLastInputStatuses was called before input star was added.\n");
            exit(1);
        }

        if (rLen != numInputConstraints)
        {
            printf(
                "Fatal Error: num row statuses (%d) in setLastInputStatuses was not equal to "
                "number of number of input constaints (%d)\n",
                rLen, numInputConstraints);
            exit(1);
        }

        if (cLen != numInputs)
        {
            printf(
                "Fatal Error: num column statuses (%d) in setLastInputStatuses was not equal to "
                "number of number of inputs (%d)\n",
                cLen, numInputs);
            exit(1);
        }

        int lpRows = glp_get_num_rows(lp);
        int lpCols = glp_get_num_cols(lp);

        for (int r = 0; r < rLen; ++r)
            glp_set_row_stat(lp, lpRows - rLen + r + 1, rowStats[r]);

        for (int c = 0; c < cLen; ++c)
            glp_set_col_stat(lp, lpCols - cLen + c + 1, colStats[c]);
    }

    // returns 0 on success
    // returns 1 on unsat
    int minimize(double* direction, int dirLen, double* result, int resLen)
    {
        ++global.optimizations;

        if (dirLen != numStandardVars)
        {
            printf("Fatal Error: dirLen(%d) is not equal to numStandardVars(%d) in call to minimize()\n", dirLen, numStandardVars);
            exit(1);
        }

        for (int i = 0; i < numStandardVars; ++i)
            glp_set_obj_coef(lp, 1 + i, direction[i]);
            
        for (int i = 0; i < numBasisVars; ++i)
            glp_set_obj_coef(lp, 1 + numStandardVars + i, 0);

        int startIterations = glp_get_it_cnt(lp);

        int simplexRes = glp_simplex(lp, &params);
        
        int newIterations = glp_get_it_cnt(lp) - startIterations;
        global.iterations += newIterations;
        
        return processSimplexResult(simplexRes, result, resLen);
    }
    
/////////////////////////////////
   private:
    bool addedInput = false;  // have we added any input stars yet?
    int numStandardVars = 0;  // number of standard variables
    int numBasisVars = -1;    // number of basis variables
    int numInputConstraints = -1;

    int numInputs = 0;
    glp_prob* lp = nullptr;
    glp_smcp params;

    vector<vector<int>> basisConstraintCols;
    vector<vector<double>> basisConstraintVals;
    
    vector<int> standardConstraintRows; //  use size() when numStandardConstraints is needed
    vector<int> basisConstraintRows; //  use size() when numBasisConstraints is needed

    void addRows(int numRows, int num, double* bound)
    {
        glp_add_rows(lp, num);

        for (int r = 0; r < num; ++r)
            glp_set_row_bnds(lp, numRows + r + 1, GLP_UP, 0, bound[r]);  // row <= constraint_b
    }

    void addCols(int numCols, int num)
    {
        glp_add_cols(lp, num);

        for (int c = 0; c < num; ++c)
            glp_set_col_bnds(lp, numCols + c + 1, GLP_FR, 0, 0);  // free variable (-inf to inf)
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

    // input basis matrix is pre-transposed:
    // number of rows = # input variables, number cols = #standard variables
    void populateInputConstraints(int lpRows, int lpCols, double* aMatrixT, int aWidth, int aHeight,
                                  double* basisMatrix, int bmWidth, int bmHeight)
    {
        int inds[bmWidth + aHeight + 1];
        double vals[bmWidth + aHeight + 1];

        for (int r = 0; r < bmHeight; ++r)
        {
            int index = 1;

            // first add a column of the input basis matrix
            for (int c = 0; c < bmWidth; ++c)
            {
                double val = basisMatrix[r * bmWidth + c];

                if (val != 0)
                {
                    inds[index] = c + 1;
                    vals[index++] = val;
                }
            }

            // next, add the column of the constraint matrix
            for (int c = 0; c < aWidth; ++c)
            {
                double val = aMatrixT[r * aWidth + c];

                if (val != 0)
                {
                    inds[index] = lpRows + c + 1;
                    vals[index++] = val;
                }
            }

            glp_set_mat_col(lp, lpCols + r + 1, index - 1, inds, vals);
        }
    }

    void copyLastInputSolution(int lpRows, int lpCols, int aWidth, int aHeight)
    {
        // lpRows and lpCols are the counts BEFORE we added the input star

        for (int r = 0; r < aHeight; ++r)
        {
            int status = glp_get_row_stat(lp, lpRows - aHeight + r + 1);
            glp_set_row_stat(lp, lpRows + r + 1, status);
        }

        for (int c = 0; c < aWidth; ++c)
        {
            int status = glp_get_col_stat(lp, lpCols - aWidth + c + 1);
            glp_set_col_stat(lp, lpCols + c + 1, status);
        }
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

            printf("Fatal Error: glp_simplex returned nonzero status (%s) in minimize(): %d\n", msg, simplexRes);
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

                // print result
                /*printf("lp result = ");
                for (int col = 0; col < glp_get_num_cols(lp); ++col)
                    printf("%f ", glp_get_col_prim(lp, col + 1));

                printf("\n");*/
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

                printf("Fatal Error: LP Status after solving in minimize() was '%s': %d\n", message, status);
                exit(1);
            }
        }

        return rv;
    }
};

#endif
