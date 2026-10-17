#pragma once

#include <vector>
#include <string>
#include <mkl.h>
#include <stdexcept>



class MatrixBDBT {
private:
    std::vector<double> values, d0, d1, dy, dy_m1, dz, dz_m1;;  // non-zero values + diags
    std::vector<MKL_INT> row_ptr;   // row start indices
    std::vector<MKL_INT> col_indx;  // column indices
    std::vector<double> inverseD;
    bool bOrderingChecked = false;
    bool bRowsSorted = true;
    MKL_INT n, nx, nxy;
public:
    // Constructor computes the product B*D^-1*B^T
    MatrixBDBT(const std::vector<double>& values_csr,         // Stores the non-zero values
        const std::vector<MKL_INT>& col_indx,    // Stores the column indices of non-zero values
        const std::vector<MKL_INT>& rows_start,
        const std::vector<MKL_INT>& rows_end,
        const std::vector<double>& D,
        size_t numRows,
        size_t numCols,
        int nxExt, int  ny, int  nz);

    // Recompute the fixed-stencil coefficients after D changes without
    // rebuilding the sparse product through MKL sparse SYRK.
    void updateDiagonals(
        const std::vector<double>& bValues,
        const std::vector<MKL_INT>& bColumns,
        const std::vector<MKL_INT>& bRowsStart,
        const std::vector<MKL_INT>& bRowsEnd,
        const std::vector<double>& D);

    void spmv_diags(const double* x, double* y) const;
    //std::vector<double> multiplyMatrix(const std::vector<double>& x) const; 

    // Saves the matrix to a file
    void saveMatrix() const;

    // Creates MUMPS object, calculates LU decomposition and returns MUMPS object
    //void makeLU(mumps_obj&);

    // Deallocates the matrix BDBT
    //void deallocate();

    // Getter methods to retrieve the result
    const std::vector<double>& getValues() const;
    const std::vector<MKL_INT>& getRowPtr() const;
    const std::vector<MKL_INT>& getColIndices() const;
    const std::vector<double>& getDiag0() const { return d0; }
    MKL_INT size() const { return n; }
};
