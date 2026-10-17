
#include "MatrixBDBT.h"
#include <fstream>
#include <iostream>
#include <cmath> 
#include <mkl_spblas.h>
#include <mkl.h>
#include <set>
#include "PerformanceProfiler.h"

MatrixBDBT::MatrixBDBT(const std::vector<double>& values_csr,         // Stores the non-zero values
    const std::vector<MKL_INT>& col_indexs,    // Stores the column indices of non-zero values
    const std::vector<MKL_INT>& rows_start,
    const std::vector<MKL_INT>& rows_end,
    const std::vector<double>& D,
    size_t numRows,
    size_t numCols,
    int nxExt, int  ny, int  nz
) {
    std::vector<double> B_values;
    std::vector<MKL_INT> B_row_indices_start;
    std::vector<MKL_INT> B_row_indices_end;
    std::vector<MKL_INT> B_column_indices;
    std::vector<double> BD_values;
    {
        ScopedProfilePhase timer(ProfilePhase::BDBTScale);
        B_values = values_csr;
        B_row_indices_start = rows_start;
        B_row_indices_end = rows_end;
        B_column_indices = col_indexs;
        n = numRows;
        nx = nxExt;
        nxy = (nz > 1) ? nx * ny : 0;
        // Compute B * D^-1 in CSR format.
        BD_values.resize(B_values.size());
        for (size_t k = 0; k < B_values.size(); ++k) {
            int i = B_column_indices[k];
            BD_values[k] = B_values[k] / sqrt(D[i]);
        }
    }

    // Create BD in CSR format
    sparse_matrix_t BD;
    sparse_status_t status = mkl_sparse_d_create_csr(&BD, SPARSE_INDEX_BASE_ZERO, numRows, numCols,
        B_row_indices_start.data(), B_row_indices_end.data(), B_column_indices.data(), BD_values.data());

     status = mkl_sparse_order(BD);
    if (status != SPARSE_STATUS_SUCCESS) {
        std::cerr << "Failed to create BD in CSR format. Status: " << status << std::endl;
        return;
    }


    // Compute the matrix product BD * B^T
    sparse_matrix_t BDBT;
    {
        ScopedProfilePhase timer(ProfilePhase::BDBTSyrk);
        status = mkl_sparse_syrk(SPARSE_OPERATION_NON_TRANSPOSE, BD, &BDBT);
    }


    if (status != SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("Error in mkl_sparse_syrk. Status: " + std::to_string(status));
    }

    // Export BDBT to CSR format
    sparse_index_base_t indexing;
    MKL_INT rows, cols;
    MKL_INT* rows_start_, * rows_end_, * col_indx_;
    double* values_;
    {
        ScopedProfilePhase timer(ProfilePhase::BDBTExport);
        status = mkl_sparse_d_export_csr(BDBT, &indexing, &rows, &cols, &rows_start_, &rows_end_, &col_indx_, &values_);
    }

    if (status != SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("Error in mkl_sparse_d_export_csr");
    }

    // Copy data to member variables
    int nnz = rows_end_[rows - 1];
    {
    ScopedProfilePhase timer(ProfilePhase::BDBTExport);
    values.assign(values_, values_ + nnz);
    row_ptr.assign(rows_end_, rows_end_ + rows);
    row_ptr.insert(row_ptr.begin(), 0);
    col_indx.assign(col_indx_,col_indx_ + nnz);
    }
    // Destroy the matrix handles
    //mkl_sparse_destroy(BT_coo);
    //mkl_sparse_destroy(BT);
    mkl_sparse_destroy(BD);
    mkl_sparse_destroy(BDBT);
    // inside MatrixBDBT constructor after you set row_ptr/col_indx/values and n
    d0.assign(n, 0.0);
    d1.assign(n, 0.0);
    dy.assign(n, 0.0);
    dy_m1.assign(n, 0.0);
    dz.assign(n, 0.0);
    dz_m1.assign(n, 0.0);

    {
        ScopedProfilePhase timer(ProfilePhase::BDBTDiagonalExtract);
#pragma omp parallel for schedule(static)
        for (MKL_INT r = 0; r < n; ++r) {
            for (MKL_INT t = row_ptr[r]; t < row_ptr[r + 1]; ++t) {
                const MKL_INT c = col_indx[t];
                const double  a = values[t];
                const MKL_INT delta = c - r;

                if (delta == 0) d0[r] = a;
                else if (delta == 1) d1[r] = a;
                else if (delta == nx)      dy[r] = a;
                else if (delta == nx - 1)  dy_m1[r] = a;
                else if (delta == nxy)     dz[r] = a;
                else if (delta == nxy - 1) dz_m1[r] = a;
                // else ignore
            }
        }
    }


}

void MatrixBDBT::saveMatrix() const {
    std::ofstream file("matrix_BDBT.dat");
    if (file.is_open()) {
        // Save matrix in CSR format (1-indexed for MATLAB)
        for (int i = 0; i < row_ptr.size() - 1; ++i) {
            for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
                // Add 1 to row and column indices to convert from 0-based to 1-based indexing
                file << i + 1 << " " << col_indx[j] + 1 << " " << values[j] << "\n";
            }
        }
        file.close();
    }
    else {
        std::cerr << "Unable to open file for writing." << std::endl;
    }
}
// Assuming: BDBT is symmetric 7-diagonal stored as CSR with only upper triangle
/*std::vector<double> MatrixBDBT::multiplyMatrix(const std::vector<double>& x) const {
    const int m = static_cast<int>(row_ptr.size()) - 1;
    std::vector<double> result(m, 0.0);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; ++i) {
        for (int jj = row_ptr[i]; jj < row_ptr[i + 1]; ++jj) {
            int j = col_indx[jj];
            double Aij = values[jj];

            if (i == j) {
                result[i] += Aij * x[j];
            } else {
                // Accumulate both symmetric contributions
                result[i] += Aij * x[j];
                result[j] += Aij * x[i];
            }
        }
    }

    return result;
}*/

void MatrixBDBT::spmv_diags(const double* x, double* y) const {
    if (!x || !y) throw std::runtime_error("spmv_diags: null pointer");
    // Contract: caller provides arrays of length n.

    const auto evaluateBoundaryRow = [this, x, y](MKL_INT i) {
        double s = d0[i] * x[i];

        // +/- 1
        if (i + 1 < n) s += d1[i] * x[i + 1];
        if (i >= 1)    s += d1[i - 1] * x[i - 1];

        // +/- nx
        if (nx > 0) {
            if (i + nx < n) s += dy[i] * x[i + nx];
            if (i >= nx)    s += dy[i - nx] * x[i - nx];
        }

        // +/- (nx-1)
        if (nx > 1) {
            const MKL_INT sh = nx - 1;
            if (i + sh < n) s += dy_m1[i] * x[i + sh];
            if (i >= sh)    s += dy_m1[i - sh] * x[i - sh];
        }

        // +/- nxy
        if (nxy > 0) {
            if (i + nxy < n) s += dz[i] * x[i + nxy];
            if (i >= nxy)    s += dz[i - nxy] * x[i - nxy];
        }

        // +/- (nxy-1)
        if (nxy > 1) {
            const MKL_INT sh = nxy - 1;
            if (i + sh < n) s += dz_m1[i] * x[i + sh];
            if (i >= sh)    s += dz_m1[i - sh] * x[i - sh];
        }

        y[i] = s;
    };

    // The diagonal arrays already contain zero for stencil entries that do not
    // exist. Only the first/last maximum-shift rows need range checks; the large
    // middle region can retain the exact accumulation order without branching.
    const bool regular3D = nxy > 1 && nx > 1 && n > 2 * nxy;
    const bool regular2D = nxy == 0 && nx > 1 && n > 2 * nx;
    if (!regular3D && !regular2D) {
#pragma omp parallel for schedule(static)
        for (MKL_INT i = 0; i < n; ++i) evaluateBoundaryRow(i);
        return;
    }

    const MKL_INT interiorBegin = regular3D ? nxy : nx;
    const MKL_INT interiorEnd = n - interiorBegin;
    const double* const d0Data = d0.data();
    const double* const d1Data = d1.data();
    const double* const dyData = dy.data();
    const double* const dyM1Data = dy_m1.data();
    const double* const dzData = dz.data();
    const double* const dzM1Data = dz_m1.data();

#pragma omp parallel
    {
#pragma omp for schedule(static) nowait
        for (MKL_INT i = 0; i < interiorBegin; ++i) evaluateBoundaryRow(i);

        if (regular3D) {
#pragma omp for simd schedule(static) nowait
            for (MKL_INT i = interiorBegin; i < interiorEnd; ++i) {
                double s = d0Data[i] * x[i];
                s += d1Data[i] * x[i + 1];
                s += d1Data[i - 1] * x[i - 1];
                s += dyData[i] * x[i + nx];
                s += dyData[i - nx] * x[i - nx];
                const MKL_INT yDiagonalShift = nx - 1;
                s += dyM1Data[i] * x[i + yDiagonalShift];
                s += dyM1Data[i - yDiagonalShift] * x[i - yDiagonalShift];
                s += dzData[i] * x[i + nxy];
                s += dzData[i - nxy] * x[i - nxy];
                const MKL_INT zDiagonalShift = nxy - 1;
                s += dzM1Data[i] * x[i + zDiagonalShift];
                s += dzM1Data[i - zDiagonalShift] * x[i - zDiagonalShift];
                y[i] = s;
            }
        }
        else {
#pragma omp for simd schedule(static) nowait
            for (MKL_INT i = interiorBegin; i < interiorEnd; ++i) {
                double s = d0Data[i] * x[i];
                s += d1Data[i] * x[i + 1];
                s += d1Data[i - 1] * x[i - 1];
                s += dyData[i] * x[i + nx];
                s += dyData[i - nx] * x[i - nx];
                const MKL_INT yDiagonalShift = nx - 1;
                s += dyM1Data[i] * x[i + yDiagonalShift];
                s += dyM1Data[i - yDiagonalShift] * x[i - yDiagonalShift];
                y[i] = s;
            }
        }

#pragma omp for schedule(static)
        for (MKL_INT i = interiorEnd; i < n; ++i) evaluateBoundaryRow(i);
    }
}

void MatrixBDBT::updateDiagonals(
    const std::vector<double>& bValues,
    const std::vector<MKL_INT>& bColumns,
    const std::vector<MKL_INT>& bRowsStart,
    const std::vector<MKL_INT>& bRowsEnd,
    const std::vector<double>& D) {
    if (bRowsStart.size() < static_cast<std::size_t>(n) ||
        bRowsEnd.size() < static_cast<std::size_t>(n) ||
        bValues.size() != bColumns.size()) {
        throw std::invalid_argument("MatrixBDBT::updateDiagonals received inconsistent CSR data");
    }

    inverseD.resize(D.size());
#pragma omp parallel for schedule(static)
    for (MKL_INT i = 0; i < static_cast<MKL_INT>(D.size()); ++i) {
        inverseD[i] = 1.0 / D[i];
    }

    if (!bOrderingChecked) {
        bRowsSorted = true;
        for (MKL_INT row = 0; row < n && bRowsSorted; ++row) {
            for (MKL_INT entry = bRowsStart[row] + 1; entry < bRowsEnd[row]; ++entry) {
                if (bColumns[entry - 1] > bColumns[entry]) {
                    bRowsSorted = false;
                    break;
                }
            }
        }
        bOrderingChecked = true;
    }

    const auto weightedRowDot = [&](MKL_INT rowA, MKL_INT rowB) {
        if (!bRowsSorted) {
            double sum = 0.0;
            for (MKL_INT a = bRowsStart[rowA]; a < bRowsEnd[rowA]; ++a) {
                for (MKL_INT b = bRowsStart[rowB]; b < bRowsEnd[rowB]; ++b) {
                    if (bColumns[a] == bColumns[b]) {
                        sum += bValues[a] * bValues[b] * inverseD[bColumns[a]];
                    }
                }
            }
            return sum;
        }

        MKL_INT a = bRowsStart[rowA];
        MKL_INT b = bRowsStart[rowB];
        const MKL_INT aEnd = bRowsEnd[rowA];
        const MKL_INT bEnd = bRowsEnd[rowB];
        double sum = 0.0;
        while (a < aEnd && b < bEnd) {
            const MKL_INT columnA = bColumns[a];
            const MKL_INT columnB = bColumns[b];
            if (columnA < columnB) {
                ++a;
            }
            else if (columnB < columnA) {
                ++b;
            }
            else {
                sum += bValues[a] * bValues[b] * inverseD[columnA];
                ++a;
                ++b;
            }
        }
        return sum;
    };

#pragma omp parallel for schedule(static)
    for (MKL_INT row = 0; row < n; ++row) {
        d0[row] = weightedRowDot(row, row);
        d1[row] = row + 1 < n ? weightedRowDot(row, row + 1) : 0.0;
        dy[row] = nx > 0 && row + nx < n ? weightedRowDot(row, row + nx) : 0.0;
        const MKL_INT yDiagonalShift = nx - 1;
        dy_m1[row] = nx > 1 && row + yDiagonalShift < n
            ? weightedRowDot(row, row + yDiagonalShift) : 0.0;
        dz[row] = nxy > 0 && row + nxy < n ? weightedRowDot(row, row + nxy) : 0.0;
        const MKL_INT zDiagonalShift = nxy - 1;
        dz_m1[row] = nxy > 1 && row + zDiagonalShift < n
            ? weightedRowDot(row, row + zDiagonalShift) : 0.0;
    }

    // Keep the diagnostic CSR representation coherent for saveMatrix/getters.
#pragma omp parallel for schedule(static)
    for (MKL_INT row = 0; row < n; ++row) {
        for (MKL_INT entry = row_ptr[row]; entry < row_ptr[row + 1]; ++entry) {
            const MKL_INT delta = col_indx[entry] - row;
            if (delta == 0) values[entry] = d0[row];
            else if (delta == 1) values[entry] = d1[row];
            else if (delta == nx) values[entry] = dy[row];
            else if (delta == nx - 1) values[entry] = dy_m1[row];
            else if (delta == nxy) values[entry] = dz[row];
            else if (delta == nxy - 1) values[entry] = dz_m1[row];
        }
    }
}


//Getters
const std::vector<double>& MatrixBDBT::getValues() const {
    return values;
}

const std::vector<MKL_INT>& MatrixBDBT::getRowPtr() const {
    return row_ptr;
}

const std::vector<MKL_INT>& MatrixBDBT::getColIndices() const {
    return col_indx;
}
