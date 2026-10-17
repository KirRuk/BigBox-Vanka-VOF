#include "PressureMatrix.h"
#include <fstream>
#include <algorithm>
#include "mkl.h"
#include "mkl_spblas.h"
#include "Point.h"
#include <limits>
#include <stdexcept>
#include <utility>

namespace {
struct ScopedSparseMatrix {
    sparse_matrix_t handle = nullptr;

    ScopedSparseMatrix() = default;

    ~ScopedSparseMatrix() {
        if (handle != nullptr) {
            mkl_sparse_destroy(handle);
        }
    }

    ScopedSparseMatrix(const ScopedSparseMatrix&) = delete;
    ScopedSparseMatrix& operator=(const ScopedSparseMatrix&) = delete;
};

matrix_descr generalMatrixDescriptor() {
    matrix_descr descriptor{};
    descriptor.type = SPARSE_MATRIX_TYPE_GENERAL;
    descriptor.mode = SPARSE_FILL_MODE_FULL;
    descriptor.diag = SPARSE_DIAG_NON_UNIT;
    return descriptor;
}
} // namespace

bool existsInDirichletRow(const std::vector<std::vector<MKL_INT>>& dirichletRow, int i, int j, int k) {
    // Construct a temporary vector with the current i, j, k values
    std::vector<MKL_INT> currentRow = { i, j, k };

    // Check if this vector exists in dirichletRow
    return std::find(dirichletRow.begin(), dirichletRow.end(), currentRow) != dirichletRow.end();
}

PressureMatrix::PressureMatrix(const std::vector<std::vector<std::vector<Point>>>& u,
    const std::vector<std::vector<std::vector<Point>>>& v,
    const std::vector<std::vector<std::vector<Point>>>& w,
    std::vector<std::vector<std::vector<Point>>>& p,
    BoundaryContainer& BC, DiagonalMatrix& D, bool symetry) {
    std::vector<double> values;
    std::vector<MKL_INT> column_indices, row_indices;
    int number_of_u = u.back().back().back().getUniqueNumber()+1;
    int number_of_uv = number_of_u + v.back().back().back().getUniqueNumber()+1;
    int sizeZ = w.size();
    bool is3D = sizeZ > 1;
    std::vector<MKL_INT> dirichletPoint;
    std::vector<std::vector<MKL_INT>> dirichletRow;
    if (BC.getBoundary(PointBoundary::E).type == BoundaryKind::DirichletPoint) {
        dirichletPoint.push_back(1);
        dirichletPoint.push_back(BC.getBoundary(PointBoundary::W).value);//X coordinate
        dirichletPoint.push_back(BC.getBoundary(PointBoundary::N).value);//Y coordinate
        dirichletPoint.push_back(BC.getBoundary(PointBoundary::S).value);//Z coordinate
    }
    else if (BC.getBoundary(PointBoundary::E).type == BoundaryKind::DirichletRow) {
        dirichletPoint.push_back(2);
        dirichletPoint.push_back(BC.getBoundary(PointBoundary::W).value);//X coordinate
        dirichletPoint.push_back(BC.getBoundary(PointBoundary::N).value);//Y coordinate
        dirichletPoint.push_back(BC.getBoundary(PointBoundary::S).value);//Z coordinate
        dirichletPoint.push_back(BC.getBoundary(PointBoundary::F).value);//Direction
        dirichletPoint.push_back(BC.getBoundary(PointBoundary::B).value);//Number of points
        
        for (int n = 0; n < dirichletPoint[5]; n++) {
            std::vector<MKL_INT> currentValue;
            int dx = 0, dy = 0, dz = 0;
            switch (dirichletPoint[4]) {
            case -1: dx = -n; break;
            case 1:  dx = n;  break;
            case -2: dy = -n; break;
            case 2:  dy = n;  break;
            case -3: dz = -n; break;
            case 3:  dz = n;  break;
            default: throw std::runtime_error("Unknown direction for pressure row");
            }
            int nx = dirichletPoint[1] + dx;
            int ny = dirichletPoint[2] + dy;
            int nz = dirichletPoint[3] + dz;
            currentValue.push_back(nx);
            currentValue.push_back(ny);
            currentValue.push_back(nz);
            dirichletRow.push_back(currentValue);

        }      
    }
    else {
        dirichletPoint.push_back(0);
    }
        
    int numberDirichletPoints = 0;
    // Loop through p 
    for (int k = 0; k < p.size(); k++) {
        for (int j = 0; j < p[k].size() ; j++) {
            for (int i = 0; i < p[k][j].size() ; i++) {

                if (dirichletPoint.size() >= 4 && dirichletPoint[0]==1&& dirichletPoint[1]==i && dirichletPoint[2] == j && dirichletPoint[3] == k) {
                    D.addPressureDirichlet();
                    p[k][j][i].switchUniqueNumber(0,p);
                    numberDirichletPoints++;
                }
                else if (dirichletPoint[0] == 2 && existsInDirichletRow(dirichletRow, i, j, k)) {
                    numberDirichletPoints++;
                    D.addPressureDirichlet();
                    p[k][j][i].switchUniqueNumber(0, p);
                }
                else if (dirichletPoint[0] == 0) {
                    //To add code that initiates dirichle pressure points on outflow boundary;
                    throw std::runtime_error("Unkown pressure boundary");
                }
                else {

                    // Continuity equation
                    int row = p[k][j][i].getUniqueNumber()- numberDirichletPoints;
                    int col1 = u[(is3D ? k+1 : 0)][j+1][i+1].getUniqueNumber();
                    if(symetry && (u[(is3D ? k + 1 : 0)][j + 1][i + 1].getBoundaryType() == PointBoundary::E)) col1 = u[(is3D ? k + 1 : 0)][j + 1][0].getUniqueNumber();
                    int col2 = u[(is3D ? k+1 : 0)][j+1][i].getUniqueNumber();
                    int col3 = number_of_u + v[(is3D ? k+1 : 0)][j+1][i+1].getUniqueNumber();
                    int col4 = number_of_u + v[(is3D ? k+1 : 0)][j][i+1].getUniqueNumber();
                    int col5 = 0, col6 = 0;
                    if (is3D) {
                        col5 = number_of_uv + w[k + 1][j+1][i+1].getUniqueNumber();
                        col6 = number_of_uv + w[k][j+1][i+1].getUniqueNumber();
                    }
                    double dx = p[k][j][i].getdx();
                    double dy = p[k][j][i].getdy();
                    double dz = p[k][j][i].getdz();
                    double value1, value2, value3, value4, value5, value6;
                    if (!symetry) {
                        value1 = (u[(is3D ? k+1 : 0)][j+1][i + 1].getBoundaryType() != PointBoundary::Interior) ? 0.0 : (is3D ? -dy * dz : -dy);
                        value2 = (u[(is3D ? k+1 : 0)][j+1][i].getBoundaryType() != PointBoundary::Interior) ? 0.0 : (is3D ? dy * dz : dy);
                        value3 = (v[(is3D ? k+1 : 0)][j + 1][i+1].getBoundaryType() != PointBoundary::Interior) ? 0.0 : (is3D ? -dx * dz : -dx);
                        value4 = (v[(is3D ? k+1 : 0)][j][i+1].getBoundaryType() != PointBoundary::Interior) ? 0.0 : (is3D ? dx * dz : dx);
                        value5 = 0.0, value6 = 0.0;
                        if (is3D) {
                            value5 = (w[k + 1][j+1][i+1].getBoundaryType() != PointBoundary::Interior) ? 0.0 : -dy * dx;
                            value6 = (w[k][j+1][i+1].getBoundaryType() != PointBoundary::Interior) ? 0.0 : dy * dx;
                        }
                    }
                    else {
                        if (u[(is3D ? k + 1 : 0)][j + 1][i + 1].getBoundaryType() == PointBoundary::Interior) value1 = (is3D ? -dy * dz : -dy);
                        else if ((u[(is3D ? k + 1 : 0)][j + 1][i + 1].getBoundaryType() == PointBoundary::E)) value1 = (is3D ? -dy * dz : -dy);
                        else value1 = 0.0;
                        if (u[(is3D ? k + 1 : 0)][j + 1][i].getBoundaryType() == PointBoundary::Interior) value2 = (is3D ? dy * dz : dy);
                        else if (u[(is3D ? k + 1 : 0)][j + 1][i].getBoundaryType() == PointBoundary::W) value2 = (is3D ? dy * dz : dy);
                        else value2 = 0.0;
                        value3 = (v[(is3D ? k + 1 : 0)][j + 1][i + 1].getBoundaryType() != PointBoundary::Interior) ? 0.0 : (is3D ? -dx * dz : -dx);
                        value4 = (v[(is3D ? k + 1 : 0)][j][i + 1].getBoundaryType() != PointBoundary::Interior) ? 0.0 : (is3D ? dx * dz : dx);
                        value5 = 0.0, value6 = 0.0;
                        if (is3D) {
                            value5 = (w[k + 1][j + 1][i + 1].getBoundaryType() != PointBoundary::Interior) ? 0.0 : -dy * dx;
                            value6 = (w[k][j + 1][i + 1].getBoundaryType() != PointBoundary::Interior) ? 0.0 : dy * dx;
                        }
                    }


                    // Add the non-zero values and their column indices
                    values.push_back(value1);
                    values.push_back(value2);
                    column_indices.push_back(col1);
                    column_indices.push_back(col2);
                    row_indices.push_back(row);
                    row_indices.push_back(row);

                    values.push_back(value3);
                    values.push_back(value4);
                    column_indices.push_back(col3);
                    column_indices.push_back(col4);
                    row_indices.push_back(row);
                    row_indices.push_back(row);

                    if (is3D) {
                        values.push_back(value5);
                        values.push_back(value6);
                        column_indices.push_back(col5);
                        column_indices.push_back(col6);
                        row_indices.push_back(row);
                        row_indices.push_back(row);
                    }
                }
            }
        }
    }

    int nnz = values.size();
    rows = row_indices[row_indices.size()-1]+1;
    cols = D.getDiag().size();

    ScopedSparseMatrix coo_matrix;
    if (mkl_sparse_d_create_coo(&coo_matrix.handle, SPARSE_INDEX_BASE_ZERO, rows, cols, nnz, &row_indices[0], &column_indices[0], &values[0]) != SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to create COO matrix handle");
    }

    ScopedSparseMatrix converted_csr_matrix;
    if (mkl_sparse_convert_csr(coo_matrix.handle, SPARSE_OPERATION_NON_TRANSPOSE, &converted_csr_matrix.handle) != SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to convert matrix to CSR format");
    }

    // Export CSR matrix from internal representation
    sparse_index_base_t indexing;
    MKL_INT mkl_rows, mkl_cols;
    MKL_INT* mkl_rows_start_ptr;
    MKL_INT* mkl_rows_end_ptr;
    MKL_INT* mkl_col_indx_ptr;
    double* mkl_values_csr_ptr;

    if (mkl_sparse_d_export_csr(converted_csr_matrix.handle, &indexing, &mkl_rows, &mkl_cols, &mkl_rows_start_ptr, &mkl_rows_end_ptr, &mkl_col_indx_ptr, &mkl_values_csr_ptr) != SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to export CSR matrix");
    }

    values_csr.assign(mkl_values_csr_ptr, mkl_values_csr_ptr + nnz);
    col_indx.assign(mkl_col_indx_ptr, mkl_col_indx_ptr + nnz);
    rows_start.assign(mkl_rows_start_ptr, mkl_rows_start_ptr + mkl_rows);
    rows_end.assign(mkl_rows_end_ptr, mkl_rows_end_ptr + mkl_rows);
    rows = mkl_rows;
    cols = mkl_cols;

    createSparseHandle();
}

PressureMatrix::~PressureMatrix() noexcept {
    destroySparseHandle();
}

PressureMatrix::PressureMatrix(const PressureMatrix& other)
    : values_csr(other.values_csr),
      col_indx(other.col_indx),
      rows_start(other.rows_start),
      rows_end(other.rows_end),
      rows(other.rows),
      cols(other.cols) {
    createSparseHandle();
}

PressureMatrix& PressureMatrix::operator=(const PressureMatrix& other) {
    if (this != &other) {
        PressureMatrix copy(other);
        swap(copy);
    }
    return *this;
}

PressureMatrix::PressureMatrix(PressureMatrix&& other) noexcept
    : values_csr(std::move(other.values_csr)),
      col_indx(std::move(other.col_indx)),
      rows_start(std::move(other.rows_start)),
      rows_end(std::move(other.rows_end)),
      rows(other.rows),
      cols(other.cols),
      csr_handle(other.csr_handle) {
    other.rows = 0;
    other.cols = 0;
    other.csr_handle = nullptr;
}

PressureMatrix& PressureMatrix::operator=(PressureMatrix&& other) noexcept {
    if (this != &other) {
        PressureMatrix moved(std::move(other));
        swap(moved);
    }
    return *this;
}

void PressureMatrix::createSparseHandle() {
    if (rows == 0 && cols == 0 && values_csr.empty() && col_indx.empty() &&
        rows_start.empty() && rows_end.empty()) {
        return;
    }
    if (rows == 0 || cols == 0 || rows_start.size() != rows || rows_end.size() != rows) {
        throw std::runtime_error("Cannot create MKL handle for an invalid pressure matrix");
    }
    if (col_indx.size() != values_csr.size()) {
        throw std::runtime_error("Pressure matrix CSR columns and values have different sizes");
    }
    if (rows > static_cast<size_t>(std::numeric_limits<MKL_INT>::max()) ||
        cols > static_cast<size_t>(std::numeric_limits<MKL_INT>::max())) {
        throw std::runtime_error("Pressure matrix dimensions exceed the configured MKL integer range");
    }

    sparse_matrix_t new_handle = nullptr;
    const sparse_status_t create_status = mkl_sparse_d_create_csr(
        &new_handle,
        SPARSE_INDEX_BASE_ZERO,
        static_cast<MKL_INT>(rows),
        static_cast<MKL_INT>(cols),
        rows_start.data(),
        rows_end.data(),
        col_indx.data(),
        values_csr.data());
    if (create_status != SPARSE_STATUS_SUCCESS) {
        if (new_handle != nullptr) {
            (void)mkl_sparse_destroy(new_handle);
        }
        throw std::runtime_error(
            "Failed to create persistent pressure-matrix CSR handle. MKL status: " +
            std::to_string(static_cast<int>(create_status)));
    }

    csr_handle = new_handle;
    const matrix_descr descriptor = generalMatrixDescriptor();
    constexpr MKL_INT expected_calls = 10000;
    // Keep the very large 3D CSR arrays as the single source of matrix storage.
    // MKL may still build small workload-balancing metadata, but not a duplicate
    // internal representation of the matrix values and structure.
    (void)mkl_sparse_set_memory_hint(csr_handle, SPARSE_MEMORY_NONE);
    (void)mkl_sparse_set_mv_hint(
        csr_handle, SPARSE_OPERATION_NON_TRANSPOSE, descriptor, expected_calls);
    (void)mkl_sparse_set_mv_hint(
        csr_handle, SPARSE_OPERATION_TRANSPOSE, descriptor, expected_calls);
    (void)mkl_sparse_optimize(csr_handle);
}

void PressureMatrix::destroySparseHandle() noexcept {
    if (csr_handle != nullptr) {
        (void)mkl_sparse_destroy(csr_handle);
        csr_handle = nullptr;
    }
}

void PressureMatrix::swap(PressureMatrix& other) noexcept {
    using std::swap;
    swap(values_csr, other.values_csr);
    swap(col_indx, other.col_indx);
    swap(rows_start, other.rows_start);
    swap(rows_end, other.rows_end);
    swap(rows, other.rows);
    swap(cols, other.cols);
    swap(csr_handle, other.csr_handle);
}

double PressureMatrix::getValue(int row, int col) const {
    // Use the rows_start and rows_end vectors to determine the range to search within col_indx
    for (int i = rows_start[row]; i < rows_end[row]; ++i) {
        if (col_indx[i] == col) {
            return values_csr[i];
        }
    }

    // If column index isn't found for the given row, return 0
    return 0.0;
}

const std::vector<double>& PressureMatrix::getValues() const {
    return values_csr;
}

const std::vector<MKL_INT>& PressureMatrix::getColumnIndices() const {
    return col_indx;
}

const std::vector<MKL_INT>& PressureMatrix::getRowStart() const {
    return rows_start;
}
const std::vector<MKL_INT>& PressureMatrix::getRowEnd() const {
    return rows_end;
}
// Multiplication by a vector from the right
std::vector<double> PressureMatrix::multiply(const std::vector<double>& vec) const {
    std::vector<double> result;
    multiply(vec, result);
    return result;
}

void PressureMatrix::multiply(
    const std::vector<double>& vec,
    std::vector<double>& result) const {
    if (csr_handle == nullptr) {
        throw std::logic_error("PressureMatrix::multiply called without an MKL matrix handle");
    }
    if (vec.size() != cols) {
        throw std::invalid_argument(
            "PressureMatrix::multiply input size does not match the matrix column count");
    }

    result.resize(rows);
    const matrix_descr descriptor = generalMatrixDescriptor();
    const sparse_status_t status = mkl_sparse_d_mv(
        SPARSE_OPERATION_NON_TRANSPOSE,
        1.0,
        csr_handle,
        descriptor,
        vec.data(),
        0.0,
        result.data());
    if (status != SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error(
            "PressureMatrix::multiply failed. MKL status: " +
            std::to_string(static_cast<int>(status)));
    }
};
std::vector<double> PressureMatrix::multiplyT(const std::vector<double>& vec) const {
    std::vector<double> result;
    multiplyT(vec, result);
    return result;
}

void PressureMatrix::multiplyT(
    const std::vector<double>& vec,
    std::vector<double>& result) const {
    if (csr_handle == nullptr) {
        throw std::logic_error("PressureMatrix::multiplyT called without an MKL matrix handle");
    }
    if (vec.size() != rows) {
        throw std::invalid_argument(
            "PressureMatrix::multiplyT input size does not match the matrix row count");
    }

    result.resize(cols);
    const matrix_descr descriptor = generalMatrixDescriptor();
    const sparse_status_t status = mkl_sparse_d_mv(
        SPARSE_OPERATION_TRANSPOSE,
        1.0,
        csr_handle,
        descriptor,
        vec.data(),
        0.0,
        result.data());
    if (status != SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error(
            "PressureMatrix::multiplyT failed. MKL status: " +
            std::to_string(static_cast<int>(status)));
    }
};

void PressureMatrix::writeToFile(const std::string& filename) const {
    std::ofstream out(filename);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open file for writing");
    }

    // Write matrix dimensions first
    out << "Matrix dimensions: " << rows << " x " << cols << "\n";

    // Iterate over rows and write rows_start and rows_end
    for (size_t i = 0; i < rows; ++i) {
        out << rows_start[i] <<" "<< rows_end[i] << "\n";
    }

    // Write col_indx and values_csr pair-wise
    for (size_t i = 0; i < values_csr.size(); ++i) {
        out << col_indx[i] << " " << values_csr[i] << "\n";
    }

    out.close();
}
