#ifndef PRESSURE_MATRIX_H
#define PRESSURE_MATRIX_H

#include <vector>
#include "Point.h" 
#include "BoundaryContainer.h" 
#include "DiagonalMatrix.h"
#include <mkl.h>

class PressureMatrix {
public:
    // Constructor: Accepts arrays of points u, v, w and other required parameters
    PressureMatrix(const std::vector<std::vector<std::vector<Point>>>& u,
        const std::vector<std::vector<std::vector<Point>>>& v,
        const std::vector<std::vector<std::vector<Point>>>& w,
        std::vector<std::vector<std::vector<Point>>>& p,
        BoundaryContainer& BC, DiagonalMatrix& D, bool symetry);
    ~PressureMatrix() noexcept;

    // The MKL handle refers to the CSR arrays below. Copies therefore build their
    // own handle, while moves transfer the arrays and handle together.
    PressureMatrix(const PressureMatrix& other);
    PressureMatrix& operator=(const PressureMatrix& other);
    PressureMatrix(PressureMatrix&& other) noexcept;
    PressureMatrix& operator=(PressureMatrix&& other) noexcept;

    // Function to get value at a given row and column
    double getValue(int row, int col) const;

    // Multiplication by a vector from the right
    std::vector<double> multiply(const std::vector<double>& vec) const;
    void multiply(const std::vector<double>& vec, std::vector<double>& result) const;
    // Multiplication of Transpose by a vector from the right
    std::vector<double> multiplyT(const std::vector<double>& vec) const;
    void multiplyT(const std::vector<double>& vec, std::vector<double>& result) const;

    // Functions to retrieve internal CSR representation
    const std::vector<double>& getValues() const;
    const std::vector<MKL_INT>& getColumnIndices() const;
    const std::vector<MKL_INT>& getRowStart() const;
    const std::vector<MKL_INT>& getRowEnd() const;
    // Additional functions and members needed for your specific implementation
    void writeToFile(const std::string& filename) const;

private:
    void createSparseHandle();
    void destroySparseHandle() noexcept;
    void swap(PressureMatrix& other) noexcept;

    std::vector<double> values_csr;         // Stores the non-zero values
    std::vector<MKL_INT> col_indx;    // Stores the column indices of non-zero values
    std::vector<MKL_INT> rows_start;
    std::vector<MKL_INT> rows_end;
    size_t rows = 0;
    size_t cols = 0;
    sparse_matrix_t csr_handle = nullptr;

};

#endif // PRESSURE_MATRIX_H
