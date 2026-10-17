
// MumpsSolver.h

#ifndef MUMPS_SOLVER_H
#define MUMPS_SOLVER_H
#include <vector>
#include <string>
#include <mkl.h>
#include <dmumps_c.h>

class Mumps_Solver {
public:
    explicit Mumps_Solver(int symmetry, bool useMPI = false);  // <- NEW arg for MPI check
    ~Mumps_Solver();           // Destructor

    // Initializes MUMPS instance (Should be called before using the solver) and Sets the problem matrix (A in Ax = b) as a symmetric matrix given in full
    //Matrix is in CSR format
    void initialize(const std::vector<MKL_INT>& rowC, const std::vector<MKL_INT>& colIndices,
        const std::vector<double>& values);

    // Solves Ax = b for x
    std::vector<double> solve(const std::vector<double>& rhs);

    // Retrieves the full LDL^-1 decomposition
    void LU();

    // Saves the matrix in COO format to a file
    void saveMatrixToFile(const std::string& filename) const;

private:
    bool realMPI_ = false;          // remembers which mode we’re in
    DMUMPS_STRUC_C mumps_instance;  // The MUMPS instance
    std::vector<MKL_INT> rowC_copy;
    std::vector<MKL_INT> colIndices_copy;
    std::vector<double> values_copy;
};

#endif // MUMPS_SOLVER_H
