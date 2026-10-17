#include "Mumps_Solver.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <mkl.h>
#include <mpi.h>
#include <cstring>
#include <filesystem>

Mumps_Solver::Mumps_Solver(int symmetry, bool useMPI)
    : realMPI_(useMPI), rowC_copy(), colIndices_copy(), values_copy()
{
    mumps_instance.icntl[23] = 1;
    /* --------------- MPI handling ---------------- */
    int argc = 0; char** argv = NULL;
    if (realMPI_) {
        /*
        int initFlag;  MPI_Initialized(&initFlag);
        if (!initFlag)  MPI_Init(&argc, &argv);

        int vMaj, vMin; MPI_Get_version(&vMaj, &vMin);
        std::cout << "MPI " << vMaj << '.' << vMin << " active\n";

        mumps_instance.par = 1;
        mumps_instance.comm_fortran = MPI_Comm_c2f(MPI_COMM_WORLD);*/
    }
    else {                       // sequential / mpiseq build
        MPI_Init(&argc, &argv);

        mumps_instance.comm_fortran = -987654;   // “no-MPI” sentinel
        std::cout << "Running MUMPS in sequential (OpenMP) mode\n";
    }

    /* --------------- MUMPS init ------------------ */
    mumps_instance.sym = symmetry;//For non-symetric use 0;
    mumps_instance.par = 1;
    mumps_instance.job = -1;
    //mumps_instance.comm_fortran = MPI_COMM_WORLD;

    dmumps_c(&mumps_instance);
    if (mumps_instance.info[0] != 0) {
        std::cerr << "MUMPS -1 failed  :  "
            << "info(1)=" << mumps_instance.info[0] << "  "
            << "info(2)=" << mumps_instance.info[1] << "  "
            << "infog(1)=" << mumps_instance.infog[0] << "  "
            << "infog(2)=" << mumps_instance.infog[1] << '\n';
        return;          // or throw …
    }
    mumps_instance.icntl[0] = 0;
    mumps_instance.icntl[1] = 0;
    mumps_instance.icntl[2] = 0;
    mumps_instance.icntl[3] = 0;
    //Check if MUMPS initialized successfully
    if (mumps_instance.info[0] != 0)
        std::cerr << "MUMPS init error code: "
        << mumps_instance.info[0] << '\n';
}


Mumps_Solver::~Mumps_Solver()
{
    mumps_instance.job = -2;          // JOB_END
    dmumps_c(&mumps_instance);

    //int finFlag; MPI_Finalized(&finFlag);
    //if (!finFlag) MPI_Finalize();     // works for both real MPI and mpiseq
}

void Mumps_Solver::initialize(const std::vector<MKL_INT>& rowC, const std::vector<MKL_INT>& colIndices,
    const std::vector<double>& values) {
    // Define the order of the matrix 
    mumps_instance.n = static_cast<MUMPS_INT>(rowC.size() - 1);
    // Use the provided arrays to define the problem matrix
    mumps_instance.nnz = static_cast<MUMPS_INT>(values.size());
    mumps_instance.nz = static_cast<MUMPS_INT>(values.size());
    /* ---------- store copies --------------------------------------- */
    rowC_copy        = rowC;
    colIndices_copy  = colIndices;
    values_copy      = values;

    MKL_INT n_mkl   = static_cast<MKL_INT>(mumps_instance.n);
    MKL_INT nnz_mkl = static_cast<MKL_INT>(mumps_instance.nnz);

    MKL_INT job[6] = { 0, 0, 1, 0, static_cast<MKL_INT>(mumps_instance.nnz), 3 }; 

    // Vectors to store the COO representation
    std::vector<MKL_INT> cooRowIndices(mumps_instance.nnz);
    std::vector<MKL_INT> cooColIndices(mumps_instance.nnz);
    std::vector<double> cooValues(mumps_instance.nnz);

    // Now use the conversion function
    MKL_INT info;
    mkl_dcsrcoo(job, &n_mkl, values_copy.data(), colIndices_copy.data(), rowC_copy.data(), 
        &nnz_mkl , cooValues.data(), cooRowIndices.data(), cooColIndices.data(), &info);

    if (info != 0) {
        // Handle the error. 
        std::cout << "Matrix conversion failed with error code " << info << std::endl;
        return;
    }
 /* ---------- hand data back to MUMPS --------------------------- */
    rowC_copy       = std::move(cooRowIndices);
    colIndices_copy = std::move(cooColIndices);
    values_copy     = std::move(cooValues);

    mumps_instance.irn = reinterpret_cast<MUMPS_INT*>(rowC_copy.data());
    mumps_instance.jcn = reinterpret_cast<MUMPS_INT*>(colIndices_copy.data());
    mumps_instance.a = values_copy.data();
    // Set ICNTL(5) to 0 to use the assembled format
    mumps_instance.icntl[4] = 0;
    // Set ICNTL(7) to 0 to useApproximate Minimum Degree (AMD) [7] is used,

    mumps_instance.icntl[6] = 0;
    mumps_instance.icntl[13] = 5;
    /*// --- 2) SET UP OUT-OF-CORE ---
// 1) get the absolute path of cwd (optional but safer):
    std::string cwd = std::filesystem::current_path().string();

    // 2) copy into the fixed-length buffer:
    strncpy(mumps_instance.ooc_tmpdir, cwd.c_str(),
        sizeof(mumps_instance.ooc_tmpdir) - 1);
    mumps_instance.ooc_tmpdir[sizeof(mumps_instance.ooc_tmpdir) - 1] = '\0';

    // 3) (optional) set a prefix, otherwise MUMPS picks one:
    const char* prefix = "mumps_ooc";
    strncpy(mumps_instance.ooc_prefix, prefix,
        sizeof(mumps_instance.ooc_prefix) - 1);
    mumps_instance.ooc_prefix[sizeof(mumps_instance.ooc_prefix) - 1] = '\0';

    mumps_instance.icntl[21] = 1;*/
    // Set ICNTL(14) to 150 to use 150% of memory increas,
    //
    /*if (mumps_instance.sym == 1 || mumps_instance.sym == 2) {
        std::vector<MKL_INT> tempRow, tempCol;
        std::vector<double> tempValues;

        for (size_t i = 0; i < cooRowIndices.size(); ++i) {
            if (cooRowIndices[i] <= cooColIndices[i]) {  // Keep only the upper triangular including diagonal
                tempRow.push_back(cooRowIndices[i]);
                tempCol.push_back(cooColIndices[i]);
                tempValues.push_back(cooValues[i]);
            }
        }
        rowC_copy = tempRow;
        colIndices_copy = tempCol;
        values_copy = tempValues;

        mumps_instance.nnz = colIndices_copy.size();
        mumps_instance.nz = colIndices_copy.size();
    }*/
        
    // Set the MUMPS instance job to 1 (Analysis)
        mumps_instance.job = 1;
    dmumps_c(&mumps_instance);
        
        
}

void Mumps_Solver::LU() {
    // Call MUMPS to perform the factorization step
    mumps_instance.job = 2;
    dmumps_c(&mumps_instance);
}

std::vector<double> Mumps_Solver::solve(const std::vector<double>& rhs) {
    mumps_instance.nrhs = 1;  // Only one right-hand side
    std::vector<double> temp = rhs;
    mumps_instance.rhs = temp.data();  // Set the RHS
    // Call MUMPS to solve the system
    mumps_instance.job = 3;
    dmumps_c(&mumps_instance);

    // Return the solution vector
    return std::vector<double>(temp.begin(), temp.end());
}

void Mumps_Solver::saveMatrixToFile(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file) {
        std::cout << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    // Write the COO matrix to the file
    for (int i = 0; i < mumps_instance.nnz; ++i) {
        file << mumps_instance.irn[i] << " "
            << mumps_instance.jcn[i] << " "
            << mumps_instance.a[i] << std::endl;
    }

    file.close();
}