#include "Relax.h"
#include "mkl.h"
#include "RHSu.h"
#include "RHSr.h"
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <omp.h>  
#include "PerformanceProfiler.h"

Relax::Relax(const std::vector<std::vector<std::vector<Point>>>& u,
    const std::vector<std::vector<std::vector<Point>>>& v,
    const std::vector<std::vector<std::vector<Point>>>& w,
    const std::vector<std::vector<std::vector<Point>>>& p)
    : pressureCorrectionNorm(100.0), velocityCorrectionNorm(0.0), forceCorrectionNorm(100.0) {
    // Determine if the system is 3D
    bool is3D = (w.size() > 1);

    // Calculate the sizes of the u, v, and w vectors
    size_t u_length = u[0][0].size() * u[0].size() * (is3D ? u.size() : 1);
    size_t v_length = v[0][0].size() * v[0].size() * (is3D ? v.size() : 1);
    size_t w_length = is3D ? (w[0][0].size() * w[0].size() * w.size()) : 0;
    uLength_ = u_length;
    vLength_ = v_length;
    wLength_ = w_length;

    // Calculate the size of the p vector
    size_t p_length = p[0][0].size() * p[0].size() * (is3D ? p.size() : 1);

    // Initialize the pressureCorrection and velocityCorrection vectors with the appropriate sizes and initial values
    pressureCorrection = std::vector<double>(p_length, 0.0);
    velocityCorrection = std::vector<double>(u_length + v_length + w_length, 0.0);
}

void Relax::relax(std::vector<std::vector<std::vector<Point>>>& u,
    std::vector<std::vector<std::vector<Point>>>& v,
    std::vector<std::vector<std::vector<Point>>>& w,
    std::vector<std::vector<std::vector<Point>>>& p,
    bool periodicEW) {
    
    bool is3D = (w.size() > 1);
    size_t u_length = u[0][0].size() * u[0].size() * (is3D ? u.size() : 1);
    size_t v_length = v[0][0].size() * v[0].size() * (is3D ? v.size() : 1);

    // ======== Update u =========
    #pragma omp parallel for collapse(3)
    for (size_t k = 0; k < u.size(); ++k) {
        for (size_t j = 0; j < u[0].size(); ++j) {
            for (size_t i = 0; i < u[0][0].size(); ++i) {
                if (!is3D && k > 0) continue;
                int unique_number = u[k][j][i].getUniqueNumber();
                double newValue = u[k][j][i].getValue() + 0.7*velocityCorrection[unique_number];
                u[k][j][i].setValue(newValue);
            }
        }
    }

    // ======== Update v =========
    #pragma omp parallel for collapse(3)
    for (size_t k = 0; k < v.size(); ++k) {
        for (size_t j = 0; j < v[0].size(); ++j) {
            for (size_t i = 0; i < v[0][0].size(); ++i) {
                if (!is3D && k > 0) continue;
                int unique_number = v[k][j][i].getUniqueNumber() + u_length;
                double newValue = v[k][j][i].getValue() + 0.7 * velocityCorrection[unique_number];
                v[k][j][i].setValue(newValue);
            }
        }
    }

    // ======== Update w (if 3D) =========
    if (is3D) {
        #pragma omp parallel for collapse(3)
        for (size_t k = 0; k < w.size(); ++k) {
            for (size_t j = 0; j < w[0].size(); ++j) {
                for (size_t i = 0; i < w[0][0].size(); ++i) {
                    int unique_number = w[k][j][i].getUniqueNumber() + u_length + v_length;
                    double newValue = w[k][j][i].getValue() + 0.7 * velocityCorrection[unique_number];
                    w[k][j][i].setValue(newValue);
                }
            }
        }
    }

    if (periodicEW) {
        const auto matchEastToWest = [](auto& component) {
            if (component.empty()) return;
            const size_t last = component[0][0].size() - 1;
            for (auto& plane : component) {
                for (auto& row : plane) {
                    if (row[0].getBoundaryType() == PointBoundary::W &&
                        row[last].getBoundaryType() == PointBoundary::E)
                        row[last].setValue(row[0].getValue());
                }
            }
        };
        matchEastToWest(u);
        matchEastToWest(v);
        if (is3D) matchEastToWest(w);
    }

    // ======== Update p =========
    #pragma omp parallel for collapse(3)
    for (size_t k = 0; k < p.size(); ++k) {
        for (size_t j = 0; j < p[0].size(); ++j) {
            for (size_t i = 0; i < p[0][0].size(); ++i) {
                if (!is3D && k > 0) continue;
                int unique_number = p[k][j][i].getUniqueNumber();
                double newValue = p[k][j][i].getValue() + 0.7 * pressureCorrection[unique_number];
                p[k][j][i].setValue(newValue);
            }
        }
    }
}

void Relax::currentStep(
    const std::vector<std::vector<std::vector<Point>>>& u,
    const std::vector<std::vector<std::vector<Point>>>& v,
    const std::vector<std::vector<std::vector<Point>>>& w,
    const std::vector<std::vector<std::vector<Point>>>& p,
    const std::vector<std::vector<std::vector<double>>>& rhoX, const std::vector<std::vector<std::vector<double>>>& rhoY, const std::vector<std::vector<std::vector<double>>>& rhoZ,
    const std::vector<std::vector<std::vector<double>>>& muX, const std::vector<std::vector<std::vector<double>>>& muY, const std::vector<std::vector<std::vector<double>>>& muZ,
    const std::vector<std::vector<std::vector<double>>>& dmudx, const std::vector<std::vector<std::vector<double>>>& dmudy, const std::vector<std::vector<std::vector<double>>>& dmudz,
    double rhoRef, double dt, int BDBTsize,
    BoundaryContainer& boundaryContainerU,
    BoundaryContainer& boundaryContainerV,
    BoundaryContainer& boundaryContainerW,
    BoundaryContainer& boundaryContainerP,
    const MatrixBDBT& BDBT,
    const DiagonalMatrix& D,
    const PressureMatrix& B, double Re, double Fr, double We,
    RHSu& rhsu, RHSr& rhsr, double CG, int numDirichletPoints,
    const std::vector<std::vector<std::vector<double>>>& phi, const std::vector<std::vector<std::vector<double>>>& kappa)
{
    // Count the number of Dirichlet points
    const Boundary& boundaryB = boundaryContainerP.getBoundary(PointBoundary::B);
    if (boundaryB.type == BoundaryKind::DirichletRow) {
        numDirichletPoints = static_cast<int>(boundaryB.value);
    }


    {
        ScopedProfilePhase timer(ProfilePhase::RHSu);
        rhsu.updateCurrent(u, v, w, p, rhoX, rhoY, rhoZ, muX, muY, muZ, dmudx, dmudy, dmudz, rhoRef, dt, boundaryContainerU, boundaryContainerV, boundaryContainerW, Re, numDirichletPoints, Fr, We, phi, kappa);
    }

    {
        ScopedProfilePhase timer(ProfilePhase::RHSr);
        rhsr.updateCurrent(p, u, v, w, boundaryContainerP, numDirichletPoints);
    }

    // Reuse the projection work vectors across nonlinear iterations.
    {
        ScopedProfilePhase timer(ProfilePhase::DMultiplyRHS);
        D.multiply_with_vector(rhsu.getRHSu(), projectionDRHSu);
    }

    // Call PressureMatrix::multiply method to get B*D*RHSu
    {
        ScopedProfilePhase timer(ProfilePhase::BMultiply);
        B.multiply(projectionDRHSu, projectionPressureRhs);
    }

    // Subtract RHSr vector from the result (B*D*RHSu - RHSr)
    const std::vector<double>& rhsrVector = rhsr.getRHSr();

    double a = -1.0;
    int incx = 1;
    int incy = 1;
    int n = static_cast<int>(projectionPressureRhs.size());


    // Perform finalVector = (B*D*RHSu - RHSr)
    cblas_daxpy(n, a, rhsrVector.data(), incx, projectionPressureRhs.data(), incy);
    const std::vector<double>* pressureTempPtr = nullptr;
    {
        ScopedProfilePhase timer(ProfilePhase::PCG);
        pressureTempPtr = &conjugateGradientPCG(
            BDBT,
            projectionPressureRhs,
            CG,
            pressureCorrection.data() + numDirichletPoints,
            0.3);
    }
    const std::vector<double>& pressureTemp = *pressureTempPtr;


    // Update pressureCorrection vector starting from numDirichletPointsint 
    int nnp = static_cast<int>(pressureTemp.size());
    {
    ScopedProfilePhase timer(ProfilePhase::CorrectionFinalize);
    cblas_dcopy(nnp, pressureTemp.data(), 1, &pressureCorrection[numDirichletPoints], 1);
    }
    // Call PressureMatrix::multiplyT method to get BT*p
    {
        ScopedProfilePhase timer(ProfilePhase::BMultiplyT);
        B.multiplyT(pressureTemp, projectionBT);
    }
    // Call the method of D for vector multiplication D*BT*p
    {
        ScopedProfilePhase timer(ProfilePhase::DMultiplyCorrection);
        D.multiply_with_vector(projectionBT, projectionDBT);
    }
    // Form the velocity correction directly in its retained destination.
    const int velocitySize = static_cast<int>(velocityCorrection.size());
    {
    ScopedProfilePhase timer(ProfilePhase::CorrectionFinalize);
    cblas_dcopy(velocitySize, projectionDRHSu.data(), 1, velocityCorrection.data(), 1);
    cblas_daxpy(velocitySize, a, projectionDBT.data(), incx, velocityCorrection.data(), incy);
    int sizePressureCorrection = static_cast<int>(pressureCorrection.size());
    int sizeVelocityCorrection = static_cast<int>(velocityCorrection.size());
    this->pressureCorrectionNorm = sqrt(cblas_ddot(sizePressureCorrection, pressureCorrection.data(), 1, pressureCorrection.data(), 1) / sizePressureCorrection);
    this->velocityCorrectionNorm = sqrt(cblas_ddot(sizeVelocityCorrection, velocityCorrection.data(), 1, velocityCorrection.data(), 1) / sizeVelocityCorrection);
    this->uCorrectionNorm = std::sqrt(
        cblas_ddot(static_cast<MKL_INT>(uLength_), velocityCorrection.data(), 1,
            velocityCorrection.data(), 1) / static_cast<double>(uLength_));
    this->vCorrectionNorm = std::sqrt(
        cblas_ddot(static_cast<MKL_INT>(vLength_), velocityCorrection.data() + uLength_, 1,
            velocityCorrection.data() + uLength_, 1) / static_cast<double>(vLength_));
    this->wCorrectionNorm = wLength_ > 0
        ? std::sqrt(
            cblas_ddot(static_cast<MKL_INT>(wLength_), velocityCorrection.data() + uLength_ + vLength_, 1,
                velocityCorrection.data() + uLength_ + vLength_, 1) / static_cast<double>(wLength_))
        : 0.0;
    }
    if (!std::isfinite(pressureCorrectionNorm) || !std::isfinite(velocityCorrectionNorm)) {
        throw std::runtime_error("pressureCorrectionNorm or velocityCorrectionNorm is not a finite number");
    }    
};
const std::vector<double>& Relax::getPressureCorrection() const {
    return pressureCorrection;
}

const std::vector<double>& Relax::getVelocityCorrection() const {
    return velocityCorrection;
}

const std::vector<double>& Relax::getForceCorrection() const {
    return forceCorrection;
}

double Relax::getPressureCorrectionNorm() const {
    return pressureCorrectionNorm;
}

double Relax::getVelocityCorrectionNorm() const {
    return velocityCorrectionNorm;
}

double Relax::getUCorrectionNorm() const {
    return uCorrectionNorm;
}

double Relax::getVCorrectionNorm() const {
    return vCorrectionNorm;
}

double Relax::getWCorrectionNorm() const {
    return wCorrectionNorm;
}

double Relax::getForceCorrectionNorm() const {
    return forceCorrectionNorm;
}

// Reuse the diagonal preconditioner and work vectors during this timestep.
void Relax::preparePCGWorkspace(const MatrixBDBT& BDBT)
{
    const MKL_INT n = BDBT.size();
    const std::vector<double>& diagonal = BDBT.getDiag0();
    const double* diagonalData = diagonal.data();

    const bool sizeChanged = (pcgSize != n);
    const bool diagonalChanged =
        sizeChanged || pcgMatrix != &BDBT || pcgDiagonalData != diagonalData;

    if (sizeChanged) {
        const std::size_t workspaceSize =
            n > 0 ? static_cast<std::size_t>(n) : std::size_t{ 0 };
        pcgInvDiag.resize(workspaceSize);
        pcgX.resize(workspaceSize);
        pcgAx.resize(workspaceSize);
        pcgR.resize(workspaceSize);
        pcgZ.resize(workspaceSize);
        pcgP.resize(workspaceSize);
        pcgAp.resize(workspaceSize);
        pcgSize = n;
    }

    if (diagonalChanged && n > 0) {
#pragma omp parallel for simd schedule(static)
        for (MKL_INT i = 0; i < n; ++i) {
            pcgInvDiag[i] = 1.0 / diagonal[i];
        }
    }

    pcgMatrix = &BDBT;
    pcgDiagonalData = diagonalData;
}

const std::vector<double>& Relax::conjugateGradientPCG(
    const MatrixBDBT& BDBT,
    const std::vector<double>& b,
    double tol_abs,
    const double* x0,
    double x0Scale
) {
    const MKL_INT n = BDBT.size();
    preparePCGWorkspace(BDBT);
    PerformanceProfiler& profiler = PerformanceProfiler::instance();
    if (n <= 0) {
        profiler.recordCGSolve(0);
        return pcgX;
    }

    // tol = tol_abs * (1 + max(1, ||b||_2))
    double bnorm2;
    {
        ScopedProfilePhase timer(ProfilePhase::PCGBlas);
        bnorm2 = cblas_ddot(n, b.data(), 1, b.data(), 1);
    }
    const double bnorm = std::sqrt(std::max(0.0, bnorm2));
    const double tol = tol_abs * (1.0 + std::max(1.0, bnorm));

    // Build the same scaled initial assumption directly in the retained x
    // buffer. This replaces the former temporary allocation and copy.
    {
        ScopedProfilePhase timer(ProfilePhase::PCGBlas);
        std::fill(pcgX.begin(), pcgX.end(), 0.0);
        cblas_daxpy(n, x0Scale, x0, 1, pcgX.data(), 1);
    }

    // r = b - A*x
    {
        ScopedProfilePhase timer(ProfilePhase::PCGSpmv);
        BDBT.spmv_diags(pcgX.data(), pcgAx.data());
    }
    profiler.recordSpmvCall();
    {
        ScopedProfilePhase timer(ProfilePhase::PCGBlas);
        std::copy_n(b.data(), n, pcgR.data());
        cblas_daxpy(n, -1.0, pcgAx.data(), 1, pcgR.data(), 1);
    }

    // early exit
    {
        double rnorm2;
        {
            ScopedProfilePhase timer(ProfilePhase::PCGBlas);
            rnorm2 = cblas_ddot(n, pcgR.data(), 1, pcgR.data(), 1);
        }
        const double rnorm = std::sqrt(std::max(0.0, rnorm2));
        if (rnorm <= tol) {
            profiler.recordCGSolve(0);
            return pcgX;
        }
    }

    // z = M^{-1} r
    double rho = 0.0;
    {
        ScopedProfilePhase timer(ProfilePhase::PCGPreconditioner);
#pragma omp parallel for simd schedule(static) reduction(+:rho)
        for (MKL_INT i = 0; i < n; ++i) {
            const double z = pcgR[i] * pcgInvDiag[i];
            pcgZ[i] = z;
            pcgP[i] = z;
            rho += pcgR[i] * z;
        }
    }

    std::int64_t iterations = 0;
    for (MKL_INT k = 0; k < n; ++k) {
        // Ap = A*p
        {
            ScopedProfilePhase timer(ProfilePhase::PCGSpmv);
            BDBT.spmv_diags(pcgP.data(), pcgAp.data());
        }
        profiler.recordSpmvCall();

        double pAp;
        {
            ScopedProfilePhase timer(ProfilePhase::PCGBlas);
            pAp = cblas_ddot(n, pcgP.data(), 1, pcgAp.data(), 1);
        }
        const double alpha = rho / pAp;

        // x = x + alpha p
        double rnorm2 = 0.0;
        {
            ScopedProfilePhase timer(ProfilePhase::PCGBlas);
#pragma omp parallel for simd schedule(static) reduction(+:rnorm2)
            for (MKL_INT i = 0; i < n; ++i) {
                pcgX[i] += alpha * pcgP[i];
                const double r = pcgR[i] - alpha * pcgAp[i];
                pcgR[i] = r;
                rnorm2 += r * r;
            }
        }
        ++iterations;
        const double rnorm = std::sqrt(std::max(0.0, rnorm2));
        if (rnorm <= tol) break;

        // z = M^{-1} r
        double rho_new = 0.0;
        {
            ScopedProfilePhase timer(ProfilePhase::PCGPreconditioner);
#pragma omp parallel for simd schedule(static) reduction(+:rho_new)
            for (MKL_INT i = 0; i < n; ++i) {
                const double z = pcgR[i] * pcgInvDiag[i];
                pcgZ[i] = z;
                rho_new += pcgR[i] * z;
            }
        }

        const double beta = rho_new / rho;
        rho = rho_new;

        // p = z + beta p
        {
            ScopedProfilePhase timer(ProfilePhase::PCGBlas);
#pragma omp parallel for simd schedule(static)
            for (MKL_INT i = 0; i < n; ++i) {
                pcgP[i] = pcgZ[i] + beta * pcgP[i];
            }
        }
    }

    profiler.recordCGSolve(iterations);
    return pcgX;
}
