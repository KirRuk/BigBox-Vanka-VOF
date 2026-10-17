#ifndef RELAX_H
#define RELAX_H

#include <vector>
#include "Point.h"
#include "DiagonalMatrix.h"
#include "MatrixBDBT.h"
#include "PressureMatrix.h"
#include "RHSu.h"
#include "RHSr.h"
#include <mkl.h>

class Relax {
private:
    // Private data members for corrections and norms
    std::vector<double> pressureCorrection, velocityCorrection, forceCorrection;
    double pressureCorrectionNorm, velocityCorrectionNorm, forceCorrectionNorm;
    double uCorrectionNorm = 0.0;
    double vCorrectionNorm = 0.0;
    double wCorrectionNorm = 0.0;
    std::size_t uLength_ = 0;
    std::size_t vLength_ = 0;
    std::size_t wLength_ = 0;

    // PCG workspace is retained for all nonlinear relaxations in one timestep.
    // MatrixBDBT is immutable while a Relax object is in use, so its object,
    // diagonal storage, and size provide a cheap cache identity.
    const MatrixBDBT* pcgMatrix = nullptr;
    const double* pcgDiagonalData = nullptr;
    MKL_INT pcgSize = 0;
    std::vector<double> pcgInvDiag;
    std::vector<double> pcgX;
    std::vector<double> pcgAx;
    std::vector<double> pcgR;
    std::vector<double> pcgZ;
    std::vector<double> pcgP;
    std::vector<double> pcgAp;
    std::vector<double> projectionDRHSu;
    std::vector<double> projectionPressureRhs;
    std::vector<double> projectionBT;
    std::vector<double> projectionDBT;

    void preparePCGWorkspace(const MatrixBDBT& BDBT);

    // Diagonally preconditioned conjugate gradient.
    const std::vector<double>& conjugateGradientPCG(
        const MatrixBDBT& BDBT,
        const std::vector<double>& b,
        double min_residual,
        const double* x_assumption,
        double x_assumption_scale);
public:
    // Constructor
    Relax(const std::vector<std::vector<std::vector<Point>>>& u,
        const std::vector<std::vector<std::vector<Point>>>& v,
        const std::vector<std::vector<std::vector<Point>>>& w,
        const std::vector<std::vector<std::vector<Point>>>& p);

    // Method to update values of u, v, w, and p using the calculated corrections
    void relax(std::vector<std::vector<std::vector<Point>>>& u,
        std::vector<std::vector<std::vector<Point>>>& v,
        std::vector<std::vector<std::vector<Point>>>& w,
        std::vector<std::vector<std::vector<Point>>>& p,
        bool periodicEW = false);

    void currentStep(
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
        RHSu& rhsu, RHSr& rhsr, double CG,  int numDirichletPoints,
        const std::vector<std::vector<std::vector<double>>>& phi, const std::vector<std::vector<std::vector<double >>>& kappa);
    // Getters for the corrections and norms
    const std::vector<double>& getPressureCorrection() const;
    const std::vector<double>& getVelocityCorrection() const;
    const std::vector<double>& getForceCorrection() const;
    double getPressureCorrectionNorm() const;
    double getVelocityCorrectionNorm() const;
    double getUCorrectionNorm() const;
    double getVCorrectionNorm() const;
    double getWCorrectionNorm() const;
    double getForceCorrectionNorm() const;
};

#endif // RELAX_H
