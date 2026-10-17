#ifndef RHSU_H
#define RHSU_H

#include <vector>
#include "DiagonalMatrix.h"
#include "Point.h"
#include "BoundaryContainer.h"

class RHSu {
public:
    static inline double AB2(double prev, double prevprev) {
        return 1.5 * prev - 0.5 * prevprev;
    }
    // Default constructor - Initializes RHScurrent and dtStep with zeros based on the size from DiagonalMatrix
    RHSu(const DiagonalMatrix& D);

    // Methods to update the current and timestep
    void updateCurrent(const std::vector<std::vector<std::vector<Point>>>& u,
        const std::vector<std::vector<std::vector<Point>>>& v,
        const std::vector<std::vector<std::vector<Point>>>& w,
        const std::vector<std::vector<std::vector<Point>>>& p,
        const std::vector<std::vector<std::vector<double>>>& rhoX, const std::vector<std::vector<std::vector<double>>>& rhoY, const std::vector<std::vector<std::vector<double>>>& rhoZ,
        const std::vector<std::vector<std::vector<double>>>& muX, const std::vector<std::vector<std::vector<double>>>& muY, const std::vector<std::vector<std::vector<double>>>& muZ,
        const std::vector<std::vector<std::vector<double>>>& dmudx, const std::vector<std::vector<std::vector<double>>>& dmudy, const std::vector<std::vector<std::vector<double>>>& dmudz, double rhoRef,
        double dt, BoundaryContainer& boundaryContainerU, BoundaryContainer& boundaryContainerV, BoundaryContainer& boundaryContainerW,double Re, int numDiorichletPoints, double Fr, double We, const std::vector<std::vector<std::vector<double>>>& phi, const std::vector<std::vector<std::vector<double>>>& kappa);
    void updateTimestep(const std::vector<std::vector<std::vector<Point>>>& u,
        const std::vector<std::vector<std::vector<Point>>>& v,
        const std::vector<std::vector<std::vector<Point>>>& w,
        double dt);
    const std::vector<double>& getRHSu();

private:
    std::vector<double> RHScurrent;
    std::vector<double> dtStep;
    std::vector<double> termBuffer;
    std::vector<double> densityBuffer;

    std::vector<double> acquireTermBuffer(std::size_t size);
    void recycleTermBuffer(std::vector<double>& buffer);

    // Helper methods used internally
    std::vector<double> LaplacianU(const std::vector<std::vector<std::vector<Point>>>& u, const std::vector<std::vector<std::vector<Point>>>& v, const std::vector<std::vector<std::vector<Point>>>& w, const std::vector<std::vector<std::vector<double>>>& mu, const std::vector<std::vector<std::vector<double>>>& dmudx, const std::vector<std::vector<std::vector<double>>>& dmudy, const std::vector<std::vector<std::vector<double>>>& dmudz, double Re);
    std::vector<double> LaplacianV(const std::vector<std::vector<std::vector<Point>>>& u, const std::vector<std::vector<std::vector<Point>>>& v, const std::vector<std::vector<std::vector<Point>>>& w, const std::vector<std::vector<std::vector<double>>>& mu, const std::vector<std::vector<std::vector<double>>>& dmudx, const std::vector<std::vector<std::vector<double>>>& dmudy, const std::vector<std::vector<std::vector<double>>>& dmudz,  double Re);
    std::vector<double> LaplacianW(const std::vector<std::vector<std::vector<Point>>>& u, const std::vector<std::vector<std::vector<Point>>>& v, const std::vector<std::vector<std::vector<Point>>>& w, const std::vector<std::vector<std::vector<double>>>& mu, const std::vector<std::vector<std::vector<double>>>& dmudx, const std::vector<std::vector<std::vector<double>>>& dmudy, const std::vector<std::vector<std::vector<double>>>& dmudz, double Re);
    std::vector<double> NonLinearU(const std::vector<std::vector<std::vector<Point>>>& u,
        const std::vector<std::vector<std::vector<Point>>>& v,
        const std::vector<std::vector<std::vector<Point>>>& w,
        const std::vector<std::vector<std::vector<double>>>& Rho);
    std::vector<double> NonLinearV(const std::vector<std::vector<std::vector<Point>>>& u,
        const std::vector<std::vector<std::vector<Point>>>& v,
        const std::vector<std::vector<std::vector<Point>>>& w,
        const std::vector<std::vector<std::vector<double>>>& Rho);
    std::vector<double> NonLinearW(const std::vector<std::vector<std::vector<Point>>>& u,
        const std::vector<std::vector<std::vector<Point>>>& v,
        const std::vector<std::vector<std::vector<Point>>>& w,
        const std::vector<std::vector<std::vector<double>>>& Rho);
    std::vector<double> gradP(const std::vector<std::vector<std::vector<Point>>>& uvw, char varType, const std::vector<std::vector<std::vector<Point>>>& p, int numDiorichletPoints);
    std::vector<double> dudT(const std::vector<std::vector<std::vector<Point>>>& uvw, double dt, const std::vector<std::vector<std::vector<double>>>& Rho, int shift);
    void applyBoundaryConditions(BoundaryContainer& boundaryContainer, const std::vector<std::vector<std::vector<Point>>>& u, const std::vector<std::vector<std::vector<Point>>>& v, const std::vector<std::vector<std::vector<Point>>>& p, const std::vector<std::vector<std::vector<double>>>& rho, const std::vector<std::vector<std::vector<double>>>& mu, int indexCorrection, bool is2D, double dt, double Re, double Fr);
    std::vector<double> NonLinearSOU(const std::vector<std::vector<std::vector<Point>>>& u, const std::vector<std::vector<std::vector<Point>>>& v, const std::vector<std::vector<std::vector<Point>>>& w, bool is3D);
    double calculateFluxSideU(int k, int j, int i, const std::vector<std::vector<std::vector<Point>>>& u, const std::vector<std::vector<std::vector<Point>>>& v, const std::vector<std::vector<std::vector<Point>>>& w, bool is3D, char side);
    std::vector<double> NonLinearSOV(const std::vector<std::vector<std::vector<Point>>>& u, const std::vector<std::vector<std::vector<Point>>>& v, const std::vector<std::vector<std::vector<Point>>>& w, bool is3D);
    double calculateFluxSideV(int k, int j, int i, const std::vector<std::vector<std::vector<Point>>>& u, const std::vector<std::vector<std::vector<Point>>>& v, const std::vector<std::vector<std::vector<Point>>>& w, bool is3D, char side);
    std::vector<double> Gravity_GradRho(const std::vector<std::vector<std::vector<Point>>>& v, const std::vector<std::vector<std::vector<double>>>& rho, double Fr, double rhoRef);
    std::vector<double> SurTension_GradPhiX(const std::vector<std::vector<std::vector<Point>>>& u, const std::vector<std::vector<std::vector<double>>>& phiS, const std::vector<std::vector<std::vector<double>>>& kappa, double We);
    std::vector<double> SurTension_GradPhiY(const std::vector<std::vector<std::vector<Point>>>& v, const std::vector<std::vector<std::vector<double>>>& phiS, const std::vector<std::vector<std::vector<double>>>& kappa, double We);
    std::vector<double> SurTension_GradPhiZ(const std::vector<std::vector<std::vector<Point>>>& w, const std::vector<std::vector<std::vector<double>>>& phiS, const std::vector<std::vector<std::vector<double>>>& kappa, double We);
};

#endif // RHSU_H
