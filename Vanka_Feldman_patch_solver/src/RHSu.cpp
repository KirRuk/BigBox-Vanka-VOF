#include "RHSu.h"
#include "Point.h"
#include <mkl.h>
#include <algorithm>
#include <cmath>
#include <vector>
#include <iostream>

namespace {
double ab2Velocity(const Point& point) {
    return 1.5 * point.getPrevValue() - 0.5 * point.getPrevPrevValue();
}

double averageAb2Velocity(const Point& first, const Point& second) {
    return 0.5 * (ab2Velocity(first) + ab2Velocity(second));
}

double averageFour(double a, double b, double c, double d) {
    return 0.25 * (a + b + c + d);
}
}

// Constructor
RHSu::RHSu(const DiagonalMatrix& D) {
    // Get the size of the diagonal from DiagonalMatrix
    size_t diagSize = D.getDiag().size();

    // Initialize RHScurrent and dtStep with zeros
    RHScurrent = std::vector<double>(diagSize, 0.0);
    dtStep = std::vector<double>(diagSize, 0.0);
}

std::vector<double> RHSu::acquireTermBuffer(std::size_t size) {
    std::vector<double> buffer;
    buffer.swap(termBuffer);
    buffer.assign(size, 0.0);
    return buffer;
}

void RHSu::recycleTermBuffer(std::vector<double>& buffer) {
    termBuffer.swap(buffer);
}

void RHSu::updateCurrent(const std::vector<std::vector<std::vector<Point>>>& u,
    const std::vector<std::vector<std::vector<Point>>>& v,
    const std::vector<std::vector<std::vector<Point>>>& w,
    const std::vector<std::vector<std::vector<Point>>>& p,
    const std::vector<std::vector<std::vector<double>>>& rhoX, const std::vector<std::vector<std::vector<double>>>& rhoY, const std::vector<std::vector<std::vector<double>>>& rhoZ,
    const std::vector<std::vector<std::vector<double>>>& muX, const std::vector<std::vector<std::vector<double>>>& muY, const std::vector<std::vector<std::vector<double>>>& muZ,
    const std::vector<std::vector<std::vector<double>>>& dmudx, const std::vector<std::vector<std::vector<double>>>& dmudy, const std::vector<std::vector<std::vector<double>>>& dmudz, double rhoRef,
    double dt, BoundaryContainer& boundaryContainerU, BoundaryContainer& boundaryContainerV, BoundaryContainer& boundaryContainerW, double Re, int numDiorichletPoints, double Fr, double We, const std::vector<std::vector<std::vector<double>>>& phiS, const std::vector<std::vector<std::vector<double>>>& kappa) {
    bool is3D = (w.size() > 1);

    size_t u_length = u[0][0].size() * u[0].size() * (is3D ? u.size() : 1);
    size_t v_length = v[0][0].size() * v[0].size() * (is3D ? v.size() : 1);
    // Compute and add the Laplacian of U term to the RHS
    {
        std::vector<double> laplacianU = LaplacianU(u,v,w, muX, dmudx, dmudy, dmudz, Re);
        // Copying the Laplacian of U to the beginning of RHScurrent
        cblas_dcopy(laplacianU.size(), &laplacianU[0], 1, &RHScurrent[0], 1);
        recycleTermBuffer(laplacianU);
    }

    // Compute and add the NonLinearU term to the RHS
    {
        std::vector<double> nonLinearU = NonLinearU(u, v, w, rhoX);
        // Element-wise addition of NonLinearU to RHScurrent
        vdAdd(nonLinearU.size(), nonLinearU.data(), RHScurrent.data(), RHScurrent.data());
        recycleTermBuffer(nonLinearU);
    }
    // Compute and add the dudTu term (time-derivative) to the RHS
    {
        std::vector<double> dudTu = dudT(u, dt, rhoX, 0);
        // Element-wise addition of dudTu to RHScurrent
        vdAdd(dudTu.size(), dudTu.data(), RHScurrent.data(), RHScurrent.data());
        recycleTermBuffer(dudTu);
    }

    // Compute and add the gradient of pressure (GradPu) term to the RHS
    {
        std::vector<double> gradPu = gradP(u, 'u', p, numDiorichletPoints);
        // Element-wise addition of GradPu to RHScurrent
        vdAdd(gradPu.size(), gradPu.data(), RHScurrent.data(), RHScurrent.data());
        recycleTermBuffer(gradPu);
    }
    // compute and the surface tension gradient of phi term to the RHS
    {
        std::vector<double> Grad_phi = SurTension_GradPhiX(u, phiS, kappa, We);
        // Element-wise addition of surface_Tension to RHScurrent offset by u_length
        vdAdd(Grad_phi.size(), Grad_phi.data(), RHScurrent.data(), RHScurrent.data());
        recycleTermBuffer(Grad_phi);
    }
    // Compute and add the Laplacian of V term to the RHS
    {
        std::vector<double> laplacianV = LaplacianV(u, v, w, muY, dmudx, dmudy, dmudz, Re);
        // Copying the Laplacian of V to RHScurrent offset by u_length
        cblas_dcopy(laplacianV.size(), &laplacianV[0], 1, &RHScurrent[u_length], 1);
        recycleTermBuffer(laplacianV);
    }
    // Compute and add the NonLinearV term to the RHS
    {
        std::vector<double> nonLinearV = NonLinearV(u, v, w, rhoY);
        // Element-wise addition of NonLinearV to RHScurrent offset by u_length
        vdAdd(nonLinearV.size(), nonLinearV.data(), &RHScurrent[u_length], &RHScurrent[u_length]);
        recycleTermBuffer(nonLinearV);
    }

    // Compute and add the dudTv term (time-derivative) to the RHS
    {
        std::vector<double> dudTv = dudT(v, dt, rhoY, u_length);
        // Element-wise addition of dudTv to RHScurrent offset by u_length
        vdAdd(dudTv.size(), dudTv.data(), &RHScurrent[u_length], &RHScurrent[u_length]);
        recycleTermBuffer(dudTv);
    }

    // Compute and add the gradient of pressure (GradPv) term to the RHS
    {
        std::vector<double> gradPv = gradP(v, 'v', p, numDiorichletPoints);
        // Element-wise addition of GradPv to RHScurrent offset by u_length
        vdAdd(gradPv.size(), gradPv.data(), &RHScurrent[u_length], &RHScurrent[u_length]);
        recycleTermBuffer(gradPv);
    }
    // compute and add the gravity term to  the RHS
    {
        std::vector<double> Grad_rho = Gravity_GradRho(v, rhoY, Fr, rhoRef);
        // Element-wise addition of NonLinearV to RHScurrent offset by u_length
        vdAdd(Grad_rho.size(), Grad_rho.data(), &RHScurrent[u_length], &RHScurrent[u_length]);
        recycleTermBuffer(Grad_rho);
    }
    // compute and add the surface_tension term to  the RHS Y mom
    {
        std::vector<double> Grad_phi = SurTension_GradPhiY(v, phiS, kappa, We);
        // Element-wise addition of surfaceTension to RHScurrent offset by u_length
        vdAdd(Grad_phi.size(), Grad_phi.data(), &RHScurrent[u_length], &RHScurrent[u_length]);
        recycleTermBuffer(Grad_phi);
    }

    if (is3D) {

        // Compute and add the Laplacian of W term to the RHS for 3D scenarios
        {
            std::vector<double> laplacianW = LaplacianW(u, v, w, muZ, dmudx, dmudy, dmudz, Re);
            // Copying the Laplacian of W to RHScurrent offset by u_length + v_length
            cblas_dcopy(laplacianW.size(), &laplacianW[0], 1, &RHScurrent[u_length + v_length], 1);
            recycleTermBuffer(laplacianW);
        }

        // Compute and add the NonLinearW term to the RHS
        {
            std::vector<double> nonLinearW = NonLinearW(u, v, w, rhoZ);
            // Element-wise addition of NonLinearW to RHScurrent offset by u_length + v_length
            vdAdd(nonLinearW.size(), nonLinearW.data(), &RHScurrent[u_length + v_length], &RHScurrent[u_length + v_length]);
            recycleTermBuffer(nonLinearW);
        }

        // Compute and add the dudTw term (time-derivative) to the RHS
        {
            std::vector<double> dudTw = dudT(w, dt, rhoZ, (u_length + v_length));
            // Element-wise addition of dudTw to RHScurrent offset by u_length + v_length
            vdAdd(dudTw.size(), dudTw.data(), &RHScurrent[u_length + v_length], &RHScurrent[u_length + v_length]);
            recycleTermBuffer(dudTw);
        }

        // Compute and add the gradient of pressure (GradPw) term to the RHS
        {
            std::vector<double> gradPw = gradP(w, 'w', p, numDiorichletPoints);
            // Element-wise addition of GradPw to RHScurrent offset by u_length + v_length
            vdAdd(gradPw.size(), gradPw.data(), &RHScurrent[u_length + v_length], &RHScurrent[u_length + v_length]);
            recycleTermBuffer(gradPw);
        }
        // compute and add the surface_tension term to  the RHS Y mom
        {
            std::vector<double> Grad_phi = SurTension_GradPhiZ(w, phiS, kappa, We);
            // Element-wise addition of surfaceTension to RHScurrent offset by u_length
            vdAdd(Grad_phi.size(), Grad_phi.data(), &RHScurrent[u_length + v_length], &RHScurrent[u_length + v_length]);
            recycleTermBuffer(Grad_phi);
        }

    }


    // Applying boundary conditions to the variables u and v
    applyBoundaryConditions(boundaryContainerU, u, v, p, rhoX, muX, 0,  (!is3D), dt, Re, Fr);

    if (is3D &&
        boundaryContainerU.getBoundary(PointBoundary::W).type == BoundaryKind::Symmetry &&
        boundaryContainerU.getBoundary(PointBoundary::E).type == BoundaryKind::Symmetry) {
        // U[0] and U[last] are the same periodic face. U[0] carries its
        // momentum equation; U[last] carries the equality constraint above.
        const int lastU = static_cast<int>(u[0][0].size()) - 1;
        const int lastV = static_cast<int>(v[0][0].size()) - 2;
        const int lastW = static_cast<int>(w[0][0].size()) - 2;
        const int lastP = static_cast<int>(p[0][0].size()) - 1;
        const int lastGradient = static_cast<int>(dmudy[0][0].size()) - 2;
#pragma omp parallel for collapse(2) schedule(static)
        for (int k = 1; k < static_cast<int>(u.size()) - 1; ++k) {
            for (int j = 1; j < static_cast<int>(u[k].size()) - 1; ++j) {
                const Point& center = u[k][j][0];
                if (center.getBoundaryType() != PointBoundary::W) continue;
                const Point& right = u[k][j][1];
                const Point& left = u[k][j][lastU - 1];
                const double dxp = right.getX() - center.getX();
                const double dxm = u[k][j][lastU].getX() - left.getX();
                const double dx = 0.5 * (dxp + dxm);
                const double dy = center.getdy();
                const double dz = center.getdz();
                const double dyp = u[k][j + 1][0].getY() - center.getY();
                const double dym = center.getY() - u[k][j - 1][0].getY();
                const double dzp = u[k + 1][j][0].getZ() - center.getZ();
                const double dzm = center.getZ() - u[k - 1][j][0].getZ();
                const double value = center.getValue();
                const double volume = dx * dy * dz;
                const double density = rhoX[k][j][0];

                const double inertia = density *
                    (-3.0 * value + 4.0 * center.getPrevValue() -
                        center.getPrevPrevValue()) * volume / (2.0 * dt);
                const double xConvection =
                    (std::pow(value + right.getValue(), 2) -
                        std::pow(value + left.getValue(), 2)) * dy * dz;
                const double yConvection =
                    ((v[k][j][1].getValue() + v[k][j][lastV].getValue()) *
                        (value + u[k][j + 1][0].getValue()) -
                    (v[k][j - 1][1].getValue() + v[k][j - 1][lastV].getValue()) *
                        (value + u[k][j - 1][0].getValue())) * dx * dz;
                const double zConvection =
                    ((w[k][j][1].getValue() + w[k][j][lastW].getValue()) *
                        (value + u[k + 1][j][0].getValue()) -
                    (w[k - 1][j][1].getValue() + w[k - 1][j][lastW].getValue()) *
                        (value + u[k - 1][j][0].getValue())) * dx * dy;
                const double convection =
                    -density * (xConvection + yConvection + zConvection) / 4.0;
                const double pressure =
                    -(p[k - 1][j - 1][0].getValue() -
                        p[k - 1][j - 1][lastP].getValue()) * dy * dz;

                const double mux = dmudx[k][j][0];
                const double muy = averageFour(
                    dmudy[k][j][1], dmudy[k][j][lastGradient],
                    dmudy[k][j - 1][1], dmudy[k][j - 1][lastGradient]);
                const double muz = averageFour(
                    dmudz[k][j][1], dmudz[k][j][lastGradient],
                    dmudz[k - 1][j][1], dmudz[k - 1][j][lastGradient]);
                const double ux =
                    (ab2Velocity(right) - ab2Velocity(left)) / (dxp + dxm);
                const double uy =
                    (ab2Velocity(u[k][j + 1][0]) - ab2Velocity(u[k][j - 1][0])) /
                    (dyp + dym);
                const double uz =
                    (ab2Velocity(u[k + 1][j][0]) - ab2Velocity(u[k - 1][j][0])) /
                    (dzp + dzm);
                const double vx =
                    (averageAb2Velocity(v[k][j][1], v[k][j - 1][1]) -
                        averageAb2Velocity(v[k][j][lastV], v[k][j - 1][lastV])) / dx;
                const double wx =
                    (averageAb2Velocity(w[k][j][1], w[k - 1][j][1]) -
                        averageAb2Velocity(w[k][j][lastW], w[k - 1][j][lastW])) / dx;
                const double variableMu =
                    (mux * 2.0 * ux + muy * (uy + vx) + muz * (uz + wx)) * volume;
                const double laplacian = muX[k][j][0] * (
                    (right.getValue() - value) / dxp * dy * dz +
                    (left.getValue() - value) / dxm * dy * dz +
                    (u[k][j + 1][0].getValue() - value) / dyp * dx * dz +
                    (u[k][j - 1][0].getValue() - value) / dym * dx * dz +
                    (u[k + 1][j][0].getValue() - value) / dzp * dx * dy +
                    (u[k - 1][j][0].getValue() - value) / dzm * dx * dy);

                const double interfaceGradient =
                    (We == 0.0 || phiS.empty()) ? 0.0 :
                    0.5 * (kappa[k - 1][j - 1][0] + kappa[k - 1][j - 1][lastP]) *
                    (phiS[k - 1][j - 1][0] - phiS[k - 1][j - 1][lastP]) *
                    dy * dz / We;
                RHScurrent[center.getUniqueNumber()] = inertia + convection +
                    pressure + (laplacian + variableMu) / Re + interfaceGradient;
            }
        }
    }
    applyBoundaryConditions(boundaryContainerV, v, u, p, rhoY, muY, u_length, (!is3D), dt, Re, Fr);

    // If the simulation is 3D, apply boundary conditions for the variable w
    if (is3D) {
        applyBoundaryConditions(boundaryContainerW, w, v, p, rhoZ, muZ, u_length + v_length, (!is3D), dt, Re, Fr);
    }
    

}

void RHSu::updateTimestep(const std::vector<std::vector<std::vector<Point>>>& u,
    const std::vector<std::vector<std::vector<Point>>>& v,
    const std::vector<std::vector<std::vector<Point>>>& w,
    double dt) {

    // Check if the simulation is 3D
    bool is3D = w.size() > 1;

    // Calculate the length of u, v, and w in dtStep
    size_t u_length = u[0][0].size() * u[0].size() * (is3D ? u.size() : 1);
    size_t v_length = v[0][0].size() * v[0].size() * (is3D ? v.size() : 1);


    // We should first calculate and store the updated values for u
   #pragma omp parallel for collapse(3)
    for (size_t k = 0; k < u.size(); k++) {
        for (size_t j = 0; j < u[0].size(); j++) {
            for (size_t i = 0; i < u[0][0].size(); i++) {
                int index = u[k][j][i].getUniqueNumber();

                double dx = u[k][j][i].getdx();
                double dy = u[k][j][i].getdy();
                double dz = u[k][j][i].getdz();
                if (!is3D) dz = 1; // Handle 2D case

                dtStep[index] = (2.0 * u[k][j][i].getPrevValue() - 0.5 * u[k][j][i].getPrevPrevValue()) / dt * (dx * dy * dz);
            }
        }
    }
    // Next, calculate and store the updated values for v
    #pragma omp parallel for collapse(3)
    for (size_t i = 0; i < v.size(); i++) {
        for (size_t j = 0; j < v[0].size(); j++) {
            for (size_t k = 0; k < v[0][0].size(); k++) {
                int index = u_length + v[i][j][k].getUniqueNumber();

                double dx = v[i][j][k].getdx();
                double dy = v[i][j][k].getdy();
                double dz = v[i][j][k].getdz();
                if (!is3D) dz = 1; // Handle 2D case

                dtStep[index] = (2.0 * v[i][j][k].getPrevValue() - 0.5 * v[i][j][k].getPrevPrevValue()) / dt * (dx * dy * dz);
            }
        }
    }

    // Lastly, calculate and store the updated values for w if it's a 3D case
    if (is3D) {
        #pragma omp parallel for collapse(3)
        for (size_t i = 0; i < w.size(); i++) {
            for (size_t j = 0; j < w[0].size(); j++) {
                for (size_t k = 0; k < w[0][0].size(); k++) {
                    int index = u_length + v_length + w[i][j][k].getUniqueNumber();

                    double dx = w[i][j][k].getdx();
                    double dy = w[i][j][k].getdy();
                    double dz = w[i][j][k].getdz();

                    dtStep[index] = (2.0 * w[i][j][k].getPrevValue() - 0.5 * w[i][j][k].getPrevPrevValue()) / dt * (dx * dy * dz);
                }
            }
        }
    }
}

const std::vector<double>& RHSu::getRHSu() {
    return RHScurrent;
}

std::vector<double> RHSu::LaplacianU(const std::vector<std::vector<std::vector<Point>>>& u, const std::vector<std::vector<std::vector<Point>>>& v, const std::vector<std::vector<std::vector<Point>>>& w, const std::vector<std::vector<std::vector<double>>>& mu, const std::vector<std::vector<std::vector<double>>>& dmudx, const std::vector<std::vector<std::vector<double>>>& dmudy, const std::vector<std::vector<std::vector<double>>>& dmudz, double Re) {
    size_t depth = u.size();
    size_t height = u[0].size();
    size_t width = u[0][0].size();
    std::vector<double> result = acquireTermBuffer(
        depth <= 1 ? height * width : depth * height * width);

    // 2D Case
    if (depth <= 1) {
        #pragma omp parallel for collapse(2)
        for (size_t j = 1; j < height - 1; ++j) {
            for (size_t i = 1; i < width - 1; ++i) {
                if (u[0][j][i].getBoundaryType() == PointBoundary::Interior) {
                    int index = u[0][j][i].getUniqueNumber();
                    // cell sizes around (0,j,i)
                    double dxp = u[0][j][i + 1].getX() - u[0][j][i].getX();
                    double dxm = u[0][j][i].getX() - u[0][j][i - 1].getX();
                    double dyp = u[0][j + 1][i].getY() - u[0][j][i].getY();
                    double dym = u[0][j][i].getY() - u[0][j - 1][i].getY();
                    // ---- gradients of mu at u-cell center (central differences) ----
                    double mux = dmudx[0][j][i];
                    double muy = (dmudy[0][j][i+1] + dmudy[0][j][i] + dmudy[0][j - 1][i + 1] + dmudy[0][j - 1][i])/4.0;
                    // ---- velocity gradients for u at center ----
                    double ux = (ab2Velocity(u[0][j][i + 1]) - ab2Velocity(u[0][j][i - 1])) / (dxp + dxm);
                    double uy = (ab2Velocity(u[0][j + 1][i]) - ab2Velocity(u[0][j - 1][i])) / (dyp + dym);
                    // ---- v_x at u-location: average in y, central diff in x ----
                    double dxv = v[0][j][i + 1].getX() - v[0][j][i].getX();
                    double vx =
                        (averageAb2Velocity(v[0][j][i + 1], v[0][j - 1][i + 1])
                            - averageAb2Velocity(v[0][j][i], v[0][j - 1][i]))
                        / dxv;
                    // cell volume (for FV scaling)
                    double vol = u[0][j][i].getdx() * u[0][j][i].getdy();

                    // --- variable-viscosity contribution: for x-component ---
                    double termMu = mux * (2.0 * ux)       // �_x * 2 u_x
                        + muy * (uy + vx);        // �_y * (u_y + v_x)


                    double term1 = (u[0][j][i + 1].getValue() - u[0][j][i].getValue()) / dxp * u[0][j][i].getdy();
                    double term2 = (u[0][j][i - 1].getValue() - u[0][j][i].getValue()) / dxm * u[0][j][i].getdy();
                    double term3 = (u[0][j + 1][i].getValue() - u[0][j][i].getValue()) / dyp * u[0][j][i].getdx();
                    double term4 = (u[0][j - 1][i].getValue() - u[0][j][i].getValue()) / dym * u[0][j][i].getdx();
                    result[index] = (mu[0][j][i]*(term1 + term2 + term3 + term4) + termMu*vol)/Re;
                }
            }
        }
    }

    // ===================== LaplacianU : 3D part (AB2) =====================
    else {
#pragma omp parallel for collapse(3)
        for (size_t k = 1; k < depth - 1; ++k) {
            for (size_t j = 1; j < height - 1; ++j) {
                for (size_t i = 1; i < width - 1; ++i) {

                    if (u[k][j][i].getBoundaryType() != PointBoundary::Interior) continue;
                    const int index = u[k][j][i].getUniqueNumber();

                    // cell sizes around (k,j,i)
                    const double dxp = u[k][j][i + 1].getX() - u[k][j][i].getX();
                    const double dxm = u[k][j][i].getX() - u[k][j][i - 1].getX();
                    const double dyp = u[k][j + 1][i].getY() - u[k][j][i].getY();
                    const double dym = u[k][j][i].getY() - u[k][j - 1][i].getY();
                    const double dzp = u[k + 1][j][i].getZ() - u[k][j][i].getZ();
                    const double dzm = u[k][j][i].getZ() - u[k - 1][j][i].getZ();

                    // ---- prepared gradients of mu, collocated at the U point ----
                    double mux = dmudx[k][j][i];
                    double muy = averageFour(
                        dmudy[k][j][i],
                        dmudy[k][j][i + 1],
                        dmudy[k][j - 1][i],
                        dmudy[k][j - 1][i + 1]);
                    double muz = averageFour(
                        dmudz[k][j][i],
                        dmudz[k][j][i + 1],
                        dmudz[k - 1][j][i],
                        dmudz[k - 1][j][i + 1]);

                    // ---- AB2 history-blended velocity gradients at the U point ----
                    double ux = (ab2Velocity(u[k][j][i + 1]) - ab2Velocity(u[k][j][i - 1])) / (dxp + dxm);
                    double uy = (ab2Velocity(u[k][j + 1][i]) - ab2Velocity(u[k][j - 1][i])) / (dyp + dym);
                    double uz = (ab2Velocity(u[k + 1][j][i]) - ab2Velocity(u[k - 1][j][i])) / (dzp + dzm);

                    // ---- v_x at u-location: average in y, central diff in x ----
                    double dxv = v[k][j][i + 1].getX() - v[k][j][i].getX();
                    double vx =
                        (averageAb2Velocity(v[k][j][i + 1], v[k][j - 1][i + 1])
                            - averageAb2Velocity(v[k][j][i], v[k][j - 1][i]))
                        / dxv;

                    // ---- w_x at u-location: average in z, central diff in x ----
                    double dxw = w[k][j][i + 1].getX() - w[k][j][i].getX();
                    double wx =
                        (averageAb2Velocity(w[k][j][i + 1], w[k - 1][j][i + 1])
                            - averageAb2Velocity(w[k][j][i], w[k - 1][j][i]))
                        / dxw;

                    // cell volume (for FV scaling)
                    const double vol = u[k][j][i].getdx() * u[k][j][i].getdy() * u[k][j][i].getdz();

                    // --- variable-viscosity contribution: for x-component ---
                    const double termMu = mux * (2.0 * ux)
                        + muy * (uy + vx)
                        + muz * (uz + wx);

                    // flux-like Laplacian part (leave as you had it)
                    const double term1 = (u[k][j][i + 1].getValue() - u[k][j][i].getValue()) / dxp * u[k][j][i].getdy() * u[k][j][i].getdz();
                    const double term2 = (u[k][j][i - 1].getValue() - u[k][j][i].getValue()) / dxm * u[k][j][i].getdy() * u[k][j][i].getdz();
                    const double term3 = (u[k][j + 1][i].getValue() - u[k][j][i].getValue()) / dyp * u[k][j][i].getdx() * u[k][j][i].getdz();
                    const double term4 = (u[k][j - 1][i].getValue() - u[k][j][i].getValue()) / dym * u[k][j][i].getdx() * u[k][j][i].getdz();
                    const double term5 = (u[k + 1][j][i].getValue() - u[k][j][i].getValue()) / dzp * u[k][j][i].getdx() * u[k][j][i].getdy();
                    const double term6 = (u[k - 1][j][i].getValue() - u[k][j][i].getValue()) / dzm * u[k][j][i].getdx() * u[k][j][i].getdy();

                    result[index] = (mu[k][j][i] * (term1 + term2 + term3 + term4 + term5 + term6) + termMu * vol) / Re;
                }
            }
        }
    }

    return result;
}
std::vector<double> RHSu::LaplacianV(const std::vector<std::vector<std::vector<Point>>>& u, const std::vector<std::vector<std::vector<Point>>>& v, const std::vector<std::vector<std::vector<Point>>>& w, const std::vector<std::vector<std::vector<double>>>& mu, const std::vector<std::vector<std::vector<double>>>& dmudx, const std::vector<std::vector<std::vector<double>>>& dmudy, const std::vector<std::vector<std::vector<double>>>& dmudz, double Re) {
    size_t depth = v.size();
    size_t height = v[0].size();
    size_t width = v[0][0].size();
    std::vector<double> result = acquireTermBuffer(
        depth <= 1 ? height * width : depth * height * width);

    // 2D Case
    if (depth <= 1) {
#pragma omp parallel for collapse(2)
        for (size_t j = 1; j < height - 1; ++j) {
            for (size_t i = 1; i < width - 1; ++i) {
                if (v[0][j][i].getBoundaryType() == PointBoundary::Interior) {
                    int index = v[0][j][i].getUniqueNumber();
                    // local cell sizes around v[0][j][i]
                    double dxp = v[0][j][i + 1].getX() - v[0][j][i].getX();
                    double dxm = v[0][j][i].getX() - v[0][j][i - 1].getX();
                    double dyp = v[0][j + 1][i].getY() - v[0][j][i].getY();
                    double dym = v[0][j][i].getY() - v[0][j - 1][i].getY();

                    double mux = (dmudx[0][j + 1][i] + dmudx[0][j][i] + dmudx[0][j + 1][i - 1] + dmudx[0][j][i - 1]) / 4.0;
                    double muy = dmudy[0][j][i];

                    double vx = (ab2Velocity(v[0][j][i + 1]) - ab2Velocity(v[0][j][i - 1])) / (dxp + dxm);
                    double vy = (ab2Velocity(v[0][j + 1][i]) - ab2Velocity(v[0][j - 1][i])) / (dyp + dym);

                    // U rows j and j+1 straddle V[j][i]. This derivative is
                    // collocated at V, not the j-1/j+1 derivative at U[j][i].
                    double dyu = u[0][j + 1][i].getY() - u[0][j][i].getY();
                    double uy =
                        (averageAb2Velocity(u[0][j + 1][i], u[0][j + 1][i - 1])
                            - averageAb2Velocity(u[0][j][i], u[0][j][i - 1]))
                        / dyu;

                    double termMuV = mux * (vx + uy) +
                        muy * (2.0 * vy);

                    double vol = v[0][j][i].getdx() * v[0][j][i].getdy();

                    double term1 = (v[0][j][i + 1].getValue() - v[0][j][i].getValue()) / dxp * v[0][j][i].getdy();
                    double term2 = (v[0][j][i - 1].getValue() - v[0][j][i].getValue()) / dxm * v[0][j][i].getdy();
                    double term3 = (v[0][j + 1][i].getValue() - v[0][j][i].getValue()) / dyp * v[0][j][i].getdx();
                    double term4 = (v[0][j - 1][i].getValue() - v[0][j][i].getValue()) / dym * v[0][j][i].getdx();

                    result[index] = (mu[0][j][i] * (term1 + term2 + term3 + term4) + termMuV*vol) / Re;
                }
            }
        }
    }

    // ===================== LaplacianV : 3D part (AB2) =====================
    else {
#pragma omp parallel for collapse(3)
        for (size_t k = 1; k < depth - 1; ++k) {
            for (size_t j = 1; j < height - 1; ++j) {
                for (size_t i = 1; i < width - 1; ++i) {

                    if (v[k][j][i].getBoundaryType() != PointBoundary::Interior) continue;
                    const int index = v[k][j][i].getUniqueNumber();

                    const double dxp = v[k][j][i + 1].getX() - v[k][j][i].getX();
                    const double dxm = v[k][j][i].getX() - v[k][j][i - 1].getX();
                    const double dyp = v[k][j + 1][i].getY() - v[k][j][i].getY();
                    const double dym = v[k][j][i].getY() - v[k][j - 1][i].getY();
                    const double dzp = v[k + 1][j][i].getZ() - v[k][j][i].getZ();
                    const double dzm = v[k][j][i].getZ() - v[k - 1][j][i].getZ();

                    // ---- prepared gradients of mu, collocated at the V point ----
                    double mux = averageFour(
                        dmudx[k][j][i],
                        dmudx[k][j][i - 1],
                        dmudx[k][j + 1][i],
                        dmudx[k][j + 1][i - 1]);
                    double muy = dmudy[k][j][i];
                    double muz = averageFour(
                        dmudz[k][j][i],
                        dmudz[k][j + 1][i],
                        dmudz[k - 1][j][i],
                        dmudz[k - 1][j + 1][i]);

                    // ---- AB2 history-blended velocity gradients at the V point ----
                    double vx = (ab2Velocity(v[k][j][i + 1]) - ab2Velocity(v[k][j][i - 1])) / (dxp + dxm);
                    double vy = (ab2Velocity(v[k][j + 1][i]) - ab2Velocity(v[k][j - 1][i])) / (dyp + dym);
                    double vz = (ab2Velocity(v[k + 1][j][i]) - ab2Velocity(v[k - 1][j][i])) / (dzp + dzm);

                    // U rows j and j+1, averaged in X, straddle this V point.
                    double dyu = u[k][j + 1][i].getY() - u[k][j][i].getY();
                    double uy =
                        (averageAb2Velocity(u[k][j + 1][i], u[k][j + 1][i - 1])
                            - averageAb2Velocity(u[k][j][i], u[k][j][i - 1]))
                        / dyu;

                    // W rows j and j+1, averaged in Z, straddle this V point.
                    double dyw = w[k][j + 1][i].getY() - w[k][j][i].getY();
                    double wy =
                        (averageAb2Velocity(w[k][j + 1][i], w[k - 1][j + 1][i])
                            - averageAb2Velocity(w[k][j][i], w[k - 1][j][i]))
                        / dyw;

                    const double termMuV = mux * (vx + uy)
                        + muy * (2.0 * vy)
                        + muz * (vz + wy);
                    const double vol = v[k][j][i].getdx() * v[k][j][i].getdy() * v[k][j][i].getdz();

                    const double term1 = (v[k][j][i + 1].getValue() - v[k][j][i].getValue()) / dxp * v[k][j][i].getdy() * v[k][j][i].getdz();
                    const double term2 = (v[k][j][i - 1].getValue() - v[k][j][i].getValue()) / dxm * v[k][j][i].getdy() * v[k][j][i].getdz();
                    const double term3 = (v[k][j + 1][i].getValue() - v[k][j][i].getValue()) / dyp * v[k][j][i].getdx() * v[k][j][i].getdz();
                    const double term4 = (v[k][j - 1][i].getValue() - v[k][j][i].getValue()) / dym * v[k][j][i].getdx() * v[k][j][i].getdz();
                    const double term5 = (v[k + 1][j][i].getValue() - v[k][j][i].getValue()) / dzp * v[k][j][i].getdx() * v[k][j][i].getdy();
                    const double term6 = (v[k - 1][j][i].getValue() - v[k][j][i].getValue()) / dzm * v[k][j][i].getdx() * v[k][j][i].getdy();

                    result[index] = (mu[k][j][i] * (term1 + term2 + term3 + term4 + term5 + term6) + termMuV * vol) / Re;
                }
            }
        }
    }


    return result;
}

std::vector<double> RHSu::LaplacianW(const std::vector<std::vector<std::vector<Point>>>& u, const std::vector<std::vector<std::vector<Point>>>& v, const std::vector<std::vector<std::vector<Point>>>& w, const std::vector<std::vector<std::vector<double>>>& mu, const std::vector<std::vector<std::vector<double>>>& dmudx, const std::vector<std::vector<std::vector<double>>>& dmudy, const std::vector<std::vector<std::vector<double>>>& dmudz, double Re) {
    size_t depth = w.size();
    size_t height = w[0].size();
    size_t width = w[0][0].size();
    std::vector<double> result = acquireTermBuffer(depth * height * width);
    #pragma omp parallel for collapse(3)
    for (size_t k = 1; k < depth - 1; ++k) {
        for (size_t j = 1; j < height - 1; ++j) {
            for (size_t i = 1; i < width - 1; ++i) {
                if (w[k][j][i].getBoundaryType() == PointBoundary::Interior) {
                    int index = w[k][j][i].getUniqueNumber();
                    double dxp = w[k][j][i + 1].getX() - w[k][j][i].getX();
                    double dxm = w[k][j][i].getX() - w[k][j][i - 1].getX();
                    double dyp = w[k][j + 1][i].getY() - w[k][j][i].getY();
                    double dym = w[k][j][i].getY() - w[k][j - 1][i].getY();
                    double dzp = w[k + 1][j][i].getZ() - w[k][j][i].getZ();
                    double dzm = w[k][j][i].getZ() - w[k - 1][j][i].getZ();

                    // ---- prepared gradients of mu, collocated at the W point ----
                    double mux = averageFour(
                        dmudx[k][j][i],
                        dmudx[k][j][i - 1],
                        dmudx[k + 1][j][i],
                        dmudx[k + 1][j][i - 1]);
                    double muy = averageFour(
                        dmudy[k][j][i],
                        dmudy[k][j - 1][i],
                        dmudy[k + 1][j][i],
                        dmudy[k + 1][j - 1][i]);
                    double muz = dmudz[k][j][i];

                    // ---- AB2 history-blended velocity gradients at the W point ----
                    double wx = (ab2Velocity(w[k][j][i + 1]) - ab2Velocity(w[k][j][i - 1])) / (dxp + dxm);
                    double wy = (ab2Velocity(w[k][j + 1][i]) - ab2Velocity(w[k][j - 1][i])) / (dyp + dym);
                    double wz = (ab2Velocity(w[k + 1][j][i]) - ab2Velocity(w[k - 1][j][i])) / (dzp + dzm);

                    // U layers k and k+1, averaged in X, straddle this W point.
                    double dzu = u[k + 1][j][i].getZ() - u[k][j][i].getZ();
                    double uz =
                        (averageAb2Velocity(u[k + 1][j][i], u[k + 1][j][i - 1])
                            - averageAb2Velocity(u[k][j][i], u[k][j][i - 1]))
                        / dzu;

                    // V layers k and k+1, averaged in Y, straddle this W point.
                    double dzv = v[k + 1][j][i].getZ() - v[k][j][i].getZ();
                    double vz =
                        (averageAb2Velocity(v[k + 1][j][i], v[k + 1][j - 1][i])
                            - averageAb2Velocity(v[k][j][i], v[k][j - 1][i]))
                        / dzv;

                    double termMuW =
                        mux * (wx + uz) +
                        muy * (wy + vz) +
                        muz * (2.0 * wz);

                    double vol = w[k][j][i].getdx() * w[k][j][i].getdy() * w[k][j][i].getdz();

                    double term1 = (w[k][j][i + 1].getValue() - w[k][j][i].getValue()) / dxp * w[k][j][i].getdy() * w[k][j][i].getdz();
                    double term2 = (w[k][j][i - 1].getValue() - w[k][j][i].getValue()) / dxm * w[k][j][i].getdy() * w[k][j][i].getdz();
                    double term3 = (w[k][j + 1][i].getValue() - w[k][j][i].getValue()) / dyp * w[k][j][i].getdx() * w[k][j][i].getdz();
                    double term4 = (w[k][j - 1][i].getValue() - w[k][j][i].getValue()) / dym * w[k][j][i].getdx() * w[k][j][i].getdz();
                    double term5 = (w[k + 1][j][i].getValue() - w[k][j][i].getValue()) / dzp * w[k][j][i].getdx() * w[k][j][i].getdy();
                    double term6 = (w[k - 1][j][i].getValue() - w[k][j][i].getValue()) / dzm * w[k][j][i].getdx() * w[k][j][i].getdy();

                    result[index] = (mu[k][j][i]*(term1 + term2 + term3 + term4 + term5 + term6) + termMuW*vol)/Re;
                }
            }
        }
    }
 

    return result;
}


std::vector<double> RHSu::gradP(const std::vector<std::vector<std::vector<Point>>>& uvw, char varType, const std::vector<std::vector<std::vector<Point>>>& p, int numDiorichletPoints) {
    size_t depth = uvw.size();
    size_t height = uvw[0].size();
    size_t width = uvw[0][0].size();

    enum DimensionType { U, V, W, U2D, V2D } dimType;

    // Determine the case (U, V, W, U2D, V2D)
    if (depth <= 1) {
        // 2D cases
        dimType = (varType == 'u') ? U2D : V2D;
    }
    else {
        // 3D cases
        dimType = (varType == 'u') ? U : ((varType == 'v') ? V : W);
    }

    std::vector<double> result = acquireTermBuffer(
        (dimType == U || dimType == V || dimType == W)
            ? depth * height * width
            : height * width);

    // Compute based on the case
    switch (dimType) {
    case U:
        #pragma omp parallel for collapse(3)
        for (size_t k = 1; k < depth - 1; ++k) {
            for (size_t j = 1; j < height - 1; ++j) {
                for (size_t i = 1; i < width - 1; ++i) {
                    if (uvw[k][j][i].getBoundaryType() == PointBoundary::Interior) {
                        int index = uvw[k][j][i].getUniqueNumber();
                        result[index] = -(p[k-1][j-1][i ].getValue() - p[k-1][j-1][i - 1].getValue()) * uvw[k][j][i].getdy() * uvw[k][j][i].getdz();
                    }
                }
            }
        }
        break;
    case V:
        #pragma omp parallel for collapse(3)
        for (size_t k = 1; k < depth - 1; ++k) {
            for (size_t j = 1; j < height - 1; ++j) {
                for (size_t i = 1; i < width - 1; ++i) {
                    if (uvw[k][j][i].getBoundaryType() == PointBoundary::Interior) {
                        int index = uvw[k][j][i].getUniqueNumber();
                        result[index] = -(p[k-1][j ][i-1].getValue() - p[k-1][j - 1][i-1].getValue()) * uvw[k][j][i].getdx() * uvw[k][j][i].getdz();
                    }
                }
            }
        }
        break;
    case W:
        #pragma omp parallel for collapse(3)
        for (size_t k = 1; k < depth - 1; ++k) {
            for (size_t j = 1; j < height - 1; ++j) {
                for (size_t i = 1; i < width - 1; ++i) {
                    if (uvw[k][j][i].getBoundaryType() == PointBoundary::Interior) {
                        int index = uvw[k][j][i].getUniqueNumber();
                        result[index] = -(p[k ][j-1][i-1].getValue() - p[k - 1][j-1][i-1].getValue()) * uvw[k][j][i].getdy() * uvw[k][j][i].getdx();
                    }
                }
            }
        }
        break;
    case U2D:
        #pragma omp parallel for collapse(2)
        for (size_t j = 1; j < height - 1; ++j) {
            for (size_t i = 1; i < width - 1; ++i) {
                if (uvw[0][j][i].getBoundaryType() == PointBoundary::Interior) {
                    int index = uvw[0][j][i].getUniqueNumber();
                    result[index] = -(p[0][j - 1][i].getValue() - p[0][j-1][i - 1].getValue()) * uvw[0][j][i].getdy();
                }
            }
        }
        break;
    case V2D:
        #pragma omp parallel for collapse(2)
        for (size_t j = 1; j < height - 1; ++j) {
            for (size_t i = 1; i < width - 1; ++i) {
                if (uvw[0][j][i].getBoundaryType() == PointBoundary::Interior) {
                    int index = uvw[0][j][i].getUniqueNumber();
                    result[index] = -(p[0][j][i - 1].getValue() - p[0][j - 1][i - 1].getValue()) * uvw[0][j][i].getdx();
                }
            }
        }
        break;
    }

    return result;
}


std::vector<double> RHSu::dudT(const std::vector<std::vector<std::vector<Point>>>& uvw, double dt, const std::vector<std::vector<std::vector<double>>>& rho, int shift) {
    size_t depth = uvw.size();
    size_t height = uvw[0].size();
    size_t width = uvw[0][0].size();

    bool is2D = (depth <= 1);
    const std::size_t resultSize =
        is2D ? height * width : depth * height * width;
    std::vector<double> result = acquireTermBuffer(resultSize);
    densityBuffer.assign(resultSize, 1.0);
    std::vector<double>& rhoFlat = densityBuffer;
    if (is2D) {
        #pragma omp parallel for collapse(2)
        for (size_t j = 0; j < height; ++j) {
            for (size_t i = 0; i < width; ++i) {
                if (uvw[0][j][i].getBoundaryType() == PointBoundary::Interior) {
                    int index = uvw[0][j][i].getUniqueNumber();
                    double u_current = uvw[0][j][i].getValue();
                    result[index] = -1.5 * u_current / dt * uvw[0][j][i].getdx() * uvw[0][j][i].getdy();
                    rhoFlat[index] = rho[0][j][i];
                }
            }
        }
    }
    else {
        #pragma omp parallel for collapse(3)
        for (size_t k = 0; k < depth; ++k) {
            for (size_t j = 0; j < height; ++j) {
                for (size_t i = 0; i < width; ++i) {
                    if (uvw[k][j][i].getBoundaryType() == PointBoundary::Interior) {
                        int index = uvw[k][j][i].getUniqueNumber();
                        double u_current = uvw[k][j][i].getValue();
                        result[index] = -1.5 * u_current / dt * uvw[k][j][i].getdx() * uvw[k][j][i].getdy() * uvw[k][j][i].getdz();
                        rhoFlat[index] = rho[k][j][i];
                    }
                }
            }
        }
    }

    // Adding the dtStep contribution


    vdAdd(result.size(), result.data(), dtStep.data() + shift, result.data());
    vdMul(result.size(), result.data(), rhoFlat.data(), result.data());
    return result;
}


std::vector<double> RHSu::NonLinearU(const std::vector<std::vector<std::vector<Point>>>& u,
    const std::vector<std::vector<std::vector<Point>>>& v,
    const std::vector<std::vector<std::vector<Point>>>& w,
    const std::vector<std::vector<std::vector<double>>>& rho) {
    size_t depth = u.size();
    size_t height = u[0].size();
    size_t width = u[0][0].size();

    bool is2D = (depth <= 1);

    std::vector<double> result = acquireTermBuffer(
        is2D ? height * width : depth * height * width);

    if (is2D) {
        // 2D case
        #pragma omp parallel for collapse(2)
        for (size_t j = 1; j < height - 1; ++j) {
            for (size_t i = 1; i < width - 1; ++i) {
                if (u[0][j][i].getBoundaryType() == PointBoundary::Interior) {
                    int index = u[0][j][i].getUniqueNumber();
                    double dx = u[0][j][i].getdx();
                    double dy = u[0][j][i].getdy();
                    
                    result[index] = -rho[0][j][i]*((u[0][j][i].getValue() + u[0][j][i + 1].getValue()) * (u[0][j][i].getValue() + u[0][j][i + 1].getValue()) * dy
                        - (u[0][j][i].getValue() + u[0][j][i - 1].getValue()) * (u[0][j][i].getValue() + u[0][j][i - 1].getValue()) * dy
                        + (v[0][j][i].getValue() + v[0][j][i + 1].getValue()) * (u[0][j][i].getValue() + u[0][j + 1][i].getValue()) * dx
                        - (v[0][j - 1][i].getValue() + v[0][j - 1][i + 1].getValue()) * (u[0][j][i].getValue() + u[0][j - 1][i].getValue()) * dx) / 4.0;
                }
            }
        }
    }
    else {
        // 3D case
        #pragma omp parallel for collapse(3)
        for (size_t k = 1; k < depth - 1; ++k) {
            for (size_t j = 1; j < height - 1; ++j) {
                for (size_t i = 1; i < width - 1; ++i) {
                    if (u[k][j][i].getBoundaryType() == PointBoundary::Interior) {
                        int index = u[k][j][i].getUniqueNumber();
                        double dx = u[k][j][i].getdx();
                        double dy = u[k][j][i].getdy();
                        double dz = u[k][j][i].getdz();

                        result[index] = -rho[k][j][i] * (((u[k][j][i].getValue() + u[k][j][i + 1].getValue()) * (u[k][j][i].getValue() + u[k][j][i + 1].getValue())
                            - (u[k][j][i].getValue() + u[k][j][i - 1].getValue()) * (u[k][j][i].getValue() + u[k][j][i - 1].getValue())) * dy * dz
                            + ((v[k][j][i].getValue() + v[k][j][i + 1].getValue()) * (u[k][j][i].getValue() + u[k][j + 1][i].getValue())
                            -(v[k][j - 1][i].getValue() + v[k][j - 1][i + 1].getValue()) * (u[k][j][i].getValue() + u[k][j - 1][i].getValue())) * dx * dz
                                + ((w[k][j][i].getValue() + w[k][j][i + 1].getValue()) * (u[k][j][i].getValue() + u[k + 1][j][i].getValue())
                                - (w[k - 1][j][i].getValue() + w[k - 1][j][i + 1].getValue()) * (u[k][j][i].getValue() + u[k - 1][j][i].getValue())) * dx * dy) / 4.0;
                    }
                }
            }
        }
    }
    return result;
}


std::vector<double> RHSu::NonLinearV(const std::vector<std::vector<std::vector<Point>>>& u,
    const std::vector<std::vector<std::vector<Point>>>& v,
    const std::vector<std::vector<std::vector<Point>>>& w,
    const std::vector<std::vector<std::vector<double>>>& rho) {
    size_t depth = v.size();
    size_t height = v[0].size();
    size_t width = v[0][0].size();

    bool is2D = (depth <= 1);

    std::vector<double> result = acquireTermBuffer(
        is2D ? height * width : depth * height * width);

    if (is2D) {
        // 2D case
        #pragma omp parallel for collapse(2)
        for (size_t j = 1; j < height - 1; ++j) {
            for (size_t i = 1; i < width - 1; ++i) {
                if (v[0][j][i].getBoundaryType() == PointBoundary::Interior) {
                    int index = v[0][j][i].getUniqueNumber();
                    double dx = v[0][j][i].getdx();
                    double dy = v[0][j][i].getdy();

                    result[index] = -rho[0][j][i]*((v[0][j][i].getValue() + v[0][j + 1][i].getValue()) * (v[0][j][i].getValue() + v[0][j + 1][i].getValue()) * dx
                        - (v[0][j][i].getValue() + v[0][j - 1][i].getValue()) * (v[0][j][i].getValue() + v[0][j - 1][i].getValue()) * dx
                        + (v[0][j][i].getValue() + v[0][j][i + 1].getValue()) * (u[0][j + 1][i].getValue() + u[0][j][i].getValue()) * dy
                        - (v[0][j][i].getValue() + v[0][j][i - 1].getValue()) * (u[0][j + 1][i - 1].getValue() + u[0][j][i - 1].getValue()) * dy) / 4.0;
                }
            }
        }
    }
    else {
        // 3D case
        #pragma omp parallel for collapse(3)
        for (size_t k = 1; k < depth - 1; ++k) {
            for (size_t j = 1; j < height - 1; ++j) {
                for (size_t i = 1; i < width - 1; ++i) {
                    if (v[k][j][i].getBoundaryType() == PointBoundary::Interior) {
                        int index = v[k][j][i].getUniqueNumber();
                        double dx = v[k][j][i].getdx();
                        double dy = v[k][j][i].getdy();
                        double dz = v[k][j][i].getdz();

                        result[index] = -rho[k][j][i] * ((v[k][j][i].getValue() + v[k][j + 1][i].getValue()) * (v[k][j][i].getValue() + v[k][j + 1][i].getValue()) * dx * dz
                            - (v[k][j][i].getValue() + v[k][j - 1][i].getValue()) * (v[k][j][i].getValue() + v[k][j - 1][i].getValue()) * dx * dz
                            + (v[k][j][i].getValue() + v[k][j][i + 1].getValue()) * (u[k][j + 1][i].getValue() + u[k][j][i].getValue()) * dy * dz
                            - (v[k][j][i].getValue() + v[k][j][i - 1].getValue()) * (u[k][j + 1][i - 1].getValue() + u[k][j][i - 1].getValue()) * dy * dz
                            + (w[k][j][i].getValue() + w[k][j + 1][i].getValue()) * (v[k][j][i].getValue() + v[k + 1][j][i].getValue()) * dx * dy
                            - (w[k - 1][j][i].getValue() + w[k - 1][j + 1][i].getValue()) * (v[k][j][i].getValue() + v[k - 1][j][i].getValue()) * dx * dy) / 4.0;
                    }
                }
            }
        }
    }
    return result;
}



std::vector<double> RHSu::NonLinearW(const std::vector<std::vector<std::vector<Point>>>& u,
    const std::vector<std::vector<std::vector<Point>>>& v,
    const std::vector<std::vector<std::vector<Point>>>& w,
    const std::vector<std::vector<std::vector<double>>>& rho) {
    size_t depth = w.size();
    size_t height = w[0].size();
    size_t width = w[0][0].size();


    std::vector<double> result = acquireTermBuffer(depth * height * width);
    // 3D case
    #pragma omp parallel for collapse(3)
    for (size_t k = 1; k < depth - 1; ++k) {
        for (size_t j = 1; j < height - 1; ++j) {
            for (size_t i = 1; i < width - 1; ++i) {
                if (w[k][j][i].getBoundaryType() == PointBoundary::Interior) {
                    int index = w[k][j][i].getUniqueNumber();
                    double dx = w[k][j][i].getdx();
                    double dy = w[k][j][i].getdy();
                    double dz = w[k][j][i].getdz();

                    result[index] = -rho[k][j][i]*((w[k][j][i].getValue() + w[k + 1][j][i].getValue()) * (w[k][j][i].getValue() + w[k + 1][j][i].getValue()) * dx * dz
                        - (w[k][j][i].getValue() + w[k - 1][j][i].getValue()) * (w[k][j][i].getValue() + w[k - 1][j][i].getValue()) * dx * dz
                        + (w[k][j][i].getValue() + w[k][j][i + 1].getValue()) * (u[k + 1][j][i].getValue() + u[k][j][i].getValue()) * dy * dz
                        - (w[k][j][i].getValue() + w[k][j][i - 1].getValue()) * (u[k + 1][j][i - 1].getValue() + u[k][j][i - 1].getValue()) * dy * dz
                        + (w[k][j + 1][i].getValue() + w[k][j][i].getValue()) * (v[k][j][i].getValue() + v[k + 1][j][i].getValue()) * dx * dy
                        - (w[k][j - 1][i].getValue() + w[k][j][i].getValue()) * (v[k][j - 1][i].getValue() + v[k + 1][j - 1][i].getValue()) * dx * dy) / 4.0;
                }
            }
        }
    }
    return result;
}
std::vector<double> RHSu::SurTension_GradPhiX(
    const std::vector<std::vector<std::vector<Point>>>& u,
    const std::vector<std::vector<std::vector<double>>>& phiS,
    const std::vector<std::vector<std::vector<double>>>& kappa,
    double We)
{
    const size_t nz_u = u.size();
    const size_t ny_u = u[0].size();
    const size_t nx_u = u[0][0].size();

    const bool is2D = (nz_u <= 1);
    std::vector<double> result = acquireTermBuffer(
        is2D ? ny_u * nx_u : nz_u * ny_u * nx_u);

    if (We == 0.0) return result;

    if (is2D) {
#pragma omp parallel for collapse(2)
        for (size_t j = 1; j < ny_u - 1; ++j) {
            for (size_t i = 1; i < nx_u - 1; ++i) {
                if (u[0][j][i].getBoundaryType() != PointBoundary::Interior) continue;

                const int idx = u[0][j][i].getUniqueNumber();
                const double A_yz = u[0][j][i].getdy(); // 2D: dy

                // u-face sits between phi( j-1, i-1 ) and phi( j-1, i )
                const double phiL = phiS[0][j - 1][i - 1];
                const double phiR = phiS[0][j - 1][i];

                const double kapL = kappa[0][j - 1][i - 1];
                const double kapR = kappa[0][j - 1][i];
                const double kap_face = 0.5 * (kapL + kapR);

                result[idx] = (kap_face * (phiR - phiL) * A_yz) / We;
            }
        }
        return result;
    }

    // 3D
#pragma omp parallel for collapse(3)
    for (size_t k = 1; k < nz_u - 1; ++k) {
        for (size_t j = 1; j < ny_u - 1; ++j) {
            for (size_t i = 1; i < nx_u - 1; ++i) {
                if (u[k][j][i].getBoundaryType() != PointBoundary::Interior) continue;

                const int idx = u[k][j][i].getUniqueNumber();
                const double A_yz = u[k][j][i].getdy() * u[k][j][i].getdz();

                // map to pressure cells: k-1, j-1 plane, between i-1 and i
                const double phiL = phiS[k - 1][j - 1][i - 1];
                const double phiR = phiS[k - 1][j - 1][i];

                const double kapL = kappa[k - 1][j - 1][i - 1];
                const double kapR = kappa[k - 1][j - 1][i];
                const double kap_face = 0.5 * (kapL + kapR);

                result[idx] = (kap_face * (phiR - phiL) * A_yz) / We;
            }
        }
    }

    return result;
}

std::vector<double> RHSu::SurTension_GradPhiY(
    const std::vector<std::vector<std::vector<Point>>>& v,
    const std::vector<std::vector<std::vector<double>>>& phiS,
    const std::vector<std::vector<std::vector<double>>>& kappa,
    double We)
{
    const size_t nz_v = v.size();
    const size_t ny_v = v[0].size();
    const size_t nx_v = v[0][0].size();

    const bool is2D = (nz_v <= 1);
    std::vector<double> result = acquireTermBuffer(
        is2D ? ny_v * nx_v : nz_v * ny_v * nx_v);

    if (We == 0.0) return result;

    if (is2D) {
#pragma omp parallel for collapse(2)
        for (size_t j = 1; j < ny_v - 1; ++j) {
            for (size_t i = 1; i < nx_v - 1; ++i) {
                if (v[0][j][i].getBoundaryType() != PointBoundary::Interior) continue;

                const int idx = v[0][j][i].getUniqueNumber();
                const double A_xz = v[0][j][i].getdx(); // 2D: dx

                // v-face sits between phi( j-1, i-1 ) and phi( j, i-1 )
                const double phiB = phiS[0][j - 1][i - 1];
                const double phiT = phiS[0][j][i - 1];

                const double kapB = kappa[0][j - 1][i - 1];
                const double kapT = kappa[0][j][i - 1];
                const double kap_face = 0.5 * (kapB + kapT);

                result[idx] = (kap_face * (phiT - phiB) * A_xz) / We;
            }
        }
        return result;
    }

    // 3D
#pragma omp parallel for collapse(3)
    for (size_t k = 1; k < nz_v - 1; ++k) {
        for (size_t j = 1; j < ny_v - 1; ++j) {
            for (size_t i = 1; i < nx_v - 1; ++i) {
                if (v[k][j][i].getBoundaryType() != PointBoundary::Interior) continue;

                const int idx = v[k][j][i].getUniqueNumber();
                const double A_xz = v[k][j][i].getdx() * v[k][j][i].getdz();

                // map to pressure cells: k-1 slab, between j-1 and j at fixed i-1
                const double phiB = phiS[k - 1][j - 1][i - 1];
                const double phiT = phiS[k - 1][j][i - 1];

                const double kapB = kappa[k - 1][j - 1][i - 1];
                const double kapT = kappa[k - 1][j][i - 1];
                const double kap_face = 0.5 * (kapB + kapT);

                result[idx] = (kap_face * (phiT - phiB) * A_xz) / We;
            }
        }
    }

    return result;
}

// You also need W (z component) in 3D:
std::vector<double> RHSu::SurTension_GradPhiZ(
    const std::vector<std::vector<std::vector<Point>>>& w,
    const std::vector<std::vector<std::vector<double>>>& phiS,
    const std::vector<std::vector<std::vector<double>>>& kappa,
    double We)
{
    const size_t nz_w = w.size();
    const size_t ny_w = w[0].size();
    const size_t nx_w = w[0][0].size();

    std::vector<double> result = acquireTermBuffer(nz_w * ny_w * nx_w);
    if (We == 0.0) return result;

#pragma omp parallel for collapse(3)
    for (size_t k = 1; k < nz_w - 1; ++k) {
        for (size_t j = 1; j < ny_w - 1; ++j) {
            for (size_t i = 1; i < nx_w - 1; ++i) {
                if (w[k][j][i].getBoundaryType() != PointBoundary::Interior) continue;

                const int idx = w[k][j][i].getUniqueNumber();
                const double A_xy = w[k][j][i].getdx() * w[k][j][i].getdy();

                // w-face between phi(k-1, j-1, i-1) and phi(k, j-1, i-1)
                const double phiB = phiS[k - 1][j - 1][i - 1];
                const double phiF = phiS[k][j - 1][i - 1];

                const double kapB = kappa[k - 1][j - 1][i - 1];
                const double kapF = kappa[k][j - 1][i - 1];
                const double kap_face = 0.5 * (kapB + kapF);

                result[idx] = (kap_face * (phiF - phiB) * A_xy) / We;
            }
        }
    }

    return result;
}


std::vector<double> RHSu::Gravity_GradRho(const std::vector<std::vector<std::vector<Point>>>& v, const std::vector<std::vector<std::vector<double>>>& rho, double Fr, double rhoRef) {
    size_t depth = v.size();
    size_t height = v[0].size();
    size_t width = v[0][0].size();

    bool is2D = (depth <= 1);

    std::vector<double> result = acquireTermBuffer(
        is2D ? height * width : depth * height * width);

    if (is2D) {
        // 2D case
        #pragma omp parallel for collapse(2)
        for (size_t j = 1; j < height - 1; ++j) {
            for (size_t i = 1; i < width - 1; ++i) {
                if (v[0][j][i].getBoundaryType() == PointBoundary::Interior) {
                    int index = v[0][j][i].getUniqueNumber();
                    double dx = v[0][j][i].getdx();
                    double dy = v[0][j][i].getdy();
                    result[index] = - (rho[0][j][i] - rhoRef)*dx*dy / pow(Fr, 2);
                }
            }
        }
    }
    else {
        // 3D case
        #pragma omp parallel for collapse(3)
        for (size_t k = 1; k < depth - 1; ++k) {
            for (size_t j = 1; j < height - 1; ++j) {
                for (size_t i = 1; i < width - 1; ++i) {
                        int index = v[k][j][i].getUniqueNumber();
                        double dx = v[k][j][i].getdx();
                        double dy = v[k][j][i].getdy();
                        double dz = v[k][j][i].getdz();

                        result[index] = -(rho[k][j][i] - rhoRef)* dx * dy * dz/ pow(Fr, 2);
                }
            }
        }
    }
    return result;
}
void RHSu::applyBoundaryConditions(BoundaryContainer& boundaryContainer, const std::vector<std::vector<std::vector<Point>>>& u, const std::vector<std::vector<std::vector<Point>>>& v, const std::vector<std::vector<std::vector<Point>>>& p,const std::vector<std::vector<std::vector<double>>>& rho, const std::vector<std::vector<std::vector<double>>>& mu, int indexCorrection, bool is2D, double dt, double Re, double Fr) {
    // The function applies boundary conditions to the right-hand side vector (RHSu) based on the given boundary conditions.
    // boundaryContainer is an object that holds the boundary conditions.
    // u is the array containing values of the variables at each grid point.
    // is2D is a boolean variable to check if the simulation is 2D or 3D.
    // dt is the time step.

    // 3D Boundary Conditions
    if (!is2D) {
        // Processing East boundary
        const Boundary& Wtype = boundaryContainer.getBoundary(PointBoundary::W);
        if (Wtype.type == BoundaryKind::FixedValue) {
            // Fixed value boundary condition for East boundary
            for (int k = 0; k < u.size(); ++k) {
                for (int j = 0; j < u[0].size(); ++j) {
                    if (u[k][j][0].getBoundaryType() == PointBoundary::W) {
                        int index = u[k][j][0].getUniqueNumber() + indexCorrection;
                        RHScurrent[index] = 0;
                    }
                }
            }
        }
        else if (Wtype.type == BoundaryKind::Convective) {
            // Convective boundary condition for East boundary
            for (int k = 0; k < u.size(); ++k) {
                for (int j = 0; j < u[0].size(); ++j) {
                    if (u[k][j][0].getBoundaryType() == PointBoundary::W) {
                        int index = u[k][j][0].getUniqueNumber() + indexCorrection;
                        double prevValue = u[k][j][0].getPrevValue();
                        double prevPrevValue = u[k][j][0].getPrevPrevValue();
                        double value = u[k][j][0].getValue();
                        double dudx = (-value + u[k][j][1].getValue()) / u[k][j][0].getdx();
                        RHScurrent[index] = (3 * value - 4 * prevValue + prevPrevValue) / (2 * dt) + dudx;
                    }
                }
            }
        }
        else if (Wtype.type == BoundaryKind::Slip) {
            // Convective boundary condition for West boundary
            for (int k = 0; k < u.size(); ++k) {
                for (int j = 0; j < u[0].size(); ++j) {
                    if (u[k][j][0].getBoundaryType() == PointBoundary::W) {
                        int index = u[k][j][0].getUniqueNumber() + indexCorrection;

                        double value = u[k][j][0].getValue();
                        double value1 = u[k][j][1].getValue();
                        double value2 = u[k][j][2].getValue();

                        RHScurrent[index] = -value + (9.0 * value1 - value2) / 8.0;
                    }
                }
            }
        }
        else if (Wtype.type == BoundaryKind::Symmetry) {
            // Tangential velocities live on the X boundary plane. The normal
            // U momentum equation is assembled below with its periodic X
            // neighbours and the full 3D terms.
            if (indexCorrection != 0) {
                const int last = static_cast<int>(u[0][0].size()) - 1;
                for (int k = 0; k < static_cast<int>(u.size()); ++k) {
                    for (int j = 0; j < static_cast<int>(u[k].size()); ++j) {
                        if (u[k][j][0].getBoundaryType() != PointBoundary::W) continue;
                        const int index = u[k][j][0].getUniqueNumber() + indexCorrection;
                        const double seamValue = last >= 4
                            ? 9.0 * (u[k][j][1].getValue() + u[k][j][last - 1].getValue()) / 16.0
                                - (u[k][j][2].getValue() + u[k][j][last - 2].getValue()) / 16.0
                            : 0.5 * (u[k][j][1].getValue() + u[k][j][last - 1].getValue());
                        RHScurrent[index] = seamValue - u[k][j][0].getValue();
                    }
                }
            }
        }


        // Processing other boundaries in a similar fashion

        //West boundary
        int last = u[0][0].size() - 1; // index of last element in the inner-most array
        const Boundary& Etype = boundaryContainer.getBoundary(PointBoundary::E);
        if (Etype.type == BoundaryKind::FixedValue) {
            for (int k = 0; k < u.size(); ++k) {
                for (int j = 0; j < u[0].size(); ++j) {
                    if (u[k][j][last].getBoundaryType() == PointBoundary::E) {
                        int index = u[k][j][last].getUniqueNumber() + indexCorrection;
                        RHScurrent[index] = 0;
                    }
                }
            }
        }
        else if (Etype.type == BoundaryKind::Convective) {
            for (int k = 0; k < u.size(); ++k) {
                for (int j = 0; j < u[0].size(); ++j) {
                    if (u[k][j][last].getBoundaryType() == PointBoundary::E) {
                        int index = u[k][j][last].getUniqueNumber() + indexCorrection;
                        double prevValue = u[k][j][last].getPrevValue();
                        double prevPrevValue = u[k][j][last].getPrevPrevValue();
                        double value = u[k][j][last].getValue();
                        double dudx = (-u[k][j][last - 1].getValue() + value) / u[k][j][last].getdx();
                        RHScurrent[index] = (3 * value - 4 * prevValue + prevPrevValue) / (2 * dt) + dudx;
                    }
                }
            }
        }
        else if (Etype.type == BoundaryKind::Slip) {
            // Convective boundary condition for West boundary
            for (int k = 0; k < u.size(); ++k) {
                for (int j = 0; j < u[0].size(); ++j) {
                    if (u[k][j][last].getBoundaryType() == PointBoundary::E) {
                        int index = u[k][j][last].getUniqueNumber() + indexCorrection;

                        double value = u[k][j][last].getValue();
                        double value1 = u[k][j][last - 1].getValue();
                        double value2 = u[k][j][last - 2].getValue();

                        RHScurrent[index] = -value + (9.0 * value1 - value2) / 8.0;
                    }
                }
            }
        }
        else if (Etype.type == BoundaryKind::Symmetry) {
            for (int k = 0; k < static_cast<int>(u.size()); ++k) {
                for (int j = 0; j < static_cast<int>(u[k].size()); ++j) {
                    if (u[k][j][last].getBoundaryType() != PointBoundary::E) continue;
                    const int index = u[k][j][last].getUniqueNumber() + indexCorrection;
                    RHScurrent[index] = u[k][j][0].getValue() - u[k][j][last].getValue();
                }
            }
        }
        // North boundary
        int lastY = u[0].size() - 1; // index of the last element in the middle array (Y dimension)
        const Boundary& Ntype = boundaryContainer.getBoundary(PointBoundary::N);
        if (Ntype.type == BoundaryKind::FixedValue) {
            for (int k = 0; k < u.size(); ++k) {
                for (int i = 0; i < u[0][0].size(); ++i) {
                    if (u[k][lastY][i].getBoundaryType() == PointBoundary::N) {
                        int index = u[k][lastY][i].getUniqueNumber() + indexCorrection;
                        RHScurrent[index] = 0;
                    }
                }
            }
        }
        else if (Ntype.type == BoundaryKind::Convective) {
            for (int k = 0; k < u.size(); ++k) {
                for (int i = 0; i < u[0][0].size(); ++i) {
                    if (u[k][lastY][i].getBoundaryType() == PointBoundary::N) {
                        int index = u[k][lastY][i].getUniqueNumber() + indexCorrection;
                        double prevValue = u[k][lastY][i].getPrevValue();
                        double prevPrevValue = u[k][lastY][i].getPrevPrevValue();
                        double value = u[k][lastY][i].getValue();
                        double dudy = (-u[k][lastY - 1][i].getValue() + value) / u[k][lastY][i].getdy();
                        RHScurrent[index] = (3 * value - 4 * prevValue + prevPrevValue) / (2 * dt) + dudy;
                    }
                }
            }
        }
        // South boundary
        const Boundary& Stype = boundaryContainer.getBoundary(PointBoundary::S);
        if (Stype.type == BoundaryKind::FixedValue) {
            for (int k = 0; k < u.size(); ++k) {
                for (int i = 0; i < u[0][0].size(); ++i) {
                    if (u[k][0][i].getBoundaryType() == PointBoundary::S) {
                        int index = u[k][0][i].getUniqueNumber() + indexCorrection;
                        RHScurrent[index] = 0;
                    }
                }
            }
        }
        else if (Stype.type == BoundaryKind::Convective) {
            for (int k = 0; k < u.size(); ++k) {
                for (int i = 0; i < u[0][0].size(); ++i) {
                    if (u[k][0][i].getBoundaryType() == PointBoundary::S) {
                        int index = u[k][0][i].getUniqueNumber() + indexCorrection;
                        double prevValue = u[k][0][i].getPrevValue();
                        double prevPrevValue = u[k][0][i].getPrevPrevValue();
                        double value = u[k][0][i].getValue();
                        double dudy = (u[k][1][i].getValue() - value) / u[k][0][i].getdy();
                        RHScurrent[index] = (3 * value - 4 * prevValue + prevPrevValue) / (2 * dt) + dudy;
                    }
                }
            }
        }
        // Forward boundary
        int lastZ = u.size() - 1; // index of the last element in the outer array (Z dimension)
        const Boundary& Ftype = boundaryContainer.getBoundary(PointBoundary::F);
        if (Ftype.type == BoundaryKind::FixedValue) {
            for (int j = 0; j < u[0].size(); ++j) {
                for (int i = 0; i < u[0][0].size(); ++i) {
                    if (u[lastZ][j][i].getBoundaryType() == PointBoundary::F) {
                        int index = u[lastZ][j][i].getUniqueNumber() + indexCorrection;
                        RHScurrent[index] = 0;
                    }
                }
            }
        }
        else if (Ftype.type == BoundaryKind::Convective) {
            for (int j = 0; j < u[0].size(); ++j) {
                for (int i = 0; i < u[0][0].size(); ++i) {
                    if (u[lastZ][j][i].getBoundaryType() == PointBoundary::F) {
                        int index = u[lastZ][j][i].getUniqueNumber() + indexCorrection;
                        double prevValue = u[lastZ][j][i].getPrevValue();
                        double prevPrevValue = u[lastZ][j][i].getPrevPrevValue();
                        double value = u[lastZ][j][i].getValue();
                        double dudz = (-u[lastZ - 1][j][i].getValue() + value) / u[lastZ][j][i].getdz();
                        RHScurrent[index] = (3 * value - 4 * prevValue + prevPrevValue) / (2 * dt) + dudz;
                    }
                }
            }
        }
        else if (Ftype.type == BoundaryKind::Slip) {
            // Convective boundary condition for West boundary
            for (int j = 0; j < u[0].size(); ++j) {
                for (int i = 0; i < u[0][0].size(); ++i) {
                    if (u[lastZ][j][i].getBoundaryType() == PointBoundary::F) {
                        int index = u[lastZ][j][i].getUniqueNumber() + indexCorrection;

                        double value = u[lastZ][j][i].getValue();
                        double value1 = u[lastZ - 1][j][i].getValue();
                        double value2 = u[lastZ - 2][j][i].getValue();

                        RHScurrent[index] = -value + (9.0 * value1 - value2) / 8.0;
                    }
                }
            }
        }




        // Backward boundary
        const Boundary& Btype = boundaryContainer.getBoundary(PointBoundary::B);
        if (Btype.type == BoundaryKind::FixedValue) {
            for (int j = 0; j < u[0].size(); ++j) {
                for (int i = 0; i < u[0][0].size(); ++i) {
                    if (u[0][j][i].getBoundaryType() == PointBoundary::B) {
                        int index = u[0][j][i].getUniqueNumber() + indexCorrection;
                        RHScurrent[index] = 0;
                    }
                }
            }
        }
        else if (Btype.type == BoundaryKind::Convective) {
            for (int j = 0; j < u[0].size(); ++j) {
                for (int i = 0; i < u[0][0].size(); ++i) {
                    if (u[0][j][i].getBoundaryType() == PointBoundary::B) {
                        int index = u[0][j][i].getUniqueNumber() + indexCorrection;
                        double prevValue = u[0][j][i].getPrevValue();
                        double prevPrevValue = u[0][j][i].getPrevPrevValue();
                        double value = u[0][j][i].getValue();
                        double dudz = (u[1][j][i].getValue() - value) / u[0][j][i].getdz();
                        RHScurrent[index] = (3 * value - 4 * prevValue + prevPrevValue) / (2 * dt) + dudz;
                    }
                }
            }
        }
        else if (Btype.type == BoundaryKind::Slip) {
            // Convective boundary condition for West boundary
            for (int j = 0; j < u[0].size(); ++j) {
                for (int i = 0; i < u[0][0].size(); ++i) {
                    if (u[0][j][i].getBoundaryType() == PointBoundary::B) {
                        int index = u[0][j][i].getUniqueNumber() + indexCorrection;

                        double value = u[0][j][i].getValue();
                        double value1 = u[1][j][i].getValue();
                        double value2 = u[2][j][i].getValue();

                        RHScurrent[index] = -value + (9.0 * value1 - value2) / 8.0;
                    }
                }
            }
        }

    } else {
    // 2D Boundary Conditions

    // Processing West boundary
    const Boundary& Wtype = boundaryContainer.getBoundary(PointBoundary::W);
    if (Wtype.type == BoundaryKind::FixedValue) {
        // Fixed value boundary condition for West boundary
        for (int j = 0; j < u[0].size(); ++j) {
            if (u[0][j][0].getBoundaryType() == PointBoundary::W) {
                int index = u[0][j][0].getUniqueNumber() + indexCorrection;
                RHScurrent[index] = 0;
            }
        }
    }
    else if (Wtype.type == BoundaryKind::Slip) {
        // Convective boundary condition for West boundary
        for (int j = 0; j < u[0].size(); ++j) {
            if (u[0][j][0].getBoundaryType() == PointBoundary::W) {
                int index = u[0][j][0].getUniqueNumber() + indexCorrection;
 
                double value = u[0][j][0].getValue();
                double value1 = u[0][j][1].getValue();
                double value2 = u[0][j][2].getValue();


                RHScurrent[index] = -value + (9.0 * value1 - value2) / 8.0;
            }
        }
    }
    else if (Wtype.type == BoundaryKind::Convective) {
        // Convective boundary condition for West boundary
        for (int j = 0; j < u[0].size(); ++j) {
            if (u[0][j][0].getBoundaryType() == PointBoundary::W) {
                int index = u[0][j][0].getUniqueNumber() + indexCorrection;
                double prevValue = u[0][j][0].getPrevValue();
                double prevPrevValue = u[0][j][0].getPrevPrevValue();
                double value = u[0][j][0].getValue();
                double dx1 = (u[0][j][0].getX() - u[0][j][1].getX());
                double dx2 = (u[0][j][0].getX() - u[0][j][2].getX());
                double dx3 = (u[0][j][1].getX() - u[0][j][2].getX());
                double a = +(dx1  + dx2 ) / (dx1 * dx2 );
                double b = -dx2 / (dx1 * dx3);
                double c = +dx1 / (dx2 * dx3);
                double dudx = (a*value + b*u[0][j][1].getValue() + c*u[0][j][2].getValue());
                RHScurrent[index] = (-3.0 * value + 4.0 * prevValue - prevPrevValue) / (2.0 * dt) + dudx;
            }
        }
    }
    else if (Wtype.type == BoundaryKind::Symmetry) {
        for (int j = 0; j < u[0].size(); ++j) {
            if (u[0][j][0].getBoundaryType() == PointBoundary::W) {
                if (indexCorrection == 0) {
                    double Inertia, convectiveTerm, pressureGrad, Laplas, gravity;
                    int index = u[0][j][0].getUniqueNumber() + indexCorrection;
                    int last = u[0][j].size() - 1;
                    double prevValue = u[0][j][0].getPrevValue();
                    double prevPrevValue = u[0][j][0].getPrevPrevValue();
                    double value = u[0][j][0].getValue();
                    // cell sizes around (0,j,i)
                    double dxp = u[0][j][1].getX() - u[0][j][0].getX();
                    double dxm = u[0][j][last].getX() - u[0][j][last - 1].getX();
                    double dyp = u[0][j + 1][0].getY() - u[0][j][0].getY();
                    double dym = u[0][j][0].getY() - u[0][j - 1][0].getY();
                    double dx = (dxp + dxm) / 2;
                    // Point::dy is the periodic U-control-volume height.
                    // Averaging neighbour-coordinate distances gives 3h/4
                    // beside a physical N/S wall even though the CV is h high.
                    double dy = u[0][j][0].getdy();
                    Inertia = rho[0][j][0] * (-3.0 * value + 4.0 * prevValue - prevPrevValue) / (2.0 * dt) * dx*dy;
                    convectiveTerm = -rho[0][j][0] * ((u[0][j][0].getValue() + u[0][j][1].getValue()) * (u[0][j][0].getValue() + u[0][j][1].getValue()) * dy/2.0
                        - (u[0][j][0].getValue() + u[0][j][last-1].getValue()) * (u[0][j][0].getValue() + u[0][j][last-1].getValue()) * dy/2.0
                        + (v[0][j][0].getValue()) * (u[0][j][0].getValue() + u[0][j + 1][0].getValue()) * dx
                        - (v[0][j - 1][0].getValue()) * (u[0][j][0].getValue() + u[0][j - 1][0].getValue()) * dx) / 2.0;
                    pressureGrad = -(p[0][j - 1][0].getValue() - p[0][j - 1][last - 1].getValue()) * dy;


                    // ---- gradients of mu at u-cell center (central differences) ----
                    double mux = (mu[0][j][1] - mu[0][j][last - 1]) / (dxp + dxm);
                    double muy = (mu[0][j + 1][0] - mu[0][j - 1][0]) / (dyp + dym);
                    // ---- velocity gradients for u at center ----
                    double ux = (u[0][j][1].getValue() - u[0][j][last - 1].getValue()) / (dxp + dxm);
                    double uy = (u[0][j + 1][0].getValue() - u[0][j - 1][0].getValue()) / (dyp + dym);
                    // ---- v_x at u-location: average in y, central diff in x ----
                    double dxv = 2.0 * (v[0][j][1].getX() - v[0][j][0].getX());
                    double vx =
                        ((v[0][j][1].getValue() + v[0][j - 1][1].getValue())
                            - (v[0][j][last].getValue() + v[0][j - 1][last].getValue()))
                        / (2.0 * dxv);
                    // cell volume (for FV scaling)
                    double vol = dx * dy;

                    // --- variable-viscosity contribution: for x-component ---
                    double termMu = mux * (2.0 * ux)       // �_x * 2 u_x
                        + muy * (uy + vx);        // �_y * (u_y + v_x)


                    double term1 = (u[0][j][1].getValue() - u[0][j][0].getValue()) / dxp * u[0][j][0].getdy();
                    double term2 = (u[0][j][last - 1].getValue() - u[0][j][0].getValue()) / dxm * dy;
                    double term3 = (u[0][j + 1][0].getValue() - u[0][j][0].getValue()) / dyp * dx;
                    double term4 = (u[0][j - 1][0].getValue() - u[0][j][0].getValue()) / dym * dx;

                    Laplas = (mu[0][j][0] * (term1 + term2 + term3 + term4) + termMu * vol) / Re;
                    gravity = 0;//-rho[0][j][0] * dx * dy / pow(Fr, 2);
                    RHScurrent[index] = Inertia + convectiveTerm + pressureGrad + Laplas + gravity;
                }
                else {
                    int last = u[0][j].size() - 1;
                    int index = u[0][j][0].getUniqueNumber() + indexCorrection;
                    RHScurrent[index] = -u[0][j][0].getValue() + 9 * (u[0][j][1].getValue() + u[0][j][last - 1].getValue()) / 16 - (u[0][j][2].getValue() + u[0][j][last - 2].getValue()) / 16;
                }
            }
        }

    }
        //East boundary
        int last = u[0][0].size() - 1; // index of last element in the inner-most array
        const Boundary& Etype = boundaryContainer.getBoundary(PointBoundary::E);
        if (Etype.type == BoundaryKind::FixedValue) {
            
                for (int j = 0; j < u[0].size(); ++j) {
                    if (u[0][j][last].getBoundaryType() == PointBoundary::E) {
                        int index = u[0][j][last].getUniqueNumber() + indexCorrection;
                        RHScurrent[index] = 0;
                    }
                }
            
        }
        else if (Etype.type == BoundaryKind::Slip) {
            // Convective boundary condition for West boundary
            for (int j = 0; j < u[0].size(); ++j) {
                if (u[0][j][last].getBoundaryType() == PointBoundary::E) {
                    int index = u[0][j][last].getUniqueNumber() + indexCorrection;

                    double value = u[0][j][last].getValue();
                    double value1 = u[0][j][last - 1].getValue();
                    double value2 = u[0][j][last - 2].getValue();


                    RHScurrent[index] = -value + (9.0 * value1 - value2) / 8.0;
                }
            }
        }
        else if (Etype.type == BoundaryKind::Convective) {
            
                for (int j = 0; j < u[0].size(); ++j) {
                    if (u[0][j][last].getBoundaryType() == PointBoundary::E) {
                        int index = u[0][j][last].getUniqueNumber() + indexCorrection;
                        double prevValue = u[0][j][last].getPrevValue();
                        double prevPrevValue = u[0][j][last].getPrevPrevValue();
                        double value = u[0][j][last].getValue();
                        double dx1 = (u[0][j][last].getX() - u[0][j][last - 1].getX());
                        double dx2 = (u[0][j][last].getX() - u[0][j][last - 2].getX());
                        double dx3 = (u[0][j][last - 1].getX() - u[0][j][last - 2].getX());
                        double a = -(dx1 + dx2 ) / (dx1 * dx2 );
                        double b = dx2 / (dx1 * dx3);
                        double c = -dx1 / (dx2 * dx3);
                        double dudx = (a * value + b * u[0][j][last - 1].getValue() + c * u[0][j][last - 2].getValue());
                        RHScurrent[index] = (-3.0 * value + 4.0 * prevValue - prevPrevValue)/ (2.0 * dt) + dudx;
                    }
                }
            
        }
        else if (Etype.type == BoundaryKind::Symmetry) {
            for (int j = 0; j < u[0].size(); ++j) {
                int last = u[0][0].size() - 1; // index of last element in the inner-most array
                if (u[0][j][last].getBoundaryType() == PointBoundary::E) {
                    int index = u[0][j][last].getUniqueNumber() + indexCorrection;
                    RHScurrent[index] = -u[0][j][last].getValue() + u[0][j][0].getValue();
                }
            }
        }

        // North boundary
        int lastY = u[0].size() - 1; // index of the last element in the middle array (Y dimension)
        const Boundary& Ntype = boundaryContainer.getBoundary(PointBoundary::N);
        if (Ntype.type == BoundaryKind::FixedValue) {
            
                for (int i = 0; i < u[0][0].size(); ++i) {
                    if (u[0][lastY][i].getBoundaryType() == PointBoundary::N) {
                        int index = u[0][lastY][i].getUniqueNumber() + indexCorrection;
                        RHScurrent[index] = 0;
                    }
                }
            
        }
        else if (Ntype.type == BoundaryKind::Convective) {
            
                for (int i = 0; i < u[0][0].size(); ++i) {
                    if (u[0][lastY][i].getBoundaryType() == PointBoundary::N) {
                        int index = u[0][lastY][i].getUniqueNumber() + indexCorrection;
                        double prevValue = u[0][lastY][i].getPrevValue();
                        double prevPrevValue = u[0][lastY][i].getPrevPrevValue();
                        double value = u[0][lastY][i].getValue();
                        double dy1 = (u[0][lastY][i].getY() - u[0][lastY-1][i].getY());
                        double dy2 = (u[0][lastY][i].getY() - u[0][lastY-2][i].getY());
                        double dy3 = (u[0][lastY-1][i].getY() - u[0][lastY-2][i].getY());
                        double a = -(dy1  + dy2 ) / (dy1 * dy2 );
                        double b = dy2 / (dy1 * dy3);
                        double c = -dy1 / (dy2 * dy3);
                        double dudy = (a * value + b * u[0][lastY - 1][i].getValue() + c * u[0][lastY - 2][i].getValue());
                        RHScurrent[index] = (-3.0 * value + 4.0 * prevValue - prevPrevValue)  / (2.0 * dt) + dudy;
                    }
                }
        }
        // South boundary
        const Boundary& Stype = boundaryContainer.getBoundary(PointBoundary::S);
        if (Stype.type == BoundaryKind::FixedValue) {

                for (int i = 0; i < u[0][0].size(); ++i) {
                    if (u[0][0][i].getBoundaryType() == PointBoundary::S) {
                        int index = u[0][0][i].getUniqueNumber() + indexCorrection;
                        RHScurrent[index] = 0;
                    }
                }
            
        }
        else if (Stype.type == BoundaryKind::Convective) {
            
                for (int i = 0; i < u[0][0].size(); ++i) {
                    if (u[0][0][i].getBoundaryType() == PointBoundary::S) {
                        int index = u[0][0][i].getUniqueNumber() + indexCorrection;
                        double prevValue = u[0][0][i].getPrevValue();
                        double prevPrevValue = u[0][0][i].getPrevPrevValue();
                        double value = u[0][0][i].getValue();
                        double dy1 = (u[0][0][i].getY() - u[0][1][i].getY());
                        double dy2 = (u[0][0][i].getY() - u[0][2][i].getY());
                        double dy3 = (u[0][1][i].getY() - u[0][2][i].getY());
                        double a = (dy1 + dy2) / (dy1 * dy2);
                        double b = -dy2 / (dy1 * dy3);
                        double c = dy1 / (dy2 * dy3);
                        double dudy = (a*value + b*u[0][1][i].getValue() + c * u[0][2][i].getValue());
                        RHScurrent[index] = (-3.0 * value + 4.0 * prevValue - prevPrevValue) / (2.0 * dt) + dudy;
                    }
                }

        }
    }

}


// Second order upwind flux calculation
std::vector<double> RHSu::NonLinearSOU(const std::vector<std::vector<std::vector<Point>>>& u,
    const std::vector<std::vector<std::vector<Point>>>& v,
    const std::vector<std::vector<std::vector<Point>>>& w, bool is3D) {
    size_t depth = u.size();
    size_t height = u[0].size();
    size_t width = u[0][0].size();
    bool is2D = (depth <= 1);

    std::vector<double> result(is2D ? height * width : depth * height * width, 0.0);
    double Rightface, Leftface, Upface, Downface, Topface, Bottomface;
    if (is2D) {
        // 2D case
        for (size_t j = 1; j < height - 2; ++j) {
            for (size_t i = 1; i < width - 2; ++i) {
                
                int index = u[0][j][i].getUniqueNumber();
                double dx = u[0][j][i].getdx();
                double dy = u[0][j][i].getdy();
                bool nonBoundary = u[0][j][i].getBoundaryType() == PointBoundary::Interior && u[0][j][i - 1].getBoundaryType() == PointBoundary::Interior && u[0][j][i + 1].getBoundaryType() == PointBoundary::Interior
                    && u[0][j - 1][i].getBoundaryType() == PointBoundary::Interior && u[0][j + 1][i].getBoundaryType() == PointBoundary::Interior;
                if (nonBoundary) {


                    Rightface = calculateFluxSideU(0, j, i, u, v, w, is3D, 'r');
                    Leftface = calculateFluxSideU(0, j, i, u, v, w, is3D, 'l');
                    Upface = calculateFluxSideU(0, j, i, u, v, w, is3D, 'u');
                    Downface = calculateFluxSideU(0, j, i, u, v, w, is3D, 'd');
                    //std::cout << "inside " << i << " " << j << std::endl;
                }
                else {
                    if (u[0][j][i].getValue() >= 0.0) {
                        Rightface = u[0][j][i].getValue();
                    }
                    else {
                        Rightface = u[0][j][i + 1].getValue();
                    }
                    if (u[0][j][i - 1].getValue() >= 0.0) {
                        Leftface = u[0][j][i - 1].getValue();
                    }
                    else {
                        Leftface = u[0][j][i].getValue();
                    }
                    if (0.5 * (v[0][j][i].getValue() + v[0][j][i + 1].getValue()) >= 0.0) {
                        Upface = u[0][j][i].getValue();
                    }
                    else {
                        Upface = u[0][j + 1][i].getValue();
                    }
                    if (0.5 * (v[0][j - 1][i].getValue() + v[0][j - 1][i + 1].getValue()) >= 0.0) {
                        Downface = u[0][j - 1][i].getValue();
                    }
                    else {
                        Downface = u[0][j][i].getValue();
                    }
                }
                
                result[index] = -((pow(Rightface, 2) - pow(Leftface, 2)) * dy +
                    (0.5 * (v[0][j][i].getValue() + v[0][j][i + 1].getValue()) * Upface -
                        0.5 * (v[0][j - 1][i].getValue() + v[0][j - 1][i + 1].getValue()) * Downface) * dx);

            }
           //std::cout << "otside the for " << std::endl;
        }
    }
    else {
        // 3D case
        for (size_t k = 1; k < depth - 2; ++k) {
            for (size_t j = 1; j < height - 2; ++j) {
                for (size_t i = 1; i < width - 2; ++i) {

                    int index = u[k][j][i].getUniqueNumber();
                    double dx = u[k][j][i].getdx();
                    double dy = u[k][j][i].getdy();
                    double dz = u[k][j][i].getdz();

                    bool nonBoundary = u[k][j][i].getBoundaryType() == PointBoundary::Interior && u[k][j][i - 1].getBoundaryType() == PointBoundary::Interior && u[k][j][i + 1].getBoundaryType() == PointBoundary::Interior
                        && u[k][j - 1][i].getBoundaryType() == PointBoundary::Interior && u[k][j + 1][i].getBoundaryType() == PointBoundary::Interior && u[k - 1][j][i].getBoundaryType() == PointBoundary::Interior
                        && u[k + 1][j][i].getBoundaryType() == PointBoundary::Interior;
                    if (nonBoundary) {
                        Rightface = calculateFluxSideU(k, j, i, u, v, w, is3D, 'r');
                        Leftface = calculateFluxSideU(k, j, i, u, v, w, is3D, 'l');
                        Upface = calculateFluxSideU(k, j, i, u, v, w, is3D, 'u');
                        Downface = calculateFluxSideU(k, j, i, u, v, w, is3D, 'd');
                        Topface = calculateFluxSideU(k, j, i, u, v, w, is3D, 'f');
                        Bottomface = calculateFluxSideU(k, j, i, u, v, w, is3D, 'b');

                    }
                    else {
                        if (u[k][j][i].getValue() >= 0.0) {
                            Rightface = u[k][j][i].getValue();
                        }
                        else {
                            Rightface = u[k][j][i + 1].getValue();
                        }
                        if (u[k][j][i - 1].getValue() >= 0.0) {
                            Leftface = u[k][j][i - 1].getValue();
                        }
                        else {
                            Leftface = u[k][j][i].getValue();
                        }
                        if (0.5 * (v[k][j][i].getValue() + v[k][j][i + 1].getValue()) >= 0.0) {
                            Upface = u[k][j][i].getValue();
                        }
                        else {
                            Upface = u[k][j + 1][i].getValue();
                        }
                        if (0.5 * (v[k][j - 1][i].getValue() + v[k][j - 1][i + 1].getValue()) >= 0.0) {
                            Downface = u[k][j - 1][i].getValue();
                        }
                        else {
                            Downface = u[k][j][i].getValue();
                        }
                        if (0.5 * (w[k][j][i].getValue() + w[k][j][i + 1].getValue()) >= 0.0) {
                            Topface = u[k][j][i].getValue();
                        }
                        else {
                            Topface = u[k + 1][j][i].getValue();
                        }
                    }
                    result[index] = -((pow(Rightface, 2) - pow(Leftface, 2)) * dz * dy +
                        (0.5 * (v[k][j][i].getValue() + v[k][j][i + 1].getValue()) * Upface -
                            0.5 * (v[k][j - 1][i].getValue() + v[k][j - 1][i + 1].getValue()) * Downface) * dz * dx +
                        (0.5 * (w[k][j][i].getValue() + w[k][j][i + 1].getValue()) * Topface -
                            0.5 * (w[k - 1][j][i].getValue() + w[k - 1][j][i + 1].getValue()) * Bottomface) * dy * dx);
                }
            }
        }
    }
    return result;
}


std::vector<double> RHSu::NonLinearSOV(const std::vector<std::vector<std::vector<Point>>>& u,
    const std::vector<std::vector<std::vector<Point>>>& v,
    const std::vector<std::vector<std::vector<Point>>>& w, bool is3D) {
    size_t depth = v.size();
    size_t height = v[0].size();
    size_t width = v[0][0].size();

    bool is2D = (depth <= 1);

    std::vector<double> result(is2D ? height * width : depth * height * width, 0.0);
    double Rightface, Leftface, Upface, Downface, Topface, Bottomface;
    if (is2D) {

        // 2D case
        for (size_t j = 1; j < height - 2; ++j) {
            for (size_t i = 1; i < width - 2; ++i) {
                int index = v[0][j][i].getUniqueNumber();
                double dx = v[0][j][i].getdx();
                double dy = v[0][j][i].getdy();

                bool nonBoundary = v[0][j][i].getBoundaryType() == PointBoundary::Interior && v[0][j][i - 1].getBoundaryType() == PointBoundary::Interior && v[0][j][i + 1].getBoundaryType() == PointBoundary::Interior
                    && v[0][j - 1][i].getBoundaryType() == PointBoundary::Interior && v[0][j + 1][i].getBoundaryType() == PointBoundary::Interior;
                if (nonBoundary) {

                    Rightface = calculateFluxSideV(0, j, i, u, v, w, is3D, 'r');
                    Leftface = calculateFluxSideV(0, j, i, u, v, w, is3D, 'l');
                    Upface = calculateFluxSideV(0, j, i, u, v, w, is3D, 'u');
                    Downface = calculateFluxSideV(0, j, i, u, v, w, is3D, 'd');

                }
                else {

                    if (0.5 * (u[0][j + 1][i].getValue() + u[0][j][i].getValue()) >= 0.0) {
                        Rightface = v[0][j][i].getValue();
                    }
                    else {
                        Rightface = v[0][j][i + 1].getValue();
                    }
                    if (0.5 * (u[0][j + 1][i - 1].getValue() + u[0][j][i - 1].getValue()) >= 0.0) {
                        Leftface = v[0][j][i - 1].getValue();
                    }
                    else {
                        Leftface = v[0][j][i].getValue();
                    }

                    if (v[0][j][i].getValue() >= 0.0) {
                        Upface = v[0][j][i].getValue();
                    }
                    else {
                        Upface = v[0][j + 1][i].getValue();
                    }
                    if (v[0][j - 1][i].getValue() >= 0.0) {
                        Downface = v[0][j - 1][i].getValue();
                    }
                    else {
                        Downface = v[0][j][i].getValue();
                    }
                }
                result[index] = -((pow(Upface, 2) - pow(Downface, 2)) * dx +
                    (0.5 * (u[0][j + 1][i].getValue() + u[0][j][i].getValue()) * Rightface -
                        (0.5 * (u[0][j + 1][i - 1].getValue() + u[0][j][i - 1].getValue()) * Leftface)) * dy);
            }
        }
    }
    else {
        // 3D case
        for (size_t k = 1; k < depth - 2; ++k) {
            for (size_t j = 1; j < height - 2; ++j) {
                for (size_t i = 1; i < width - 2; ++i) {
                    int index = v[k][j][i].getUniqueNumber();
                    double dx = v[k][j][i].getdx();
                    double dy = v[k][j][i].getdy();
                    double dz = v[k][j][i].getdz();


                    bool nonBoundary = v[k][j][i].getBoundaryType() == PointBoundary::Interior && v[k][j][i - 1].getBoundaryType() == PointBoundary::Interior && v[k][j][i + 1].getBoundaryType() == PointBoundary::Interior
                        && v[k][j - 1][i].getBoundaryType() == PointBoundary::Interior && v[k][j + 1][i].getBoundaryType() == PointBoundary::Interior
                        && v[k - 1][j][i].getBoundaryType() == PointBoundary::Interior;
                    if (nonBoundary) {

                        Rightface = calculateFluxSideV(k, j, i, u, v, w, is3D, 'r');
                        Leftface = calculateFluxSideV(k, j, i, u, v, w, is3D, 'l');
                        Upface = calculateFluxSideV(k, j, i, u, v, w, is3D, 'u');
                        Downface = calculateFluxSideV(k, j, i, u, v, w, is3D, 'd');
                        Topface = calculateFluxSideV(k, j, i, u, v, w, is3D, 'f');
                        Bottomface = calculateFluxSideV(k, j, i, u, v, w, is3D, 'b');


                    }
                    else {
                        if (0.5 * (u[k][j + 1][i].getValue() + u[k][j][i].getValue()) >= 0.0) {
                            Rightface = v[k][j][i].getValue();
                        }
                        else {
                            Rightface = v[k][j][i + 1].getValue();
                        }
                        if (0.5 * (u[k][j + 1][i - 1].getValue() + u[k][j][i - 1].getValue()) >= 0.0) {
                            Leftface = v[k][j][i - 1].getValue();
                        }
                        else {
                            Leftface = v[k][j][i].getValue();
                        }

                        if (v[k][j][i].getValue() >= 0.0) {
                            Upface = v[k][j][i].getValue();
                        }
                        else {
                            Upface = v[k][j + 1][i].getValue();
                        }
                        if (v[k][j - 1][i].getValue() >= 0.0) {
                            Downface = v[k][j - 1][i].getValue();
                        }
                        else {
                            Downface = v[k][j][i].getValue();
                        }

                        if (0.5 * (w[k][j][i].getValue() + w[k][j + 1][i].getValue()) >= 0.0) {
                            Topface = v[k][j][i].getValue();
                        }
                        else {
                            Topface = v[k + 1][j][i].getValue();
                        }
                        if (0.5 * (w[k - 1][j][i].getValue() + w[k - 1][j + 1][i].getValue()) >= 0.0) {
                            Bottomface = v[k - 1][j][i].getValue();
                        }
                        else {
                            Bottomface = v[k][j][i].getValue();
                        }
                    }
                    result[index] = -((pow(Upface, 2) - pow(Downface, 2)) * dz * dx +
                        (0.5 * (u[k][j + 1][i].getValue() + u[k][j][i].getValue()) * Rightface -
                            (0.5 * (u[k][j + 1][i - 1].getValue() + u[k][j][i - 1].getValue()) * Leftface)) * dz * dy +
                        (0.5 * (w[k][j + 1][i].getValue() + w[k][j][i].getValue()) * Topface -
                            0.5 * (w[k - 1][j + 1][i].getValue() + w[k - 1][j][i + 1].getValue()) * Bottomface) * dx * dy);
                }
            }
        }
    }
    return result;
}


double RHSu::calculateFluxSideU(int k, int j, int i, const std::vector<std::vector<std::vector<Point>>>& u, const std::vector<std::vector<std::vector<Point>>>& v, const std::vector<std::vector<std::vector<Point>>>& w, bool is3D, char side) {
    std::vector<bool>signs;
    double face_value;

    switch (side)
    {
    case 'r':

        if (!is3D) k = 0;
        if (u[k][j][i].getValue() >= 0.0) {
            double step = std::abs((u[k][j][i].getX() - u[k][j][i - 1].getX()));
            double downWindStep = std::abs((u[k][j][i + 1].getX() - u[k][j][i].getX()));
            double modification = downWindStep / step;
            face_value = (2.0 + modification) / (2.0) * u[k][j][i].getValue() - modification / 2.0 * u[k][j][i - 1].getValue();
        }
        else {
            double step = std::abs((u[k][j][i + 1].getX() - u[k][j][i + 2].getX()));
            double downWindStep = std::abs((u[k][j][i].getX() - u[k][j][i + 1].getX()));
            double modification = downWindStep / step; // = std::abs((u[k][j][i].getX() + u[k][j][i + 1].getX() - 2 * u[k][j][i + 2].getX()));
            face_value = (2.0 + modification) / (2.0)  * u[k][j][i + 1].getValue() - modification / 2.0 * u[k][j][i + 2].getValue();
        }
        break;
    case 'l':
        if (!is3D) k = 0;
        if (u[k][j][i - 1].getValue() >= 0.0) {
            double step = std::abs((u[k][j][i - 1].getX() - u[k][j][i - 2].getX()));
            double downWindStep = std::abs((u[k][j][i].getX() - u[k][j][i - 1].getX()));
            double modification = downWindStep / step; // = std::abs((u[k][j][i].getX() + u[k][j][i - 1].getX() - 2 * u[k][j][i - 2].getX()));
            face_value = (2.0 + modification) / (2.0)  * u[k][j][i - 1].getValue() - modification / 2.0 * u[k][j][i - 2].getValue();
        }
        else {
            double step = std::abs((u[k][j][i].getX() - u[k][j][i + 1].getX()));
            double downWindStep = std::abs((u[k][j][i - 1].getX() - u[k][j][i].getX()));
            double modification = downWindStep / step; // = std::abs((u[k][j][i - 1].getX() + u[k][j][i].getX() - 2 * u[k][j][i + 1].getX()));
            face_value = (2.0 + modification) / (2.0)  * u[k][j][i].getValue() - modification / 2.0 * u[k][j][i + 1].getValue();
        }

        break;
    case'u':
        if (!is3D) k = 0;
        if (0.5 * (v[k][j][i].getValue() + v[k][j][i + 1].getValue()) >= 0.0) {
            double step = std::abs((u[k][j][i].getY() - u[k][j - 1][i].getY()));
            double downWindStep = std::abs((u[k][j + 1][i].getY() - u[k][j][i].getY()));
            double modification = downWindStep / step; // = std::abs((u[k][j + 1][i].getY() + u[k][j][i].getY() - 2 * u[k][j - 1][i].getY()));
            face_value = (2.0 + modification) / (2.0)  * u[k][j][i].getValue() - modification / 2.0 * u[k][j - 1][i].getValue();
        }
        else {
            double step = std::abs((u[k][j + 1][i].getY() - u[k][j + 2][i].getY()));
            double downWindStep = std::abs((u[k][j][i].getY() - u[k][j + 1][i].getY()));
            double modification = downWindStep / step; // = std::abs((u[k][j][i].getY() + u[k][j + 1][i].getY() - 2 * u[k][j + 2][i].getY()));
            face_value = (2.0 + modification) / (2.0)  * u[k][j + 1][i].getValue() - modification / 2.0 * u[k][j + 2][i].getValue();
        }
        break;
    case'd':
        if (!is3D) k = 0;
        if (0.5 * (v[k][j - 1][i].getValue() + v[k][j - 1][i + 1].getValue()) >= 0.0) {
            double step = std::abs((u[k][j - 1][i].getY() - u[k][j - 2][i].getY()));
            double downWindStep = std::abs((u[k][j][i].getY() - u[k][j - 1][i].getY()));
            double modification = downWindStep / step; // = std::abs((u[k][j][i].getY() + u[k][j - 1][i].getY() - 2 * u[k][j - 2][i].getY()));
            face_value = (2.0 + modification) / (2.0)  * u[k][j - 1][i].getValue() - modification / 2.0 * u[k][j - 2][i].getValue();
        }
        else {
            double step = std::abs((u[k][j][i].getY() - u[k][j + 1][i].getY()));
            double downWindStep = std::abs((u[k][j - 1][i].getY() - u[k][j][i].getY()));
            double modification = downWindStep / step; // = std::abs((u[k][j - 1][i].getY() + u[k][j][i].getY() - 2 * u[k][j + 1][i].getY()));
            face_value = (2.0 + modification) / (2.0)  * u[k][j][i].getValue() - modification / 2.0 * u[k][j + 1][i].getValue();
        }
        break;

    case'f':
        if (is3D) {
            if (0.5 * (w[k][j][i].getValue() + w[k][j][i + 1].getValue()) >= 0.0) {
                double step = std::abs((u[k][j][i].getZ() - u[k - 1][j][i].getZ()));
                double downWindStep = std::abs((u[k + 1][j][i].getZ() - u[k][j][i].getZ()));
                double modification = downWindStep / step; // = std::abs((u[k + 1][j][i].getZ() + u[k][j][i].getZ() - 2 * u[k - 1][j][i].getZ()));
                face_value = (2.0 + modification) / (2.0)  * u[k][j][i].getValue() - modification / 2.0 * u[k - 1][j][i].getValue();
            }
            else {
                double step = std::abs((u[k + 1][j][i].getZ() - u[k + 2][j][i].getZ()));
                double downWindStep = std::abs((u[k][j][i].getZ() - u[k + 1][j][i].getZ()));
                double modification = downWindStep / step; // = std::abs((u[k][j][i].getZ() + u[k + 1][j][i].getZ() - 2 * u[k + 2][j][i].getZ()));
                face_value = (2.0 + modification) / (2.0)  * u[k + 1][j][i].getValue() - modification / 2.0 * u[k + 2][j][i].getValue();
            }
        }
        else { //2D
            face_value = 0.0;
        }
        break;
    case'b':
        if (is3D) {
            if (0.5 * (w[k - 1][j][i].getValue() + w[k - 1][j][i + 1].getValue()) >= 0.0) {
                double step = std::abs((u[k - 1][j][i].getZ() - u[k - 2][j][i].getZ()));
                double downWindStep = std::abs((u[k][j][i].getZ() - u[k - 1][j][i].getZ()));
                double modification = downWindStep / step; // = std::abs((u[k][j][i].getZ() + u[k - 1][j][i].getZ() - 2 * u[k - 2][j][i].getZ()));
                face_value = (2.0 + modification) / (2.0)  * u[k - 1][j][i].getValue() - modification / 2.0 * u[k - 2][j][i].getValue();
            }
            else {
                double step = std::abs((u[k][j][i].getZ() - u[k + 1][j][i].getZ()));
                double downWindStep = std::abs((u[k - 1][j][i].getZ() - u[k][j][i].getZ()));
                double modification = downWindStep / step; // = std::abs((u[k - 1][j][i].getZ() + u[k][j][i].getZ() - 2 * u[k + 1][j][i].getZ()));
                face_value = (2.0 + modification) / (2.0)  * u[k][j][i].getValue() - modification / 2.0 * u[k + 1][j][i].getValue();
            }
        }
        else {
            face_value = 0.0;
        }
        break;
    default:
        break;
    }
    return face_value;
}

double RHSu::calculateFluxSideV(int k, int j, int i, const std::vector<std::vector<std::vector<Point>>>& u, const std::vector<std::vector<std::vector<Point>>>& v, const std::vector<std::vector<std::vector<Point>>>& w, bool is3D, char side) {
    std::vector<bool>signs;
    double face_value;
    switch (side)
    {
    case 'r':
        if (!is3D) k = 0;
            if (0.5 * (u[k][j + 1][i].getValue() + u[k][j][i].getValue()) >= 0.0) {
                double step = std::abs((v[k][j][i].getX() - v[k][j][i - 1].getX()));
                double downWindStep = std::abs((v[k][j][i + 1].getX() - v[k][j][i].getX()));
                double modification = downWindStep / step; // = std::abs((v[k][j][i + 1].getX() + v[k][j][i].getX() - 2 * u[k][j][i - 1].getX()));
                face_value = (2.0 + modification) / (2.0) * v[k][j][i].getValue() - modification / 2.0 * v[0][j][i - 1].getValue();
            }
            else {
                double step = std::abs((v[k][j][i + 1].getX() - v[k][j][i + 2].getX()));
                double downWindStep = std::abs((v[k][j][i].getX() - v[k][j][i + 1].getX()));
                double modification = downWindStep / step; // = std::abs((v[k][j][i].getX() + v[k][j][i + 1].getX() - 2 * v[k][j][i + 2].getX()));
                face_value = (2.0 + modification) / (2.0) * v[k][j][i + 1].getValue() - modification / 2.0 * v[k][j][i + 2].getValue();
            }
       
        break;
    case 'l':
        if (!is3D) k = 0;
            if (0.5 * (u[k][j + 1][i - 1].getValue() + u[k][j][i - 1].getValue()) >= 0.0) {
                double step = std::abs((v[k][j][i - 1].getX() - v[k][j][i - 2].getX()));
                double downWindStep = std::abs((v[k][j][i].getX() - v[k][j][i - 1].getX()));
                double modification = downWindStep / step; // = std::abs((v[k][j][i].getX() + v[k][j][i - 1].getX() - 2 * v[k][j][i - 2].getX()));
                face_value = (2.0 + modification) / (2.0) * v[k][j][i - 1].getValue() - modification / 2.0 * v[k][j][i - 2].getValue();
            }
            else {
                double step = std::abs((u[k][j][i].getX() - u[k][j][i + 1].getX()));
                double downWindStep = std::abs((v[k][j][i - 1].getX() - v[k][j][i].getX()));
                double modification = downWindStep / step; // = std::abs((v[k][j][i - 1].getX() + v[k][j][i].getX() - 2 * v[k][j][i + 1].getX()));
                face_value = (2.0 + modification) / (2.0) * v[k][j][i].getValue() - modification / 2.0 * v[k][j][i + 1].getValue();
            }
        
        break;
    case'u':
        if (!is3D) k = 0;
            if (v[k][j][i].getValue() >= 0.0) {
                double step = std::abs((v[k][j][i].getY() - v[k][j - 1][i].getY()));
                double downWindStep = std::abs((v[k][j + 1][i].getY() - v[k][j][i].getY()));
                double modification = downWindStep / step; // = std::abs((v[k][j + 1][i].getY() + v[k][j][i].getY() - 2 * v[k][j - 1][i].getY()));
                face_value = (2.0 + modification) / (2.0) * v[k][j][i].getValue() - modification / 2.0 * v[k][j - 1][i].getValue();
            }
            else {
                double step = std::abs((v[k][j + 1][i].getY() - v[k][j + 2][i].getY()));
                double downWindStep = std::abs((v[k][j][i].getY() - v[k][j + 1][i].getY()));
                double modification = downWindStep / step; // = std::abs((v[k][j][i].getY() + v[k][j + 1][i].getY() - 2 * v[k][j + 2][i].getY()));
                face_value = (2.0 + modification) / (2.0) * v[k][j + 1][i].getValue() - modification / 2.0 * v[k][j + 2][i].getValue();
            }
        break;

    case'd':
        if (!is3D) k = 0;
            if (v[k][j - 1][i].getValue() >= 0.0) {
                double step = std::abs((v[k][j - 1][i].getY() - v[k][j - 2][i].getY()));
                double downWindStep = std::abs((v[k][j][i].getY() - v[k][j - 1][i].getY()));
                double modification = downWindStep / step; // = std::abs((v[k][j][i].getY() + v[k][j - 1][i].getY() - 2 * v[k][j - 2][i].getY()));
                face_value = (2.0 + modification) / (2.0) * v[k][j - 1][i].getValue() - modification / 2.0 * v[k][j - 2][i].getValue();
            }
            else {
                double step = std::abs((v[k][j][i].getY() - v[k][j + 1][i].getY()));
                double downWindStep = std::abs((v[k][j - 1][i].getY() - v[k][j][i].getY()));
                double modification = downWindStep / step; // = std::abs((v[k][j - 1][i].getY() + v[k][j][i].getY() - 2 * v[k][j + 1][i].getY()));
                face_value = (2.0 + modification) / (2.0) * v[k][j][i].getValue() - modification / 2.0 * v[k][j + 1][i].getValue();
            }
        break;

    case'f':
        if (is3D) {
            if (0.5 * (w[k][j][i].getValue() + w[k][j + 1][i].getValue()) >= 0.0) {
                double step = std::abs((v[k - 1][j][i].getZ() - v[k - 2][j][i].getZ()));
                double downWindStep = std::abs((v[k][j][i].getZ() - v[k - 1][j][i].getZ()));
                double modification = downWindStep / step; // = std::abs((v[k][j][i].getZ() + v[k - 1][j][i].getZ() - 2 * v[k - 2][j][i].getZ()));
                face_value = (2.0 + modification) / (2.0) * v[k][j][i].getValue() - modification / 2.0 * v[k - 1][j][i].getValue();
            }
            else {
                double step = std::abs((v[k + 1][j][i].getZ() - v[k + 2][j][i].getZ()));
                double downWindStep = std::abs((v[k][j][i].getZ() - v[k + 1][j][i].getZ()));
                double modification = downWindStep / step; // = std::abs((v[k][j][i].getZ() + v[k + 1][j][i].getZ() - 2 * v[k + 2][j][i].getZ()));
                face_value = (2.0 + modification) / (2.0) * v[k + 1][j][i].getValue() - modification / 2.0 * v[k + 2][j][i].getValue();
            }
        }
        else { //2D
            face_value = 0.0;
        }
        break;
    case'b':
        if (is3D) {
            if (0.5 * (w[k - 1][j][i].getValue() + w[k - 1][j + 1][i].getValue()) >= 0.0) {
                double step = std::abs((v[k - 1][j][i].getZ() - v[k - 2][j][i].getZ()));
                double downWindStep = std::abs((v[k][j][i].getZ() - v[k - 1][j][i].getZ()));
                double modification = downWindStep / step; // = std::abs((v[k][j][i].getZ() + v[k - 1][j][i].getZ() - 2 * v[k - 2][j][i].getZ()));
                face_value = (2.0 + modification) / (2.0) * v[k - 1][j][i].getValue() - modification / 2.0 * v[k - 2][j][i].getValue();
            }
            else {
                double step = std::abs((v[k][j][i].getZ() - v[k + 1][j][i].getZ()));
                double downWindStep = std::abs((v[k - 1][j][i].getZ() - v[k][j][i].getZ()));
                double modification = downWindStep / step; // = std::abs((v[k - 1][j][i].getZ() + v[k][j][i].getZ() - 2 * v[k + 1][j][i].getZ()));
                face_value = (2.0 + modification) / (2.0) * v[k][j][i].getValue() - modification / 2.0 * v[k + 1][j][i].getValue();
            }
        }
        else {
            face_value = 0.0;
        }
        break;
    default:
        break;

    }
    return face_value;
}
