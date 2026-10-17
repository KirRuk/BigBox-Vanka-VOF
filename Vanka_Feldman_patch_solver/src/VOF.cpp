#include "VOF.h"
#include "PerformanceProfiler.h"
#include <tuple>
#include "Initialize.h"
#include "Point.h"
#include <iostream>
#include <cmath>
#include "BoundaryContainer.h"
#include <iomanip>
#include <vector>
#include <mkl.h>
#include <numeric>
#include <omp.h>
#include "output.h"
#include <chrono>
#include <stdexcept>
#include <cstring>
#include <cfloat>
#include <filesystem>
#include <sstream>
#include <string>
#include <algorithm>

namespace {
bool hasShape(
    const ScalarField3D& field,
    size_t depth,
    size_t height,
    size_t width)
{
    if (field.size() != depth) {
        return false;
    }
    for (const auto& layer : field) {
        if (layer.size() != height) {
            return false;
        }
        for (const auto& row : layer) {
            if (row.size() != width) {
                return false;
            }
        }
    }
    return true;
}

void resetField(
    ScalarField3D& field,
    size_t depth,
    size_t height,
    size_t width,
    double value)
{
    if (!hasShape(field, depth, height, width)) {
        ScalarField3D replacement(
            depth,
            std::vector<std::vector<double>>(
                height,
                std::vector<double>(width, value)));
        field.swap(replacement);
        return;
    }

#pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < static_cast<int>(depth); ++k) {
        for (int j = 0; j < static_cast<int>(height); ++j) {
            std::fill(field[k][j].begin(), field[k][j].end(), value);
        }
    }
}

// The same reconstructed value must be used by the two cells sharing a face.
// This is the existing normalized-variable VOF reconstruction expressed as a
// face value instead of as two independently assembled matrix rows.
double reconstructedFaceValue(double farLeft, double left, double right,
                              double farRight, double velocity) {
    const double upstream = velocity >= 0.0 ? left : right;
    const double farUpstream = velocity >= 0.0 ? farLeft : farRight;
    const double downstream = velocity >= 0.0 ? right : left;
    const double span = downstream - farUpstream;
    const double normalized = span == 0.0 ? 1.0 : (upstream - farUpstream) / span;
    if (normalized > 0.0 && normalized <= 2.0 / 13.0)
        return 3.0 * upstream - 2.0 * farUpstream;
    if (normalized > 2.0 / 13.0 && normalized <= 0.8)
        return (5.0 * upstream - farUpstream) / 6.0 + downstream / 3.0;
    if (normalized > 0.8 && normalized <= 1.0)
        return downstream;
    return upstream;
}
}

// Natural cubic spline on uniform grid xi = 0,1,...,n-1
struct UniformCubicSpline {
    std::vector<double> a; // a[j] = y_j
    std::vector<double> b, c, d; // size n-1 for b,c,d; size n for a

    explicit UniformCubicSpline(const std::vector<double>& y) {
        const size_t n = y.size();
        if (n < 2) {
            throw std::runtime_error("UniformCubicSpline: need at least 2 points");
        }
        a = y;
        b.resize(n - 1);
        c.resize(n);
        d.resize(n - 1);

        // h = 1 everywhere
        std::vector<double> alpha(n - 1);
        for (size_t i = 1; i < n - 1; ++i) {
            alpha[i] = 3.0 * (a[i + 1] - 2.0 * a[i] + a[i - 1]); // 3*(Delta�a_i)
        }

        std::vector<double> l(n), mu(n), z(n);
        l[0] = 1.0;
        mu[0] = 0.0;
        z[0] = 0.0;

        for (size_t i = 1; i < n - 1; ++i) {
            l[i] = 4.0 - mu[i - 1];        // 2*(1+1) - 1*mu_{i-1}
            mu[i] = 1.0 / l[i];
            z[i] = (alpha[i] - z[i - 1]) / l[i];
        }

        l[n - 1] = 1.0;
        z[n - 1] = 0.0;
        c[n - 1] = 0.0;

        for (int j = static_cast<int>(n) - 2; j >= 0; --j) {
            c[j] = z[j] - mu[j] * c[j + 1];
            b[j] = (a[j + 1] - a[j]) - (2.0 * c[j] + c[j + 1]) / 3.0;
            d[j] = (c[j + 1] - c[j]) / 3.0;
        }
    }

    // evaluate at xi in [0, n-1]
    double eval(double xi) const {
        size_t n = a.size();
        if (xi <= 0.0) {
            xi = 0.0;
        }
        else if (xi >= static_cast<double>(n - 1)) {
            xi = static_cast<double>(n - 1);
        }

        size_t j = static_cast<size_t>(std::floor(xi));
        if (j >= n - 1) j = n - 2;

        double dx = xi - static_cast<double>(j);
        return a[j] + b[j] * dx + c[j] * dx * dx + d[j] * dx * dx * dx;
    }
};

// bisection for spline(xi) - 0.5 on [xi0, xi1]
static double findRootOnInterval(const UniformCubicSpline& spl,
    double xi0, double xi1,
    double tol = 1e-10, int maxIter = 60)
{
    double f0 = spl.eval(xi0) - 0.5;
    double f1 = spl.eval(xi1) - 0.5;

    if (std::abs(f0) < tol) return xi0;
    if (std::abs(f1) < tol) return xi1;
    if (f0 * f1 > 0.0) {
        throw std::runtime_error("findRootOnInterval: no sign change");
    }

    double a = xi0, b = xi1;
    for (int it = 0; it < maxIter; ++it) {
        double m = 0.5 * (a + b);
        double fm = spl.eval(m) - 0.5;
        if (std::abs(fm) < tol || 0.5 * (b - a) < tol) {
            return m;
        }
        if (f0 * fm <= 0.0) {
            b = m;
            f1 = fm;
        }
        else {
            a = m;
            f0 = fm;
        }
    }
    return 0.5 * (a + b);
}

const std::vector<std::vector<std::vector<Point>>>& VOF::getPhi()
{
    return phi;
}

[[nodiscard]] std::vector<double> VOF::getPhiV() const {
    const bool is3D = (phi.size() > 2);
    const int nz = is3D ? int(phi.size()) : 1;
    const int ny = int(phi[0].size());
    const int nx = int(phi[0][0].size());

    std::vector<double> phiV(RHS.size());
#pragma omp parallel for collapse(3) schedule(static)
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const int Eq = phi[k][j][i].getUniqueNumber();
                phiV[Eq] = phi[k][j][i].getValue(); // use operator[] for speed
            }
    return phiV; // caller can bind to const if they like: const auto x = getPhiV();
}


const std::vector<std::vector<std::vector<double>>>& VOF::getKappa()
{
    return kappa;
}

const std::vector<std::vector<std::vector<double>>>& VOF::getPhiS()
{
    return phiS;
}
VOF::VOF(const std::string& filename, double ds, double Lx, double Ly, double Lz, double phase)//With check and unequal grid suitable
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(1); // Exiting if the file fails to open.
    }

    std::string line;
    // Skip the header.
    std::getline(file, line);
    if (!std::getline(file, line)) {
        std::cerr << "Error: Unable to read first line or file is empty." << std::endl;
        return;
    }

    // Extract I, J (and optionally K) values from the second line.
   // std::getline(file, line);
    int I, J, K; // Default K = 1 for 2D.

    bool is3D = extractIJK(line, I, J, K);

    int nx = I;
    int ny = J;
    int nz = K;
    int unique_number = 0; // Counter for unique numbers.

    // Check the consistency of nx, ny, nz values with the expected grid dimensions.
    int expected_nx = static_cast<int>(std::round(Lx / ds));
    int expected_ny = static_cast<int>(std::round(Ly / ds));
    int expected_nz = static_cast<int>(std::round(Lz / ds));
    if (I != expected_nx || J != expected_ny || (nz > 1 && K != expected_nz)) {
        std::cerr << "Mismatch in expected grid dimensions for scalar field in file: " << filename << std::endl;
        std::cerr << "Expected I=" << expected_nx << ", J=" << expected_ny;
        if (nz > 1) std::cerr << ", K=" << expected_nz;
        std::cerr << " but got I=" << I << ", J=" << J;
        if (nz > 1) std::cerr << ", K=" << K;
        std::cerr << std::endl;
        exit(1);
    }

    // Variables to hold the previous line's x, y (and z if 3D) values.
    std::getline(file, line);
    std::stringstream ss(line);
    double x, y, z, phi_val;  // Default z to 0.0 for 2D cases

    if (nz > 1) {
        ss >> x >> y >> z >> phi_val;  // Read x, y, z, and phi for 3D
    }
    else {
        ss >> x >> y >> phi_val;       // Read x, y, and phi for 2D
    }
    double prev_x = 0, prev_y = 0, prev_z = 0, y_down = 0, z_down = 0;

    // Loop through the file's data lines and populate the vector.

    for (int k = 0; k < (nz > 1 ? nz : 1); ++k) { // Handle both 2D and 3D cases
        std::vector<std::vector<Point>> zLayer;
        double dz = ds;
        for (int j = 0; j < ny; ++j) {
            std::vector<Point> yLayer;
            double dy = ds;
            for (int i = 0; i < nx; ++i) {
                if (std::getline(file, line)) {
                    std::stringstream ss(line);
                    double dx, x_next, y_next, z_next, phi_val_next;
                    std::stringstream ss_new(line);
                    if (nz > 1) {
                        ss_new >> x_next >> y_next >> z_next >> phi_val_next;
                    }
                    else {
                        ss_new >> x_next >> y_next >> phi_val_next;
                        z_next = 0.0; 
                    }

                    // Calculate dx and dy using the average distance.

                    dx = ds;
                    Point point(i, j, k, unique_number++, dx, dy);
                    point.setX(x);
                    point.setY(y);
                    point.setZ(z); 

                    // Set face boundary conditions (applies to both 2D and 3D cases)
                    if (i == 0 || i == 1) {
                        if (i == 0 && j == 0) {}
                        else if ((i == 0 && j == ny - 1)) {}
                        else if ((nz > 1 && i == 0 && k == 0)) {}
                        else if ((nz > 1 && i == 0 && k == nz - 1)) {}
                        else point.setBoundaryType(PointBoundary::W); // West face
                    }
                    if (i == nx - 1 || i == nx - 2) {
                        if (i == nx - 1 && j == 0) {} //T
                        else if (i == nx - 1 && j == ny - 1) {}//Y
                        else if (nz > 1 && i == nx - 1 && k == 0) {}
                        else if (nz > 1 && i == nx - 1 && k == nz - 1) {}
                        point.setBoundaryType(PointBoundary::E); // East face
                    }
                    if (j == 0 || j == 1) {
                        if (i == 0 || i == nx - 1) {}
                        else if (nz > 1 && j == 0 && k == 0) {}
                        else if (nz > 1 && j == 0 && k == nz - 1) {}
                        else point.setBoundaryType(PointBoundary::S); // South face
                    }
                    if (j == ny - 1 || j == ny - 2)
                    {
                        if (i == 0 || i == nx - 1) {}
                        else if (nz > 1 && j == ny - 1 && k == 0) {}
                        else if (nz > 1 && j == ny - 1 && k == nz - 1) {}
                        else point.setBoundaryType(PointBoundary::N); // North face
                    }
                    if (nz > 1) {
                        if (k == 0 || k == 1) {
                            point.setBoundaryType(PointBoundary::B); // Bottom face
                        }
                        if (k == nz - 1 || k == nz - 2) {
                            point.setBoundaryType(PointBoundary::F); // Top face
                        }
                    }

                    // Handle edges (only in 3D cases)
                    // These are edges where two boundary faces meet
                    if (i == 0 && j == 0) point.setBoundaryType(PointBoundary::D); // West-South edge
                    if (i == 0 && j == ny - 1) point.setBoundaryType(PointBoundary::U); // West-North edge
                    if (i == nx - 1 && j == 0) point.setBoundaryType(PointBoundary::T); // East-South edge
                    if (i == nx - 1 && j == ny - 1) point.setBoundaryType(PointBoundary::Y); // East-North edge
                    if (nz > 1) {
                        if (i == 0 && k == 0) point.setBoundaryType(PointBoundary::A); // West-Bottom edge
                        if (i == 0 && k == nz - 1) point.setBoundaryType(PointBoundary::C); // West-Top edge
                        if (i == nx - 1 && k == 0) point.setBoundaryType(PointBoundary::G); // East-Bottom edge
                        if (i == nx - 1 && k == nz - 1) point.setBoundaryType(PointBoundary::H); // East-Top edge
                        if (j == 0 && k == 0) point.setBoundaryType(PointBoundary::R); // South-Bottom edge
                        if (j == 0 && k == nz - 1) point.setBoundaryType(PointBoundary::V); // South-Top edge
                        if (j == ny - 1 && k == 0) point.setBoundaryType(PointBoundary::Z); // North-Bottom edge
                        if (j == ny - 1 && k == nz - 1) point.setBoundaryType(PointBoundary::X); // North-Top edge
                    }

                    // Handle corners (only in 3D case)
                    // These are points where three boundary faces meet
                    if (nz > 1) {
                        if (i == 0 && j == 0 && k == 0) point.setBoundaryType(PointBoundary::I); // West-South-Bottom corner
                        if (i == 0 && j == ny - 1 && k == 0) point.setBoundaryType(PointBoundary::J); // West-North-Bottom corner
                        if (i == 0 && j == 0 && k == nz - 1) point.setBoundaryType(PointBoundary::K); // West-South-Top corner
                        if (i == 0 && j == ny - 1 && k == nz - 1) point.setBoundaryType(PointBoundary::L); // West-North-Top corner
                        if (i == nx - 1 && j == 0 && k == 0) point.setBoundaryType(PointBoundary::M); // East-South-Bottom corner
                        if (i == nx - 1 && j == ny - 1 && k == 0) point.setBoundaryType(PointBoundary::O); // East-North-Bottom corner
                        if (i == nx - 1 && j == 0 && k == nz - 1) point.setBoundaryType(PointBoundary::P); // East-South-Top corner
                        if (i == nx - 1 && j == ny - 1 && k == nz - 1) point.setBoundaryType(PointBoundary::Q); // East-North-Top corner
                    }

                    point.setValue(phi_val);
                    point.setPrevValue(phi_val);
                    point.setPrevPrevValue(phi_val);
                    yLayer.push_back(point); // Add point to yLayer.
                    phi_val = phi_val_next;
                    prev_x = x; prev_y = y; prev_z = z;
                    x = x_next; y = y_next;  z = z_next;


                }
                else {
                    double dx;
                    dx = ds;


                    Point point(i, j, k, unique_number++, dx, dy);
                    point.setX(x);
                    point.setY(y);
                    point.setZ(z);  // Setting Z value to 0, assuming 2D.

                    // Set face boundary conditions (applies to both 2D and 3D cases)
                    if (i == 0 || i == 1) {
                        if (i == 0 && j == 0) {}
                        else if ((i == 0 && j == ny - 1)) {}
                        else if ((nz > 1 && i == 0 && k == 0)) {}
                        else if ((nz > 1 && i == 0 && k == nz - 1)) {}
                        else point.setBoundaryType(PointBoundary::W); // West face
                    }
                    if (i == nx - 1 || i == nx - 2) {
                        if (i == nx - 1 && j == 0) {} //T
                        else if (i == nx - 1 && j == ny - 1) {}//Y
                        else if (nz > 1 && i == nx - 1 && k == 0) {}
                        else if (nz > 1 && i == nx - 1 && k == nz - 1) {}
                        point.setBoundaryType(PointBoundary::E); // East face
                    }
                    if (j == 0 || j == 1) {
                        if (i == 0 || i == nx - 1) {}
                        else if (nz > 1 && j == 0 && k == 0) {}
                        else if (nz > 1 && j == 0 && k == nz - 1) {}
                        else point.setBoundaryType(PointBoundary::S); // South face
                    }
                    if (j == ny - 1 || j == ny - 2)
                    {
                        if (i == 0 || i == nx - 1) {}
                        else if (nz > 1 && j == ny - 1 && k == 0) {}
                        else if (nz > 1 && j == ny - 1 && k == nz - 1) {}
                        else point.setBoundaryType(PointBoundary::N); // North face
                    }
                    if (nz > 1) {
                        if (k == 0 || k == 1) {
                            point.setBoundaryType(PointBoundary::B); // Bottom face
                        }
                        if (k == nz - 1 || k == nz - 2) {
                            point.setBoundaryType(PointBoundary::F); // Top face
                        }
                    }

                    // Handle edges (only in 3D cases)
                    // These are edges where two boundary faces meet
                    if (i == 0 && j == 0) point.setBoundaryType(PointBoundary::D); // West-South edge
                    if (i == 0 && j == ny - 1) point.setBoundaryType(PointBoundary::U); // West-North edge
                    if (i == nx - 1 && j == 0) point.setBoundaryType(PointBoundary::T); // East-South edge
                    if (i == nx - 1 && j == ny - 1) point.setBoundaryType(PointBoundary::Y); // East-North edge
                    if (nz > 1) {
                        if (i == 0 && k == 0) point.setBoundaryType(PointBoundary::A); // West-Bottom edge
                        if (i == 0 && k == nz - 1) point.setBoundaryType(PointBoundary::C); // West-Top edge
                        if (i == nx - 1 && k == 0) point.setBoundaryType(PointBoundary::G); // East-Bottom edge
                        if (i == nx - 1 && k == nz - 1) point.setBoundaryType(PointBoundary::H); // East-Top edge
                        if (j == 0 && k == 0) point.setBoundaryType(PointBoundary::R); // South-Bottom edge
                        if (j == 0 && k == nz - 1) point.setBoundaryType(PointBoundary::V); // South-Top edge
                        if (j == ny - 1 && k == 0) point.setBoundaryType(PointBoundary::Z); // North-Bottom edge
                        if (j == ny - 1 && k == nz - 1) point.setBoundaryType(PointBoundary::X); // North-Top edge
                    }

                    // Handle corners (only in 3D case)
                    // These are points where three boundary faces meet
                    if (nz > 1) {
                        if (i == 0 && j == 0 && k == 0) point.setBoundaryType(PointBoundary::I); // West-South-Bottom corner
                        if (i == 0 && j == ny - 1 && k == 0) point.setBoundaryType(PointBoundary::J); // West-North-Bottom corner
                        if (i == 0 && j == 0 && k == nz - 1) point.setBoundaryType(PointBoundary::K); // West-South-Top corner
                        if (i == 0 && j == ny - 1 && k == nz - 1) point.setBoundaryType(PointBoundary::L); // West-North-Top corner
                        if (i == nx - 1 && j == 0 && k == 0) point.setBoundaryType(PointBoundary::M); // East-South-Bottom corner
                        if (i == nx - 1 && j == ny - 1 && k == 0) point.setBoundaryType(PointBoundary::O); // East-North-Bottom corner
                        if (i == nx - 1 && j == 0 && k == nz - 1) point.setBoundaryType(PointBoundary::P); // East-South-Top corner
                        if (i == nx - 1 && j == ny - 1 && k == nz - 1) point.setBoundaryType(PointBoundary::Q); // East-North-Top corner
                    }


                    point.setValue(phi_val);         // Set current value.
                    point.setPrevValue(phi_val);     // Assuming current value is also the previous value.
                    point.setPrevPrevValue(phi_val); // Assuming current value is the previous-previous value.

                    yLayer.push_back(point);  // Add point to yLayer.
                    prev_x = x; prev_y = y; prev_z = z;
                }
            }
            zLayer.push_back(yLayer); //Add line to zLayer
        }
        phi.push_back(zLayer); // Add yLayer to the 3D vector.
    }
    file.close();  // Close the file after reading. 

    std::vector<std::vector<std::vector<double>>>tempKappa(nz, std::vector<std::vector<double>>(ny, std::vector<double>(nx, 0.0)));
    //   this->kappa = tempKappa;
       //  this->phi_intial = tempKappa;
   //    calculateKappa();
       // calculateCOM();
        //  calculateKappa_new();

    size_t Neq = phi.back().back().back().getUniqueNumber() + 1;
    std::vector<double> RHS_t(Neq, 0.0);
    this->RHS = RHS_t;

    // Compute initial mass


    initialMass = 0.0;
    for (int k = 0; k < nz; k++) {

        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                double value = std::abs((1.0 - phase) - phi[k][j][i].getValue());  // Volume fraction
                double dx = phi[k][j][i].getdx();
                double dy = phi[k][j][i].getdy();
                double dz = phi[k][j][i].getdz();

                initialMass += value * dx * dy * (nz > 1 ? dz : 1);

            }
        }
    }
    this->calculateKappa(2,1E-10,1,2.0/3.0);

}

VOF::VOF(const std::string& filename, double ds)//to update
{

}

void VOF::setPeriodicX(bool periodic) {
    if (periodicX == periodic) return;
    periodicX = periodic;
    calculateKappa(2, 1E-10, 1, 2.0 / 3.0);
}

void VOF::updateRHS(const BoundaryContainer& BCu, const std::vector<std::vector<std::vector<Point>>> u, double dt)
{

    bool is3D = (phi.size() > 2);
    int nz = is3D ? phi.size() : 1;
    int ny = phi[0].size();
    int nx = phi[0][0].size();

    double term_x = 0.0;
    double term_y = 0.0;
    std::vector<std::vector<std::vector<double>>> dphiX(nz, std::vector<std::vector<double>>(ny, std::vector<double>(nx, 0.0)));
    std::vector<std::vector<std::vector<double>>> dphiY(nz, std::vector<std::vector<double>>(ny, std::vector<double>(nx, 0.0)));

#pragma omp parallel for collapse(3) schedule(dynamic) // // Parallel update for phi

    for (int k = 0; k < nz; k++) {

        for (int j = 0; j < ny; j++) {

            for (int i = 0; i < nx; i++)
            {

                int Eq = phi[is3D ? k : 0][j][i].getUniqueNumber();
                double dx = phi[is3D ? k : 0][j][i].getdx();
                double dy = phi[is3D ? k : 0][j][i].getdy();
                double cellVolume = dx * dy * (is3D ? phi[k][j][i].getdz() : 1.0);

                this->RHS[Eq] = (2.0 * phi[(is3D ? k : 0)][j][i].getValue() - 0.50 * phi[(is3D ? k : 0)][j][i].getPrevValue()) * cellVolume / (dt);
                //this->RHS[Eq] = 1.0 * phi[(is3D ? k : 0)][j][i].getPrevValue() * cellVolume / dt;
            }
        }
    }
}

void VOF::timeStep(std::vector<double> phiVetor)
{
    bool is3D = (phi.size() > 2);
    int nx = phi[0][0].size();
    int ny = phi[0].size();
    int nz = is3D ? phi.size() : 1;

#pragma omp parallel for collapse(3) schedule(dynamic) // // Parallel update for phi

    for (int k = 0; k < nz; k++) {

        for (int j = 0; j < ny; j++) {

            for (int i = 0; i < nx; i++)
            {

                int Eq = phi[k][j][i].getUniqueNumber();
                double prevValue = phi[k][j][i].getPrevValue();
                double Value = phi[k][j][i].getValue();

                phi[k][j][i].setPrevPrevValue(prevValue);
                phi[k][j][i].setPrevValue(Value);
                phi[k][j][i].setValue(phiVetor.at(Eq));
            }
        }
    }
    //this->calculateKappa();
}

void VOF::updateLHS(BoundaryContainer& boundaryContainerU, const std::vector<std::vector<std::vector<Point>>>& u, const std::vector<std::vector<std::vector<Point>>>& v, const std::vector<std::vector<std::vector<Point>>>& w, double dt)
{
    bool is3D = (phi.size() > 2);
    int nx = phi[0][0].size();
    int ny = phi[0].size();
    int nz = is3D ? phi.size() : 1;
    double valueAtCenter, valueDown, valueUp, valueRight, valueLeft, valuefacedown, valuefaceup;
    LHS.setn(phi.back().back().back().getUniqueNumber() + 1);
    std::vector<MKL_INT> row_ptr_t(LHS.getSize() + 1, 0); //temperary vector for row conter
    std::vector<MKL_INT> col_indices_t; std::vector<double> values_t; //another temporary vectors

    for (int k = 0; k < nz; k++) {

        for (int j = 0; j < ny; j++) {

            for (int i = 0; i < nx; i++)
            {

                int Eq = phi[k][j][i].getUniqueNumber();
                char Case = boundaryLetter(phi[k][j][i].getBoundaryType());
                double cellVolume = phi[k][j][i].getdx() * phi[k][j][i].getdy() * (is3D ? phi[k][j][i].getdz() : 1.0);
                int count;
                std::vector<int> cols; std::vector<double> vals; //values and columns per equation
                std::vector<double> allVals;
                std::vector<double> Vector_new;
                std::vector<bool>  signs;
                switch (Case)
                {
                case 'E':
                    if (nz > 1) {
                        if (i == nx - 1) {
                            cols = { Eq - nx * ny,Eq - nx, Eq - 1, Eq,   Eq + nx ,Eq + nx * ny };
                            allVals.clear();
                            allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                            vals = { -allVals.at(21),-allVals.at(13),-allVals.at(5),-allVals.at(6) + allVals.at(9) - allVals.at(14) + allVals.at(17) - allVals.at(22) + 1.5 * cellVolume / dt,allVals.at(10),allVals.at(18) };
                            count = cols.size();
                        }
                        else {
                            cols = { Eq - nx * ny,Eq - nx, Eq - 1, Eq, Eq + 1, Eq + nx,Eq + nx * ny };
                            allVals.clear();
                            allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                            vals = { -allVals.at(21),-allVals.at(13),-allVals.at(5),-allVals.at(6) + allVals.at(1) + allVals.at(9) - allVals.at(14) + allVals.at(17) - allVals.at(22) + 1.5 * cellVolume / dt,allVals.at(2),allVals.at(10),allVals.at(18) };
                            count = cols.size();
                        }

                    }
                    else {
                        if (boundaryContainerU.getBoundary(PointBoundary::E).type == BoundaryKind::Symmetry) {
                            //  cols = { Eq - nx, Eq - 1, Eq,   Eq + nx };
                            if (i == nx - 1) {
                                cols = { Eq - 2 * nx, Eq - nx, Eq - 2, Eq - 1, Eq, Eq - nx + 1, Eq - nx + 2, Eq + nx, Eq + 2 * nx };
                            }
                            else {
                                cols = { Eq - 2 * nx, Eq - nx, Eq - 2, Eq - 1, Eq, Eq + 1, Eq - nx + 1, Eq + nx, Eq + 2 * nx };
                            }

                            allVals.clear();
                            //   allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                             //  if (boundaryContainerU.getBoundary(PointBoundary::E).type == BoundaryKind::Symmetry && i == nx-1) {
                            Vector_new = symetry(k, j, i, u, v, w, is3D, 'r');
                            allVals.insert(allVals.begin(), Vector_new.begin(), Vector_new.end());
                            Vector_new = symetry(k, j, i, u, v, w, is3D, 'l');
                            allVals.insert(allVals.end(), Vector_new.begin(), Vector_new.end());
                            Vector_new = symetry(k, j, i, u, v, w, is3D, 'u');
                            allVals.insert(allVals.end(), Vector_new.begin(), Vector_new.end());
                            Vector_new = symetry(k, j, i, u, v, w, is3D, 'd');
                            allVals.insert(allVals.end(), Vector_new.begin(), Vector_new.end());
                            //  }
                            valueAtCenter = allVals.at(1) - allVals.at(6) + allVals.at(9) - allVals.at(14) + 1.5 * cellVolume / dt;
                            valueDown = allVals.at(8) - allVals.at(13);
                            valueLeft = allVals.at(0) - allVals.at(5);
                            valueRight = allVals.at(2) - allVals.at(7);
                            valueUp = allVals.at(10) - allVals.at(15);

                            vals = { -allVals.at(12),valueDown,-allVals.at(4),valueLeft, valueAtCenter,valueRight,allVals.at(3),valueUp,allVals.at(11) };
                            count = cols.size();

                            //  vals = { -allVals.at(13),-allVals.at(5),-allVals.at(6) + allVals.at(1) + allVals.at(9) - allVals.at(14) + 1.5 * cellVolume / dt,allVals.at(10) };//numbering knows only Mukesh

                          //    if (allVals.at(2) != 0.0 && i != (nx - 1)) { //This condition should be modified for non-zero penetration of the fluid
                         //         ((boundaryContainerU.getBoundary(PointBoundary::E).type == BoundaryKind::Symmetry) && i == nx-1) ? cols.insert(cols.begin() + 1, Eq - nx - 1) : cols.insert(cols.end() - 1, Eq + 1);

                          //        vals.insert(vals.end() - 1, allVals.at(2));
                           //   }

                        }
                        else if (boundaryContainerU.getBoundary(PointBoundary::E).type == BoundaryKind::Convective) {
                            if (i == nx - 1) {
                                cols = { Eq - nx, Eq - 1, Eq,   Eq + nx };
                                allVals.clear();
                                allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                                vals = { -allVals.at(13),-allVals.at(5),-allVals.at(6) + allVals.at(5) + allVals.at(9) - allVals.at(14) + 1.5 * cellVolume / dt,allVals.at(10) };
                                count = cols.size();
                            }
                            else {
                                cols = { Eq - nx, Eq - 1, Eq, Eq + 1, Eq + nx };
                                allVals.clear();
                                allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                                vals = { -allVals.at(13),-allVals.at(5),-allVals.at(6) + allVals.at(1) + allVals.at(9) - allVals.at(14) + 1.5 * cellVolume / dt,allVals.at(2),allVals.at(10) };
                                count = cols.size();
                            }
                        }
                        else {
                            if (i == nx - 1) {
                                cols = { Eq - nx, Eq - 1, Eq,   Eq + nx };
                                allVals.clear();
                                allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                                vals = { -allVals.at(13),-allVals.at(5),-allVals.at(6) + allVals.at(1) + allVals.at(9) - allVals.at(14) + 1.5 * cellVolume / dt,allVals.at(10) };
                                count = cols.size();
                            }
                            else {
                                cols = { Eq - nx, Eq - 1, Eq, Eq + 1, Eq + nx };
                                allVals.clear();
                                allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                                vals = { -allVals.at(13),-allVals.at(5),-allVals.at(6) + allVals.at(1) + allVals.at(9) - allVals.at(14) + 1.5 * cellVolume / dt,allVals.at(2),allVals.at(10) };
                                count = cols.size();
                            }

                        }
                    }
                    break;
                case 'W':
                    if (nz > 1) {
                        if (i == 0) {
                            cols = { Eq - nx * ny,Eq - nx,  Eq, Eq + 1,  Eq + nx,Eq + nx * ny };
                            allVals.clear();
                            allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                            vals = { -allVals.at(21),-allVals.at(13), allVals.at(1) + allVals.at(9) - allVals.at(14) + allVals.at(17) - allVals.at(22) + 1.5 * cellVolume / dt,allVals.at(2),allVals.at(10),allVals.at(18) };
                            count = cols.size();
                        }
                        else {
                            cols = { Eq - nx * ny,Eq - nx,Eq - 1,  Eq, Eq + 1,  Eq + nx ,Eq + nx * ny };
                            allVals.clear();
                            allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                            vals = { -allVals.at(21),-allVals.at(13),-allVals.at(5),-allVals.at(6) + allVals.at(1) + allVals.at(9) - allVals.at(14) + allVals.at(17) - allVals.at(22) + 1.5 * cellVolume / dt,allVals.at(2),allVals.at(10),allVals.at(18) };
                            count = cols.size();
                        }
                    }
                    else {

                        if (boundaryContainerU.getBoundary(PointBoundary::W).type == BoundaryKind::Symmetry) {

                            if (i == 1) {
                                cols = { Eq - 2 * nx, Eq - nx, Eq + nx - 2, Eq + nx - 1, Eq, Eq + 1, Eq + 2, Eq + nx, Eq + 2 * nx };
                            }
                            else {
                                cols = { Eq - 2 * nx, Eq - nx, Eq + nx - 1, Eq - 1, Eq, Eq + 1, Eq + 2, Eq + nx, Eq + 2 * nx };
                            }

                            allVals.clear();
                            //   
                          //      if (boundaryContainerU.getBoundary(PointBoundary::W).type == BoundaryKind::Symmetry && i == 0) {
                            Vector_new = symetry(k, j, i, u, v, w, is3D, 'r');
                            allVals.insert(allVals.begin(), Vector_new.begin(), Vector_new.end());
                            Vector_new = symetry(k, j, i, u, v, w, is3D, 'l');
                            allVals.insert(allVals.end(), Vector_new.begin(), Vector_new.end());
                            Vector_new = symetry(k, j, i, u, v, w, is3D, 'u');
                            allVals.insert(allVals.end(), Vector_new.begin(), Vector_new.end());
                            Vector_new = symetry(k, j, i, u, v, w, is3D, 'd');
                            allVals.insert(allVals.end(), Vector_new.begin(), Vector_new.end());
                            //   }
                            valueAtCenter = allVals.at(1) - allVals.at(6) + allVals.at(9) - allVals.at(14) + 1.5 * cellVolume / dt;
                            valueDown = allVals.at(8) - allVals.at(13);
                            valueLeft = allVals.at(0) - allVals.at(5);
                            valueRight = allVals.at(2) - allVals.at(7);
                            valueUp = allVals.at(10) - allVals.at(15);

                            vals = { -allVals.at(12),valueDown,-allVals.at(4),valueLeft, valueAtCenter,valueRight,allVals.at(3),valueUp,allVals.at(11) };
                            count = cols.size();
                        }
                        else {
                            if (i == 0) {
                                cols = { Eq - nx,  Eq, Eq + 1,  Eq + nx };
                                allVals.clear();
                                allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                                vals = { -allVals.at(13), -allVals.at(6) + allVals.at(1) + allVals.at(9) - allVals.at(14) + 1.5 * cellVolume / dt,allVals.at(2),allVals.at(10) };
                                count = cols.size();
                            }
                            else {
                                cols = { Eq - nx,Eq - 1,  Eq, Eq + 1,  Eq + nx };
                                allVals.clear();
                                allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                                vals = { -allVals.at(13),-allVals.at(5),-allVals.at(6) + allVals.at(1) + allVals.at(9) - allVals.at(14) + 1.5 * cellVolume / dt,allVals.at(2),allVals.at(10) };
                                count = cols.size();
                            }

                        }
                    }
                    //     if (allVals.at(5) != 0.0 && i != 0) { //This condition should be modified for non-zero penetration of the fluid
                    //         ((boundaryContainerU.getBoundary(PointBoundary::W).type == BoundaryKind::Symmetry) && i == 0)? cols.insert(cols.begin() + 1, Eq + nx - 1) : cols.insert(cols.begin() + 1, Eq - 1);
                    //         vals.insert(vals.begin() + 1, -allVals.at(5));
                    //     }

                    break;
                case 'N':
                    if (nz > 1) {
                        if (j == ny - 1) {
                            cols = { Eq - nx * ny ,Eq - nx, Eq - 1, Eq, Eq + 1,Eq + nx * ny };
                            allVals.clear();
                            allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                            vals = { -allVals.at(21) ,-allVals.at(13),-allVals.at(5),-allVals.at(6) + allVals.at(1) - allVals.at(14) + allVals.at(17) - allVals.at(22) + 1.5 * cellVolume / dt,allVals.at(2),allVals.at(18) };
                            count = cols.size();
                        }
                        else {
                            cols = { Eq - nx * ny ,Eq - nx, Eq - 1, Eq, Eq + 1,Eq + nx,Eq + nx * ny };
                            allVals.clear();
                            allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                            vals = { -allVals.at(21) ,-allVals.at(13),-allVals.at(5),-allVals.at(6) + allVals.at(1) + allVals.at(9) - allVals.at(14) + allVals.at(17) - allVals.at(22) + 1.5 * cellVolume / dt,allVals.at(2),allVals.at(10),allVals.at(18) };
                            count = cols.size();
                        }
                    }
                    else {
                        cols = { Eq - nx, Eq - 1, Eq, Eq + 1 };
                        allVals.clear();
                        allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                        vals = { -allVals.at(13),-allVals.at(5),-allVals.at(6) + allVals.at(1) + allVals.at(9) - allVals.at(14) + 1.5 * cellVolume / dt,allVals.at(2) };//numbering knows only Mukesh
                        if (allVals.at(10) != 0.0 && (Eq + nx) < nx * ny * (nz == 0 ? 1 : nz)) { //This condition should be modified for non-zero penetration of the fluid
                            cols.insert(cols.end(), Eq + nx);
                            vals.insert(vals.end(), allVals.at(10));
                        }
                        count = cols.size();
                    }
                    break;
                case 'S':
                    if (nz > 1) {
                        if (j == 0) {
                            cols = { Eq - nx * ny,Eq - 1, Eq, Eq + 1,  Eq + nx,Eq + nx * ny };
                            allVals.clear();
                            allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                            vals = { -allVals.at(21),-allVals.at(5),-allVals.at(6) + allVals.at(1) + allVals.at(9) + allVals.at(17) - allVals.at(22) + 1.5 * cellVolume / dt,allVals.at(2), allVals.at(10),allVals.at(18) };
                            count = cols.size();
                        }
                        else {
                            cols = { Eq - nx * ny,Eq - nx,Eq - 1, Eq, Eq + 1,  Eq + nx,Eq + nx * ny };
                            allVals.clear();
                            allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                            vals = { -allVals.at(21),-allVals.at(13) ,-allVals.at(5),-allVals.at(6) + allVals.at(1) + allVals.at(9) - allVals.at(14) + allVals.at(17) - allVals.at(22) + 1.5 * cellVolume / dt,allVals.at(2), allVals.at(10),allVals.at(18) };
                            count = cols.size();
                        }

                    }
                    else {
                        cols = { Eq - 1, Eq, Eq + 1,  Eq + nx };
                        allVals.clear();
                        allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                        vals = { -allVals.at(5),-allVals.at(6) + allVals.at(1) + allVals.at(9) - allVals.at(14) + 1.5 * cellVolume / dt,allVals.at(2), allVals.at(10) };//numbering knows only Mukesh
                        if (allVals.at(13) != 0.0 && (Eq - nx) > 0) { //This condition should be modified for non-zero penetration of the fluid
                            cols.insert(cols.begin(), Eq - nx);
                            vals.insert(vals.begin(), -allVals.at(13));
                        }
                        count = cols.size();
                    }


                    break;
                case'B':
                    if (k == 0) {
                        cols = { Eq - nx,Eq - 1, Eq, Eq + 1,  Eq + nx,Eq + nx * ny };
                        allVals.clear();
                        allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                        vals = { -allVals.at(13),-allVals.at(5),-allVals.at(6) + allVals.at(1) + allVals.at(9) - allVals.at(14) + allVals.at(17) + 1.5 * cellVolume / dt,allVals.at(2), allVals.at(10),allVals.at(18) };
                        count = cols.size();
                    }
                    else {
                        cols = cols = { Eq - nx * ny, Eq - nx,Eq - 1, Eq, Eq + 1,  Eq + nx,Eq + nx * ny };
                        allVals.clear();
                        allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                        vals = { -allVals.at(21), -allVals.at(13),-allVals.at(5),-allVals.at(6) + allVals.at(1) + allVals.at(9) - allVals.at(14) + allVals.at(17) - allVals.at(22) + 1.5 * cellVolume / dt,allVals.at(2), allVals.at(10),allVals.at(18) };
                        count = cols.size();
                    }
                    break;
                case'F':
                    if (k == nz - 1) {
                        cols = { Eq - nx * ny,Eq - nx,Eq - 1, Eq, Eq + 1,  Eq + nx };
                        allVals.clear();
                        allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                        vals = { -allVals.at(21) ,-allVals.at(13),-allVals.at(5),-allVals.at(6) + allVals.at(1) + allVals.at(9) - allVals.at(14) - allVals.at(22) + 1.5 * cellVolume / dt,allVals.at(2), allVals.at(10) };
                        count = cols.size();
                    }
                    else {
                        cols = cols = { Eq - nx * ny, Eq - nx,Eq - 1, Eq, Eq + 1,  Eq + nx,Eq + nx * ny };
                        allVals.clear();
                        allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                        vals = { -allVals.at(21), -allVals.at(13),-allVals.at(5),-allVals.at(6) + allVals.at(1) + allVals.at(9) - allVals.at(14) + allVals.at(17) - allVals.at(22) + 1.5 * cellVolume / dt,allVals.at(2), allVals.at(10),allVals.at(18) };
                        count = cols.size();
                    }
                    break;
                    //courener cases
                case 'D'://lower left corner for 2D  // West-South edge for 3D (i == 0 && j == 0)
                    if (nz > 1) {
                        cols = { Eq - nx * ny, Eq, Eq + 1,  Eq + nx,Eq + nx * ny };
                        allVals.clear();
                        allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                        vals = { -allVals.at(21), allVals.at(1) + allVals.at(9) + allVals.at(17) - allVals.at(22) + 1.5 * cellVolume / dt,allVals.at(2),allVals.at(10),allVals.at(18) };
                        count = cols.size();
                    }
                    else {
                        cols = { Eq, Eq + 1,  Eq + nx };
                        allVals.clear();
                        allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                        vals = { allVals.at(1) + allVals.at(9) - allVals.at(14) + 1.5 * cellVolume / dt,allVals.at(2),allVals.at(10) };//numbering knows only Mukesh                               
                        count = cols.size();
                    }
                    break;
                case 'T'://lower right corner for 2D // East-South edge for 3D (i == nx - 1 && j == 0)
                    if (nz > 1) {
                        cols = { Eq - nx * ny,Eq - 1, Eq,   Eq + nx,Eq + nx * ny };
                        allVals.clear();
                        allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                        vals = { -allVals.at(21), -allVals.at(5),-allVals.at(6) + allVals.at(9) + allVals.at(17) - allVals.at(22) + 1.5 * cellVolume / dt,allVals.at(10),allVals.at(18) };
                        count = cols.size();

                    }
                    else {
                        if (boundaryContainerU.getBoundary(PointBoundary::E).type == BoundaryKind::Convective) {
                            cols = { Eq - 1, Eq,   Eq + nx };
                            allVals.clear();
                            allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                            vals = { -allVals.at(5),-allVals.at(6) + allVals.at(5) + allVals.at(9) - allVals.at(14) + 1.5 * cellVolume / dt,allVals.at(10) };
                            count = cols.size();
                        }
                        else {
                            cols = { Eq - 1, Eq,   Eq + nx };
                            allVals.clear();
                            allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                            vals = { -allVals.at(5),-allVals.at(6) + allVals.at(9) + 1.5 * cellVolume / dt,allVals.at(10) };//numbering knows only Mukesh
                            count = cols.size();
                        }
                    }
                    break;
                case 'U'://upper left corner for 2D // West-North edge for 3D (i == 0 && j == ny - 1)
                    if (nz > 1) {
                        cols = { Eq - nx * ny, Eq - nx,  Eq, Eq + 1,Eq + nx * ny };
                        allVals.clear();
                        allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                        vals = { -allVals.at(21),-allVals.at(13), allVals.at(1) - allVals.at(14) + allVals.at(17) - allVals.at(22) + 1.5 * cellVolume / dt,allVals.at(2),allVals.at(18) };
                        count = cols.size();
                    }
                    else {
                        cols = { Eq - nx,  Eq, Eq + 1 };
                        allVals.clear();
                        allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                        vals = { -allVals.at(13), allVals.at(1) - allVals.at(14) + 1.5 * cellVolume / dt,allVals.at(2) };
                        count = cols.size();
                    }

                    break;
                case 'Y'://upper right corner for 2D // East-North edge for 3D (i == nx - 1 && j == ny - 1)
                    if (nz > 1) {
                        cols = { Eq - nx * ny,Eq - nx, Eq - 1, Eq,Eq + nx * ny };
                        allVals.clear();
                        allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                        vals = { -allVals.at(21) ,-allVals.at(13),-allVals.at(5),-allVals.at(6) - allVals.at(14) + allVals.at(17) - allVals.at(22) + 1.5 * cellVolume / dt,  allVals.at(18) };//numbering knows only Mukesh
                        count = cols.size();
                    }
                    else {
                        if (boundaryContainerU.getBoundary(PointBoundary::E).type == BoundaryKind::Convective) {
                            cols = { Eq - nx, Eq - 1, Eq };
                            allVals.clear();
                            allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                            vals = { -allVals.at(13),-allVals.at(5),-allVals.at(6) + allVals.at(5) + allVals.at(9) - allVals.at(14) + 1.5 * cellVolume / dt };
                            count = cols.size();
                        }
                        else {
                            cols = { Eq - nx, Eq - 1, Eq };
                            allVals.clear();
                            allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);

                            vals = { -allVals.at(13),-allVals.at(5),-allVals.at(6) + allVals.at(9) - allVals.at(14) + 1.5 * cellVolume / dt };//numbering knows only Mukesh
                            count = cols.size();
                        }
                    }
                    break;
                case 'A':// West-Bottom edge this only for 3d (i == 0 && k == 0)
                    cols = { Eq - nx,Eq, Eq + 1, Eq + nx,Eq + nx * ny };
                    allVals.clear();
                    allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                    vals = { -allVals.at(13),allVals.at(1) + allVals.at(9) - allVals.at(14) + allVals.at(17) + 1.5 * cellVolume / dt,allVals.at(2), allVals.at(10),allVals.at(18) };//numbering knows only Mukesh
                    count = cols.size();
                    break;
                case 'C':// // West-Top edge this only for 3d (i == 0 && k == nz - 1)
                    cols = { Eq - nx * ny,Eq - nx,Eq, Eq + 1, Eq + nx };
                    allVals.clear();
                    allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                    vals = { -allVals.at(21), -allVals.at(13),allVals.at(1) + allVals.at(9) - allVals.at(14) - allVals.at(22) + 1.5 * cellVolume / dt,allVals.at(2), allVals.at(10) };//numbering knows only Mukesh
                    count = cols.size();
                    break;
                case 'G':// // East-Bottom edge  (i == nx - 1 && k == 0) 
                    cols = { Eq - nx,Eq - 1, Eq, Eq + nx,Eq + nx * ny };
                    allVals.clear();
                    allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                    vals = { -allVals.at(13), -allVals.at(5),-allVals.at(6) + allVals.at(9) - allVals.at(14) + allVals.at(17) + 1.5 * cellVolume / dt, allVals.at(10),allVals.at(18) };//numbering knows only Mukesh
                    count = cols.size();
                    break;
                case 'H'://  if (i == nx - 1 && k == nz - 1) point.setBoundaryType(PointBoundary::H); // East-Top edge
                    cols = { Eq - nx * ny, Eq - nx,Eq - 1, Eq, Eq + nx };
                    allVals.clear();
                    allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                    vals = { -allVals.at(21),-allVals.at(13), -allVals.at(5), -allVals.at(6) + allVals.at(9) - allVals.at(14) - allVals.at(22) + 1.5 * cellVolume / dt, allVals.at(10) };//numbering knows only Mukesh
                    count = cols.size();
                    break;
                case 'R'://  (j == 0 && k == 0) point.setBoundaryType(PointBoundary::R); // South-Bottom edge
                    cols = { Eq - 1, Eq,Eq + 1, Eq + nx,Eq + nx * ny };
                    allVals.clear();
                    allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                    vals = { -allVals.at(5),allVals.at(1) - allVals.at(6) + allVals.at(9) + allVals.at(17) + 1.5 * cellVolume / dt,allVals.at(2), allVals.at(10),allVals.at(18) };//numbering knows only Mukesh
                    count = cols.size();
                    break;
                case 'V'://   (j == 0 && k == nz - 1) point.setBoundaryType(PointBoundary::V); // South-Top edge
                    cols = { Eq - nx * ny,Eq - 1, Eq,Eq + 1, Eq + nx };
                    allVals.clear();
                    allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                    vals = { -allVals.at(21) ,-allVals.at(5),allVals.at(1) - allVals.at(6) + allVals.at(9) - allVals.at(22) + 1.5 * cellVolume / dt,allVals.at(2), allVals.at(10) };//numbering knows only Mukesh
                    count = cols.size();
                    break;
                case 'Z'://  (j == ny - 1 && k == 0) point.setBoundaryType(PointBoundary::Z); // North-Bottom edge
                    cols = { Eq - nx,Eq - 1, Eq,Eq + 1,Eq + nx * ny };
                    allVals.clear();
                    allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                    vals = { -allVals.at(13) ,-allVals.at(5),allVals.at(1) - allVals.at(6) - allVals.at(14) + allVals.at(17) + 1.5 * cellVolume / dt,allVals.at(2),allVals.at(18) };
                    count = cols.size();
                    break;
                case 'X'://  (j == ny - 1 && k == nz - 1) point.setBoundaryType(PointBoundary::X); // North-Top edge
                    cols = { Eq - nx * ny,Eq - nx,Eq - 1, Eq,Eq + 1 };
                    allVals.clear();
                    allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                    vals = { -allVals.at(21),-allVals.at(13) ,-allVals.at(5),allVals.at(1) - allVals.at(6) - allVals.at(14) - allVals.at(22) + 1.5 * cellVolume / dt,allVals.at(2) };
                    count = cols.size();
                    break;
                case 'I'://  (i == 0 && j == 0 && k == 0) point.setBoundaryType(PointBoundary::I); // West-South-Bottom corner
                    cols = { Eq,Eq + 1,Eq + nx,Eq + nx * ny };
                    allVals.clear();
                    allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                    vals = { allVals.at(1) + allVals.at(9) + allVals.at(17) + 1.5 * cellVolume / dt,allVals.at(2),allVals.at(10),allVals.at(18) };
                    count = cols.size();
                    break;
                case 'J'://  (i == 0 && j == ny - 1 && k == 0) point.setBoundaryType(PointBoundary::J); // West-North-Bottom corner
                    cols = { Eq - nx, Eq,Eq + 1,Eq + nx * ny };
                    allVals.clear();
                    allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                    vals = { -allVals.at(13) ,allVals.at(1) - allVals.at(14) + allVals.at(17) + 1.5 * cellVolume / dt,allVals.at(2),allVals.at(18) };
                    count = cols.size();
                    break;
                case 'K':// (i == 0 && j == 0 && k == nz - 1) point.setBoundaryType(PointBoundary::K); // West-South-Top corner
                    cols = { Eq - nx * ny,Eq,Eq + 1,Eq + nx };
                    allVals.clear();
                    allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                    vals = { -allVals.at(21),allVals.at(1) + allVals.at(9) - allVals.at(22) + 1.5 * cellVolume / dt,allVals.at(2),allVals.at(10) };
                    count = cols.size();
                    break;
                case 'L':// (i == 0 && j == ny - 1 && k == nz - 1) point.setBoundaryType(PointBoundary::L); // West-North-Top corner
                    cols = { Eq - nx * ny,Eq - nx,Eq,Eq + 1 };
                    allVals.clear();
                    allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                    vals = { -allVals.at(21),-allVals.at(13),allVals.at(1) - allVals.at(14) - allVals.at(22) + 1.5 * cellVolume / dt,allVals.at(2) };
                    count = cols.size();
                    break;
                case 'M': //i == nx - 1 && j == 0 && k == 0) point.setBoundaryType(PointBoundary::M); // East-South-Bottom corner
                    cols = { Eq - 1,Eq,Eq + nx,Eq + nx * ny };
                    allVals.clear();
                    allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                    vals = { -allVals.at(5) ,-allVals.at(6) + allVals.at(9) + allVals.at(17) + 1.5 * cellVolume / dt,allVals.at(10),allVals.at(18) };
                    count = cols.size();
                    break;
                case 'O': //(i == nx - 1 && j == ny - 1 && k == 0) point.setBoundaryType(PointBoundary::O); // East-North-Bottom corner
                    cols = { Eq - nx,Eq - 1,Eq,Eq + nx * ny };
                    allVals.clear();
                    allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                    vals = { -allVals.at(13),-allVals.at(5) ,-allVals.at(6) - allVals.at(14) + allVals.at(17) + 1.5 * cellVolume / dt,allVals.at(18) };
                    count = cols.size();
                    break;
                case 'P': //(i == nx - 1 && j == 0 && k == nz - 1) point.setBoundaryType(PointBoundary::P); // East-South-Top corner
                    cols = { Eq - nx * ny,Eq - 1,Eq,Eq + nx };
                    allVals.clear();
                    allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                    vals = { -allVals.at(21),-allVals.at(5) , -allVals.at(6) + allVals.at(9) - allVals.at(22) + 1.5 * cellVolume / dt,allVals.at(10) };
                    count = cols.size();
                    break;
                case 'Q':  // (i == nx - 1 && j == ny - 1 && k == nz - 1) point.setBoundaryType(PointBoundary::Q); // East-North-Top corner
                    cols = { Eq - nx * ny,Eq - nx,Eq - 1,Eq };
                    allVals.clear();
                    allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                    vals = { -allVals.at(21), -allVals.at(13) ,-allVals.at(5) ,-allVals.at(6) - allVals.at(14) - allVals.at(22) + 1.5 * cellVolume / dt };
                    count = cols.size();
                    break;
                default:

                    if (nz > 1) {
                        cols = { Eq - 2 * nx * ny,Eq - nx * ny, Eq - 2 * nx, Eq - nx, Eq - 2, Eq - 1, Eq, Eq + 1, Eq + 2, Eq + nx, Eq + 2 * nx,Eq + nx * ny,Eq + 2 * nx * ny }; // Higher Order Scheme
                    }
                    else {
                        cols = { Eq - 2 * nx, Eq - nx, Eq - 2, Eq - 1, Eq, Eq + 1, Eq + 2, Eq + nx, Eq + 2 * nx }; // Higher Order Scheme
                    }




                    allVals.clear();
                    //      allVals = calculateFlux(k, j, i, u, v, w, is3D);
                    //      

                    Vector_new = calculateFluxSide(k, j, i, u, v, w, is3D, 'r');
                    allVals.insert(allVals.begin(), Vector_new.begin(), Vector_new.end());
                    Vector_new = calculateFluxSide(k, j, i, u, v, w, is3D, 'l');
                    allVals.insert(allVals.end(), Vector_new.begin(), Vector_new.end());
                    Vector_new = calculateFluxSide(k, j, i, u, v, w, is3D, 'u');
                    allVals.insert(allVals.end(), Vector_new.begin(), Vector_new.end());
                    Vector_new = calculateFluxSide(k, j, i, u, v, w, is3D, 'd');
                    allVals.insert(allVals.end(), Vector_new.begin(), Vector_new.end());
                    if (nz > 1) {
                        Vector_new = calculateFluxSide(k, j, i, u, v, w, is3D, 'f');
                        allVals.insert(allVals.end(), Vector_new.begin(), Vector_new.end());
                        Vector_new = calculateFluxSide(k, j, i, u, v, w, is3D, 'b');
                        allVals.insert(allVals.end(), Vector_new.begin(), Vector_new.end());

                    }

                    valueAtCenter = allVals.at(1) - allVals.at(6) + allVals.at(9) - allVals.at(14) + 1.5 * cellVolume / dt;
                    valueAtCenter = valueAtCenter + (nz > 1 ? (allVals.at(17) - allVals.at(22)) : 0.0);
                    valueDown = allVals.at(8) - allVals.at(13);
                    valueLeft = allVals.at(0) - allVals.at(5);
                    valueRight = allVals.at(2) - allVals.at(7);
                    valueUp = allVals.at(10) - allVals.at(15);
                    if (nz > 1) {
                        valuefacedown = allVals.at(16) - allVals.at(21);
                        valuefaceup = allVals.at(18) - allVals.at(23);
                        vals = { -allVals.at(20), valuefacedown, -allVals.at(12),valueDown,-allVals.at(4),valueLeft, valueAtCenter,valueRight,allVals.at(3),valueUp,allVals.at(11), valuefaceup,allVals.at(19) };
                    }
                    else {
                        vals = { -allVals.at(12),valueDown,-allVals.at(4),valueLeft, valueAtCenter,valueRight,allVals.at(3),valueUp,allVals.at(11) };
                    }
                    count = cols.size();
                    break;
                }

                col_indices_t.insert(col_indices_t.begin() + row_ptr_t[Eq], cols.begin(), cols.end());
                values_t.insert(values_t.begin() + row_ptr_t[Eq], vals.begin(), vals.end());
                row_ptr_t[Eq + 1] = row_ptr_t[Eq] + count;
            }
        }
    }

    std::vector<MKL_INT> rows_end_t(row_ptr_t.begin() + 1, row_ptr_t.end());

    LHS.setCols(col_indices_t); LHS.setValues(values_t); LHS.setRows(row_ptr_t); LHS.setRowsEnd(rows_end_t);
    sparse_status_t status = mkl_sparse_d_create_csr(
        &LHS_mkl, SPARSE_INDEX_BASE_ZERO,
        LHS.getSize(), LHS.getSize(),
        const_cast<MKL_INT*>(LHS.getRowPtr().data()),
        const_cast<MKL_INT*>(LHS.getRowPtr_end().data()),
        const_cast<MKL_INT*>(LHS.getColIndices().data()),
        const_cast<double*>(LHS.getValues().data())
    );


}

void VOF::updateLHS(BoundaryContainer& boundaryContainerU, const std::vector<std::vector<std::vector<Point>>>& u, const std::vector<std::vector<std::vector<Point>>>& v, const std::vector<std::vector<std::vector<Point>>>& w)
{
    bool is3D = (phi.size() > 2);
    int nx = phi[0][0].size();
    int ny = phi[0].size();
    int nz = is3D ? phi.size() : 1;
    double valueAtCenter, valueDown, valueUp, valueRight, valueLeft, valuefacedown, valuefaceup;
    LHS.setn(phi.back().back().back().getUniqueNumber() + 1);
    std::vector<MKL_INT> row_ptr_t(LHS.getSize() + 1, 0); //temperary vector for row conter
    std::vector<MKL_INT> col_indices_t; std::vector<double> values_t; //another temporary vectors
    
    for (int k = 0; k < nz; k++) {

        for (int j = 0; j < ny; j++) {

            for (int i = 0; i < nx; i++)
            {

                int Eq = phi[k][j][i].getUniqueNumber();
                char Case = boundaryLetter(phi[k][j][i].getBoundaryType());
                double cellVolume = phi[k][j][i].getdx() * phi[k][j][i].getdy() * (is3D ? phi[k][j][i].getdz() : 1.0);
                int count;
                std::vector<int> cols; std::vector<double> vals; //values and columns per equation
                std::vector<double> allVals;
                std::vector<double> Vector_new;
                std::vector<bool>  signs;
                switch (Case)
                {
                case 'E':
                    if (nz > 1) {
                        if (i == nx - 1) {
                            cols = { Eq - nx * ny,Eq - nx, Eq - 1, Eq,   Eq + nx ,Eq + nx * ny };
                            allVals.clear();
                            allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                            vals = { -allVals.at(21),-allVals.at(13),-allVals.at(5),-allVals.at(6) + allVals.at(9) - allVals.at(14) + allVals.at(17) - allVals.at(22),allVals.at(10),allVals.at(18) };
                            count = cols.size();
                        }
                        else {
                            cols = { Eq - nx * ny,Eq - nx, Eq - 1, Eq, Eq + 1, Eq + nx,Eq + nx * ny };
                            allVals.clear();
                            allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                            vals = { -allVals.at(21),-allVals.at(13),-allVals.at(5),-allVals.at(6) + allVals.at(1) + allVals.at(9) - allVals.at(14) + allVals.at(17) - allVals.at(22),allVals.at(2),allVals.at(10),allVals.at(18) };
                            count = cols.size();
                        }

                    }
                    else {
                        if (boundaryContainerU.getBoundary(PointBoundary::E).type == BoundaryKind::Symmetry) {
                            //  cols = { Eq - nx, Eq - 1, Eq,   Eq + nx };
                            if (i == nx - 1) {
                                cols = { Eq - 2 * nx, Eq - nx, Eq - 2, Eq - 1, Eq, Eq - nx + 1, Eq - nx + 2, Eq + nx, Eq + 2 * nx };
                            }
                            else {
                                cols = { Eq - 2 * nx, Eq - nx, Eq - 2, Eq - 1, Eq, Eq + 1, Eq - nx + 1, Eq + nx, Eq + 2 * nx };
                            }

                            allVals.clear();
                            //   allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                             //  if (boundaryContainerU.getBoundary(PointBoundary::E).type == BoundaryKind::Symmetry && i == nx-1) {
                            Vector_new = symetry(k, j, i, u, v, w, is3D, 'r');
                            allVals.insert(allVals.begin(), Vector_new.begin(), Vector_new.end());
                            Vector_new = symetry(k, j, i, u, v, w, is3D, 'l');
                            allVals.insert(allVals.end(), Vector_new.begin(), Vector_new.end());
                            Vector_new = symetry(k, j, i, u, v, w, is3D, 'u');
                            allVals.insert(allVals.end(), Vector_new.begin(), Vector_new.end());
                            Vector_new = symetry(k, j, i, u, v, w, is3D, 'd');
                            allVals.insert(allVals.end(), Vector_new.begin(), Vector_new.end());
                            //  }
                            valueAtCenter = allVals.at(1) - allVals.at(6) + allVals.at(9) - allVals.at(14);
                            valueDown = allVals.at(8) - allVals.at(13);
                            valueLeft = allVals.at(0) - allVals.at(5);
                            valueRight = allVals.at(2) - allVals.at(7);
                            valueUp = allVals.at(10) - allVals.at(15);

                            vals = { -allVals.at(12),valueDown,-allVals.at(4),valueLeft, valueAtCenter,valueRight,allVals.at(3),valueUp,allVals.at(11) };
                            count = cols.size();

                            //  vals = { -allVals.at(13),-allVals.at(5),-allVals.at(6) + allVals.at(1) + allVals.at(9) - allVals.at(14),allVals.at(10) };//numbering knows only Mukesh

                          //    if (allVals.at(2) != 0.0 && i != (nx - 1)) { //This condition should be modified for non-zero penetration of the fluid
                         //         ((boundaryContainerU.getBoundary(PointBoundary::E).type == BoundaryKind::Symmetry) && i == nx-1) ? cols.insert(cols.begin() + 1, Eq - nx - 1) : cols.insert(cols.end() - 1, Eq + 1);

                          //        vals.insert(vals.end() - 1, allVals.at(2));
                           //   }

                        }
                        else if (boundaryContainerU.getBoundary(PointBoundary::E).type == BoundaryKind::Convective) {
                            if (i == nx - 1) {
                                cols = { Eq - nx, Eq - 1, Eq,   Eq + nx };
                                allVals.clear();
                                allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                                vals = { -allVals.at(13),-allVals.at(5),-allVals.at(6) + allVals.at(5) + allVals.at(9) - allVals.at(14),allVals.at(10) };
                                count = cols.size();
                            }
                            else {
                                cols = { Eq - nx, Eq - 1, Eq, Eq + 1, Eq + nx };
                                allVals.clear();
                                allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                                vals = { -allVals.at(13),-allVals.at(5),-allVals.at(6) + allVals.at(1) + allVals.at(9) - allVals.at(14),allVals.at(2),allVals.at(10) };
                                count = cols.size();
                            }
                        }
                        else {
                            if (i == nx - 1) {
                                cols = { Eq - nx, Eq - 1, Eq,   Eq + nx };
                                allVals.clear();
                                allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                                vals = { -allVals.at(13),-allVals.at(5),-allVals.at(6) + allVals.at(1) + allVals.at(9) - allVals.at(14),allVals.at(10) };
                                count = cols.size();
                            }
                            else {
                                cols = { Eq - nx, Eq - 1, Eq, Eq + 1, Eq + nx };
                                allVals.clear();
                                allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                                vals = { -allVals.at(13),-allVals.at(5),-allVals.at(6) + allVals.at(1) + allVals.at(9) - allVals.at(14),allVals.at(2),allVals.at(10) };
                                count = cols.size();
                            }

                        }
                    }
                    break;
                case 'W':
                    if (nz > 1) {
                        if (i == 0) {
                            cols = { Eq - nx * ny,Eq - nx,  Eq, Eq + 1,  Eq + nx,Eq + nx * ny };
                            allVals.clear();
                            allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                            vals = { -allVals.at(21),-allVals.at(13), allVals.at(1) + allVals.at(9) - allVals.at(14) + allVals.at(17) - allVals.at(22),allVals.at(2),allVals.at(10),allVals.at(18) };
                            count = cols.size();
                        }
                        else {
                            cols = { Eq - nx * ny,Eq - nx,Eq - 1,  Eq, Eq + 1,  Eq + nx ,Eq + nx * ny };
                            allVals.clear();
                            allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                            vals = { -allVals.at(21),-allVals.at(13),-allVals.at(5),-allVals.at(6) + allVals.at(1) + allVals.at(9) - allVals.at(14) + allVals.at(17) - allVals.at(22) ,allVals.at(2),allVals.at(10),allVals.at(18) };
                            count = cols.size();
                        }
                    }
                    else {

                        if (boundaryContainerU.getBoundary(PointBoundary::W).type == BoundaryKind::Symmetry) {

                            if (i == 1) {
                                cols = { Eq - 2 * nx, Eq - nx, Eq + nx - 2, Eq + nx - 1, Eq, Eq + 1, Eq + 2, Eq + nx, Eq + 2 * nx };
                            }
                            else {
                                cols = { Eq - 2 * nx, Eq - nx, Eq + nx - 1, Eq - 1, Eq, Eq + 1, Eq + 2, Eq + nx, Eq + 2 * nx };
                            }

                            allVals.clear();
                            //   
                          //      if (boundaryContainerU.getBoundary(PointBoundary::W).type == BoundaryKind::Symmetry && i == 0) {
                            Vector_new = symetry(k, j, i, u, v, w, is3D, 'r');
                            allVals.insert(allVals.begin(), Vector_new.begin(), Vector_new.end());
                            Vector_new = symetry(k, j, i, u, v, w, is3D, 'l');
                            allVals.insert(allVals.end(), Vector_new.begin(), Vector_new.end());
                            Vector_new = symetry(k, j, i, u, v, w, is3D, 'u');
                            allVals.insert(allVals.end(), Vector_new.begin(), Vector_new.end());
                            Vector_new = symetry(k, j, i, u, v, w, is3D, 'd');
                            allVals.insert(allVals.end(), Vector_new.begin(), Vector_new.end());
                            //   }
                            valueAtCenter = allVals.at(1) - allVals.at(6) + allVals.at(9) - allVals.at(14);
                            valueDown = allVals.at(8) - allVals.at(13);
                            valueLeft = allVals.at(0) - allVals.at(5);
                            valueRight = allVals.at(2) - allVals.at(7);
                            valueUp = allVals.at(10) - allVals.at(15);

                            vals = { -allVals.at(12),valueDown,-allVals.at(4),valueLeft, valueAtCenter,valueRight,allVals.at(3),valueUp,allVals.at(11) };
                            count = cols.size();
                        }
                        else {
                            if (i == 0) {
                                cols = { Eq - nx,  Eq, Eq + 1,  Eq + nx };
                                allVals.clear();
                                allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                                vals = { -allVals.at(13), -allVals.at(6) + allVals.at(1) + allVals.at(9) - allVals.at(14),allVals.at(2),allVals.at(10) };
                                count = cols.size();
                            }
                            else {
                                cols = { Eq - nx,Eq - 1,  Eq, Eq + 1,  Eq + nx };
                                allVals.clear();
                                allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                                vals = { -allVals.at(13),-allVals.at(5),-allVals.at(6) + allVals.at(1) + allVals.at(9) - allVals.at(14),allVals.at(2),allVals.at(10) };
                                count = cols.size();
                            }

                        }
                    }
                    //     if (allVals.at(5) != 0.0 && i != 0) { //This condition should be modified for non-zero penetration of the fluid
                    //         ((boundaryContainerU.getBoundary(PointBoundary::W).type == BoundaryKind::Symmetry) && i == 0)? cols.insert(cols.begin() + 1, Eq + nx - 1) : cols.insert(cols.begin() + 1, Eq - 1);
                    //         vals.insert(vals.begin() + 1, -allVals.at(5));
                    //     }

                    break;
                case 'N':
                    if (nz > 1) {
                        if (j == ny - 1) {
                            cols = { Eq - nx * ny ,Eq - nx, Eq - 1, Eq, Eq + 1,Eq + nx * ny };
                            allVals.clear();
                            allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                            vals = { -allVals.at(21) ,-allVals.at(13),-allVals.at(5),-allVals.at(6) + allVals.at(1) - allVals.at(14) + allVals.at(17) - allVals.at(22),allVals.at(2),allVals.at(18) };
                            count = cols.size();
                        }
                        else {
                            cols = { Eq - nx * ny ,Eq - nx, Eq - 1, Eq, Eq + 1,Eq + nx,Eq + nx * ny };
                            allVals.clear();
                            allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                            vals = { -allVals.at(21) ,-allVals.at(13),-allVals.at(5),-allVals.at(6) + allVals.at(1) + allVals.at(9) - allVals.at(14) + allVals.at(17) - allVals.at(22),allVals.at(2),allVals.at(10),allVals.at(18) };
                            count = cols.size();
                        }
                    }
                    else {
                        cols = { Eq - nx, Eq - 1, Eq, Eq + 1 };
                        allVals.clear();
                        allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                        vals = { -allVals.at(13),-allVals.at(5),-allVals.at(6) + allVals.at(1) + allVals.at(9) - allVals.at(14),allVals.at(2) };//numbering knows only Mukesh
                        if (allVals.at(10) != 0.0 && (Eq + nx) < nx * ny * (nz == 0 ? 1 : nz)) { //This condition should be modified for non-zero penetration of the fluid
                            cols.insert(cols.end(), Eq + nx);
                            vals.insert(vals.end(), allVals.at(10));
                        }
                        count = cols.size();
                    }
                    break;
                case 'S':
                    if (nz > 1) {
                        if (j == 0) {
                            cols = { Eq - nx * ny,Eq - 1, Eq, Eq + 1,  Eq + nx,Eq + nx * ny };
                            allVals.clear();
                            allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                            vals = { -allVals.at(21),-allVals.at(5),-allVals.at(6) + allVals.at(1) + allVals.at(9) + allVals.at(17) - allVals.at(22),allVals.at(2), allVals.at(10),allVals.at(18) };
                            count = cols.size();
                        }
                        else {
                            cols = { Eq - nx * ny,Eq - nx,Eq - 1, Eq, Eq + 1,  Eq + nx,Eq + nx * ny };
                            allVals.clear();
                            allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                            vals = { -allVals.at(21),-allVals.at(13) ,-allVals.at(5),-allVals.at(6) + allVals.at(1) + allVals.at(9) - allVals.at(14) + allVals.at(17) - allVals.at(22),allVals.at(2), allVals.at(10),allVals.at(18) };
                            count = cols.size();
                        }

                    }
                    else {
                        cols = { Eq - 1, Eq, Eq + 1,  Eq + nx };
                        allVals.clear();
                        allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                        vals = { -allVals.at(5),-allVals.at(6) + allVals.at(1) + allVals.at(9) - allVals.at(14),allVals.at(2), allVals.at(10) };//numbering knows only Mukesh
                        if (allVals.at(13) != 0.0 && (Eq - nx) > 0) { //This condition should be modified for non-zero penetration of the fluid
                            cols.insert(cols.begin(), Eq - nx);
                            vals.insert(vals.begin(), -allVals.at(13));
                        }
                        count = cols.size();
                    }


                    break;
                case'B':
                    if (k == 0) {
                        cols = { Eq - nx,Eq - 1, Eq, Eq + 1,  Eq + nx,Eq + nx * ny };
                        allVals.clear();
                        allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                        vals = { -allVals.at(13),-allVals.at(5),-allVals.at(6) + allVals.at(1) + allVals.at(9) - allVals.at(14) + allVals.at(17),allVals.at(2), allVals.at(10),allVals.at(18) };
                        count = cols.size();
                    }
                    else {
                        cols = cols = { Eq - nx * ny, Eq - nx,Eq - 1, Eq, Eq + 1,  Eq + nx,Eq + nx * ny };
                        allVals.clear();
                        allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                        vals = { -allVals.at(21), -allVals.at(13),-allVals.at(5),-allVals.at(6) + allVals.at(1) + allVals.at(9) - allVals.at(14) + allVals.at(17) - allVals.at(22),allVals.at(2), allVals.at(10),allVals.at(18) };
                        count = cols.size();
                    }
                    break;
                case'F':
                    if (k == nz - 1) {
                        cols = { Eq - nx * ny,Eq - nx,Eq - 1, Eq, Eq + 1,  Eq + nx };
                        allVals.clear();
                        allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                        vals = { -allVals.at(21) ,-allVals.at(13),-allVals.at(5),-allVals.at(6) + allVals.at(1) + allVals.at(9) - allVals.at(14) - allVals.at(22),allVals.at(2), allVals.at(10) };
                        count = cols.size();
                    }
                    else {
                        cols = cols = { Eq - nx * ny, Eq - nx,Eq - 1, Eq, Eq + 1,  Eq + nx,Eq + nx * ny };
                        allVals.clear();
                        allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                        vals = { -allVals.at(21), -allVals.at(13),-allVals.at(5),-allVals.at(6) + allVals.at(1) + allVals.at(9) - allVals.at(14) + allVals.at(17) - allVals.at(22),allVals.at(2), allVals.at(10),allVals.at(18) };
                        count = cols.size();
                    }
                    break;
                    //courener cases
                case 'D'://lower left corner for 2D  // West-South edge for 3D (i == 0 && j == 0)
                    if (nz > 1) {
                        cols = { Eq - nx * ny, Eq, Eq + 1,  Eq + nx,Eq + nx * ny };
                        allVals.clear();
                        allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                        vals = { -allVals.at(21), allVals.at(1) + allVals.at(9) + allVals.at(17) - allVals.at(22),allVals.at(2),allVals.at(10),allVals.at(18) };
                        count = cols.size();
                    }
                    else {
                        cols = { Eq, Eq + 1,  Eq + nx };
                        allVals.clear();
                        allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                        vals = { allVals.at(1) + allVals.at(9) - allVals.at(14),allVals.at(2),allVals.at(10) };//numbering knows only Mukesh                               
                        count = cols.size();
                    }
                    break;
                case 'T'://lower right corner for 2D // East-South edge for 3D (i == nx - 1 && j == 0)
                    if (nz > 1) {
                        cols = { Eq - nx * ny,Eq - 1, Eq,   Eq + nx,Eq + nx * ny };
                        allVals.clear();
                        allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                        vals = { -allVals.at(21), -allVals.at(5),-allVals.at(6) + allVals.at(9) + allVals.at(17) - allVals.at(22),allVals.at(10),allVals.at(18) };
                        count = cols.size();

                    }
                    else {
                        if (boundaryContainerU.getBoundary(PointBoundary::E).type == BoundaryKind::Convective) {
                            cols = { Eq - 1, Eq,   Eq + nx };
                            allVals.clear();
                            allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                            vals = { -allVals.at(5),-allVals.at(6) + allVals.at(5) + allVals.at(9) - allVals.at(14),allVals.at(10) };
                            count = cols.size();
                        }
                        else {
                            cols = { Eq - 1, Eq,   Eq + nx };
                            allVals.clear();
                            allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                            vals = { -allVals.at(5),-allVals.at(6) + allVals.at(9),allVals.at(10) };//numbering knows only Mukesh
                            count = cols.size();
                        }
                    }
                    break;
                case 'U'://upper left corner for 2D // West-North edge for 3D (i == 0 && j == ny - 1)
                    if (nz > 1) {
                        cols = { Eq - nx * ny, Eq - nx,  Eq, Eq + 1,Eq + nx * ny };
                        allVals.clear();
                        allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                        vals = { -allVals.at(21),-allVals.at(13), allVals.at(1) - allVals.at(14) + allVals.at(17) - allVals.at(22),allVals.at(2),allVals.at(18) };
                        count = cols.size();
                    }
                    else {
                        cols = { Eq - nx,  Eq, Eq + 1 };
                        allVals.clear();
                        allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                        vals = { -allVals.at(13), allVals.at(1) - allVals.at(14),allVals.at(2) };
                        count = cols.size();
                    }

                    break;
                case 'Y'://upper right corner for 2D // East-North edge for 3D (i == nx - 1 && j == ny - 1)
                    if (nz > 1) {
                        cols = { Eq - nx * ny,Eq - nx, Eq - 1, Eq,Eq + nx * ny };
                        allVals.clear();
                        allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                        vals = { -allVals.at(21) ,-allVals.at(13),-allVals.at(5),-allVals.at(6) - allVals.at(14) + allVals.at(17) - allVals.at(22),  allVals.at(18) };//numbering knows only Mukesh
                        count = cols.size();
                    }
                    else {
                        if (boundaryContainerU.getBoundary(PointBoundary::E).type == BoundaryKind::Convective) {
                            cols = { Eq - nx, Eq - 1, Eq };
                            allVals.clear();
                            allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                            vals = { -allVals.at(13),-allVals.at(5),-allVals.at(6) + allVals.at(5) + allVals.at(9) - allVals.at(14) };
                            count = cols.size();
                        }
                        else {
                            cols = { Eq - nx, Eq - 1, Eq };
                            allVals.clear();
                            allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);

                            vals = { -allVals.at(13),-allVals.at(5),-allVals.at(6) + allVals.at(9) - allVals.at(14) };//numbering knows only Mukesh
                            count = cols.size();
                        }
                    }
                    break;
                case 'A':// West-Bottom edge this only for 3d (i == 0 && k == 0)
                    cols = { Eq - nx,Eq, Eq + 1, Eq + nx,Eq + nx * ny };
                    allVals.clear();
                    allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                    vals = { -allVals.at(13),allVals.at(1) + allVals.at(9) - allVals.at(14) + allVals.at(17),allVals.at(2), allVals.at(10),allVals.at(18) };//numbering knows only Mukesh
                    count = cols.size();
                    break;
                case 'C':// // West-Top edge this only for 3d (i == 0 && k == nz - 1)
                    cols = { Eq - nx * ny,Eq - nx,Eq, Eq + 1, Eq + nx };
                    allVals.clear();
                    allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                    vals = { -allVals.at(21), -allVals.at(13),allVals.at(1) + allVals.at(9) - allVals.at(14) - allVals.at(22),allVals.at(2), allVals.at(10) };//numbering knows only Mukesh
                    count = cols.size();
                    break;
                case 'G':// // East-Bottom edge  (i == nx - 1 && k == 0) 
                    cols = { Eq - nx,Eq - 1, Eq, Eq + nx,Eq + nx * ny };
                    allVals.clear();
                    allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                    vals = { -allVals.at(13), -allVals.at(5),-allVals.at(6) + allVals.at(9) - allVals.at(14) + allVals.at(17), allVals.at(10),allVals.at(18) };//numbering knows only Mukesh
                    count = cols.size();
                    break;
                case 'H'://  if (i == nx - 1 && k == nz - 1) point.setBoundaryType(PointBoundary::H); // East-Top edge
                    cols = { Eq - nx * ny, Eq - nx,Eq - 1, Eq, Eq + nx };
                    allVals.clear();
                    allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                    vals = { -allVals.at(21),-allVals.at(13), -allVals.at(5), -allVals.at(6) + allVals.at(9) - allVals.at(14) - allVals.at(22), allVals.at(10) };//numbering knows only Mukesh
                    count = cols.size();
                    break;
                case 'R'://  (j == 0 && k == 0) point.setBoundaryType(PointBoundary::R); // South-Bottom edge
                    cols = { Eq - 1, Eq,Eq + 1, Eq + nx,Eq + nx * ny };
                    allVals.clear();
                    allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                    vals = { -allVals.at(5),allVals.at(1) - allVals.at(6) + allVals.at(9) + allVals.at(17),allVals.at(2), allVals.at(10),allVals.at(18) };//numbering knows only Mukesh
                    count = cols.size();
                    break;
                case 'V'://   (j == 0 && k == nz - 1) point.setBoundaryType(PointBoundary::V); // South-Top edge
                    cols = { Eq - nx * ny,Eq - 1, Eq,Eq + 1, Eq + nx };
                    allVals.clear();
                    allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                    vals = { -allVals.at(21) ,-allVals.at(5),allVals.at(1) - allVals.at(6) + allVals.at(9) - allVals.at(22),allVals.at(2), allVals.at(10) };//numbering knows only Mukesh
                    count = cols.size();
                    break;
                case 'Z'://  (j == ny - 1 && k == 0) point.setBoundaryType(PointBoundary::Z); // North-Bottom edge
                    cols = { Eq - nx,Eq - 1, Eq,Eq + 1,Eq + nx * ny };
                    allVals.clear();
                    allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                    vals = { -allVals.at(13) ,-allVals.at(5),allVals.at(1) - allVals.at(6) - allVals.at(14) + allVals.at(17),allVals.at(2),allVals.at(18) };
                    count = cols.size();
                    break;
                case 'X'://  (j == ny - 1 && k == nz - 1) point.setBoundaryType(PointBoundary::X); // North-Top edge
                    cols = { Eq - nx * ny,Eq - nx,Eq - 1, Eq,Eq + 1 };
                    allVals.clear();
                    allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                    vals = { -allVals.at(21),-allVals.at(13) ,-allVals.at(5),allVals.at(1) - allVals.at(6) - allVals.at(14) - allVals.at(22),allVals.at(2) };
                    count = cols.size();
                    break;
                case 'I'://  (i == 0 && j == 0 && k == 0) point.setBoundaryType(PointBoundary::I); // West-South-Bottom corner
                    cols = { Eq,Eq + 1,Eq + nx,Eq + nx * ny };
                    allVals.clear();
                    allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                    vals = { allVals.at(1) + allVals.at(9) + allVals.at(17),allVals.at(2),allVals.at(10),allVals.at(18) };
                    count = cols.size();
                    break;
                case 'J'://  (i == 0 && j == ny - 1 && k == 0) point.setBoundaryType(PointBoundary::J); // West-North-Bottom corner
                    cols = { Eq - nx, Eq,Eq + 1,Eq + nx * ny };
                    allVals.clear();
                    allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                    vals = { -allVals.at(13) ,allVals.at(1) - allVals.at(14) + allVals.at(17),allVals.at(2),allVals.at(18) };
                    count = cols.size();
                    break;
                case 'K':// (i == 0 && j == 0 && k == nz - 1) point.setBoundaryType(PointBoundary::K); // West-South-Top corner
                    cols = { Eq - nx * ny,Eq,Eq + 1,Eq + nx };
                    allVals.clear();
                    allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                    vals = { -allVals.at(21),allVals.at(1) + allVals.at(9) - allVals.at(22),allVals.at(2),allVals.at(10) };
                    count = cols.size();
                    break;
                case 'L':// (i == 0 && j == ny - 1 && k == nz - 1) point.setBoundaryType(PointBoundary::L); // West-North-Top corner
                    cols = { Eq - nx * ny,Eq - nx,Eq,Eq + 1 };
                    allVals.clear();
                    allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                    vals = { -allVals.at(21),-allVals.at(13),allVals.at(1) - allVals.at(14) - allVals.at(22),allVals.at(2) };
                    count = cols.size();
                    break;
                case 'M': //i == nx - 1 && j == 0 && k == 0) point.setBoundaryType(PointBoundary::M); // East-South-Bottom corner
                    cols = { Eq - 1,Eq,Eq + nx,Eq + nx * ny };
                    allVals.clear();
                    allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                    vals = { -allVals.at(5) ,-allVals.at(6) + allVals.at(9) + allVals.at(17),allVals.at(10),allVals.at(18) };
                    count = cols.size();
                    break;
                case 'O': //(i == nx - 1 && j == ny - 1 && k == 0) point.setBoundaryType(PointBoundary::O); // East-North-Bottom corner
                    cols = { Eq - nx,Eq - 1,Eq,Eq + nx * ny };
                    allVals.clear();
                    allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                    vals = { -allVals.at(13),-allVals.at(5) ,-allVals.at(6) - allVals.at(14) + allVals.at(17),allVals.at(18) };
                    count = cols.size();
                    break;
                case 'P': //(i == nx - 1 && j == 0 && k == nz - 1) point.setBoundaryType(PointBoundary::P); // East-South-Top corner
                    cols = { Eq - nx * ny,Eq - 1,Eq,Eq + nx };
                    allVals.clear();
                    allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                    vals = { -allVals.at(21),-allVals.at(5) , -allVals.at(6) + allVals.at(9) - allVals.at(22),allVals.at(10) };
                    count = cols.size();
                    break;
                case 'Q':  // (i == nx - 1 && j == ny - 1 && k == nz - 1) point.setBoundaryType(PointBoundary::Q); // East-North-Top corner
                    cols = { Eq - nx * ny,Eq - nx,Eq - 1,Eq };
                    allVals.clear();
                    allVals = calculateFluxFOU(k, j, i, u, v, w, is3D);
                    vals = { -allVals.at(21), -allVals.at(13) ,-allVals.at(5) ,-allVals.at(6) - allVals.at(14) - allVals.at(22) };
                    count = cols.size();
                    break;
                default:

                    if (nz > 1) {
                        cols = { Eq - 2 * nx * ny,Eq - nx * ny, Eq - 2 * nx, Eq - nx, Eq - 2, Eq - 1, Eq, Eq + 1, Eq + 2, Eq + nx, Eq + 2 * nx,Eq + nx * ny,Eq + 2 * nx * ny }; // Higher Order Scheme
                    }
                    else {
                        cols = { Eq - 2 * nx, Eq - nx, Eq - 2, Eq - 1, Eq, Eq + 1, Eq + 2, Eq + nx, Eq + 2 * nx }; // Higher Order Scheme
                    }




                    allVals.clear();
                    //      allVals = calculateFlux(k, j, i, u, v, w, is3D);
                    //      

                    Vector_new = calculateFluxSide(k, j, i, u, v, w, is3D, 'r');
                    allVals.insert(allVals.begin(), Vector_new.begin(), Vector_new.end());
                    Vector_new = calculateFluxSide(k, j, i, u, v, w, is3D, 'l');
                    allVals.insert(allVals.end(), Vector_new.begin(), Vector_new.end());
                    Vector_new = calculateFluxSide(k, j, i, u, v, w, is3D, 'u');
                    allVals.insert(allVals.end(), Vector_new.begin(), Vector_new.end());
                    Vector_new = calculateFluxSide(k, j, i, u, v, w, is3D, 'd');
                    allVals.insert(allVals.end(), Vector_new.begin(), Vector_new.end());
                    if (nz > 1) {
                        Vector_new = calculateFluxSide(k, j, i, u, v, w, is3D, 'f');
                        allVals.insert(allVals.end(), Vector_new.begin(), Vector_new.end());
                        Vector_new = calculateFluxSide(k, j, i, u, v, w, is3D, 'b');
                        allVals.insert(allVals.end(), Vector_new.begin(), Vector_new.end());

                    }

                    valueAtCenter = allVals.at(1) - allVals.at(6) + allVals.at(9) - allVals.at(14) ;
                    valueAtCenter = valueAtCenter + (nz > 1 ? (allVals.at(17) - allVals.at(22)) : 0.0);
                    valueDown = allVals.at(8) - allVals.at(13);
                    valueLeft = allVals.at(0) - allVals.at(5);
                    valueRight = allVals.at(2) - allVals.at(7);
                    valueUp = allVals.at(10) - allVals.at(15);
                    if (nz > 1) {
                        valuefacedown = allVals.at(16) - allVals.at(21);
                        valuefaceup = allVals.at(18) - allVals.at(23);
                        vals = { -allVals.at(20), valuefacedown, -allVals.at(12),valueDown,-allVals.at(4),valueLeft, valueAtCenter,valueRight,allVals.at(3),valueUp,allVals.at(11), valuefaceup,allVals.at(19) };
                    }
                    else {
                        vals = { -allVals.at(12),valueDown,-allVals.at(4),valueLeft, valueAtCenter,valueRight,allVals.at(3),valueUp,allVals.at(11) };
                    }
                    count = cols.size();
                    break;
                }

                col_indices_t.insert(col_indices_t.begin() + row_ptr_t[Eq], cols.begin(), cols.end());
                values_t.insert(values_t.begin() + row_ptr_t[Eq], vals.begin(), vals.end());
                row_ptr_t[Eq + 1] = row_ptr_t[Eq] + count;
            }
        }
    }

    std::vector<MKL_INT> rows_end_t(row_ptr_t.begin() + 1, row_ptr_t.end());

    LHS.setCols(col_indices_t); LHS.setValues(values_t); LHS.setRows(row_ptr_t); LHS.setRowsEnd(rows_end_t);
    sparse_status_t status = mkl_sparse_d_create_csr(
        &LHS_mkl, SPARSE_INDEX_BASE_ZERO,
        LHS.getSize(), LHS.getSize(),
        const_cast<MKL_INT*>(LHS.getRowPtr().data()),
        const_cast<MKL_INT*>(LHS.getRowPtr_end().data()),
        const_cast<MKL_INT*>(LHS.getColIndices().data()),
        const_cast<double*>(LHS.getValues().data())
    );


}

std::vector<bool> VOF::checkDirection(int k, int j, int i,
    const std::vector<std::vector<std::vector<Point>>>& u, const std::vector<std::vector<std::vector<Point>>>& v, const std::vector<std::vector<std::vector<Point>>>& w, bool is3D) const {
    std::vector<bool> directions(6, true);
    if (u[(is3D ? k + 1 : 0)][j + 1][i + 1].getValue() < 0.0) directions[2] = false;
    if (u[(is3D ? k + 1 : 0)][j + 1][i].getValue() < 0.0) directions[1] = false;
    if (v[(is3D ? k + 1 : 0)][j + 1][i + 1].getValue() < 0.0) directions[3] = false;
    if (v[(is3D ? k + 1 : 0)][j][i + 1].getValue() < 0.0) directions[0] = false;
    if (is3D && k + 1 < w.size() && w[k + 1][j + 1][i + 1].getValue() < 0.0) directions[5] = false;
    if (is3D && k < w.size() && w[k][j + 1][i + 1].getValue() < 0.0) directions[4] = false;
    return directions;

}
std::vector<double> VOF::calculateFluxSide(int k, int j, int i,
    const std::vector<std::vector<std::vector<Point>>>& u, const std::vector<std::vector<std::vector<Point>>>& v, const std::vector<std::vector<std::vector<Point>>>& w, bool is3D, char side) const {
    std::vector<double> phi_flux_OneD(4, 0.0); // Initialize phi_flux vector with zeros
    std::vector<bool>signs;

    double phi_telda1 = 0.0;
    double dot_px1;
    signs = checkDirection(k, j, i, u, v, w, is3D);
    bool sign;
    double velocity;
    double face_area;

    double dx = phi[(is3D ? k : 0)][j][i].getdx();
    double dy = phi[(is3D ? k : 0)][j][i].getdy();
    double dz = phi[(is3D ? k : 0)][j][i].getdz();
    double surfaceX = (is3D ? (dz * dy) : dy);
    double surfaceY = (is3D ? (dz * dx) : dx);
    double surfaceZ = dx * dy;

    switch (side)
    {
    case 'r':

        velocity = u[(is3D ? (k + 1) : 0)][j + 1][i + 1].getValue();
        sign = signs[2];
        face_area = (is3D ? (dz * dy) : dy);
        if (sign == true) {
            dot_px1 = (phi[(is3D ? k : 0)][j][i + 1].getValue() - (phi[(is3D ? k : 0)][j][i - 1].getValue()));
            phi_telda1 = (dot_px1 == 0) ? 1.0 : ((phi[(is3D ? k : 0)][j][i].getValue() - (phi[(is3D ? k : 0)][j][i - 1].getValue())) / dot_px1);
        }
        else {
            dot_px1 = (phi[(is3D ? k : 0)][j][i].getValue() - phi[(is3D ? k : 0)][j][i + 2].getValue());
            phi_telda1 = (dot_px1 == 0) ? 1.0 : ((phi[(is3D ? k : 0)][j][i + 1].getValue() - phi[(is3D ? k : 0)][j][i + 2].getValue()) / dot_px1);
        }
        break;
    case 'l':

        velocity = u[(is3D ? (k + 1) : 0)][j + 1][i].getValue();
        sign = signs[1];
        face_area = (is3D ? (dz * dy) : dy);
        if (sign == true) {
            dot_px1 = (phi[(is3D ? k : 0)][j][i].getValue() - (phi[(is3D ? k : 0)][j][i - 2].getValue()));
            phi_telda1 = (dot_px1 == 0) ? 1.0 : (((phi[(is3D ? k : 0)][j][i - 1].getValue()) - (phi[(is3D ? k : 0)][j][i - 2].getValue())) / dot_px1);
        }
        else {
            dot_px1 = (phi[(is3D ? k : 0)][j][i - 1].getValue() - phi[(is3D ? k : 0)][j][i + 1].getValue());
            phi_telda1 = (dot_px1 == 0) ? 1.0 : ((phi[(is3D ? k : 0)][j][i].getValue() - phi[(is3D ? k : 0)][j][i + 1].getValue()) / dot_px1);
        }
        break;
    case 'u':

        velocity = v[(is3D ? (k + 1) : 0)][j + 1][i + 1].getValue();
        sign = signs[3];
        face_area = (is3D ? (dz * dx) : dx);
        if (sign == true) {
            dot_px1 = (phi[(is3D ? k : 0)][j + 1][i].getValue() - (j==0?1:phi[(is3D ? k : 0)][j - 1][i].getValue()));
            phi_telda1 = (dot_px1 == 0) ? 1.0 : ((phi[(is3D ? k : 0)][j][i].getValue() - (phi[(is3D ? k : 0)][j - 1][i].getValue())) / dot_px1);
        }
        else {
            dot_px1 = (phi[(is3D ? k : 0)][j][i].getValue() - phi[(is3D ? k : 0)][j + 2][i].getValue());
            phi_telda1 = (dot_px1 == 0) ? 1.0 : ((phi[(is3D ? k : 0)][j + 1][i].getValue() - phi[(is3D ? k : 0)][j + 2][i].getValue()) / dot_px1);
        }
        break;
    case 'd':
        velocity = v[(is3D ? (k + 1) : 0)][j][i + 1].getValue();
        sign = signs[0];
        face_area = (is3D ? (dz * dx) : dx);
        if (sign == true) {
            dot_px1 = (phi[(is3D ? k : 0)][j][i].getValue() - (phi[(is3D ? k : 0)][j - 2][i].getValue()));
            phi_telda1 = (dot_px1 == 0) ? 1.0 : ((( phi[(is3D ? k : 0)][j - 1][i].getValue()) - (phi[(is3D ? k : 0)][j - 2][i].getValue())) / dot_px1);
        }
        else {
            dot_px1 = (phi[(is3D ? k : 0)][j - 1][i].getValue() - phi[(is3D ? k : 0)][j + 1][i].getValue());
            phi_telda1 = (dot_px1 == 0) ? 1.0 : ((phi[(is3D ? k : 0)][j][i].getValue() - phi[(is3D ? k : 0)][j + 1][i].getValue()) / dot_px1);
        }
        break;
    case 'f'://face
        velocity = w[k + 1][j + 1][i + 1].getValue();
        sign = signs[5];
        face_area = dx * dy;
        if (w[k + 1][j + 1][i + 1].getValue() >= 0.0) {
            dot_px1 = (phi[k + 1][j][i].getPrevValue() - (k==0?1:phi[k - 1][j][i].getPrevValue()));
            phi_telda1 = (dot_px1 == 0) ? 1.0 : (phi[k][j][i].getPrevValue() - ( phi[k - 1][j][i].getPrevValue())) / dot_px1;
        }
        else {
            dot_px1 = (phi[k][j][i].getPrevValue() - phi[k + 2][j][i].getPrevValue());
            phi_telda1 = (dot_px1 == 0) ? 1.0 : (phi[k + 1][j][i].getPrevValue() - phi[k + 2][j][i].getPrevValue()) / dot_px1;
        }
        break;
    case 'b'://back
        velocity = w[k][j + 1][i + 1].getValue();
        sign = signs[4];
        face_area = dx * dy;
        if (w[k][j + 1][i + 1].getValue() >= 0.0) {
            dot_px1 = (phi[k][j][i].getPrevValue() -(k<3?1: phi[k - 2][j][i].getPrevValue()));
            phi_telda1 = (dot_px1 == 0) ? 1.0 : ((phi[k - 1][j][i].getPrevValue()) - (phi[k - 2][j][i].getPrevValue())) / dot_px1;
        }
        else {
            dot_px1 = (phi[k - 1][j][i].getPrevValue() - phi[k + 1][j][i].getPrevValue());
            phi_telda1 = (dot_px1 == 0) ? 1.0 : (phi[k][j][i].getPrevValue() - phi[k + 1][j][i].getPrevValue()) / dot_px1;
        }
        break;
    default:

        break;
    }
    // Calculate phi_telda1

    if (sign == true) {
        if (phi_telda1 > 0.0 && phi_telda1 <= 2.0 / 13.0) {
            phi_flux_OneD[1] = 3.0 * velocity * face_area;
            phi_flux_OneD[0] = -2.0 * velocity * face_area;
        }
        else if (phi_telda1 > 2.0 / 13.0 && phi_telda1 <= 0.8) {
            phi_flux_OneD[1] = 5.0 / 6.0 * velocity * face_area;
            phi_flux_OneD[0] = -1.0 / 6.0 * velocity * face_area;
            phi_flux_OneD[2] = 1.0 / 3.0 * velocity * face_area;
        }
        else if (phi_telda1 > 0.8 && phi_telda1 <= 1.0) {
            phi_flux_OneD[2] = velocity * face_area;
        }
        else {
            phi_flux_OneD[1] = velocity * face_area;
        }
    }
    else {
        if (phi_telda1 > 0.0 && phi_telda1 <= 2.0 / 13.0) {
            phi_flux_OneD[2] = 3.0 * velocity * face_area;
            phi_flux_OneD[3] = -2.0 * velocity * face_area;
        }
        else if (phi_telda1 > 2.0 / 13.0 && phi_telda1 <= 0.8) {
            phi_flux_OneD[1] = 1.0 / 3.0 * velocity * face_area;
            phi_flux_OneD[2] = 5.0 / 6.0 * velocity * face_area;
            phi_flux_OneD[3] = -1.0 / 6.0 * velocity * face_area;
        }
        else if (phi_telda1 > 0.8 && phi_telda1 <= 1.0) {
            phi_flux_OneD[1] = velocity * face_area;
        }
        else {
            phi_flux_OneD[2] = velocity * face_area;
        }
    }
    return phi_flux_OneD;
}




std::vector<double> VOF::calculateFluxFOU(int k, int j, int i,
    const std::vector<std::vector<std::vector<Point>>>& u, const std::vector<std::vector<std::vector<Point>>>& v, const std::vector<std::vector<std::vector<Point>>>& w, bool is3D) const {
    std::vector<double> phi_flux(26, 0.0); // Initialize phi_flux vector with zeros

    std::vector<bool>signs;
    signs = checkDirection(k, j, i, u, v, w, is3D);
    double rhoRight, rhoLeft, rhoUp, rhoDown, rhoUplan, rhoDplan;
    double dx = phi[(is3D ? k : 0)][j][i].getdx();
    double dy = phi[(is3D ? k : 0)][j][i].getdy();
    double dz = phi[(is3D ? k : 0)][j][i].getdz();

    //calculate flux using FOU
    if (signs[2] == true) {

        phi_flux[1] = u[(is3D ? k + 1 : 0)][j + 1][i + 1].getValue() * (is3D ? dy * dz : dy);
    }
    else {
        phi_flux[2] = u[(is3D ? k + 1 : 0)][j + 1][i + 1].getValue() * (is3D ? dy * dz : dy);
    }
    if (signs[1] == true) {

        phi_flux[5] = u[(is3D ? k + 1 : 0)][j + 1][i].getValue() * (is3D ? dy * dz : dy);
    }
    else {
        phi_flux[6] = u[(is3D ? k + 1 : 0)][j + 1][i].getValue() * (is3D ? dy * dz : dy);
    }
    if (signs[3] == true) {

        phi_flux[9] = v[(is3D ? k + 1 : 0)][j + 1][i + 1].getValue() * (is3D ? dx * dz : dx);
    }
    else {
        phi_flux[10] = v[(is3D ? k + 1 : 0)][j + 1][i + 1].getValue() * (is3D ? dx * dz : dx);

    }
    if (signs[0] == true) {

        phi_flux[13] = v[(is3D ? k + 1 : 0)][j][i + 1].getValue() * (is3D ? dx * dz : dx);
    }
    else {
        phi_flux[14] = v[(is3D ? k + 1 : 0)][j][i + 1].getValue() * (is3D ? dx * dz : dx);

    }
    if (is3D == true) {
        if (signs[5] == true) {

            phi_flux[17] = w[k + 1][j + 1][i + 1].getValue() * dx * dy;
        }
        else {
            phi_flux[18] = w[k + 1][j + 1][i + 1].getValue() * dx * dy;
        }

        if (signs[4] == true) {

            phi_flux[21] = w[k][j + 1][i + 1].getValue() * dx * dy;
        }
        else {
            phi_flux[22] = w[k][j + 1][i + 1].getValue() * dx * dy;
        }

    }
    return phi_flux;
}
// Matrix-free y = LHS * x with full boundary handling (2D + 3D).
// Mirrors the column/value patterns in updateLHS; uses calculateFluxFOU/symetry/calculateFluxSide.
// Assumes these helpers and Point getters are const/thread-safe.
// NOTE: pass x by const ref to avoid copy.

std::vector<double> VOF::productLHS(
    const BoundaryContainer& boundaryContainerU,
    const std::vector<std::vector<std::vector<Point>>>& u,
    const std::vector<std::vector<std::vector<Point>>>& v,
    const std::vector<std::vector<std::vector<Point>>>& w,
    const std::vector<double>& x) const
{
    const bool is3D = (phi.size() > 2);
    const int nx = static_cast<int>(phi[0][0].size());
    const int ny = static_cast<int>(phi[0].size());
    const int nz = is3D ? static_cast<int>(phi.size()) : 1;
    const int nxny = nx * ny;
    const int N = nxny * nz;

    std::vector<double> y(N, 0.0);

    const bool periodicEast =
        boundaryContainerU.getBoundary(PointBoundary::E).type == BoundaryKind::Symmetry;
    const bool periodicWest =
        boundaryContainerU.getBoundary(PointBoundary::W).type == BoundaryKind::Symmetry;
    if (periodicEast != periodicWest)
        throw std::invalid_argument("VOF requires periodic U boundaries on both E and W");

    if (periodicEast) {
        // Evaluate each face from the same pair of cells on either side. The
        // west and east U faces represent one periodic face; PressureMatrix
        // uses the west U unknown for that face as well.
        auto value = [&](int k, int j, int i) {
            return x[phi[k][j][i].getUniqueNumber()];
        };
        auto xFlux = [&](int k, int j, int face) {
            const int wrappedFace = face == nx ? 0 : face;
            const int left = (wrappedFace + nx - 1) % nx;
            const int right = wrappedFace;
            const double velocity = u[is3D ? k + 1 : 0][j + 1][wrappedFace].getValue();
            const double fraction = nx > 2
                ? reconstructedFaceValue(value(k, j, (left + nx - 1) % nx),
                    value(k, j, left), value(k, j, right),
                    value(k, j, (right + 1) % nx), velocity)
                : value(k, j, velocity >= 0.0 ? left : right);
            const Point& cell = phi[k][j][left];
            return velocity * fraction * cell.getdy() *
                (is3D ? cell.getdz() : 1.0);
        };
        auto yFlux = [&](int k, int j, int i) {
            const double velocity = v[is3D ? k + 1 : 0][j][i + 1].getValue();
            const int lower = std::max(0, j - 1);
            const int upper = std::min(ny - 1, j);
            double fraction = value(k, velocity >= 0.0 ? lower : upper, i);
            if (j >= 2 && j <= ny - 2)
                fraction = reconstructedFaceValue(value(k, j - 2, i),
                    value(k, j - 1, i), value(k, j, i),
                    value(k, j + 1, i), velocity);
            const Point& cell = phi[k][lower][i];
            return velocity * fraction * cell.getdx() *
                (is3D ? cell.getdz() : 1.0);
        };
        auto zFlux = [&](int k, int j, int i) {
            const double velocity = w[k][j + 1][i + 1].getValue();
            const int back = std::max(0, k - 1);
            const int front = std::min(nz - 1, k);
            double fraction = value(velocity >= 0.0 ? back : front, j, i);
            if (k >= 2 && k <= nz - 2)
                fraction = reconstructedFaceValue(value(k - 2, j, i),
                    value(k - 1, j, i), value(k, j, i),
                    value(k + 1, j, i), velocity);
            const Point& cell = phi[back][j][i];
            return velocity * fraction * cell.getdx() * cell.getdy();
        };

#pragma omp parallel for collapse(3) schedule(static)
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    y[phi[k][j][i].getUniqueNumber()] =
                        xFlux(k, j, i + 1) - xFlux(k, j, i) +
                        yFlux(k, j + 1, i) - yFlux(k, j, i) +
                        (is3D ? zFlux(k + 1, j, i) - zFlux(k, j, i) : 0.0);
                }
            }
        }
        return y;
    }

    // Cheap inline dot over small stacks.
    auto dot_cols_vals = [&](const int* cols, const double* vals, int m) -> double {
        double s = 0.0;
        for (int t = 0; t < m; ++t) {
            const int c = cols[t];
            s += vals[t] * x[c];
        }
        return s;
    };

    // Hoist string checks for 2D east/west types
    const bool eastSym = (!is3D) && (boundaryContainerU.getBoundary(PointBoundary::E).type == BoundaryKind::Symmetry);
    const bool eastConv = (!is3D) && (boundaryContainerU.getBoundary(PointBoundary::E).type == BoundaryKind::Convective);
    const bool westSym = (!is3D) && (boundaryContainerU.getBoundary(PointBoundary::W).type == BoundaryKind::Symmetry);

    if (!is3D) {
        // ===================== 2D =====================
#pragma omp parallel for collapse(2) schedule(static)
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int k = 0;
                const int Eq = phi[k][j][i].getUniqueNumber();
                const char Case = boundaryLetter(phi[k][j][i].getBoundaryType());

                // Small fixed buffers (max 9 nnz in 2D cases here)
                int    cols[9];  double vals[9];  int m = 0;

                switch (Case)
                {
                    // ---------- East ----------
                case 'E': {
                    if (eastSym && j > 1 && j < ny - 2) {
                        std::vector<double> allVals; allVals.reserve(16);
                        auto Vn = symetry(k, j, i, u, v, w, false, 'r'); allVals.insert(allVals.end(), Vn.begin(), Vn.end());
                        Vn = symetry(k, j, i, u, v, w, false, 'l'); allVals.insert(allVals.end(), Vn.begin(), Vn.end());
                        Vn = symetry(k, j, i, u, v, w, false, 'u'); allVals.insert(allVals.end(), Vn.begin(), Vn.end());
                        Vn = symetry(k, j, i, u, v, w, false, 'd'); allVals.insert(allVals.end(), Vn.begin(), Vn.end());

                        const double cC = allVals[1] - allVals[6] + allVals[9] - allVals[14];
                        const double cD = allVals[8] - allVals[13];
                        const double cL = allVals[0] - allVals[5];
                        const double cR = allVals[2] - allVals[7];
                        const double cU = allVals[10] - allVals[15];

                        if (i == nx - 1) {
                            int tmpCols[9] = { Eq - 2 * nx, Eq - nx, Eq - 2, Eq - 1, Eq, Eq - nx + 1, Eq - nx + 2, Eq + nx, Eq + 2 * nx };
                            double tmpVals[9] = { -allVals[12], cD, -allVals[4], cL, cC, cR, allVals[3], cU, allVals[11] };
                            std::memcpy(cols, tmpCols, sizeof(tmpCols)); std::memcpy(vals, tmpVals, sizeof(tmpVals)); m = 9;
                        }
                        else {
                            int tmpCols[9] = { Eq - 2 * nx, Eq - nx, Eq - 2, Eq - 1, Eq, Eq + 1, Eq - nx + 1, Eq + nx, Eq + 2 * nx };
                            double tmpVals[9] = { -allVals[12], cD, -allVals[4], cL, cC, cR, allVals[3], cU, allVals[11] };
                            std::memcpy(cols, tmpCols, sizeof(tmpCols)); std::memcpy(vals, tmpVals, sizeof(tmpVals)); m = 9;
                        }
                    }
                    else if (eastConv) {
                        std::vector<double> a = calculateFluxFOU(k, j, i, u, v, w, false);
                        if (i == nx - 1) {
                            int tmpCols[4] = { Eq - nx, Eq - 1, Eq, Eq + nx };
                            double tmpVals[4] = { -a[13], -a[5], -a[6] + a[5] + a[9] - a[14], a[10] };
                            std::memcpy(cols, tmpCols, sizeof(tmpCols)); std::memcpy(vals, tmpVals, sizeof(tmpVals)); m = 4;
                        }
                        else {
                            int tmpCols[5] = { Eq - nx, Eq - 1, Eq, Eq + 1, Eq + nx };
                            double tmpVals[5] = { -a[13], -a[5], -a[6] + a[1] + a[9] - a[14], a[2], a[10] };
                            std::memcpy(cols, tmpCols, sizeof(tmpCols)); std::memcpy(vals, tmpVals, sizeof(tmpVals)); m = 5;
                        }
                    }
                    else { // default
                        std::vector<double> a = calculateFluxFOU(k, j, i, u, v, w, false);
                        if (i == nx - 1) {
                            int tmpCols[4] = { Eq - nx, Eq - 1, Eq, Eq + nx };
                            double tmpVals[4] = { -a[13], -a[5], -a[6] + a[1] + a[9] - a[14], a[10] };
                            std::memcpy(cols, tmpCols, sizeof(tmpCols)); std::memcpy(vals, tmpVals, sizeof(tmpVals)); m = 4;
                        }
                        else {
                            int tmpCols[5] = { Eq - nx, Eq - 1, Eq, Eq + 1, Eq + nx };
                            double tmpVals[5] = { -a[13], -a[5], -a[6] + a[1] + a[9] - a[14], a[2], a[10] };
                            std::memcpy(cols, tmpCols, sizeof(tmpCols)); std::memcpy(vals, tmpVals, sizeof(tmpVals)); m = 5;
                        }
                    }
                } break;

                    // ---------- West ----------
                case 'W': {
                    if (westSym && j > 1 && j < ny - 2) {
                        std::vector<double> allVals; allVals.reserve(16);
                        auto Vn = symetry(k, j, i, u, v, w, false, 'r'); allVals.insert(allVals.begin(), Vn.begin(), Vn.end());
                        Vn = symetry(k, j, i, u, v, w, false, 'l'); allVals.insert(allVals.end(), Vn.begin(), Vn.end());
                        Vn = symetry(k, j, i, u, v, w, false, 'u'); allVals.insert(allVals.end(), Vn.begin(), Vn.end());
                        Vn = symetry(k, j, i, u, v, w, false, 'd'); allVals.insert(allVals.end(), Vn.begin(), Vn.end());

                        const double cC = allVals[1] - allVals[6] + allVals[9] - allVals[14];
                        const double cD = allVals[8] - allVals[13];
                        const double cL = allVals[0] - allVals[5];
                        const double cR = allVals[2] - allVals[7];
                        const double cU = allVals[10] - allVals[15];

                        if (i == 1) {
                            int tmpCols[9] = { Eq - 2 * nx, Eq - nx, Eq + nx - 2, Eq + nx - 1, Eq, Eq + 1, Eq + 2, Eq + nx, Eq + 2 * nx };
                            double tmpVals[9] = { -allVals[12], cD, -allVals[4], cL, cC, cR, allVals[3], cU, allVals[11] };
                            std::memcpy(cols, tmpCols, sizeof(tmpCols)); std::memcpy(vals, tmpVals, sizeof(tmpVals)); m = 9;
                        }
                        else {
                            int tmpCols[9] = { Eq - 2 * nx, Eq - nx, Eq + nx - 1, Eq - 1, Eq, Eq + 1, Eq + 2, Eq + nx, Eq + 2 * nx };
                            double tmpVals[9] = { -allVals[12], cD, -allVals[4], cL, cC, cR, allVals[3], cU, allVals[11] };
                            std::memcpy(cols, tmpCols, sizeof(tmpCols)); std::memcpy(vals, tmpVals, sizeof(tmpVals)); m = 9;
                        }
                    }
                    else {
                        std::vector<double> a = calculateFluxFOU(k, j, i, u, v, w, false);
                        if (i == 0) {
                            int tmpCols[4] = { Eq - nx, Eq, Eq + 1, Eq + nx };
                            double tmpVals[4] = { -a[13], -a[6] + a[1] + a[9] - a[14], a[2], a[10] };
                            std::memcpy(cols, tmpCols, sizeof(tmpCols)); std::memcpy(vals, tmpVals, sizeof(tmpVals)); m = 4;
                        }
                        else {
                            int tmpCols[5] = { Eq - nx, Eq - 1, Eq, Eq + 1, Eq + nx };
                            double tmpVals[5] = { -a[13], -a[5], -a[6] + a[1] + a[9] - a[14], a[2], a[10] };
                            std::memcpy(cols, tmpCols, sizeof(tmpCols)); std::memcpy(vals, tmpVals, sizeof(tmpVals)); m = 5;
                        }
                    }
                } break;

                    // ---------- North ----------
                case 'N': {
                    std::vector<double> a = calculateFluxFOU(k, j, i, u, v, w, false);
                    int tmpCols[5] = { Eq - nx, Eq - 1, Eq, Eq + 1, 0 };
                    double tmpVals[5] = { -a[13], -a[5], -a[6] + a[1] + a[9] - a[14], a[2], 0.0 };
                    m = 4;
                    if (a[10] != 0.0 && (Eq + nx) < nxny) { tmpCols[m] = Eq + nx; tmpVals[m] = a[10]; ++m; }
                    std::memcpy(cols, tmpCols, m * sizeof(int)); std::memcpy(vals, tmpVals, m * sizeof(double));
                } break;

                    // ---------- South ----------
                case 'S': {
                    std::vector<double> a = calculateFluxFOU(k, j, i, u, v, w, false);
                    // base 4
                    int tmpCols[5] = { Eq - 1, Eq, Eq + 1, Eq + nx, 0 };
                    double tmpVals[5] = { -a[5], -a[6] + a[1] + a[9] - a[14], a[2], a[10], 0.0 };
                    m = 4;
                    // optional Eq-nx in front if used
                    if (a[13] != 0.0 && (Eq - nx) >= 0) {
                        // shift right 1
                        for (int t = m; t > 0; --t) { tmpCols[t] = tmpCols[t - 1]; tmpVals[t] = tmpVals[t - 1]; }
                        tmpCols[0] = Eq - nx; tmpVals[0] = -a[13]; ++m;
                    }
                    std::memcpy(cols, tmpCols, m * sizeof(int)); std::memcpy(vals, tmpVals, m * sizeof(double));
                } break;

                    // ---------- Corners ----------
                case 'D': { // (i==0, j==0)
                    std::vector<double> a = calculateFluxFOU(k, j, i, u, v, w, false);
                    int tmpCols[3] = { Eq, Eq + 1, Eq + nx };
                    double tmpVals[3] = { a[1] + a[9] - a[14], a[2], a[10] };
                    std::memcpy(cols, tmpCols, sizeof(tmpCols)); std::memcpy(vals, tmpVals, sizeof(tmpVals)); m = 3;
                } break;

                case 'T': { // (i==nx-1, j==0)
                    std::vector<double> a = calculateFluxFOU(k, j, i, u, v, w, false);
                    int tmpCols[3] = { Eq - 1, Eq, Eq + nx };
                    double tmpVals[3];
                    if (eastConv) {
                        double tv[3] = { -a[5], -a[6] + a[5] + a[9] - a[14], a[10] };
                        std::memcpy(tmpVals, tv, sizeof(tv));
                    }
                    else {
                        double tv[3] = { -a[5], -a[6] + a[9], a[10] };
                        std::memcpy(tmpVals, tv, sizeof(tv));
                    }
                    std::memcpy(cols, tmpCols, sizeof(tmpCols)); std::memcpy(vals, tmpVals, sizeof(tmpVals)); m = 3;
                } break;

                case 'U': { // (i==0, j==ny-1)
                    std::vector<double> a = calculateFluxFOU(k, j, i, u, v, w, false);
                    int tmpCols[3] = { Eq - nx, Eq, Eq + 1 };
                    double tmpVals[3] = { -a[13], a[1] - a[14], a[2] };
                    std::memcpy(cols, tmpCols, sizeof(tmpCols)); std::memcpy(vals, tmpVals, sizeof(tmpVals)); m = 3;
                } break;

                case 'Y': { // (i==nx-1, j==ny-1)
                    std::vector<double> a = calculateFluxFOU(k, j, i, u, v, w, false);
                    int tmpCols[3] = { Eq - nx, Eq - 1, Eq };
                    double tmpVals[3];
                    if (eastConv) {
                        double tv[3] = { -a[13], -a[5], -a[6] + a[5] + a[9] - a[14] };
                        std::memcpy(tmpVals, tv, sizeof(tv));
                    }
                    else {
                        double tv[3] = { -a[13], -a[5], -a[6] + a[9] - a[14] };
                        std::memcpy(tmpVals, tv, sizeof(tv));
                    }
                    std::memcpy(cols, tmpCols, sizeof(tmpCols)); std::memcpy(vals, tmpVals, sizeof(tmpVals)); m = 3;
                } break;

                    // ---------- Interior (default 2D) ----------
                default: {
                    // High-order scheme using calculateFluxSide
                    std::vector<double> al; al.reserve(24);
                    auto vtmp = calculateFluxSide(k, j, i, u, v, w, false, 'r'); al.insert(al.end(), vtmp.begin(), vtmp.end());
                    vtmp = calculateFluxSide(k, j, i, u, v, w, false, 'l'); al.insert(al.end(), vtmp.begin(), vtmp.end());
                    vtmp = calculateFluxSide(k, j, i, u, v, w, false, 'u'); al.insert(al.end(), vtmp.begin(), vtmp.end());
                    vtmp = calculateFluxSide(k, j, i, u, v, w, false, 'd'); al.insert(al.end(), vtmp.begin(), vtmp.end());

                    const double cC = al[1] - al[6] + al[9] - al[14];
                    const double cD = al[8] - al[13];
                    const double cL = al[0] - al[5];
                    const double cR = al[2] - al[7];
                    const double cU = al[10] - al[15];

                    int tmpCols[9] = { Eq - 2 * nx, Eq - nx, Eq - 2, Eq - 1, Eq, Eq + 1, Eq + 2, Eq + nx, Eq + 2 * nx };
                    double tmpVals[9] = { -al[12],   cD,     -al[4],   cL,    cC,  cR,    al[3],   cU,     al[11] };
                    std::memcpy(cols, tmpCols, sizeof(tmpCols)); std::memcpy(vals, tmpVals, sizeof(tmpVals)); m = 9;
                } break;
                }

                y[Eq] = dot_cols_vals(cols, vals, m);
            }
        }
    }
    else {
        // ===================== 3D =====================
#pragma omp parallel for collapse(3) schedule(static)
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    const int Eq = phi[k][j][i].getUniqueNumber();
                    const char Case = boundaryLetter(phi[k][j][i].getBoundaryType());

                    // Max ~13 terms for these stencils
                    int    cols[13]; double vals[13]; int m = 0;

                    switch (Case)
                    {
                        // ---------- Faces ----------
                    case 'E': { // east face (i boundary)
                        std::vector<double> a = calculateFluxFOU(k, j, i, u, v, w, true);
                        if (i == nx - 1) {
                            int tmpCols[6] = { Eq - nxny, Eq - nx, Eq - 1, Eq,        Eq + nx,  Eq + nxny };
                            double tmpVals[6] = { -a[21],   -a[13],  -a[5], -a[6] + a[9] - a[14] + a[17] - a[22], a[10], a[18] };
                            std::memcpy(cols, tmpCols, sizeof(tmpCols)); std::memcpy(vals, tmpVals, sizeof(tmpVals)); m = 6;
                        }
                        else {
                            int tmpCols[7] = { Eq - nxny, Eq - nx, Eq - 1, Eq,        Eq + 1,   Eq + nx,  Eq + nxny };
                            double tmpVals[7] = { -a[21],  -a[13],  -a[5], -a[6] + a[1] + a[9] - a[14] + a[17] - a[22], a[2], a[10], a[18] };
                            std::memcpy(cols, tmpCols, sizeof(tmpCols)); std::memcpy(vals, tmpVals, sizeof(tmpVals)); m = 7;
                        }
                    } break;

                    case 'W': { // west face
                        std::vector<double> a = calculateFluxFOU(k, j, i, u, v, w, true);
                        if (i == 0) {
                            int tmpCols[6] = { Eq - nxny,  Eq - nx,      Eq,       Eq + 1,    Eq + nx,  Eq + nxny };
                            double tmpVals[6] = { -a[21],   -a[13], -a[6] + a[1] + a[9] - a[14] + a[17] - a[22], a[2], a[10], a[18] };
                            std::memcpy(cols, tmpCols, sizeof(tmpCols)); std::memcpy(vals, tmpVals, sizeof(tmpVals)); m = 6;
                        }
                        else {
                            int tmpCols[7] = { Eq - nxny,  Eq - nx, Eq - 1,  Eq,        Eq + 1,  Eq + nx,  Eq + nxny };
                            double tmpVals[7] = { -a[21],   -a[13], -a[5], -a[6] + a[1] + a[9] - a[14] + a[17] - a[22], a[2], a[10], a[18] };
                            std::memcpy(cols, tmpCols, sizeof(tmpCols)); std::memcpy(vals, tmpVals, sizeof(tmpVals)); m = 7;
                        }
                    } break;

                    case 'N': { // north 
                        std::vector<double> a = calculateFluxFOU(k, j, i, u, v, w, true);
                        if (j == ny - 1) {
                            int tmpCols[6] = { Eq - nxny,  Eq - nx,  Eq - 1, Eq,         Eq + 1,    Eq + nxny };
                            double tmpVals[6] = { -a[21],   -a[13],  -a[5], -a[6] + a[1] - a[14] + a[17] - a[22], a[2], a[18] };
                            std::memcpy(cols, tmpCols, sizeof(tmpCols)); std::memcpy(vals, tmpVals, sizeof(tmpVals)); m = 6;
                        }
                        else {
                            int tmpCols[7] = { Eq - nxny,  Eq - nx,  Eq - 1, Eq,         Eq + 1,    Eq + nx,  Eq + nxny };
                            double tmpVals[7] = { -a[21],   -a[13],  -a[5], -a[6] + a[1] + a[9] - a[14] + a[17] - a[22], a[2], a[10], a[18] };
                            std::memcpy(cols, tmpCols, sizeof(tmpCols)); std::memcpy(vals, tmpVals, sizeof(tmpVals)); m = 7;
                        }
                    } break;

                    case 'S': { // south 
                        std::vector<double> a = calculateFluxFOU(k, j, i, u, v, w, true);
                        if (j == 0) {
                            int tmpCols[6] = { Eq - nxny,       Eq - 1,  Eq,         Eq + 1,   Eq + nx,  Eq + nxny };
                            double tmpVals[6] = { -a[21],        -a[5], -a[6] + a[1] + a[9] + a[17] - a[22], a[2], a[10], a[18] };
                            std::memcpy(cols, tmpCols, sizeof(tmpCols)); std::memcpy(vals, tmpVals, sizeof(tmpVals)); m = 6;
                        }
                        else {
                            int tmpCols[7] = { Eq - nxny,  Eq - nx,  Eq - 1,  Eq,         Eq + 1,  Eq + nx,  Eq + nxny };
                            double tmpVals[7] = { -a[21],   -a[13], -a[5], -a[6] + a[1] + a[9] - a[14] + a[17] - a[22], a[2], a[10], a[18] };
                            std::memcpy(cols, tmpCols, sizeof(tmpCols)); std::memcpy(vals, tmpVals, sizeof(tmpVals)); m = 7;
                        }
                    } break;

                    case 'B': { // back (k==0)
                        std::vector<double> a = calculateFluxFOU(k, j, i, u, v, w, true);
                        if (k == 0) {
                            int tmpCols[6] = { Eq - nx,  Eq - 1, Eq,         Eq + 1,   Eq + nx,  Eq + nxny };
                            double tmpVals[6] = { -a[13],    -a[5], -a[6] + a[1] + a[9] - a[14] + a[17], a[2], a[10], a[18] };
                            std::memcpy(cols, tmpCols, sizeof(tmpCols)); std::memcpy(vals, tmpVals, sizeof(tmpVals)); m = 6;
                        }
                        else {
                            int tmpCols[7] = { Eq - nxny, Eq - nx,  Eq - 1,  Eq,         Eq + 1,  Eq + nx,  Eq + nxny };
                            double tmpVals[7] = { -a[21],  -a[13], -a[5], -a[6] + a[1] + a[9] - a[14] + a[17] - a[22], a[2], a[10], a[18] };
                            std::memcpy(cols, tmpCols, sizeof(tmpCols)); std::memcpy(vals, tmpVals, sizeof(tmpVals)); m = 7;
                        }
                    } break;

                    case 'F': { // front (k==nz-1)
                        std::vector<double> a = calculateFluxFOU(k, j, i, u, v, w, true);
                        if (k == nz - 1) {
                            int tmpCols[6] = { Eq - nxny,  Eq - nx,  Eq - 1,  Eq,         Eq + 1,  Eq + nx };
                            double tmpVals[6] = { -a[21],   -a[13],  -a[5], -a[6] + a[1] + a[9] - a[14] - a[22], a[2], a[10] };
                            std::memcpy(cols, tmpCols, sizeof(tmpCols)); std::memcpy(vals, tmpVals, sizeof(tmpVals)); m = 6;
                        }
                        else {
                            int tmpCols[7] = { Eq - nxny,  Eq - nx,  Eq - 1,  Eq,         Eq + 1,  Eq + nx,  Eq + nxny };
                            double tmpVals[7] = { -a[21],   -a[13],  -a[5], -a[6] + a[1] + a[9] - a[14] + a[17] - a[22], a[2], a[10], a[18] };
                            std::memcpy(cols, tmpCols, sizeof(tmpCols)); std::memcpy(vals, tmpVals, sizeof(tmpVals)); m = 7;
                        }
                    } break;

                        // ---------- Edges ----------
                    case 'D': { // (South West i==0, j==0)
                        std::vector<double> a = calculateFluxFOU(k, j, i, u, v, w, true);

                        int tmpCols[5] = { Eq - nx * ny, Eq, Eq + 1,  Eq + nx, Eq + nx * ny };
                        double tmpVals[5] = { -a[21], a[1] + a[9] + a[17] - a[22],a[2],a[10],a[18]};
                        std::memcpy(cols, tmpCols, sizeof(tmpCols)); std::memcpy(vals, tmpVals, sizeof(tmpVals)); m = 5;
                    } break;

                    case 'T': { // (South East i==nx-1, j==0)
                        std::vector<double> a = calculateFluxFOU(k, j, i, u, v, w, true);

                        int tmpCols[5] = { Eq - nx * ny,Eq - 1, Eq,   Eq + nx,Eq + nx * ny };
                        double tmpVals[5] = { -a[21], - a[5], -a[6] + a[9] + a[17] - a[22], a[10],a[18]};

                        std::memcpy(cols, tmpCols, sizeof(tmpCols)); std::memcpy(vals, tmpVals, sizeof(tmpVals)); m = 5;
                    } break;

                    case 'U': { // (North-West i==0, j==ny-1)
                        std::vector<double> a = calculateFluxFOU(k, j, i, u, v, w, true);
                        int tmpCols[5] = { Eq - nx * ny, Eq - nx,  Eq, Eq + 1,Eq + nx * ny };
                        double tmpVals[5] = {-a[21], - a[13], a[1] - a[14]+a[17]-a[22], a[2], a[18]};
                        std::memcpy(cols, tmpCols, sizeof(tmpCols)); std::memcpy(vals, tmpVals, sizeof(tmpVals)); m = 5;
                    } break;

                    case 'Y': { // (North-East i==nx-1, j==ny-1)
                        std::vector<double> a = calculateFluxFOU(k, j, i, u, v, w, true);
                        int tmpCols[5] = { Eq - nx * ny,Eq - nx, Eq - 1, Eq,Eq + nx * ny };
                        double tmpVals[5] = { -a[21],-a[13], -a[5], -a[6] + a[9] - a[14] + a[17] - a[22],a[18]};

                        std::memcpy(cols, tmpCols, sizeof(tmpCols)); std::memcpy(vals, tmpVals, sizeof(tmpVals)); m = 5;
                    } break;

                    case 'A': { // West-Back (i==0,k==0)
                        std::vector<double> a = calculateFluxFOU(k, j, i, u, v, w, true);
                        int tmpCols[5] = { Eq - nx,   Eq,       Eq + 1,  Eq + nx,  Eq + nxny };
                        double tmpVals[5] = { -a[13], a[1] + a[9] - a[14] + a[17], a[2], a[10], a[18] };
                        std::memcpy(cols, tmpCols, sizeof(tmpCols)); std::memcpy(vals, tmpVals, sizeof(tmpVals)); m = 5;
                    } break;

                    case 'C': { // West-Front (i==0,k==nz-1)
                        std::vector<double> a = calculateFluxFOU(k, j, i, u, v, w, true);
                        int tmpCols[5] = { Eq - nxny,  Eq - nx,  Eq,       Eq + 1,  Eq + nx };
                        double tmpVals[5] = { -a[21],   -a[13], a[1] + a[9] - a[14] - a[22], a[2], a[10] };
                        std::memcpy(cols, tmpCols, sizeof(tmpCols)); std::memcpy(vals, tmpVals, sizeof(tmpVals)); m = 5;
                    } break;

                    case 'G': { // East-Back (i==nx-1,k==0)
                        std::vector<double> a = calculateFluxFOU(k, j, i, u, v, w, true);
                        int tmpCols[5] = { Eq - nx,   Eq - 1,  Eq,         Eq + nx,  Eq + nxny };
                        double tmpVals[5] = { -a[13],  -a[5], -a[6] + a[9] - a[14] + a[17], a[10], a[18] };
                        std::memcpy(cols, tmpCols, sizeof(tmpCols)); std::memcpy(vals, tmpVals, sizeof(tmpVals)); m = 5;
                    } break;

                    case 'H': { // East-Front (i==nx-1,k==nz-1)
                        std::vector<double> a = calculateFluxFOU(k, j, i, u, v, w, true);
                        int tmpCols[5] = { Eq - nxny,  Eq - nx,  Eq - 1,  Eq,         Eq + nx };
                        double tmpVals[5] = { -a[21],   -a[13],  -a[5], -a[6] + a[9] - a[14] - a[22], a[10] };
                        std::memcpy(cols, tmpCols, sizeof(tmpCols)); std::memcpy(vals, tmpVals, sizeof(tmpVals)); m = 5;
                    } break;

                    case 'R': { // South-Back (j==0,k==0)
                        std::vector<double> a = calculateFluxFOU(k, j, i, u, v, w, true);
                        int tmpCols[5] = { Eq - 1,  Eq,         Eq + 1,  Eq + nx,  Eq + nxny };
                        double tmpVals[5] = { -a[5], a[1] - a[6] + a[9] + a[17], a[2], a[10], a[18] };
                        std::memcpy(cols, tmpCols, sizeof(tmpCols)); std::memcpy(vals, tmpVals, sizeof(tmpVals)); m = 5;
                    } break;

                    case 'V': { // South-Front (j==0,k==nz-1)
                        std::vector<double> a = calculateFluxFOU(k, j, i, u, v, w, true);
                        int tmpCols[5] = { Eq - nxny,  Eq - 1,  Eq,         Eq + 1,  Eq + nx };
                        double tmpVals[5] = { -a[21],   -a[5],  a[1] - a[6] + a[9] - a[22], a[2], a[10] };
                        std::memcpy(cols, tmpCols, sizeof(tmpCols)); std::memcpy(vals, tmpVals, sizeof(tmpVals)); m = 5;
                    } break;

                    case 'Z': { // North-Back (j==ny-1,k==0)
                        std::vector<double> a = calculateFluxFOU(k, j, i, u, v, w, true);
                        int tmpCols[5] = { Eq - nx,  Eq - 1,  Eq,         Eq + 1,  Eq + nxny };
                        double tmpVals[5] = { -a[13], -a[5], a[1] - a[6] - a[14] + a[17], a[2], a[18] };
                        std::memcpy(cols, tmpCols, sizeof(tmpCols)); std::memcpy(vals, tmpVals, sizeof(tmpVals)); m = 5;
                    } break;

                    case 'X': { // North-Front (j==ny-1,k==nz-1)
                        std::vector<double> a = calculateFluxFOU(k, j, i, u, v, w, true);
                        int tmpCols[5] = { Eq - nxny,  Eq - nx,  Eq - 1,  Eq,         Eq + 1 };
                        double tmpVals[5] = { -a[21],   -a[13],  -a[5], a[1] - a[6] - a[14] - a[22], a[2] };
                        std::memcpy(cols, tmpCols, sizeof(tmpCols)); std::memcpy(vals, tmpVals, sizeof(tmpVals)); m = 5;
                    } break;

                        // ---------- Corners ----------
                    case 'I': { // West-South-Bottom (i==0,j==0,k==0)
                        std::vector<double> a = calculateFluxFOU(k, j, i, u, v, w, true);
                        int tmpCols[4] = { Eq,    Eq + 1,  Eq + nx,  Eq + nxny };
                        double tmpVals[4] = { a[1] + a[9] + a[17], a[2], a[10], a[18] };
                        std::memcpy(cols, tmpCols, sizeof(tmpCols)); std::memcpy(vals, tmpVals, sizeof(tmpVals)); m = 4;
                    } break;

                    case 'J': { // West-North-Bottom (i==0,j==ny-1,k==0)
                        std::vector<double> a = calculateFluxFOU(k, j, i, u, v, w, true);
                        int tmpCols[4] = { Eq - nx,  Eq,        Eq + 1,  Eq + nxny };
                        double tmpVals[4] = { -a[13], a[1] - a[14] + a[17], a[2], a[18] };
                        std::memcpy(cols, tmpCols, sizeof(tmpCols)); std::memcpy(vals, tmpVals, sizeof(tmpVals)); m = 4;
                    } break;

                    case 'K': { // West-South-Top (i==0,j==0,k==nz-1)
                        std::vector<double> a = calculateFluxFOU(k, j, i, u, v, w, true);
                        int tmpCols[4] = { Eq - nxny,  Eq,        Eq + 1,  Eq + nx };
                        double tmpVals[4] = { -a[21],   a[1] + a[9] - a[22], a[2], a[10] };
                        std::memcpy(cols, tmpCols, sizeof(tmpCols)); std::memcpy(vals, tmpVals, sizeof(tmpVals)); m = 4;
                    } break;

                    case 'L': { // West-North-Top (i==0,j==ny-1,k==nz-1)
                        std::vector<double> a = calculateFluxFOU(k, j, i, u, v, w, true);
                        int tmpCols[4] = { Eq - nxny,  Eq - nx,  Eq,        Eq + 1 };
                        double tmpVals[4] = { -a[21],   -a[13],  a[1] - a[14] - a[22], a[2] };
                        std::memcpy(cols, tmpCols, sizeof(tmpCols)); std::memcpy(vals, tmpVals, sizeof(tmpVals)); m = 4;
                    } break;

                    case 'M': { // East-South-Bottom (i==nx-1,j==0,k==0)
                        std::vector<double> a = calculateFluxFOU(k, j, i, u, v, w, true);
                        int tmpCols[4] = { Eq - 1,  Eq,         Eq + nx,  Eq + nxny };
                        double tmpVals[4] = { -a[5], -a[6] + a[9] + a[17], a[10], a[18] };
                        std::memcpy(cols, tmpCols, sizeof(tmpCols)); std::memcpy(vals, tmpVals, sizeof(tmpVals)); m = 4;
                    } break;

                    case 'O': { // East-North-Bottom (i==nx-1,j==ny-1,k==0)
                        std::vector<double> a = calculateFluxFOU(k, j, i, u, v, w, true);
                        int tmpCols[4] = { Eq - nx,  Eq - 1,  Eq,         Eq + nxny };
                        double tmpVals[4] = { -a[13], -a[5], -a[6] - a[14] + a[17], a[18] };
                        std::memcpy(cols, tmpCols, sizeof(tmpCols)); std::memcpy(vals, tmpVals, sizeof(tmpVals)); m = 4;
                    } break;

                    case 'P': { // East-South-Top (i==nx-1,j==0,k==nz-1)
                        std::vector<double> a = calculateFluxFOU(k, j, i, u, v, w, true);
                        int tmpCols[4] = { Eq - nxny,  Eq - 1,  Eq,         Eq + nx };
                        double tmpVals[4] = { -a[21],   -a[5], -a[6] + a[9] - a[22], a[10] };
                        std::memcpy(cols, tmpCols, sizeof(tmpCols)); std::memcpy(vals, tmpVals, sizeof(tmpVals)); m = 4;
                    } break;

                    case 'Q': { // East-North-Top (i==nx-1,j==ny-1,k==nz-1)
                        std::vector<double> a = calculateFluxFOU(k, j, i, u, v, w, true);
                        int tmpCols[4] = { Eq - nxny,  Eq - nx,  Eq - 1,  Eq };
                        double tmpVals[4] = { -a[21],   -a[13],  -a[5], -a[6] - a[14] - a[22] };
                        std::memcpy(cols, tmpCols, sizeof(tmpCols)); std::memcpy(vals, tmpVals, sizeof(tmpVals)); m = 4;
                    } break;

                        // ---------- Interior (default 3D) ----------
                    default: {
                        std::vector<double> al; al.reserve(32);
                        auto vt = calculateFluxSide(k, j, i, u, v, w, true, 'r'); al.insert(al.end(), vt.begin(), vt.end());
                        vt = calculateFluxSide(k, j, i, u, v, w, true, 'l'); al.insert(al.end(), vt.begin(), vt.end());
                        vt = calculateFluxSide(k, j, i, u, v, w, true, 'u'); al.insert(al.end(), vt.begin(), vt.end());
                        vt = calculateFluxSide(k, j, i, u, v, w, true, 'd'); al.insert(al.end(), vt.begin(), vt.end());
                        vt = calculateFluxSide(k, j, i, u, v, w, true, 'f'); al.insert(al.end(), vt.begin(), vt.end());
                        vt = calculateFluxSide(k, j, i, u, v, w, true, 'b'); al.insert(al.end(), vt.begin(), vt.end());

                        const double cC = al[1] - al[6] + al[9] - al[14] + (al[17] - al[22]);
                        const double cD = al[8] - al[13];
                        const double cL = al[0] - al[5];
                        const double cR = al[2] - al[7];
                        const double cU = al[10] - al[15];
                        const double cBk = al[16] - al[21];
                        const double cFr = al[18] - al[23];

                        int tmpCols[13] = { Eq - 2 * nxny, Eq - nxny, Eq - 2 * nx, Eq - nx,  Eq - 2, Eq - 1,
                                             Eq,          Eq + 1,  Eq + 2,   Eq + nx,  Eq + 2 * nx,  Eq + nxny,  Eq + 2 * nxny };
                        double tmpVals[13] = { -al[20],    cBk,       -al[12],   cD,      -al[4],  cL,
                                              cC,         cR,       al[3],    cU,       al[11],    cFr,        al[19] };
                        std::memcpy(cols, tmpCols, sizeof(tmpCols)); std::memcpy(vals, tmpVals, sizeof(tmpVals)); m = 13;
                    } break;
                    }

                    y[Eq] = dot_cols_vals(cols, vals, m);
                }
            }
        }
    }

    return y;
}


std::vector<double> VOF::symetry(int k, int j, int i,
    const std::vector<std::vector<std::vector<Point>>>& u, const std::vector<std::vector<std::vector<Point>>>& v, const std::vector<std::vector<std::vector<Point>>>& w,
    bool is3D, char side) const{
    std::vector<double> phi_flux_OneD(4, 0.0); // Initialize phi_flux vector with zeros
    std::vector<bool>signs;

    double phi_telda1;
    double dot_px1;
    signs = checkDirection(k, j, i, u, v, w, is3D);
    bool sign;
    double velocity;
    double face_area = 0.0;

    size_t height = v[0].size();
    size_t width = u[0][0].size();

    int nx = phi[0][0].size();
    int ny = phi[0].size();
    double in_cell, dwind_cell, upwd_cell;
    double dx = phi[(is3D ? k : 0)][j][i].getdx();
    double dy = phi[(is3D ? k : 0)][j][i].getdy();
    double dz = phi[(is3D ? k : 0)][j][i].getdz();


    switch (side)
    {
    case 'r':

        velocity = u[(is3D ? k + 1 : 0)][j + 1][i + 1].getValue();
        sign = signs[2];
        face_area = (is3D ? dz * dy : dy);
        if (sign == true) {
            if (i == nx - 1) {
                in_cell = phi[(is3D ? k : 0)][j][nx - 1].getValue();
                dwind_cell = phi[(is3D ? k : 0)][j][nx - 2].getValue();
                upwd_cell = phi[(is3D ? k : 0)][j][0].getValue();
            }
            else if (i == 0) {
                in_cell = phi[(is3D ? k : 0)][j][0].getValue();
                dwind_cell = phi[(is3D ? k : 0)][j][width - 2].getValue();
                upwd_cell = phi[(is3D ? k : 0)][j][1].getValue();

            }
            else {
                in_cell = phi[(is3D ? k : 0)][j][i].getValue();
                dwind_cell = phi[(is3D ? k : 0)][j][i - 1].getValue();
                upwd_cell = phi[(is3D ? k : 0)][j][i + 1].getValue();
            }

        }
        else {
            if (i == nx - 1) {
                in_cell = phi[(is3D ? k : 0)][j][0].getValue();
                dwind_cell = phi[(is3D ? k : 0)][j][1].getValue();
                upwd_cell = phi[(is3D ? k : 0)][j][nx - 1].getValue();
            }
            else if (i == nx - 2) {
                in_cell = phi[(is3D ? k : 0)][j][nx - 1].getValue();
                dwind_cell = phi[(is3D ? k : 0)][j][0].getValue();
                upwd_cell = phi[(is3D ? k : 0)][j][nx - 2].getValue();

            }
            else {
                in_cell = phi[(is3D ? k : 0)][j][i + 1].getValue();
                dwind_cell = phi[(is3D ? k : 0)][j][i + 2].getValue();
                upwd_cell = phi[(is3D ? k : 0)][j][i].getValue();

            }
        }
        break;
    case 'l':
        velocity = u[(is3D ? k + 1 : 0)][j + 1][i].getValue();
        sign = signs[1];
        face_area = (is3D ? dz * dy : dy);
        if (sign == true) {
            if (i == 0) {
                in_cell = phi[(is3D ? k : 0)][j][nx - 1].getValue();
                dwind_cell = phi[(is3D ? k : 0)][j][nx - 2].getValue();
                upwd_cell = phi[(is3D ? k : 0)][j][0].getValue();

            }
            else if (i == 1) {
                in_cell = phi[(is3D ? k : 0)][j][0].getValue();
                dwind_cell = phi[(is3D ? k : 0)][j][nx - 1].getValue();
                upwd_cell = phi[(is3D ? k : 0)][j][1].getValue();

            }
            else {
                in_cell = phi[(is3D ? k : 0)][j][i - 1].getValue();
                dwind_cell = phi[(is3D ? k : 0)][j][i - 2].getValue();
                upwd_cell = phi[(is3D ? k : 0)][j][i].getValue();
            }

        }
        else {
            if (i == nx - 1) {
                in_cell = phi[(is3D ? k : 0)][j][nx - 1].getValue();
                dwind_cell = phi[(is3D ? k : 0)][j][0].getValue();
                upwd_cell = phi[(is3D ? k : 0)][j][nx - 2].getValue();
            }
            else if (i == 0) {
                in_cell = phi[(is3D ? k : 0)][j][0].getValue();
                dwind_cell = phi[(is3D ? k : 0)][j][1].getValue();
                upwd_cell = phi[(is3D ? k : 0)][j][nx - 1].getValue();
            }
            else {
                in_cell = phi[(is3D ? k : 0)][j][i].getValue();
                dwind_cell = phi[(is3D ? k : 0)][j][i + 1].getValue();
                upwd_cell = phi[(is3D ? k : 0)][j][i - 1].getValue();
            }
        }


        break;
    case 'u':
        velocity = v[(is3D ? k + 1 : 0)][j + 1][i + 1].getValue();
        sign = signs[3];
        face_area = (is3D ? dz * dx : dx);
        if (sign == true) {
            in_cell = phi[(is3D ? k : 0)][j][i].getValue();
            dwind_cell = (j == 0) ? phi[(is3D ? k : 0)][j][i].getValue() : phi[(is3D ? k : 0)][j - 1][i].getValue();
            upwd_cell = (j == ny - 1) ? phi[(is3D ? k : 0)][j][i].getValue() : phi[(is3D ? k : 0)][j + 1][i].getValue();
        }
        else {
            in_cell = (j == ny - 1 || j == ny - 2) ? phi[(is3D ? k : 0)][j][i].getValue() : phi[(is3D ? k : 0)][j + 1][i].getValue();
            dwind_cell = (j == ny - 1 || j == ny - 2) ? phi[(is3D ? k : 0)][j][i].getValue() : phi[(is3D ? k : 0)][j + 2][i].getValue();
            upwd_cell = phi[(is3D ? k : 0)][j][i].getValue();

        }
        break;
    case 'd':
        velocity = v[(is3D ? k + 1 : 0)][j][i + 1].getValue();
        sign = signs[0];
        face_area = (is3D ? dz * dx : dx);
        if (sign == true) {
            in_cell = (j == 0 || j == 1) ? phi[(is3D ? k : 0)][j][i].getValue() : phi[(is3D ? k : 0)][j - 1][i].getValue();
            dwind_cell = (j == 0 || j == 1) ? phi[(is3D ? k : 0)][j][i].getValue() : phi[(is3D ? k : 0)][j - 2][i].getValue();
            upwd_cell = phi[(is3D ? k : 0)][j][i].getValue();
        }
        else {
            in_cell = phi[(is3D ? k : 0)][j][i].getValue();
            dwind_cell = (j == ny - 1) ? phi[(is3D ? k : 0)][j][i].getValue() : phi[(is3D ? k : 0)][j + 1][i].getValue();
            upwd_cell = phi[(is3D ? k : 0)][j - 1][i].getValue();
        }
        break;
        /*
          case 'f'://face
               velocity = w[k + 1][j + 1][i + 1].getValue();
               sign = signs[5];
               face_area = dx * dy;
               if (w[k + 1][j + 1][i + 1].getValue() >= 0.0) {
                   dot_px1 = (phi[k + 1][j][i].getPrevValue() - phi[k - 1][j][i].getPrevValue());
                   phi_telda1 = (dot_px1 == 0) ? 1.0 : (phi[k][j][i].getPrevValue() - phi[k - 1][j][i].getPrevValue()) / dot_px1;
               }
               else {
                   dot_px1 = (phi[k][j][i].getPrevValue() - phi[k + 2][j][i].getPrevValue());
                   phi_telda1 = (dot_px1 == 0) ? 1.0 : (phi[k + 1][j][i].getPrevValue() - phi[k + 2][j][i].getPrevValue()) / dot_px1;
               }
               break;
           case 'b'://back
               velocity = w[k ][j + 1][i + 1].getValue();
               sign = signs[6];
               face_area = dx * dy;
               if (w[k][j + 1][i + 1].getValue() >= 0.0) {
                   dot_px1 = (phi[k][j][i].getPrevValue() - phi[k - 2][j][i].getPrevValue());
                   phi_telda1 = (dot_px1 == 0) ? 1.0 : (phi[k - 1][j][i].getPrevValue() - phi[k - 2][j][i].getPrevValue()) / dot_px1;
               }
               else {
                   dot_px1 = (phi[k - 1][j][i].getPrevValue() - phi[k + 1][j][i].getPrevValue());
                   phi_telda1 = (dot_px1 == 0) ? 1.0 : (phi[k][j][i].getPrevValue() - phi[k + 1][j][i].getPrevValue()) / dot_px1;
               }
               break;*/
    default:
        std::cout << "Uknown cell boundary in symetry boundary conditions, Phi calculation" << std::endl;
        break;
    }
    // Calculate phi_telda1

    dot_px1 = (upwd_cell - dwind_cell);
    phi_telda1 = (dot_px1 == 0) ? 1.0 : (in_cell - dwind_cell) / dot_px1;

    if (sign == true) {
        if (phi_telda1 > 0.0 && phi_telda1 <= 2.0 / 13.0) {
            phi_flux_OneD[1] = 3.0 * velocity * face_area;
            phi_flux_OneD[0] = -2.0 * velocity * face_area;
        }
        else if (phi_telda1 > 2.0 / 13.0 && phi_telda1 <= 0.8) {
            phi_flux_OneD[1] = 5.0 / 6.0 * velocity * face_area;
            phi_flux_OneD[0] = -1.0 / 6.0 * velocity * face_area;
            phi_flux_OneD[2] = 1.0 / 3.0 * velocity * face_area;
        }
        else if (phi_telda1 > 0.8 && phi_telda1 <= 1.0) {
            phi_flux_OneD[2] = velocity * face_area;
        }
        else {
            phi_flux_OneD[1] = velocity * face_area;
        }
    }
    else {
        if (phi_telda1 > 0.0 && phi_telda1 <= 2.0 / 13.0) {
            phi_flux_OneD[2] = 3.0 * velocity * face_area;
            phi_flux_OneD[3] = -2.0 * velocity * face_area;
        }
        else if (phi_telda1 > 2.0 / 13.0 && phi_telda1 <= 0.8) {
            phi_flux_OneD[1] = 1.0 / 3.0 * velocity * face_area;
            phi_flux_OneD[2] = 5.0 / 6.0 * velocity * face_area;
            phi_flux_OneD[3] = -1.0 / 6.0 * velocity * face_area;
        }
        else if (phi_telda1 > 0.8 && phi_telda1 <= 1.0) {
            phi_flux_OneD[1] = velocity * face_area;
        }
        else {
            phi_flux_OneD[2] = velocity * face_area;
        }
    }
    return phi_flux_OneD;
}


void VOF::saveMatrix() {
    std::ofstream file("matrix_VOF.dat");



    if (file.is_open()) {
        for (int i = 0; i < LHS.getSize(); ++i) {
            for (int j = LHS.getRowPtr()[i]; j < LHS.getRowPtr()[i + 1]; ++j) {
                file << i + 1 << " "
                    << LHS.getColIndices()[j] + 1 << " "
                    << std::scientific << std::setprecision(4)
                    << LHS.getValues()[j] << "\n";

                int col = LHS.getColIndices()[j];
                if (col < 0) {
                    std::cerr << "Error: Negative column index at j=" << j
                        << ", row=" << i
                        << ", col=" << col << std::endl;
                }
            }
        }
        file.close();
    }
    else {
        std::cerr << "Unable to open file for writing." << std::endl;
    }

}

void VOF::innerLoopStep(std::vector<double> phiVetor)
{
    int nx = phi[0][0].size();
    int ny = phi[0].size();
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++)
        {
            int k = 0;
            int Eq = phi[k][j][i].getUniqueNumber();
            phi[k][j][i].setValue((abs(phiVetor.at(Eq)) > 10e-12) ? phiVetor.at(Eq) : 0.0);

        }
    }
    calculateKappa(2,1E-10,1,2.0/3.0);
}
void VOF::writecomputebubble3D(double time, double comX, double comY, double comZ, double meanVelocity, double avgvelocity, double massLoss) {
    if (time == 0.0) {
        std::remove("bubble_dynamics.dat");
    }



    std::ofstream tecplotFile("bubble_dynamics.dat", std::ios::app);

    // Check if the file was successfully opened
    if (!tecplotFile.is_open()) {
        std::cerr << "Error opening Tecplot file." << std::endl;
        return;
    }

    // If the file is empty (first time writing), add the header
    if (time == 0.0) {

        tecplotFile << "VARIABLES = \"Time\",\"comX\",\"comY\",\"comZ\",\"meanVelocity\",\"AvgVelocity\", \"Mass Leakage\"\n";
        tecplotFile << "ZONE T=\"Mass Leakage Data\", I=6, F=POINT\n";
    }

    // Write the current time and mass leakage data
    tecplotFile << std::fixed << std::setprecision(15) << time << "\t" << comX << "\t" << comY << "\t" << comZ << "\t" << meanVelocity << "\t" << avgvelocity << "\t" << massLoss << "\n";

    // Close the file
    tecplotFile.close();
}

std::tuple<double, double, double, double, double, double>
VOF::computeBubbleProperties(double time,
    const std::vector<std::vector<std::vector<Point>>>& u,
    const std::vector<std::vector<std::vector<Point>>>& v,
    const std::vector<std::vector<std::vector<Point>>>& w,
    double dt, double phase) {

    double xMoment = 0.0, yMoment = 0.0, zMoment = 0.0, mass_er = 0.0;
    double weightedVelocitySumX = 0.0, weightedVelocitySumY = 0.0, weightedVelocitySumZ = 0.0;
    double totalVolume = 0.0, volume = 0.0, surfaceArea = 0.0;

    bool is3D = (phi.size() > 2);
    int nz = is3D ? phi.size() : 1;
    int ny = phi[0].size();
    int nx = phi[0][0].size();

    double comX = -1.0, comY = -1.0, comZ = -1.0;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                double value = std::abs((1.0 - phase) - phi[k][j][i].getValue());
                double dx = phi[k][j][i].getdx();
                double dy = phi[k][j][i].getdy();
                double dz = phi[k][j][i].getdz();
                double vol = dx * dy * ((nz > 1) ? dz : 1);


                double xCoord = phi[k][j][i].getX();
                double yCoord = phi[k][j][i].getY();
                double zCoord = phi[k][j][i].getZ();

                xMoment += value * vol * xCoord;
                yMoment += value * vol * yCoord;
                zMoment += value * vol * zCoord;



                double U = 0.5 * (u[k][j][i].getValue() + u[k][j][i + 1].getValue());
                double V = 0.5 * (v[k][j][i].getValue() + v[k][j + 1][i].getValue());
                double W = 0.0;
                if (is3D) {
                    if (k + 1 < (int)w.size()) {
                        W = 0.5 * (w[k][j][i].getValue() + w[k + 1][j][i].getValue());
                    }
                    else {
                        W = w[k][j][i].getValue(); // fallback if w has only nz planes
                    }
                }
                weightedVelocitySumX += U * value * vol;
                weightedVelocitySumY += V * value * vol;
                weightedVelocitySumZ += W * value * vol;
                totalVolume += value * vol;
            }
        }
    }
    comX = (totalVolume > 0) ? xMoment / totalVolume : -1.0;
    comY = (totalVolume > 0) ? yMoment / totalVolume : -1.0;
    comZ = (totalVolume > 0) ? zMoment / totalVolume : -1.0;

    if (time == 0.0) {
        prevComX = comX;
        prevComY = comY;
        prevComZ = comZ;
    }

    double velocityX = (comX - prevComX) / dt;
    double velocityY = (comY - prevComY) / dt;
    double velocityZ = (comZ - prevComZ) / dt;

    double UCx = (totalVolume > 0) ? weightedVelocitySumX / totalVolume : 0.0;
    double UCy = (totalVolume > 0) ? weightedVelocitySumY / totalVolume : 0.0;
    double UCz = (totalVolume > 0) ? weightedVelocitySumZ / totalVolume : 0.0;

    double meanVelocity = (totalVolume > 0) ? std::sqrt(UCx * UCx + UCy * UCy + UCz * UCz) : 0.0;

    double massLoss = (initialMass > 0) ? (initialMass - totalVolume) / initialMass : 0.0;

    // Update previous center of mass
    prevComX = comX;
    prevComY = comY;
    prevComZ = comZ;

    // Write data to file
    writecomputebubble3D(time, comX, comY, comZ, meanVelocity, UCy, massLoss);

    return std::make_tuple(comX, comY, comZ, meanVelocity, UCy, massLoss);
}

void VOF::calculateKappa(int smoothingPhi, double small, int smoothingKappa, double alphaKappa)
{
    const bool is3D = (phi.size() > 2);
    const int nz = is3D ? (int)phi.size() : 1;
    const int ny = (int)phi[0].size();
    const int nx = (int)phi[0][0].size();

    // Make sure member kappa exists and sized correctly
    this->kappa.assign(nz, std::vector<std::vector<double>>(ny, std::vector<double>(nx, 0.0)));
    this->phiS.assign(nz, std::vector<std::vector<double>>(ny, std::vector<double>(nx, 0.0)));

    // =======================
    // 1) Allocate all buffers
    // =======================
    // Smoothed phi buffers (need 2 buffers for Jacobi-type smoothing)
    std::vector<std::vector<std::vector<double>>> phiS2(
        nz, std::vector<std::vector<double>>(ny, std::vector<double>(nx, 0.0))
    );

    // Normals n = grad(phiS)/|grad(phiS)| (set to 0 where |grad| < small)
    std::vector<std::vector<std::vector<double>>> nX(
        nz, std::vector<std::vector<double>>(ny, std::vector<double>(nx, 0.0))
    );
    std::vector<std::vector<std::vector<double>>> nY(
        nz, std::vector<std::vector<double>>(ny, std::vector<double>(nx, 0.0))
    );
    std::vector<std::vector<std::vector<double>>> nZ(
        nz, std::vector<std::vector<double>>(ny, std::vector<double>(nx, 0.0))
    );

    // Temporary for kappa smoothing (cannot do neighbour averaging safely in-place)
    std::vector<std::vector<std::vector<double>>> kappaTmp(
        nz, std::vector<std::vector<double>>(ny, std::vector<double>(nx, 0.0))
    );

    // ==================================================
    // 2) Copy phi to phiS (phiS initial = current phi)
    // ==================================================
#pragma omp parallel for collapse(3) schedule(static)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                phiS[k][j][i] = phi[k][j][i].getValue();
            }
        }
    }

    // ==========================
    // 3) Smooth phi (Jacobi-like)
    // ==========================
    // Pointer toggle to avoid swap/extra allocations
    auto* srcPhi = &phiS;
    auto* dstPhi = &phiS2;

    for (int p = 0; p < smoothingPhi; ++p) {

        // Copy boundaries through unchanged (optional but safe)
#pragma omp parallel for collapse(3) schedule(static)
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    if ((!periodicX && (i == 0 || i == nx - 1)) || j == 0 || j == ny - 1 || (is3D && (k == 0 || k == nz - 1))) {
                        (*dstPhi)[k][j][i] = (*srcPhi)[k][j][i];
                    }
                }
            }
        }

        const int kBeg = is3D ? 1 : 0;
        const int kEnd = is3D ? (nz - 1) : 1;

#pragma omp parallel for collapse(3) schedule(static)
        for (int k = kBeg; k < kEnd; ++k) {
            for (int j = 1; j < ny - 1; ++j) {
                for (int i = periodicX ? 0 : 1; i < (periodicX ? nx : nx - 1); ++i) {
                    double sum = (*srcPhi)[k][j][i]
                        + (*srcPhi)[k][j][periodicX ? (i + 1) % nx : i + 1]
                        + (*srcPhi)[k][j][periodicX ? (i + nx - 1) % nx : i - 1]
                        + (*srcPhi)[k][j + 1][i] + (*srcPhi)[k][j - 1][i];

                    double denom = 5.0;

                    if (is3D) {
                        sum += (*srcPhi)[k + 1][j][i] + (*srcPhi)[k - 1][j][i];
                        denom = 7.0;
                    }

                    (*dstPhi)[k][j][i] = sum / denom;
                }
            }
        }

        // toggle pointers (no swap of containers)
        auto* tmp = srcPhi;
        srcPhi = dstPhi;
        dstPhi = tmp;
    }
    if (srcPhi != &phiS) {
        phiS.swap(phiS2);   // now phiS always contains the final smoothed field
    }
    // ==========================
    // 4) Compute normals from phi
    // ==========================
    // n = grad(phiS)/|grad(phiS)|, but only where |grad| >= small.
    // Else keep n = 0 (already initialised).
    const int kBegN = is3D ? 1 : 0;
    const int kEndN = is3D ? (nz - 1) : 1;

#pragma omp parallel for collapse(3) schedule(static)
    for (int k = kBegN; k < kEndN; ++k) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = periodicX ? 0 : 1; i < (periodicX ? nx : nx - 1); ++i) {

                const int kk = is3D ? k : 0;

                const double dx2 = periodicX ? 2.0 * phi[kk][j][i].getdx() :
                    phi[kk][j][i + 1].getX() - phi[kk][j][i - 1].getX();
                const double dy2 = phi[kk][j + 1][i].getY() - phi[kk][j - 1][i].getY();

                const double dphix = ((*srcPhi)[k][j][periodicX ? (i + 1) % nx : i + 1] -
                    (*srcPhi)[k][j][periodicX ? (i + nx - 1) % nx : i - 1]) / dx2;
                const double dphiy = ((*srcPhi)[k][j + 1][i] - (*srcPhi)[k][j - 1][i]) / dy2;

                double dphiz = 0.0;
                if (is3D) {
                    const double dz2 = phi[k + 1][j][i].getZ() - phi[k - 1][j][i].getZ();
                    dphiz = ((*srcPhi)[k + 1][j][i] - (*srcPhi)[k - 1][j][i]) / dz2;
                }

                const double g2 = dphix * dphix + dphiy * dphiy + dphiz * dphiz;
                const double g = std::sqrt(g2 + DBL_EPSILON*DBL_EPSILON);


                    nX[k][j][i] = dphix / g;
                    nY[k][j][i] = dphiy / g;
                    if (is3D) nZ[k][j][i] = dphiz / g;
                // else: keep zeros
            }
        }
    }
    // ==========================
    // 5) Curvature: kappa = -div(n)
    // ==========================
    // Write raw curvature directly into this->kappa (no redundant kappaRaw).
    const int kBegK = is3D ? 1 : 0;
    const int kEndK = is3D ? (nz - 1) : 1;

#pragma omp parallel for collapse(3) schedule(static)
    for (int k = kBegK; k < kEndK; ++k) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = periodicX ? 0 : 1; i < (periodicX ? nx : nx - 1); ++i) {

                const int kk = is3D ? k : 0;
                if (phiS[k][j][i] > small && phiS[k][j][i] < 1 - small) {

                    const double dx2 = periodicX ? 2.0 * phi[kk][j][i].getdx() :
                        phi[kk][j][i + 1].getX() - phi[kk][j][i - 1].getX();
                    const double dy2 = phi[kk][j + 1][i].getY() - phi[kk][j - 1][i].getY();

                    const double dnx_dx = (nX[k][j][periodicX ? (i + 1) % nx : i + 1] -
                        nX[k][j][periodicX ? (i + nx - 1) % nx : i - 1]) / dx2;
                    const double dny_dy = (nY[k][j + 1][i] - nY[k][j - 1][i]) / dy2;

                    double dnz_dz = 0.0;
                    if (is3D) {
                        const double dz2 = phi[k + 1][j][i].getZ() - phi[k - 1][j][i].getZ();
                        dnz_dz = (nZ[k + 1][j][i] - nZ[k - 1][j][i]) / dz2;
                    }

                    const double divn = dnx_dx + dny_dy + dnz_dz;

                    // IMPORTANT sign convention:
                    // kappa = -div(n) tends to give the "expected" sign for phi=1 inside, 0 outside
                    this->kappa[k][j][i] = -divn;
                }
            }
        }
    }

    // ==========================
    // 6) Optional smoothing of kappa
    // ==========================
    // This is purely noise control. Even if phi is smoothed, kappa smoothing often helps
    // for sharp algebraic VOF because div(n) amplifies local noise.
    // If you do not want it: pass smoothingKappa=0.
    if (smoothingKappa > 0) {

        auto* srcK = &this->kappa;
        auto* dstK = &kappaTmp;

        for (int pass = 0; pass < smoothingKappa; ++pass) {

            // Copy boundaries through unchanged (or set to 0 if you prefer)
#pragma omp parallel for collapse(3) schedule(static)
            for (int k = 0; k < nz; ++k) {
                for (int j = 0; j < ny; ++j) {
                    for (int i = 0; i < nx; ++i) {
                        if ((!periodicX && (i == 0 || i == nx - 1)) || j == 0 || j == ny - 1 || (is3D && (k == 0 || k == nz - 1))) {
                            (*dstK)[k][j][i] = (*srcK)[k][j][i];
                        }
                    }
                }
            }

#pragma omp parallel for collapse(3) schedule(static)
            for (int k = kBegK; k < kEndK; ++k) {
                for (int j = 1; j < ny - 1; ++j) {
                    for (int i = periodicX ? 0 : 1; i < (periodicX ? nx : nx - 1); ++i) {

                        // Neighbour mean (4-neighbour in 2D, 6-neighbour in 3D)
                        double sum = (*srcK)[k][j][periodicX ? (i + 1) % nx : i + 1] +
                            (*srcK)[k][j][periodicX ? (i + nx - 1) % nx : i - 1]
                            + (*srcK)[k][j + 1][i] + (*srcK)[k][j - 1][i];
                        double denom = 4.0;

                        if (is3D) {
                            sum += (*srcK)[k + 1][j][i] + (*srcK)[k - 1][j][i];
                            denom = 6.0;
                        }

                        const double neigh = sum / denom;
                        (*dstK)[k][j][i] = alphaKappa * (*srcK)[k][j][i] + (1.0 - alphaKappa) * neigh;
                    }
                }
            }

            // toggle pointers without swap() of containers
            auto* tmp = srcK;
            srcK = dstK;
            dstK = tmp;
        }

        // If the last pass ended with srcK pointing to kappaTmp, copy it back into member.
        if (srcK != &this->kappa) {
#pragma omp parallel for collapse(3) schedule(static)
            for (int k = 0; k < nz; ++k)
                for (int j = 0; j < ny; ++j)
                    for (int i = 0; i < nx; ++i)
                        this->kappa[k][j][i] = kappaTmp[k][j][i];
        }
    }
}


std::vector<double> VOF::solveVOF(int maxIterations, double tolerance, std::vector<double>& InitialGuess, int outputLevel) {
    const int n = RHS.size();
    std::vector<double> x(n, 0.0);
    cblas_dcopy(static_cast<MKL_INT>(InitialGuess.size()), InitialGuess.data(), 1, x.data(), 1);
    double test = cblas_ddot(n, x.data(), 1, x.data(), 1);
    std::vector<double> r(n), r0(n), p(n), ph(n), s(n), sh(n), t(n), th(n), v(n);
    matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    // Initial residual: r = b - A*x
    sparse_status_t statusR = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, LHS_mkl, descr, x.data(), 0.0, r.data());
    vdSub(n, RHS.data(), r.data(), r.data());
    r0 = r; // Reference residual (r^)

    double rho_prev = 1.0, rho = 1.0, alpha = 1.0, omega = 1.0, omega_prev = 1.0;
    double norm_b = cblas_dnrm2(n, RHS.data(), 1);
    if (norm_b == 0.0) norm_b = 1.0;

    double resid = cblas_dnrm2(n, r.data(), 1) / norm_b;
    if (resid < tolerance) {
        std::cout << "Initial guess is already within tolerance.\n";
        return x;
    }

    int actualIterations = 0;
    for (int iter = 0; iter < maxIterations; ++iter) {
        actualIterations = iter + 1;

        // BiCG-like step (ODD step of BiCGSTAB2)
        // Compute scalar rho = (r^ * r)
        rho = cblas_ddot(n, r0.data(), 1, r.data(), 1);
        if (rho == 0.0) break;

        if (iter == 0) {
            p = r;
        }
        else {
            double beta = (rho / rho_prev) * (alpha / omega_prev);// Step 1: p = p - omega_prev * v
            // This computes the intermediate vector: p = p - omega_prev * v
            cblas_daxpy(n, -omega_prev, v.data(), 1, p.data(), 1);

            // Step 2: temp = r
            // We create a temporary vector `temp` and initialize it with r
            std::vector<double> temp(n, 0.0);
            cblas_dcopy(n, r.data(), 1, temp.data(), 1);

            // Step 3: temp = temp + beta * p
            // Now compute: temp = r + beta * (p - omega_prev * v), which is our updated p
            cblas_daxpy(n, beta, p.data(), 1, temp.data(), 1);

            // Step 4: p = temp
            // Copy the result back into p
            cblas_dcopy(n, temp.data(), 1, p.data(), 1);
        }

        // v = A * p
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, LHS_mkl, descr, p.data(), 0.0, v.data());

        // alpha = rho / (r^ * v)
        double denom = cblas_ddot(n, r0.data(), 1, v.data(), 1);
        if (denom == 0.0) break;
        alpha = rho / denom;

        // s = r - alpha * v
        // Step 1: s = r
        // Initialize s with r
        cblas_dcopy(n, r.data(), 1, s.data(), 1);

        // Step 2: s = s - alpha * v
        // Equivalent to s = r - alpha * v
        cblas_daxpy(n, -alpha, v.data(), 1, s.data(), 1);

        // Early convergence check
        double s_norm = cblas_dnrm2(n, s.data(), 1);
        if (s_norm / norm_b < tolerance) {
            // Perform x = x + alpha * p in place
            cblas_daxpy(n, alpha, p.data(), 1, x.data(), 1);
            resid = s_norm / norm_b;
            break;
        }

        // EVEN step (2nd-order BiCGSTAB2 correction)
        // t = A * s
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, LHS_mkl, descr, s.data(), 0.0, t.data());

        // omega = (t * s) / (t * t)
        double t_dot_t = cblas_ddot(n, t.data(), 1, t.data(), 1);
        if (t_dot_t == 0.0) {
            resid = 0.0;
            break;
        }
        omega = cblas_ddot(n, t.data(), 1, s.data(), 1) / t_dot_t;

        // Step 1: x = x + alpha * p
        // Perform x = x + alpha * p in place
        cblas_daxpy(n, alpha, p.data(), 1, x.data(), 1);

        // Step 2: x = x + omega * s
        // Perform x = x + omega * s in place
        cblas_daxpy(n, omega, s.data(), 1, x.data(), 1);

        // r = s - omega * t

        // Step 1: r = s
        // Initialize r with s 
        cblas_dcopy(n, s.data(), 1, r.data(), 1);

        // Step 2: r = r - omega * t
        // Equivalent to r = s - omega * t
        cblas_daxpy(n, -omega, t.data(), 1, r.data(), 1);

        // Check convergence
        resid = cblas_dnrm2(n, r.data(), 1) / norm_b;
        if (resid < tolerance) break;

        // Update values for next iteration
        rho_prev = rho;
        omega_prev = omega;
    }
    if (actualIterations == maxIterations) {
        statusR = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, LHS_mkl, descr, x.data(), 0.0, r.data());
        vdSub(n, RHS.data(), r.data(), r.data());
        double actualResidual = cblas_dnrm2(n, r.data(), 1) / norm_b;
        std::cout << "Actual iterations " << actualIterations << std::endl;
        std::cout << "Actual residual " << actualResidual << std::endl;
    }
    if (outputLevel > 1) {
        std::cout << "BiCGSTAB2 finished in " << actualIterations
            << " iterations with relative residual " << resid << std::endl;
    }
    if (outputLevel > 2) {
        statusR = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, LHS_mkl, descr, x.data(), 0.0, r.data());
        vdSub(n, RHS.data(), r.data(), r.data());
        double actualResidual = cblas_dnrm2(n, r.data(), 1) / norm_b;
        std::cout << "Actual residual " << actualResidual << std::endl;
    }

    return x;
}

std::vector<double> VOF::explicitVOF(BoundaryContainer& BCu, const std::vector<std::vector<std::vector<Point>>>& u, const std::vector<std::vector<std::vector<Point>>>& v, const std::vector<std::vector<std::vector<Point>>>& w, double dt)
{
    const bool eastPeriodic = BCu.getBoundary(PointBoundary::E).type == BoundaryKind::Symmetry;
    const bool westPeriodic = BCu.getBoundary(PointBoundary::W).type == BoundaryKind::Symmetry;
    if (eastPeriodic != westPeriodic)
        throw std::invalid_argument("VOF requires periodic U boundaries on both E and W");
    setPeriodicX(eastPeriodic);


    std::vector<double> phiV = this->getPhiV();
    const MKL_INT n = static_cast<MKL_INT>(phiV.size());
    if (n == 0) throw std::runtime_error("explicitVOF: empty phi.");

    std::vector<double> k1(n, 0.0), k2(n, 0.0), phi1(n, 0.0);
    double Ivol;
    if (phi.size() > 1) {
        Ivol = 1/(phi[1][1][1].getdx()* phi[1][1][1].getdy()* phi[1][1][1].getdz());
    }
    else {
        double dx = phi[0][1][1].getdx(); double dy = phi[0][1][1].getdy();
        Ivol = 1 / (dy *dx);
    }
    // k1 = A*phi

    k1 = this->productLHS(BCu,u,v,w,phiV);

    // phi1 = phi - dt*k1
    cblas_dcopy(n, phiV.data(), 1, phi1.data(), 1);           // phi1 = phi
    cblas_daxpy(n, -dt* Ivol, k1.data(), 1, phi1.data(), 1);       // phi1 += -dt*k1
    this->timeStep(phi1);
    // k2 = A*phi1
    k2 = this->productLHS(BCu, u, v, w, phi1);

    // phi = phi - 0.5*dt*(k1 + k2)
    // do k1 <- k1 + k2 to reduce to one BLAS axpy on phi
    cblas_daxpy(n, 1.0, k2.data(), 1, k1.data(), 1);         // k1 += k2
    cblas_daxpy(n, -0.5 * dt* Ivol, k1.data(), 1, phiV.data(), 1);    // phi += -0.5*dt*k1

    // Clipping can otherwise remove or create phase volume even though all
    // periodic face fluxes cancel. Preserve the RK2 sum while restoring the
    // physical bounds. The explicit solver uses one uniform cell volume.
    const double massBeforeClipping = eastPeriodic
        ? std::accumulate(phiV.begin(), phiV.end(), 0.0) : 0.0;
    std::vector<double> z(n, 0.0), o(n, 1.0);
    vdFmax(n, phiV.data(), z.data(), phiV.data());  // phi = max(phi, 0)
    vdFmin(n, phiV.data(), o.data(), phiV.data());  // phi = min(phi, 1)
    if (eastPeriodic) {
        const double clippedMass = std::accumulate(phiV.begin(), phiV.end(), 0.0);
        const double correction = massBeforeClipping - clippedMass;
        if (correction != 0.0) {
            double capacity = 0.0;
            if (correction > 0.0) {
                for (double fraction : phiV) capacity += 1.0 - fraction;
            } else {
                for (double fraction : phiV) capacity += fraction;
            }
            if (capacity < std::abs(correction) - 1e-12)
                throw std::runtime_error("VOF bound correction cannot conserve phase volume");
            const double scale = capacity > 0.0
                ? std::clamp(correction / capacity, -1.0, 1.0) : 0.0;
            for (double& fraction : phiV)
                fraction += scale * (correction > 0.0 ? 1.0 - fraction : fraction);
        }
    }
    this->timeStep(phiV);
    {
        ScopedProfilePhase timer(ProfilePhase::VOFCurvature);
        this->calculateKappa(4, 2E-1, 1, 2.0 / 3.0);
    }
    return phiV;
}




void VOF::savePhi(double time, MKL_INT timeStep) {
    writeTecplotFile(this->phi, "Phi", "Phi.dat", time, timeStep);
    writeTecplotFile(this->phi, this->kappa, "Kappa", "Kappa.dat", time, timeStep);

}

void VOF::saveTecplotSnapshot(double time, MKL_INT timeStep) {
    namespace fs = std::filesystem;

    fs::path dir("tecplot");
    fs::create_directories(dir);

    std::ostringstream name;
    name << "phi_step"
        << std::setfill('0') << std::setw(8) << timeStep
        << "_t" << std::fixed << std::setprecision(6) << time
        << ".szplt";

    const bool snapshotWritten = writeTecplotBinarySnapshot(
        this->phi,
        "Phi",
        (dir / name.str()).string(),
        time,
        static_cast<int>(timeStep));
#ifdef VANKA_HAVE_TECIO
    if (!snapshotWritten) {
        throw std::runtime_error("Unable to write the requested Tecplot SZL snapshot.");
    }
#else
    (void)snapshotWritten;
#endif
}


void VOF::updateRho(
    double rhoRelation,
    double muRelation,
    bool symetry,
    MaterialFields& fields)
{
    setPeriodicX(symetry);
    size_t nz = (this->phi.size() > 1) ? this->phi.size() : 1;
    size_t ny = (this->phi.size() > 1) ? this->phi[1].size() : this->phi[0].size();
    size_t nx = (this->phi.size() > 1) ? this->phi[1][1].size() : this->phi[0][1].size();

    const bool is3D = this->phi.size() > 1;
    const size_t xyDepth = is3D ? nz + 2 : 1;
    const size_t zDepth = is3D ? nz + 1 : 1;

    resetField(fields.rhoX, xyDepth, ny + 2, nx + 1, 1.0);
    resetField(fields.rhoY, xyDepth, ny + 1, nx + 2, 1.0);
    resetField(fields.rhoZ, zDepth, ny + 2, nx + 2, 1.0);
    resetField(fields.muX, xyDepth, ny + 2, nx + 1, 1.0);
    resetField(fields.muY, xyDepth, ny + 1, nx + 2, 1.0);
    resetField(fields.muZ, zDepth, ny + 2, nx + 2, 1.0);
    resetField(fields.dmudx, xyDepth, ny + 2, nx + 1, 0.0);
    resetField(fields.dmudy, xyDepth, ny + 1, nx + 2, 0.0);
    resetField(fields.dmudz, zDepth, ny + 2, nx + 2, 0.0);
    resetField(fields.muS, is3D ? nz : 1, ny, nx, 1.0);

    auto& rhoX = fields.rhoX;
    auto& rhoY = fields.rhoY;
    auto& rhoZ = fields.rhoZ;
    auto& muX = fields.muX;
    auto& muY = fields.muY;
    auto& muZ = fields.muZ;
    auto& dmdx = fields.dmudx;
    auto& dmdy = fields.dmudy;
    auto& dmdz = fields.dmudz;
    auto& muS = fields.muS;

    if (this->phi.size() > 1)
    {
    #pragma omp parallel for collapse(3) schedule(static)
        for (int k = 0; k < nz; k++) {
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    double localPhi = this->phi[k][j][i].getValue();
                    if (i < nx - 1) {
                        double phiUpX = this->phi[k][j][i + 1].getValue();
                        if ((localPhi + phiUpX)   > (1e-8) ) {
                            rhoX[k + 1][j + 1][i + 1] = 1.0 + 2.0 * (rhoRelation - 1) * phiUpX * localPhi / (localPhi + phiUpX);
                            muX[k + 1][j + 1][i + 1] = 1.0 + 2.0 * (muRelation - 1) * phiUpX * localPhi / (localPhi + phiUpX);
                        }
                    }

                    if (j < ny - 1) {
                        double phiUpY = this->phi[k][j + 1][i].getValue();
                        if ((localPhi + phiUpY)   > (1e-8) ) {
                            rhoY[k + 1][j + 1][i + 1] = 1.0 + 2.0 * (rhoRelation - 1) * phiUpY * localPhi / (localPhi + phiUpY);
                            muY[k + 1][j + 1][i + 1] = 1.0 + 2.0 * (muRelation - 1) * phiUpY * localPhi / (localPhi + phiUpY);
                        }
                    }

                    if (k < nz - 1) {
                        double phiUpZ = this->phi[k + 1][j][i].getValue();
                        if ((localPhi + phiUpZ)   > (1e-8) ) {
                            rhoZ[k + 1][j + 1][i + 1] = 1.0 + 2.0 * (rhoRelation - 1) * phiUpZ * localPhi / (localPhi + phiUpZ);
                            muZ[k + 1][j + 1][i + 1] = 1.0 + 2.0 * (muRelation - 1) * phiUpZ * localPhi / (localPhi + phiUpZ);
                        }
                    }
                    muS[k][j][i] = 1.0 + (muRelation - 1) * phiS[k][j][i];
                }
            }
        }

        // Extend the interior face properties to every physical/ghost plane.
        // The old 3D branch left these entries at 1.0, which introduced a
        // spurious viscosity jump whenever the interface approached a wall.
        auto extendBoundaryValues = [](
            auto& field,
            size_t kFirst, size_t kLast,
            size_t jFirst, size_t jLast,
            size_t iFirst, size_t iLast) {
                // X boundary strips on otherwise-interior rows.
                for (size_t k = kFirst; k <= kLast; ++k) {
                    for (size_t j = jFirst; j <= jLast; ++j) {
                        for (size_t i = 0; i < iFirst; ++i) {
                            field[k][j][i] = field[k][j][iFirst];
                        }
                        for (size_t i = iLast + 1; i < field[k][j].size(); ++i) {
                            field[k][j][i] = field[k][j][iLast];
                        }
                    }
                }

                // Y boundary rows, including the X strips just populated.
                for (size_t k = kFirst; k <= kLast; ++k) {
                    for (size_t j = 0; j < jFirst; ++j) {
                        for (size_t i = 0; i < field[k][j].size(); ++i) {
                            field[k][j][i] = field[k][jFirst][i];
                        }
                    }
                    for (size_t j = jLast + 1; j < field[k].size(); ++j) {
                        for (size_t i = 0; i < field[k][j].size(); ++i) {
                            field[k][j][i] = field[k][jLast][i];
                        }
                    }
                }

                // Z boundary layers, including their already-complete X/Y edges.
                for (size_t k = 0; k < kFirst; ++k) {
                    for (size_t j = 0; j < field[k].size(); ++j) {
                        for (size_t i = 0; i < field[k][j].size(); ++i) {
                            field[k][j][i] = field[kFirst][j][i];
                        }
                    }
                }
                for (size_t k = kLast + 1; k < field.size(); ++k) {
                    for (size_t j = 0; j < field[k].size(); ++j) {
                        for (size_t i = 0; i < field[k][j].size(); ++i) {
                            field[k][j][i] = field[kLast][j][i];
                        }
                    }
                }
            };

        if (symetry) {
            // Periodic E/W face for the X-aligned properties.
            for (size_t k = 0; k < nz; ++k) {
                for (size_t j = 0; j < ny; ++j) {
                    const double phiWest = this->phi[k][j][0].getValue();
                    const double phiEast = this->phi[k][j][nx - 1].getValue();
                    if (phiWest + phiEast > 1e-8) {
                        const double rhoPeriodic = 1.0 + 2.0 * (rhoRelation - 1.0) *
                            phiWest * phiEast / (phiWest + phiEast);
                        const double muPeriodic = 1.0 + 2.0 * (muRelation - 1.0) *
                            phiWest * phiEast / (phiWest + phiEast);
                        rhoX[k + 1][j + 1][0] = rhoPeriodic;
                        rhoX[k + 1][j + 1][nx] = rhoPeriodic;
                        muX[k + 1][j + 1][0] = muPeriodic;
                        muX[k + 1][j + 1][nx] = muPeriodic;
                    }
                }
            }

            // Interpolate cell-centred X boundary values from both periodic sides.
            auto setPeriodicXBoundary = [nx](
                auto& field,
                size_t kFirst, size_t kLast,
                size_t jFirst, size_t jLast) {
                    for (size_t k = kFirst; k <= kLast; ++k) {
                        for (size_t j = jFirst; j <= jLast; ++j) {
                            double boundaryValue = 0.5 * (field[k][j][1] + field[k][j][nx]);
                            if (nx > 2) {
                                boundaryValue =
                                    9.0 * (field[k][j][1] + field[k][j][nx]) / 16.0 -
                                    (field[k][j][2] + field[k][j][nx - 1]) / 16.0;
                            }
                            field[k][j][0] = boundaryValue;
                            field[k][j][nx + 1] = boundaryValue;
                        }
                    }
                };

            setPeriodicXBoundary(rhoY, 1, nz, 1, ny - 1);
            setPeriodicXBoundary(muY, 1, nz, 1, ny - 1);
            setPeriodicXBoundary(rhoZ, 1, nz - 1, 1, ny);
            setPeriodicXBoundary(muZ, 1, nz - 1, 1, ny);

            extendBoundaryValues(rhoX, 1, nz, 1, ny, 0, nx);
            extendBoundaryValues(muX, 1, nz, 1, ny, 0, nx);
            extendBoundaryValues(rhoY, 1, nz, 1, ny - 1, 0, nx + 1);
            extendBoundaryValues(muY, 1, nz, 1, ny - 1, 0, nx + 1);
            extendBoundaryValues(rhoZ, 1, nz - 1, 1, ny, 0, nx + 1);
            extendBoundaryValues(muZ, 1, nz - 1, 1, ny, 0, nx + 1);
        }
        else {
            extendBoundaryValues(rhoX, 1, nz, 1, ny, 1, nx - 1);
            extendBoundaryValues(muX, 1, nz, 1, ny, 1, nx - 1);
            extendBoundaryValues(rhoY, 1, nz, 1, ny - 1, 1, nx);
            extendBoundaryValues(muY, 1, nz, 1, ny - 1, 1, nx);
            extendBoundaryValues(rhoZ, 1, nz - 1, 1, ny, 1, nx);
            extendBoundaryValues(muZ, 1, nz - 1, 1, ny, 1, nx);
        }

    }
    else
    {
    #pragma omp parallel for collapse(2) schedule(static)
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                double localPhi = this->phi[0][j][i].getValue();
                // Cell-centred viscosity is required below to construct the
                // U- and V-located viscosity gradients in the 2D RHS.
                muS[0][j][i] = 1.0 + (muRelation - 1.0) * phiS[0][j][i];
                if (i < nx - 1) {
                    double phiUpX = this->phi[0][j][i + 1].getValue();
                    if ((localPhi + phiUpX)   > (1e-8) ) {
                        rhoX[0][j + 1][i + 1] = 1.0 + 2.0 * (rhoRelation - 1) * phiUpX * localPhi / (localPhi + phiUpX);
                        muX[0][j + 1][i + 1] = 1.0 + 2.0 * (muRelation - 1) * phiUpX * localPhi / (localPhi + phiUpX);
                    }
                }

                if (j < ny - 1) {
                    double phiUpY = this->phi[0][j + 1][i].getValue();
                    if ((localPhi + phiUpY)  > (1e-8) ) {
                        rhoY[0][j + 1][i + 1] = 1.0 + 2.0 * (rhoRelation - 1) * phiUpY * localPhi / (localPhi + phiUpY);
                        muY[0][j + 1][i + 1] = 1.0 + 2.0 * (muRelation - 1) * phiUpY * localPhi / (localPhi + phiUpY);
                    }
                }
            }
        }
        rhoY[0][0][0] = rhoY[0][1][1];
        muY[0][0][0] = muY[0][1][1];
        rhoX[0][0][0] = rhoX[0][1][1];
        muX[0][0][0] = muX[0][1][1];
        if (symetry) {
            for (int j = 0; j < ny; j++) {
                double localPhi = this->phi[0][j][nx - 1].getValue();
                double phiUpX = this->phi[0][j][0].getValue();
                if ((localPhi + phiUpX)   > (1e-8) ) {
                    rhoX[0][j + 1][0] = 1.0 + 2.0 * (rhoRelation - 1) * phiUpX * localPhi / (localPhi + phiUpX);
                    rhoX[0][j + 1][nx] = 1.0 + 2.0 * (rhoRelation - 1) * phiUpX * localPhi / (localPhi + phiUpX);
                    muX[0][j + 1][0] = 1.0 + 2.0 * (muRelation - 1) * phiUpX * localPhi / (localPhi + phiUpX);
                    muX[0][j + 1][nx] = 1.0 + 2.0 * (muRelation - 1) * phiUpX * localPhi / (localPhi + phiUpX);
                }
                rhoY[0][j + 1][0] = 9 * (rhoY[0][j + 1][1] + rhoY[0][j + 1][nx]) / 16 - (rhoY[0][j + 1][2] + rhoY[0][j + 1][nx - 1]) / 16;
                rhoY[0][j + 1][nx + 1] = rhoY[0][j + 1][0];
                muY[0][j + 1][0] = 9 * (muY[0][j + 1][1] + muY[0][j + 1][nx]) / 16 - (muY[0][j + 1][2] + muY[0][j + 1][nx - 1]) / 16;
                muY[0][j + 1][nx + 1] = muY[0][j + 1][0];
            }
        }
        else {
            for (int j = 0; j < ny; j++) {
                rhoX[0][j + 1][0] = rhoX[0][j + 1][1];
                rhoX[0][j + 1][nx] = rhoX[0][j + 1][nx - 1];
                muX[0][j + 1][0] = muX[0][j + 1][1];
                muX[0][j + 1][nx] = muX[0][j + 1][nx - 1];
                rhoY[0][j + 1][0] = rhoY[0][j + 1][1];
                rhoY[0][j + 1][nx + 1] = rhoY[0][j + 1][nx];
                muY[0][j + 1][0] = muY[0][j + 1][1];
                muY[0][j + 1][nx + 1] = muY[0][j + 1][nx];
            }
        }
        for (int i = 0; i < nx; i++) {
            rhoX[0][0][i + 1] = rhoX[0][1][i + 1];
            rhoX[0][ny + 1][i + 1] = rhoX[0][ny][i + 1];
            muX[0][0][i + 1] = muX[0][1][i + 1];
            muX[0][ny + 1][i + 1] = muX[0][ny][i + 1];
            rhoY[0][ny][i + 1] = rhoY[0][ny - 1][i + 1];
            rhoY[0][0][i + 1] = rhoY[0][1][i + 1];
            muY[0][ny][i + 1] = muY[0][ny - 1][i + 1];
            muY[0][0][i + 1] = muY[0][1][i + 1];
        }
    }
    #pragma omp parallel for collapse(3) 
    for (size_t k = 0; k < nz; ++k) {
        for (size_t j = 0; j < ny; ++j) {
            for (size_t i = 0; i < nx; ++i) {
                int kk = (nz <= 1) ? -1 : k;
                const double dx =(i < nx - 1) ? (phi[k][j][i + 1].getX() - phi[k][j][i].getX()) : phi[k][j][i].getdx()/2.0;
                const double dy = (j < ny - 1) ? phi[k][j + 1][i].getY() - phi[k][j][i].getY() : phi[k][j][i].getdy() / 2.0;
                const double dz = (k < nz - 1) ? ((nz >= 1) ? phi[kk + 1][j][i].getZ() - phi[k][j][i].getZ() : 1.0) : phi[k][j][i].getdz() / 2.0;

                if (i < nx - 1) dmdx[kk + 1][j + 1][i + 1] = (muS[k][j][i + 1] - muS[k][j][i]) / dx;
                if (j < ny - 1) dmdy[kk + 1][j + 1][i + 1] = (muS[k][j + 1][i] - muS[k][j][i]) / dy;
                if (k < nz - 1) dmdz[kk + 1][j + 1][i + 1] = (nz >= 1) ? (muS[kk + 1][j][i] - muS[k][j][i]) / dz : 0.0;
                
            }
        }
    }

    if (symetry) {
        for (size_t k = 0; k < nz; ++k) {
            for (size_t j = 0; j < ny; ++j) {
                const double periodicDx =
                    0.5 * (phi[k][j][0].getdx() + phi[k][j][nx - 1].getdx());
                const double periodicGradient =
                    (muS[k][j][0] - muS[k][j][nx - 1]) / periodicDx;
                const size_t gradientK = (nz > 1) ? k + 1 : 0;
                dmdx[gradientK][j + 1][0] = periodicGradient;
                dmdx[gradientK][j + 1][nx] = periodicGradient;
            }
        }
    }

}

double VOF::findInterfaceY_cubicSpline(int k, int i,
    double /*eps_band*/,
    int /*maxPts*/) const
{
    // assume 2D or given k-layer
    if (k < 0 || k >= static_cast<int>(phi.size())) {
        throw std::runtime_error("findInterfaceY_cubicSpline: invalid k");
    }
    const size_t ny = phi[k].size();
    if (ny < 2) {
        throw std::runtime_error("findInterfaceY_cubicSpline: ny < 2");
    }
    if (i < 0 || i >= static_cast<int>(phi[k][0].size())) {
        throw std::runtime_error("findInterfaceY_cubicSpline: invalid i");
    }

    // 1) find first j where (phi_j - 0.5) changes sign with (phi_{j+1} - 0.5)
    int j_cross = -1;
    for (size_t j = 0; j + 1 < ny; ++j) {
        double f1 = phi[k][j][i].getValue() - 0.5;
        double f2 = phi[k][j + 1][i].getValue() - 0.5;

        if (std::abs(f1) < 1e-14) {
            // exactly on 0.5 at node j
            return phi[k][j][i].getY();
        }
        if (std::abs(f2) < 1e-14) {
            // exactly on 0.5 at node j+1
            return phi[k][j + 1][i].getY();
        }
        if (f1 * f2 < 0.0) {
            j_cross = static_cast<int>(j);
            break;
        }
    }

    if (j_cross < 0) {
        throw std::runtime_error("findInterfaceY_cubicSpline: no phi = 0.5 crossing in column");
    }

    // 2) build a small local stencil around j_cross
    //    use indices [j_start .. j_end], at most 4 points
    int j_start = std::max(j_cross - 1, 0);
    int j_end = std::min(j_cross + 2, static_cast<int>(ny) - 1);
    int m_int = j_end - j_start + 1;
    if (m_int < 2) {
        throw std::runtime_error("findInterfaceY_cubicSpline: less than 2 points in stencil");
    }
    size_t m = static_cast<size_t>(m_int);

    std::vector<double> phi_vals(m);
    for (size_t s = 0; s < m; ++s) {
        int jj = j_start + static_cast<int>(s);
        phi_vals[s] = phi[k][jj][i].getValue();
    }

    // 3) construct uniform spline in local xi = 0 .. m-1
    UniformCubicSpline spline(phi_vals);

    // 4) local indices of the crossing inside this stencil
    double xi0 = static_cast<double>(j_cross - j_start);
    double xi1 = static_cast<double>(j_cross + 1 - j_start);

    // safety: ensure sign change on these bounds
    double f0 = spline.eval(xi0) - 0.5;
    double f1 = spline.eval(xi1) - 0.5;
    if (f0 * f1 > 0.0) {
        // fall back to simple linear interpolation in case spline is odd
        double y0 = phi[k][j_cross][i].getY();
        double y1 = phi[k][j_cross + 1][i].getY();
        double phi0 = phi[k][j_cross][i].getValue();
        double phi1v = phi[k][j_cross + 1][i].getValue();
        double t = (0.5 - phi0) / (phi1v - phi0);
        return y0 + t * (y1 - y0);
    }

    double xi_root = findRootOnInterval(spline, xi0, xi1);

    // 5) map xi_root back to physical y:
    // y = y_start + xi_root * dy, dy uniform
    double y_start = phi[k][j_start][i].getY();
    double dy = phi[k][j_start + 1][i].getY() - phi[k][j_start][i].getY();
    return y_start + xi_root * dy;
}
