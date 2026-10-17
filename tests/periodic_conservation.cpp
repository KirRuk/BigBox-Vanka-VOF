#include "VOF.h"
#include "Initialize.h"
#include <mkl.h>
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <stdexcept>
#include <string>

namespace {
void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void run(bool is3D, double direction, bool sharp) {
    constexpr int nx = 7, ny = 7, nz3 = 7;
    constexpr double ds = 0.1, dt = 0.001;
    const int nz = is3D ? nz3 : 1;
    const auto filename = std::filesystem::path(is3D ? "periodic_3d.dat" : "periodic_2d.dat");
    {
        std::ofstream file(filename);
        file << std::setprecision(17) << "VARIABLES=X,Y,Z,Phi\nZONE I="
             << nx << ", J=" << ny;
        if (is3D) file << ", K=" << nz;
        file << ", F=POINT\n";
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i) {
                    file << (i + 0.5) * ds << ' ' << (j + 0.5) * ds << ' ';
                    if (is3D) file << (k + 0.5) * ds << ' ';
                    file << (sharp ? (i < nx / 2 ? 0.0 : 1.0) :
                        0.45 + 0.2 * std::sin(0.65 * i + 0.3 * j + 0.2 * k)) << '\n';
                }
    }

    VOF vof(filename.string(), ds, nx * ds, ny * ds, is3D ? nz * ds : 0.0, 1.0);
    std::filesystem::remove(filename);
    BoundaryContainer bcu, bcv, bcw;
    for (auto* bc : {&bcu, &bcv, &bcw}) {
        bc->addBoundary("E", {"symetry", 0.0});
        bc->addBoundary("W", {"symetry", 0.0});
        bc->addBoundary("N", {"fixed_value", 0.0});
    }
    auto [u, v, w, p] = initialize(bcu, bcv, bcw,
        nx * ds, ny * ds, is3D ? nz * ds : 0.0, ds,
        0.0, 0.0, 0.0, vof.getPhi(), 1.0, 1.0, false);

    for (size_t k = 0; k < u.size(); ++k)
        for (size_t j = 0; j < u[k].size(); ++j)
            for (size_t i = 0; i < u[k][j].size(); ++i) {
                const size_t face = i == u[k][j].size() - 1 ? 0 : i;
                u[k][j][i].setValue(direction *
                    (0.7 + 0.2 * std::sin(0.4 * face + 0.2 * j + 0.3 * k)));
            }
    for (size_t k = 0; k < v.size(); ++k)
        for (size_t j = 0; j < v[k].size(); ++j)
            for (size_t i = 0; i < v[k][j].size(); ++i)
                v[k][j][i].setValue(j == 0 || j + 1 == v[k].size()
                    ? 0.0 : 0.15 * std::sin(0.3 * i + 0.2 * k));
    if (is3D)
        for (size_t k = 0; k < w.size(); ++k)
            for (size_t j = 0; j < w[k].size(); ++j)
                for (size_t i = 0; i < w[k][j].size(); ++i)
                    w[k][j][i].setValue(k == 0 || k + 1 == w.size()
                        ? 0.0 : 0.12 * std::cos(0.2 * i + 0.3 * j));

    const auto initialValues = vof.getPhiV();
    const double initial = std::accumulate(initialValues.begin(),
        initialValues.end(), 0.0);
    for (int step = 0; step < 5; ++step) {
        const auto values = vof.getPhiV();
        const auto transport = vof.productLHS(bcu, u, v, w, values);
        const double netFlux = std::accumulate(transport.begin(), transport.end(), 0.0);
        check(std::abs(netFlux) < 1e-11, "periodic VOF face fluxes do not cancel");
        const auto next = vof.explicitVOF(bcu, u, v, w, sharp ? 0.02 : dt);
        const double mass = std::accumulate(next.begin(), next.end(), 0.0);
        check(std::abs(mass - initial) < 1e-10, "periodic VOF lost mass");
        const auto [low, high] = std::minmax_element(next.begin(), next.end());
        check(*low >= 0.0 && *high <= 1.0, "VOF left the physical range");
    }
}
}

void runPeriodicConservation() {
    for (int threads : {1, 4}) {
        omp_set_num_threads(threads);
        mkl_set_num_threads(threads);
        run(false, 1.0, false);
        run(false, -1.0, false);
        run(true, 1.0, false);
        run(true, -1.0, false);
        run(false, 1.0, true);
        run(false, -1.0, true);
        run(true, 1.0, true);
        run(true, -1.0, true);
    }
}
