// Vanka_Feldman_patch_solver.cpp : Defines the entry point for the application.
//

#ifndef RESOURCE_DIR
#define RESOURCE_DIR "./"
#endif

#include "Vanka_Feldman_patch_solver.h"
#include <iostream>
#include "Point.h"
#include "Initialize.h"
#include "DiagonalMatrix.h"
#include "read.h"
#include "BoundaryContainer.h"
#include "PressureMatrix.h"
#include "MatrixBDBT.h"
#include "output.h"
#include <cmath>
#include "Relax.h"
#include "nextTimeStep.h"
#include <omp.h>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include "VOF.h"
#include "PerformanceProfiler.h"
static inline double wtime_safe() {
    volatile double t = omp_get_wtime();  // simplest form
    return (double)t;
}

namespace {
double environmentDouble(const char* name, double fallback) {
    const char* text = std::getenv(name);
    if (text == nullptr || text[0] == '\0') return fallback;
    char* end = nullptr;
    const double value = std::strtod(text, &end);
    if (end == text || *end != '\0' || !std::isfinite(value) || value <= 0.0) {
        throw std::runtime_error(std::string("Invalid positive value for ") + name);
    }
    return value;
}

int environmentPositiveInt(const char* name, int fallback) {
    const char* text = std::getenv(name);
    if (text == nullptr || text[0] == '\0') return fallback;
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value <= 0 ||
        value > std::numeric_limits<int>::max()) {
        throw std::runtime_error(std::string("Invalid positive integer for ") + name);
    }
    return static_cast<int>(value);
}

double vofMass(const std::vector<std::vector<std::vector<Point>>>& phi) {
    double mass = 0.0;
    #pragma omp parallel for collapse(3) reduction(+:mass)
    for (std::int64_t k = 0; k < static_cast<std::int64_t>(phi.size()); ++k) {
        for (std::int64_t j = 0; j < static_cast<std::int64_t>(phi[k].size()); ++j) {
            for (std::int64_t i = 0; i < static_cast<std::int64_t>(phi[k][j].size()); ++i) {
                const Point& cell = phi[k][j][i];
                mass += cell.getValue() * cell.getdx() * cell.getdy() *
                    (phi.size() > 1 ? cell.getdz() : 1.0);
            }
        }
    }
    return mass;
}

std::pair<double, double> divergenceDiagnostics(
    const std::vector<std::vector<std::vector<Point>>>& u,
    const std::vector<std::vector<std::vector<Point>>>& v,
    const std::vector<std::vector<std::vector<Point>>>& w,
    const std::vector<std::vector<std::vector<Point>>>& p) {
    double sumSquares = 0.0;
    double maximum = 0.0;
    const bool is3D = w.size() > 1;
    const std::int64_t count = static_cast<std::int64_t>(p.size()) *
        static_cast<std::int64_t>(p[0].size()) *
        static_cast<std::int64_t>(p[0][0].size());
    #pragma omp parallel for collapse(3) reduction(+:sumSquares) reduction(max:maximum)
    for (std::int64_t k = 0; k < static_cast<std::int64_t>(p.size()); ++k) {
        for (std::int64_t j = 0; j < static_cast<std::int64_t>(p[k].size()); ++j) {
            for (std::int64_t i = 0; i < static_cast<std::int64_t>(p[k][j].size()); ++i) {
                const Point& cell = p[k][j][i];
                const std::size_t velocityK = is3D ? static_cast<std::size_t>(k + 1) : 0;
                const double dudx =
                    (u[velocityK][j + 1][i + 1].getValue() -
                        u[velocityK][j + 1][i].getValue()) / cell.getdx();
                const double dvdy =
                    (v[velocityK][j + 1][i + 1].getValue() -
                        v[velocityK][j][i + 1].getValue()) / cell.getdy();
                const double dwdz = is3D
                    ? (w[k + 1][j + 1][i + 1].getValue() -
                        w[k][j + 1][i + 1].getValue()) / cell.getdz()
                    : 0.0;
                const double divergence = dudx + dvdy + dwdz;
                sumSquares += divergence * divergence;
                maximum = std::max(maximum, std::abs(divergence));
            }
        }
    }
    return {std::sqrt(sumSquares / static_cast<double>(count)), maximum};
}
} // namespace

int main()
{
    // start timing
    double start = omp_get_wtime();
    
    std::string userFile = std::string(RESOURCE_DIR) + "user.choice";
    UserChoices choices = read(userFile);
    choices.convgCrit = environmentDouble("VANKA_OUTER_TOL", choices.convgCrit);
    choices.CG = environmentDouble("VANKA_CG_TOL", choices.CG);
    omp_set_num_threads(choices.threads);
    mkl_set_dynamic(0);
    mkl_set_num_threads(choices.threads);

    int nx = static_cast<int>(std::round(choices.Lx / choices.ds)) + 1,
        ny = static_cast<int>(std::round(choices.Ly / choices.ds)) + 1,
        nz = static_cast<int>(std::round(choices.Lz / choices.ds)) + 1;  // Number of grid points in each direction
    std::vector<std::vector<std::vector<Point>>> u,v,w,p;

    BoundaryContainer BCu, BCv, BCw, BCp;
    std::string BCufile = std::string(RESOURCE_DIR) + "boundariesU.inp";
    BCu.loadFromFile(BCufile);
    std::string BCvfile = std::string(RESOURCE_DIR) + "boundariesV.inp";
    BCv.loadFromFile(BCvfile);
    std::string BCwfile = std::string(RESOURCE_DIR) + "boundariesW.inp";
    BCw.loadFromFile(BCwfile);
    std::string BCpfile = std::string(RESOURCE_DIR) + "boundariesP.inp";
    BCp.loadFromFile(BCpfile);
    const auto isPeriodic = [](const BoundaryContainer& bc, PointBoundary face) {
        return bc.getBoundary(face).type == BoundaryKind::Symmetry;
    };
    const bool periodicU = isPeriodic(BCu, PointBoundary::E) || isPeriodic(BCu, PointBoundary::W);
    const bool periodicV = isPeriodic(BCv, PointBoundary::E) || isPeriodic(BCv, PointBoundary::W);
    const bool periodicW = isPeriodic(BCw, PointBoundary::E) || isPeriodic(BCw, PointBoundary::W);
    const bool symetry = periodicU || periodicV || periodicW;
    if (symetry &&
        (!isPeriodic(BCu, PointBoundary::E) || !isPeriodic(BCu, PointBoundary::W) ||
         !isPeriodic(BCv, PointBoundary::E) || !isPeriodic(BCv, PointBoundary::W) ||
         (!choices.twoDimensional &&
             (!isPeriodic(BCw, PointBoundary::E) || !isPeriodic(BCw, PointBoundary::W))))) {
        throw std::invalid_argument(
            "Periodic E/W boundaries must be set on both faces for U, V, and (in 3D) W");
    }
    std::string VOFinput;
    if (choices.twoDimensional) VOFinput = std::string(RESOURCE_DIR) + "PhiInit.dat";
    if (!choices.twoDimensional) VOFinput = std::string(RESOURCE_DIR) + "PhiInit3D.dat";
    VOF phi(VOFinput, choices.ds, choices.Lx, choices.Ly, choices.Lz, 1.0);
    
    if (choices.read_data) {
        std::string Uinput = std::string(RESOURCE_DIR) + "Vx.dat";
        std::string Vinput = std::string(RESOURCE_DIR) + "Vy.dat";
        std::string Winput = std::string(RESOURCE_DIR) + "Vz.dat";
        std::string Pinput = std::string(RESOURCE_DIR) + "P.dat";
        u = loadU(Uinput, choices.ds);
        v = loadV(Vinput, choices.ds);
        w = loadW(Winput, choices.ds);
        p = loadP(Pinput, choices.ds, choices.Lx, choices.Ly, choices.Lz,u,v,w);
    }
    else {
        std::tuple<std::vector<std::vector<std::vector<Point>>>, std::vector<std::vector<std::vector<Point>>>, std::vector<std::vector<std::vector<Point>>>, std::vector<std::vector<std::vector<Point>>>> data = initialize(BCu,BCv,BCw,choices.Lx, choices.Ly, choices.Lz, choices.ds,0.0,0.0,0.0, phi.getPhi(), choices.Rho,choices.Fr, choices.developed);
        u = std::get<0>(data);
        v = std::get<1>(data);
        w = std::get<2>(data);
        p = std::get<3>(data);
    }
    MaterialFields materialFields;
    phi.updateRho(choices.Rho, choices.Mu, symetry, materialFields);

    auto& rhoX = materialFields.rhoX;
    auto& rhoY = materialFields.rhoY;
    auto& rhoZ = materialFields.rhoZ;
    auto& muX = materialFields.muX;
    auto& muY = materialFields.muY;
    auto& muZ = materialFields.muZ;
    auto& dmudx = materialFields.dmudx;
    auto& dmudy = materialFields.dmudy;
    auto& dmudz = materialFields.dmudz;
    //Output
    std::cout << "Welcome to the Vanka-Feldman Solver (License: MIT)" << std::endl;
    std::cout << "---------------------------------------------" << std::endl;
    std::cout << "CPU threads:" << choices.threads <<std::endl;
    std::cout << "Domain Dimensions:" << std::endl;
    std::cout << "   Length (X-axis): " << choices.Lx << " units" << std::endl;
    std::cout << "   Width (Y-axis): " << choices.Ly << " units" << std::endl;
    if (nz > 1) {
        std::cout << "   Height (Z-axis): " << choices.Lz << " units" << std::endl;
    }

    std::cout << "\nGrid Resolution:" << std::endl;
    std::cout << "   Grid Points (X-axis): " << nx << std::endl;
    std::cout << "   Grid Points (Y-axis): " << ny << std::endl;
    if (nz > 1) {
        std::cout << "   Grid Points (Z-axis): " << nz << std::endl;
    }

    std::cout << "\nPhysical Parameters:" << std::endl;
    std::cout << "   Reynolds Number (Re): " << choices.Re << std::endl;
    std::cout << "   Froude Number (Fr): " << choices.Fr << std::endl;
    std::cout << "   Webber Number (We): " << choices.We << std::endl;
    std::cout << "   Density ratio: " << choices.Rho << std::endl;
    std::cout << "   Viscosity ratio: " << choices.Mu << std::endl;

    std::cout << "\nSimulation Details:" << std::endl;

    std::cout << "---------------------------------------------" << std::endl;

    std::cout << "Simulation of Rayleigh-Taylor instability" << std::endl;
    std::cout << "Initial shape is given by cos function" << std::endl;
    std::cout << "E/W boundaries: " << (symetry ? "periodic" : "nonperiodic") << std::endl;

    std::cout << "---------------------------------------------" << std::endl;
    

    DiagonalMatrix D = DiagonalMatrix(u, v, w, BCu, BCv, BCw, choices.dt, choices.Re,rhoX,muX,rhoY,muY,rhoZ,muZ);
    PressureMatrix B = PressureMatrix(u, v, w, p, BCp, D, symetry);

    size_t BDBTsize = B.getRowStart().size();//getting matrix size

    int NumDirichletPoints = ((nx) - 1) * (ny - 1) * (nz > 1 ? nz - 1 : 1) - static_cast<int>(BDBTsize);//calculate number of Dirichle points based on matrix sizes
    MatrixBDBT BDBT = MatrixBDBT(B.getValues(), B.getColumnIndices(), B.getRowStart(), B.getRowEnd(), D.getDiag(), BDBTsize, D.getDiag().size(),nx - 1,ny - 1 ,nz - 1);
    int endTimeStep = ceil(choices.terminationTime / choices.dt);
    endTimeStep = std::min(
        endTimeStep,
        environmentPositiveInt("VANKA_MAX_TIMESTEPS", endTimeStep));
    double time = 0;
    RHSu rhsu(D);
    RHSr rhsr(BDBTsize);
    const double initialVofMass = vofMass(phi.getPhi());
    PerformanceProfiler& profiler = PerformanceProfiler::instance();
    const bool legacyBDBTRebuild = environmentFlagEnabled("VANKA_LEGACY_BDBT_REBUILD");
    const std::size_t vofCells = static_cast<std::size_t>(nx - 1) *
        static_cast<std::size_t>(ny - 1) *
        static_cast<std::size_t>(nz > 1 ? nz - 1 : 1);
    const auto nestedCount = [](const auto& grid) {
        std::size_t count = 0;
        for (const auto& plane : grid) {
            for (const auto& row : plane) count += row.size();
        }
        return count;
    };
    const std::size_t pointCount = nestedCount(u) + nestedCount(v) +
        nestedCount(w) + nestedCount(p) + nestedCount(phi.getPhi());
    const std::size_t materialDoubleCount = nestedCount(rhoX) + nestedCount(rhoY) +
        nestedCount(rhoZ) + nestedCount(muX) + nestedCount(muY) + nestedCount(muZ) +
        nestedCount(dmudx) + nestedCount(dmudy) + nestedCount(dmudz);
    profiler.emitInitialization(
        omp_get_wtime() - start,
        vofCells,
        BDBTsize,
        pointCount,
        sizeof(Point),
        materialDoubleCount);
    for (int timeStep = 1; timeStep <= endTimeStep; ++timeStep) {
        profiler.beginStep(timeStep, vofCells, BDBTsize);
        const double stepStart = omp_get_wtime();

        if (!choices.skipfluid) {
            int iteration = 0;
            rhsu.updateTimestep(u, v, w, choices.dt);
            Relax R = Relax(u, v, w, p);
            double norm = 10.0;

            while (*reinterpret_cast<volatile double*>(&norm) > choices.convgCrit) {
                double rhoRef = choices.Rho;
                R.currentStep(u, v, w, p, rhoX, rhoY, rhoZ, muX, muY, muZ, dmudx, dmudy, dmudz, rhoRef, choices.dt, BDBTsize, BCu, BCv, BCw, BCp, BDBT, D, B, choices.Re, choices.Fr, choices.We, rhsu, rhsr, choices.CG, NumDirichletPoints, phi.getPhiS(), phi.getKappa());
                {
                    ScopedProfilePhase timer(ProfilePhase::RelaxUpdate);
                    R.relax(u,v,w,p,symetry);
                }

                iteration++;
                double pNorm = R.getPressureCorrectionNorm();
                double uNorm = R.getVelocityCorrectionNorm();
                norm = (pNorm > uNorm ? pNorm : uNorm );

                if (iteration > choices.maxRelaxNumber) break;
            }
            profiler.recordOuterIterations(iteration);
            profiler.recordOuterResiduals(
                R.getUCorrectionNorm(),
                R.getVCorrectionNorm(),
                R.getWCorrectionNorm(),
                R.getPressureCorrectionNorm());

            if (iteration > choices.maxRelaxNumber) {
                std::cout << "Maximum itteration number is reached:" << std::endl;
                std::cout << iteration << " Norm reached: " << norm << std::endl;
                if (norm > 0.001) {
                    std::cout << "Stop! Timestep: " << timeStep << std::endl;
                }
            }

        }
        time = time + choices.dt;
        {
            ScopedProfilePhase timer(ProfilePhase::VOFTotal);
            phi.explicitVOF(BCu, u, v, w,choices.dt);
        }

        {
            ScopedProfilePhase timer(ProfilePhase::UpdateRho);
            phi.updateRho(choices.Rho, choices.Mu, symetry, materialFields);
        }
        if (!choices.skipfluid) {
            {
                ScopedProfilePhase timer(ProfilePhase::DiagonalUpdate);
                D.update(u, v, w, BCu, BCv, BCw, choices.dt, choices.Re, rhoX, muX, rhoY, muY, rhoZ, muZ);
            }
            {
                ScopedProfilePhase timer(ProfilePhase::BDBTRebuild);
                if (legacyBDBTRebuild) {
                    MatrixBDBT BDBTnew = MatrixBDBT(
                        B.getValues(), B.getColumnIndices(), B.getRowStart(),
                        B.getRowEnd(), D.getDiag(), BDBTsize, D.getDiag().size(),
                        nx - 1, ny - 1, nz - 1);
                    BDBT = std::move(BDBTnew);
                }
                else {
                    BDBT.updateDiagonals(
                        B.getValues(),
                        B.getColumnIndices(),
                        B.getRowStart(),
                        B.getRowEnd(),
                        D.getDiag());
                }
            }
        }

        double currentCPUtime = wtime_safe() - start;
        bool finish = nextTimeStep(u,v,w,p,phi.getPhi(),choices.dt,choices.Iprint,choices.convgCritSS, currentCPUtime, time,timeStep);
        const auto [divergenceRms, divergenceMax] = divergenceDiagnostics(u, v, w, p);
        const double currentVofMass = vofMass(phi.getPhi());
        const double vofMassError = initialVofMass != 0.0
            ? (currentVofMass - initialVofMass) / initialVofMass
            : 0.0;
        profiler.recordPhysicalDiagnostics(divergenceRms, divergenceMax, vofMassError);
        if (timeStep % choices.Iprint == 0) {
            ScopedProfilePhase timer(ProfilePhase::IO);
            phi.savePhi(time, timeStep);
            phi.saveTecplotSnapshot(time, timeStep);
        }

        profiler.finishStep(omp_get_wtime() - stepStart);

        if (finish && !choices.skipfluid){
            endTimeStep = timeStep;
            break;
        }
    }


    // end timing
    double CPUtime = omp_get_wtime() - start;
    if (environmentFlagEnabled("VANKA_TOLERANCE_STUDY")) {
        if (!writeToleranceStudySnapshot(
                u, v, w, p, phi.getPhi(), "tolerance_snapshot.bin")) {
            throw std::runtime_error("Failed to write tolerance study snapshot");
        }
    }
    if (!environmentFlagEnabled("VANKA_SKIP_FINAL_OUTPUT")) {
        const double finalIoStart = omp_get_wtime();
        allOut(u,v,w,p,phi.getPhi(), CPUtime, time, endTimeStep);
        profiler.emitFinalIO(omp_get_wtime() - finalIoStart);
    }
    std::cout << "The program was executed successfully!\n";

    return 0;
}
