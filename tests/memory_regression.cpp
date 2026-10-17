#include "Initialize.h"
#include "VOF.h"
#include "Relax.h"
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <omp.h>
#include <stdexcept>
#include <type_traits>

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void checkCompactStorage() {
    static_assert(sizeof(PointBoundary) == 1 && sizeof(BoundaryKind) == 1);
    static_assert(sizeof(Point) <= 96);
    static_assert(sizeof(Boundary) <= 16 && sizeof(BoundaryContainer) <= 96);
    static_assert(std::is_trivially_copyable_v<Point>);
    const Point defaultPoint;
    require(defaultPoint.getUniqueNumber() == -1, "Default ID sentinel changed");
    require(defaultPoint.getBoundaryType() == PointBoundary::Interior, "Default boundary changed");
    for (const char letter : std::string("0ABCDEFGHIJKLMNOPQRSTUVWXYZ")) {
        Point point(65535, 400, 65535, 4000000, 0.1);
        point.setBoundaryType(std::string(1, letter));
        const Point copy = point;
        require(boundaryLetter(copy.getBoundaryType()) == letter, "Boundary code round trip failed");
        require(copy.getI() == 65535 && copy.getJ() == 400 && copy.getK() == 65535,
            "16-bit coordinate storage truncated");
        require(copy.getUniqueNumber() == 4000000, "Linear ID incorrectly narrowed");
    }
    for (const int invalid : {-1, 65536}) {
        bool rejected = false;
        try { Point point(invalid, 0, 0, 0, 0.1); }
        catch (const std::out_of_range&) { rejected = true; }
        require(rejected, "Invalid coordinate silently truncated");
        rejected = false;
        try { Point::validateGridIndices(0, invalid, 0); }
        catch (const std::out_of_range&) { rejected = true; }
        require(rejected, "Grid validation did not reject oversized/negative coordinates");
    }
    Point sentinel(0,0,0,std::numeric_limits<int>::max(),0.1);
    require(sentinel.getUniqueNumber() == std::numeric_limits<int>::max(), "Maximum linear ID changed");
    BoundaryContainer boundaries;
    // Both initialization overloads must reject an oversized mesh before any
    // allocation or OpenMP worker constructs a Point (no cross-region throw).
    bool oversizedRejected = false;
    try { initialize(boundaries,boundaries,boundaries,6553.5,0.2,0.0,0.1); }
    catch (const std::out_of_range&) { oversizedRejected = true; }
    require(oversizedRejected, "Simple initialization did not validate its mesh");
    oversizedRejected = false;
    try {
        const std::vector<std::vector<std::vector<Point>>> emptyPhi;
        initialize(boundaries,boundaries,boundaries,6553.5,0.2,0.0,0.1,
            0.0,0.0,0.0,emptyPhi,3.0,1.0,false);
    } catch (const std::out_of_range&) { oversizedRejected = true; }
    require(oversizedRejected, "Material initialization did not validate its mesh");
    const char* names[] = {"fixed_value", "slip", "convective_boundary", "symetry",
        "dirichlet_point", "dirichlet_row"};
    const char* faces[] = {"E", "W", "N", "S", "F", "B"};
    const auto filename = std::filesystem::path("memory_regression_bc.inp");
    {
        std::ofstream file(filename);
        file << "# Legacy text input\n";
        for (int i = 0; i < 6; ++i) file << faces[i] << ' ' << names[i] << ' ' << i + 0.25 << '\n';
    }
    boundaries.loadFromFile(filename.string());
    std::filesystem::remove(filename);
    for (int i = 0; i < 6; ++i) {
        const auto& boundary = boundaries.getBoundary(pointBoundaryFromText(faces[i]));
        require(boundaryKindName(boundary.type) == std::string(names[i]), "Condition name round trip failed");
        require(boundary.value == i + 0.25, "Condition value changed");
    }
    bool rejected = false;
    try { boundaries.getBoundary(PointBoundary::D); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "An edge was incorrectly accepted as a face");
    rejected = false;
    try { boundaryKindFromText("unknown"); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Unknown condition silently accepted");
}

// Emit individual numerical results for a before/after storage-layout comparison.
// Run from a build directory: generated fixtures never touch the user's inputs.
void emit(const char* label, const std::vector<double>& values) {
    for (size_t i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values[i])) throw std::runtime_error("Non-finite regression result");
        std::cout << "RESULT " << label << ' ' << i << ' ' << values[i] << '\n';
    }
}
void emitField(const char* label, const ScalarField3D& values) {
    for (const auto& layer : values) for (const auto& row : layer) emit(label, row);
}

void runPeriodicConservation();

void runCase(bool is3D, bool periodic) {
    constexpr int nx = 6, ny = 7;
    const int nz = is3D ? 5 : 1;
    constexpr double ds = 0.1, dt = 0.001;
    const auto filename = std::filesystem::path("memory_regression_phi.dat");
    {
        std::ofstream file(filename);
        file << std::setprecision(17) << "VARIABLES=X,Y,Z,Phi\nZONE I=" << nx << ", J=" << ny;
        if (is3D) file << ", K=" << nz;
        file << ", F=POINT\n";
        for (int k = 0; k < nz; ++k) for (int j = 0; j < ny; ++j) for (int i = 0; i < nx; ++i) {
            file << (i + 0.5)*ds << ' ' << (j + 0.5)*ds << ' ';
            if (is3D) file << (k + 0.5)*ds << ' ';
            file << 0.4 + 0.1*std::sin(i + 0.7*j + 0.3*k) << '\n';
        }
    }
    VOF phi(filename.string(), ds, nx*ds, ny*ds, is3D ? nz*ds : 0.0, 1.0);
    std::filesystem::remove(filename);
    BoundaryContainer bcu, bcv, bcw, bcp;
    bcv.addBoundary("E", {"slip", 0.0});
    bcv.addBoundary("W", {"slip", 0.0});
    if (periodic) for (auto* bc : {&bcu, &bcv, &bcw}) {
        bc->addBoundary("E", {"symetry", 0.0});
        bc->addBoundary("W", {"symetry", 0.0});
        bc->addBoundary("N", {"fixed_value", 0.0});
    }
    bcp.addBoundary("E", {"dirichlet_point", 0.0});
    bcp.addBoundary("W", {"fixed_value", 0.0});
    bcp.addBoundary("N", {"fixed_value", 0.0});
    bcp.addBoundary("B", {"dirichlet_point", 1.0});
    auto [u,v,w,p] = initialize(bcu, bcv, bcw, nx*ds, ny*ds, is3D ? nz*ds : 0.0, ds,
        0.03, 0.02, 0.01, phi.getPhi(), 3.0, 1.0, false);
    MaterialFields f;
    phi.updateRho(3.0, 2.0, periodic, f);
    emitField("muS", f.muS);
    emitField("dmudx", f.dmudx);
    emitField("dmudy", f.dmudy);
    emitField("dmudz", f.dmudz);
    DiagonalMatrix d(u,v,w,bcu,bcv,bcw,dt,100.0,f.rhoX,f.muX,f.rhoY,f.muY,f.rhoZ,f.muZ);
    PressureMatrix b(u,v,w,p,bcp,d,periodic);
    const auto n = b.getRowStart().size();
    const int dirichlet = nx*ny*nz-static_cast<int>(n);
    MatrixBDBT a(b.getValues(),b.getColumnIndices(),b.getRowStart(),b.getRowEnd(),d.getDiag(),
        n,d.getDiag().size(),nx,ny,is3D ? nz : 0);
    RHSu rhs(d);
    RHSr rhsr(static_cast<int>(n));
    rhs.updateTimestep(u,v,w,dt);
    Relax r(u,v,w,p);
    for (int iteration = 0; iteration < (periodic ? 20 : 2); ++iteration) {
        r.currentStep(u,v,w,p,f.rhoX,f.rhoY,f.rhoZ,f.muX,f.muY,f.muZ,f.dmudx,f.dmudy,f.dmudz,
            3.0,dt,static_cast<int>(n),bcu,bcv,bcw,bcp,a,d,b,100.0,1.0,1e8,rhs,rhsr,1e-14,
            dirichlet,phi.getPhiS(),phi.getKappa());
        if (iteration < 2) {
            emit("rhs", rhs.getRHSu());
            emit("pressure",r.getPressureCorrection());
            emit("velocity",r.getVelocityCorrection());
        }
        r.relax(u,v,w,p,periodic);
        if (periodic) {
            const auto checkSeam = [](const auto& field) {
                for (const auto& plane : field)
                    for (const auto& row : plane)
                        if (row.front().getBoundaryType() == PointBoundary::W &&
                            row.back().getBoundaryType() == PointBoundary::E)
                            require(std::abs(row.front().getValue() - row.back().getValue()) < 1e-14,
                                "Periodic fluid faces diverged");
            };
            checkSeam(u);
            checkSeam(v);
            if (is3D) checkSeam(w);

            double netFlux = 0.0;
            double divergenceSquares = 0.0;
            double divergenceMax = 0.0;
            for (int k = 0; k < nz; ++k)
                for (int j = 0; j < ny; ++j)
                    for (int i = 0; i < nx; ++i) {
                        const Point& cell = p[k][j][i];
                        const int velocityK = is3D ? k + 1 : 0;
                        double cellFlux = (u[velocityK][j + 1][i + 1].getValue() -
                            u[velocityK][j + 1][i].getValue()) * cell.getdy() *
                            (is3D ? cell.getdz() : 1.0);
                        cellFlux += (v[velocityK][j + 1][i + 1].getValue() -
                            v[velocityK][j][i + 1].getValue()) * cell.getdx() *
                            (is3D ? cell.getdz() : 1.0);
                        if (is3D)
                            cellFlux += (w[k + 1][j + 1][i + 1].getValue() -
                                w[k][j + 1][i + 1].getValue()) *
                                cell.getdx() * cell.getdy();
                        netFlux += cellFlux;
                        const double divergence = cellFlux /
                            (cell.getdx() * cell.getdy() *
                                (is3D ? cell.getdz() : 1.0));
                        divergenceSquares += divergence * divergence;
                        divergenceMax = std::max(divergenceMax, std::abs(divergence));
                    }
            require(std::abs(netFlux) < 1e-11, "Periodic fluid has net boundary flux");
            if (iteration == 19) {
                const double rms = std::sqrt(divergenceSquares / (nx * ny * nz));
                emit(is3D ? "periodic_3d_divergence" : "periodic_2d_divergence",
                    {rms, divergenceMax});
                require(divergenceMax < 1e-8,
                    "Periodic fluid did not converge to a divergence-free field");
            }
        }
    }
    emit("D",d.getDiag());
    emit("B",b.getValues());
    emit("A",a.getDiag0());
    phi.updateLHS(bcu,u,v,w);
    emit("productLHS",phi.productLHS(bcu,u,v,w,phi.getPhiV()));
    emit("explicitVOF",phi.explicitVOF(bcu,u,v,w,dt));
    phi.updateRho(3.0,2.0,periodic,f);
    emitField("muS_after",f.muS);
}
int main(int argc, char** argv) {
    try {
        mkl_set_dynamic(0);
        checkCompactStorage();
        std::cout << std::setprecision(17) << "Point bytes: " << sizeof(Point) << '\n';
        // A single CTest process checks both settings. The optional argument
        // selects one setting for separate before/after numerical snapshots.
        const auto run = [](int threads) {
            omp_set_num_threads(threads);
            mkl_set_num_threads(threads);
            runCase(false,false);
            runCase(false,true);
            runCase(true,false);
            runCase(true,true);
        };
        if (argc > 1) run(std::stoi(argv[1]));
        else { run(1); run(4); }
        runPeriodicConservation();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
