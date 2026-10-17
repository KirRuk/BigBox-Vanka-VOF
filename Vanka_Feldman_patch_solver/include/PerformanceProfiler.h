#pragma once

#include <cstddef>
#include <cstdint>

bool environmentFlagEnabled(const char* name) noexcept;

enum class ProfilePhase : std::uint8_t {
    RHSu,
    RHSr,
    DMultiplyRHS,
    BMultiply,
    PCG,
    PCGSpmv,
    PCGBlas,
    PCGPreconditioner,
    BMultiplyT,
    DMultiplyCorrection,
    CorrectionFinalize,
    RelaxUpdate,
    VOFTotal,
    VOFCurvature,
    UpdateRho,
    DiagonalUpdate,
    BDBTRebuild,
    BDBTScale,
    BDBTSyrk,
    BDBTExport,
    BDBTDiagonalExtract,
    StepFinalize,
    IO,
    Count
};

class PerformanceProfiler {
public:
    static PerformanceProfiler& instance();

    bool enabled() const noexcept { return enabled_; }
    void beginStep(int step, std::size_t vofCells, std::size_t matrixSize);
    void add(ProfilePhase phase, double seconds) noexcept;
    void recordCGSolve(std::int64_t iterations) noexcept;
    void recordSpmvCall() noexcept;
    void recordOuterIterations(std::int64_t iterations) noexcept;
    void recordOuterResiduals(double u, double v, double w, double p) noexcept;
    void recordTimeResiduals(double u, double v, double w, double p) noexcept;
    void recordPhysicalDiagnostics(
        double divergenceRms,
        double divergenceMax,
        double vofMassError) noexcept;
    void finishStep(double totalSeconds) const;
    void emitInitialization(
        double seconds,
        std::size_t vofCells,
        std::size_t matrixSize,
        std::size_t pointCount,
        std::size_t pointBytes,
        std::size_t materialDoubleCount) const;
    void emitFinalIO(double seconds) const;

private:
    PerformanceProfiler();

    bool enabled_ = false;
    int step_ = 0;
    std::size_t vofCells_ = 0;
    std::size_t matrixSize_ = 0;
    double phaseSeconds_[static_cast<std::size_t>(ProfilePhase::Count)]{};
    std::int64_t cgCalls_ = 0;
    std::int64_t cgIterations_ = 0;
    std::int64_t spmvCalls_ = 0;
    std::int64_t outerIterations_ = 0;
    double outerResidualU_ = 0.0;
    double outerResidualV_ = 0.0;
    double outerResidualW_ = 0.0;
    double outerResidualP_ = 0.0;
    double timeResidualU_ = 0.0;
    double timeResidualV_ = 0.0;
    double timeResidualW_ = 0.0;
    double timeResidualP_ = 0.0;
    double divergenceRms_ = 0.0;
    double divergenceMax_ = 0.0;
    double vofMassError_ = 0.0;
};

class ScopedProfilePhase {
public:
    explicit ScopedProfilePhase(ProfilePhase phase) noexcept;
    ~ScopedProfilePhase();

    ScopedProfilePhase(const ScopedProfilePhase&) = delete;
    ScopedProfilePhase& operator=(const ScopedProfilePhase&) = delete;

private:
    PerformanceProfiler* profiler_ = nullptr;
    ProfilePhase phase_ = ProfilePhase::RHSu;
    double start_ = 0.0;
};
