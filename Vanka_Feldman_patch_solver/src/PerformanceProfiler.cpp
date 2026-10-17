#include "PerformanceProfiler.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <omp.h>

namespace {
std::size_t phaseIndex(ProfilePhase phase) noexcept {
    return static_cast<std::size_t>(phase);
}

double safeRate(double numerator, double seconds) noexcept {
    return seconds > 0.0 ? numerator / seconds : 0.0;
}
} // namespace

PerformanceProfiler& PerformanceProfiler::instance() {
    static PerformanceProfiler profiler;
    return profiler;
}

bool environmentFlagEnabled(const char* name) noexcept {
#ifdef _WIN32
    char* value = nullptr;
    std::size_t length = 0;
    if (_dupenv_s(&value, &length, name) != 0 || value == nullptr) return false;
    const bool enabled = value[0] != '\0' && std::strcmp(value, "0") != 0;
    std::free(value);
    return enabled;
#else
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
#endif
}

PerformanceProfiler::PerformanceProfiler() {
    enabled_ = environmentFlagEnabled("VANKA_PROFILE");
}

void PerformanceProfiler::beginStep(
    int step,
    std::size_t vofCells,
    std::size_t matrixSize) {
    if (!enabled_) return;
    step_ = step;
    vofCells_ = vofCells;
    matrixSize_ = matrixSize;
    std::fill(std::begin(phaseSeconds_), std::end(phaseSeconds_), 0.0);
    cgCalls_ = 0;
    cgIterations_ = 0;
    spmvCalls_ = 0;
    outerIterations_ = 0;
    outerResidualU_ = 0.0;
    outerResidualV_ = 0.0;
    outerResidualW_ = 0.0;
    outerResidualP_ = 0.0;
    timeResidualU_ = 0.0;
    timeResidualV_ = 0.0;
    timeResidualW_ = 0.0;
    timeResidualP_ = 0.0;
    divergenceRms_ = 0.0;
    divergenceMax_ = 0.0;
    vofMassError_ = 0.0;
}

void PerformanceProfiler::add(ProfilePhase phase, double seconds) noexcept {
    if (enabled_) phaseSeconds_[phaseIndex(phase)] += seconds;
}

void PerformanceProfiler::recordCGSolve(std::int64_t iterations) noexcept {
    if (!enabled_) return;
    ++cgCalls_;
    cgIterations_ += iterations;
}

void PerformanceProfiler::recordSpmvCall() noexcept {
    if (enabled_) ++spmvCalls_;
}

void PerformanceProfiler::recordOuterIterations(std::int64_t iterations) noexcept {
    if (enabled_) outerIterations_ += iterations;
}

void PerformanceProfiler::recordOuterResiduals(
    double u,
    double v,
    double w,
    double p) noexcept {
    if (!enabled_) return;
    outerResidualU_ = u;
    outerResidualV_ = v;
    outerResidualW_ = w;
    outerResidualP_ = p;
}

void PerformanceProfiler::recordTimeResiduals(
    double u,
    double v,
    double w,
    double p) noexcept {
    if (!enabled_) return;
    timeResidualU_ = u;
    timeResidualV_ = v;
    timeResidualW_ = w;
    timeResidualP_ = p;
}

void PerformanceProfiler::recordPhysicalDiagnostics(
    double divergenceRms,
    double divergenceMax,
    double vofMassError) noexcept {
    if (!enabled_) return;
    divergenceRms_ = divergenceRms;
    divergenceMax_ = divergenceMax;
    vofMassError_ = vofMassError;
}

void PerformanceProfiler::finishStep(double totalSeconds) const {
    if (!enabled_) return;

    const auto seconds = [this](ProfilePhase phase) {
        return phaseSeconds_[phaseIndex(phase)];
    };
    const double vofTotal = seconds(ProfilePhase::VOFTotal);
    const double curvature = seconds(ProfilePhase::VOFCurvature);
    const double vofAdvection = std::max(0.0, vofTotal - curvature);
    const double io = seconds(ProfilePhase::IO);
    const double compute = std::max(0.0, totalSeconds - io);
    const double pcg = seconds(ProfilePhase::PCG);
    const double spmv = seconds(ProfilePhase::PCGSpmv);

    std::cout << std::setprecision(17)
        << "PERF_STEP {"
        << "\"step\":" << step_
        << ",\"vof_cells\":" << vofCells_
        << ",\"matrix_size\":" << matrixSize_
        << ",\"total_s\":" << totalSeconds
        << ",\"compute_s\":" << compute
        << ",\"rhsu_s\":" << seconds(ProfilePhase::RHSu)
        << ",\"rhsr_s\":" << seconds(ProfilePhase::RHSr)
        << ",\"d_multiply_rhs_s\":" << seconds(ProfilePhase::DMultiplyRHS)
        << ",\"b_multiply_s\":" << seconds(ProfilePhase::BMultiply)
        << ",\"pcg_s\":" << pcg
        << ",\"pcg_spmv_s\":" << spmv
        << ",\"pcg_blas_s\":" << seconds(ProfilePhase::PCGBlas)
        << ",\"pcg_preconditioner_s\":" << seconds(ProfilePhase::PCGPreconditioner)
        << ",\"b_multiply_t_s\":" << seconds(ProfilePhase::BMultiplyT)
        << ",\"d_multiply_correction_s\":" << seconds(ProfilePhase::DMultiplyCorrection)
        << ",\"correction_finalize_s\":" << seconds(ProfilePhase::CorrectionFinalize)
        << ",\"relax_update_s\":" << seconds(ProfilePhase::RelaxUpdate)
        << ",\"vof_total_s\":" << vofTotal
        << ",\"vof_advection_s\":" << vofAdvection
        << ",\"vof_curvature_s\":" << curvature
        << ",\"update_rho_s\":" << seconds(ProfilePhase::UpdateRho)
        << ",\"diagonal_update_s\":" << seconds(ProfilePhase::DiagonalUpdate)
        << ",\"bdbt_rebuild_s\":" << seconds(ProfilePhase::BDBTRebuild)
        << ",\"bdbt_scale_s\":" << seconds(ProfilePhase::BDBTScale)
        << ",\"bdbt_syrk_s\":" << seconds(ProfilePhase::BDBTSyrk)
        << ",\"bdbt_export_s\":" << seconds(ProfilePhase::BDBTExport)
        << ",\"bdbt_diagonal_extract_s\":" << seconds(ProfilePhase::BDBTDiagonalExtract)
        << ",\"step_finalize_s\":" << seconds(ProfilePhase::StepFinalize)
        << ",\"io_s\":" << io
        << ",\"outer_iterations\":" << outerIterations_
        << ",\"outer_residual_u\":" << outerResidualU_
        << ",\"outer_residual_v\":" << outerResidualV_
        << ",\"outer_residual_w\":" << outerResidualW_
        << ",\"outer_residual_p\":" << outerResidualP_
        << ",\"time_residual_u\":" << timeResidualU_
        << ",\"time_residual_v\":" << timeResidualV_
        << ",\"time_residual_w\":" << timeResidualW_
        << ",\"time_residual_p\":" << timeResidualP_
        << ",\"divergence_rms\":" << divergenceRms_
        << ",\"divergence_max\":" << divergenceMax_
        << ",\"vof_mass_error\":" << vofMassError_
        << ",\"cg_calls\":" << cgCalls_
        << ",\"cg_iterations\":" << cgIterations_
        << ",\"spmv_calls\":" << spmvCalls_
        << ",\"pcg_avg_iteration_s\":"
        << safeRate(pcg, static_cast<double>(cgIterations_))
        << ",\"spmv_avg_s\":" << safeRate(spmv, static_cast<double>(spmvCalls_))
        << ",\"spmv_mpoints_per_s\":"
        << safeRate(static_cast<double>(matrixSize_) * static_cast<double>(spmvCalls_), spmv) / 1.0e6
        << ",\"pcg_munknown_iterations_per_s\":"
        << safeRate(static_cast<double>(matrixSize_) * static_cast<double>(cgIterations_), pcg) / 1.0e6
        << "}" << std::endl;
}

void PerformanceProfiler::emitInitialization(
    double seconds,
    std::size_t vofCells,
    std::size_t matrixSize,
    std::size_t pointCount,
    std::size_t pointBytes,
    std::size_t materialDoubleCount) const {
    if (!enabled_) return;
    const std::size_t pointStorageBytes = pointCount * pointBytes;
    const std::size_t materialStorageBytes = materialDoubleCount * sizeof(double);
    const std::size_t bdbtDiagonalBytes = 6 * matrixSize * sizeof(double);
    const std::size_t pcgWorkspaceBytes = 7 * matrixSize * sizeof(double);
    std::cout << std::setprecision(17)
        << "PERF_INIT {"
        << "\"seconds\":" << seconds
        << ",\"vof_cells\":" << vofCells
        << ",\"matrix_size\":" << matrixSize
        << ",\"point_count\":" << pointCount
        << ",\"sizeof_point\":" << pointBytes
        << ",\"point_storage_bytes\":" << pointStorageBytes
        << ",\"material_double_count\":" << materialDoubleCount
        << ",\"material_storage_bytes\":" << materialStorageBytes
        << ",\"bdbt_diagonal_bytes\":" << bdbtDiagonalBytes
        << ",\"pcg_workspace_bytes\":" << pcgWorkspaceBytes
        << "}" << std::endl;
}

void PerformanceProfiler::emitFinalIO(double seconds) const {
    if (!enabled_) return;
    std::cout << std::setprecision(17)
        << "PERF_FINAL_IO {\"seconds\":" << seconds << "}" << std::endl;
}

ScopedProfilePhase::ScopedProfilePhase(ProfilePhase phase) noexcept
    : profiler_(&PerformanceProfiler::instance()), phase_(phase) {
    if (profiler_->enabled()) start_ = omp_get_wtime();
}

ScopedProfilePhase::~ScopedProfilePhase() {
    if (profiler_ != nullptr && profiler_->enabled()) {
        profiler_->add(phase_, omp_get_wtime() - start_);
    }
}
