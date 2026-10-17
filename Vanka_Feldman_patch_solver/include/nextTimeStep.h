#ifndef TIMESTEP_H
#define TIMESTEP_H

#include "Point.h"
#include <vector>
#include <cmath> // for std::abs
#include <mkl.h>
#include <numeric>
#include "PerformanceProfiler.h"
extern void out(const std::vector<std::vector<std::vector<Point>>>& u,
    const std::vector<std::vector<std::vector<Point>>>& v,
    const std::vector<std::vector<std::vector<Point>>>& w,
    const std::vector<std::vector<std::vector<Point>>>& p,
    const std::vector<std::vector<std::vector<Point>>>& phi,
    double timeDerivativeU,
    double timeDerivativeV,
    double timeDerivativeW,
    double timeDerivativeP,
    double time,
    double CPUtime,
    double dt,
    int timeStep
    );

double timeDerivative(std::vector<std::vector<std::vector<Point>>>& field, bool is3D, double dt) {
    // Calculate and return the time derivative for a given 3D array (u, v, w or p).
    // Calculate total number of elements in the 3D vector
    size_t totalElements = field[0][0].size() * field[0].size() * (is3D ? field.size() : 1);

    // Preallocate memory for the 1D vector
    std::vector<double> PrevValue(totalElements,0.0);
    std::vector<double> Value(totalElements,0.0);
    std::vector<double> PrevPrevValue(totalElements,0.0);

    #pragma omp parallel for collapse(3)
    for (int k = 0; k < field.size(); ++k) {
        for (int j = 0; j < field[0].size(); ++j) {
            for (int i = 0; i < field[0][0].size(); ++i) {
                Point& elem = field[k][j][i];
                int index = elem.getUniqueNumber();  // assuming it's unique and within [0, totalElements)
    
                Value[index] = elem.getValue();
                PrevValue[index] = elem.getPrevValue();
                PrevPrevValue[index] = elem.getPrevPrevValue();
                elem.setPrevPrevValue(elem.getPrevValue());
                elem.setPrevValue(elem.getValue());
            }
        }
    }
    std::vector<double> resultingVector(totalElements);

    // 3 * Value
    cblas_dcopy(totalElements, Value.data(), 1, resultingVector.data(), 1);
    cblas_dscal(totalElements, 3.0, resultingVector.data(), 1);

    // -4 * PrevValue
    cblas_daxpy(totalElements, -4.0, PrevValue.data(), 1, resultingVector.data(), 1);

    // +1 * PrevPrevValue
    cblas_daxpy(totalElements, 1.0, PrevPrevValue.data(), 1, resultingVector.data(), 1);

    // Compute the norm of the resulting vector using MKL
    double norm = cblas_dnrm2(totalElements, resultingVector.data(), 1)/2.0/std::sqrt(totalElements);

    //auto it = std::max_element(resultingVector.begin(), resultingVector.end(),
    //    [](double a, double b) { return std::fabs(a) < std::fabs(b); });
    //double inf_norm = std::fabs(*it);

    return norm; 
}

bool nextTimeStep(std::vector<std::vector<std::vector<Point>>>& u,
    std::vector<std::vector<std::vector<Point>>>& v,
    std::vector<std::vector<std::vector<Point>>>& w,
    std::vector<std::vector<std::vector<Point>>>& p,
    const std::vector<std::vector<std::vector<Point>>>& phi,
    double dt, int Iprint, double convgCritSS,double currentCPUtime, double time, int timeStep) {

    bool is3D = (w.size() > 1);

    double timeDerivativeU;
    double timeDerivativeV;
    double timeDerivativeW = 0.0;
    double timeDerivativeP;
    {
        ScopedProfilePhase timer(ProfilePhase::StepFinalize);
        timeDerivativeU = timeDerivative(u, is3D, dt);
        timeDerivativeV = timeDerivative(v, is3D, dt);
        if (is3D) timeDerivativeW = timeDerivative(w, is3D, dt);
        timeDerivativeP = timeDerivative(p, is3D, dt);
    }
    PerformanceProfiler::instance().recordTimeResiduals(
        timeDerivativeU,
        timeDerivativeV,
        timeDerivativeW,
        timeDerivativeP);
    if (timeStep % Iprint == 0) {
        // Calculate the time derivatives

        {
            ScopedProfilePhase timer(ProfilePhase::IO);
            out(u, v, w, p, phi, timeDerivativeU, timeDerivativeV, timeDerivativeW, timeDerivativeP, time, currentCPUtime, dt, timeStep);
        }
        // Check for steady state
        if (std::abs(timeDerivativeU) < convgCritSS &&
            std::abs(timeDerivativeV) < convgCritSS &&
            (!is3D || std::abs(timeDerivativeW) < convgCritSS) &&
            std::abs(timeDerivativeP) < convgCritSS) {
            return true; // Steady state reached
        }
    }

    return false; // Steady state not reached
}

#endif // TIMESTEP_H
