#ifndef OUTPUT_H
#define OUTPUT_H

#include <vector>
#include <string>
#include "Point.h"


void writeTecplotFile(
    const std::vector<std::vector<std::vector<Point>>>& field,
    const std::string& varName,
    const std::string& fileName,
    double time,
    int timeStep
);
void writeTecplotFile(
    const std::vector<std::vector<std::vector<Point>>>& field,
    const std::vector<std::vector<std::vector<double>>>& fieldValue,
    const std::string& varName,
    const std::string& fileName,
    double time,
    int timeStep
);

bool writeTecplotBinarySnapshot(
    const std::vector<std::vector<std::vector<Point>>>& field,
    const std::string& varName,
    const std::string& fileName,
    double time,
    int timeStep
);

void writeAllTecplotFile(
    const std::vector<std::vector<std::vector<double>>>& uModified,
    const std::vector<std::vector<std::vector<double>>>& vModified,
    const std::vector<std::vector<std::vector<double>>>& wModified,
    const std::vector<std::vector<std::vector<Point>>>& p,
    const std::string& fileName,
    double time,
    int timeStep
);

void writeAllTecplotFile(
    const std::vector<std::vector<std::vector<double>>>& uModified,
    const std::vector<std::vector<std::vector<double>>>& vModified,
    const std::vector<std::vector<std::vector<double>>>& wModified,
    const std::vector<std::vector<std::vector<Point>>>& p,
    const std::vector<std::vector<std::vector<Point>>>& phi,
    const std::string& fileName,
    double time,
    int timeStep
);

void out(
    const std::vector<std::vector<std::vector<Point>>>& u,
    const std::vector<std::vector<std::vector<Point>>>& v,
    const std::vector<std::vector<std::vector<Point>>>& w,
    const std::vector<std::vector<std::vector<Point>>>& p,
    double timeDerivativeU,
    double timeDerivativeV,
    double timeDerivativeW,
    double timeDerivativeP,
    double time,
    double CPUtime,
    int timeStep,
    double dt
);
void out(
    const std::vector<std::vector<std::vector<Point>>>& u,
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

void allOut(
    const std::vector<std::vector<std::vector<Point>>>& u,
    const std::vector<std::vector<std::vector<Point>>>& v,
    const std::vector<std::vector<std::vector<Point>>>& w,
    const std::vector<std::vector<std::vector<Point>>>& p,
    double CPUtime,
    double time,
    int timeStep
);

// Compact, full-precision cell-centred state used only by controlled
// tolerance-sensitivity benchmarks.
bool writeToleranceStudySnapshot(
    const std::vector<std::vector<std::vector<Point>>>& u,
    const std::vector<std::vector<std::vector<Point>>>& v,
    const std::vector<std::vector<std::vector<Point>>>& w,
    const std::vector<std::vector<std::vector<Point>>>& p,
    const std::vector<std::vector<std::vector<Point>>>& phi,
    const std::string& fileName);
void allOut(
    const std::vector<std::vector<std::vector<Point>>>& u,
    const std::vector<std::vector<std::vector<Point>>>& v,
    const std::vector<std::vector<std::vector<Point>>>& w,
    const std::vector<std::vector<std::vector<Point>>>& p,
    const std::vector<std::vector<std::vector<Point>>>& phi,
    double CPUtime,
    double time,
    int timeStep
);
/*void VideoOut(
    const std::vector<std::vector<std::vector<Point>>>& u,
    const std::vector<std::vector<std::vector<Point>>>& v,
    const std::vector<std::vector<std::vector<Point>>>& w,
    const std::vector<std::vector<std::vector<Point>>>& p,
    const std::vector<std::vector<std::vector<Point>>>& phi,
    double time,
    int timeStep
);*/

std::vector<std::vector<std::vector<double>>> interpolateAndProcess(
    const std::vector<std::vector<std::vector<Point>>>& field,
    bool is3D,
    char dimension
);

#endif // OUTPUT_H
