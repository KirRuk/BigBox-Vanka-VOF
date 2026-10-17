#ifndef READ_H
#define READ_H

#include <iostream>
#include <map>
#include <vector>
#include <mkl.h>
struct UserChoices {
    double Lx, Ly, Lz, ds, dt, terminationTime, Re, UR, convgCrit, convgCritSS, supportCellsRad, CG, Rho, Mu, Fr, We;
    bool twoDimensional, read_data, dynamics, mooving, readBody, skipfluid, developed;
    int writeFreq, nSubdomainsX, nSubdomainsY, maxRelaxNumber, Iprint, Multigrid, threads;
    std::vector<MKL_INT> DynCoord;
};

UserChoices read(const std::string& filename);

#endif // read