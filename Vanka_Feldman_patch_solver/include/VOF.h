// VOF.h
#ifndef VOF_H
#define VOF_H

#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include "Point.h"
#include "BoundaryContainer.h"
#include <mkl.h>

using ScalarField3D = std::vector<std::vector<std::vector<double>>>;

struct MaterialFields {
    ScalarField3D rhoX;
    ScalarField3D rhoY;
    ScalarField3D rhoZ;
    ScalarField3D muX;
    ScalarField3D muY;
    ScalarField3D muZ;
    ScalarField3D dmudx;
    ScalarField3D dmudy;
    ScalarField3D dmudz;

    // Cell-centred viscosity is scratch storage used to construct dmu.
    // Keeping it here lets updateRho reuse its allocation as well.
    ScalarField3D muS;
};

struct Matrix {
private:
    std::vector<double> values;
    std::vector<MKL_INT> col_indices;
    std::vector<MKL_INT> row_ptr;
    std::vector<MKL_INT> row_ptr_end;
    int n;
public:
    const std::vector<MKL_INT>& getRowPtr() const { return row_ptr; }
    const std::vector<MKL_INT>& getRowPtr_end() const { return row_ptr_end; }
    const std::vector<MKL_INT>& getColIndices() const { return col_indices; }
    const std::vector<double>& getValues() const { return values; }
    int getSize() const { return n; }
    void setn(int Neq) { this->n = Neq; }
    void setValues(std::vector<double> Val) { this->values = Val; }
    void setRows(std::vector<MKL_INT> rows) { this->row_ptr = rows; }
    void setRowsEnd(std::vector<MKL_INT> rows) { this->row_ptr_end = rows; }
    void setCols(std::vector<MKL_INT> cols) { this->col_indices = cols; }
};
class VOF {

private:
    std::vector<std::vector<std::vector<Point>>> phi;
    std::vector<std::vector<std::vector<double>>> kappa, phiS;
    Matrix LHS;
    sparse_matrix_t LHS_mkl;
    double initialMass;
    bool periodicX = false;
    std::vector<double> RHS;
    std::vector<bool> checkDirection(int k, int j, int i, const std::vector<std::vector<std::vector<Point>>>& u, const std::vector<std::vector<std::vector<Point>>>& v, const std::vector<std::vector<std::vector<Point>>>& w, bool is3D) const;
    std::vector<double> calculateFluxSide(int k, int j, int i, const std::vector<std::vector<std::vector<Point>>>& u, const std::vector<std::vector<std::vector<Point>>>& v, const std::vector<std::vector<std::vector<Point>>>& w, bool is3D, char side) const;
    std::vector<double> calculateFluxFOU(int k, int j, int i, const std::vector<std::vector<std::vector<Point>>>& u, const std::vector<std::vector<std::vector<Point>>>& v, const std::vector<std::vector<std::vector<Point>>>& w, bool is3D) const;
    std::vector<double> symetry(int k, int j, int i, const std::vector<std::vector<std::vector<Point>>>& u, const std::vector<std::vector<std::vector<Point>>>& v, const std::vector<std::vector<std::vector<Point>>>& w, bool is3D, char side) const;
    std::vector<double> calculateFluxSECOFOU(int k, int j, int i, const std::vector<std::vector<std::vector<Point>>>& u, const std::vector<std::vector<std::vector<Point>>>& v, const std::vector<std::vector<std::vector<Point>>>& w, bool is3D) const;

    void calculateKappa(int smoothingPhi, double small, int smoothingKappa, double alphaKappa);
    double  prevComX, prevComY, prevComZ;



public:
    const std::vector<std::vector<std::vector<Point>>>& getPhi();
    std::vector<double> getPhiV() const;
    const std::vector<std::vector<std::vector<double>>>& getKappa();
    const std::vector<std::vector<std::vector<double>>>& getPhiS();
    VOF(const std::string& filename, double ds, double Lx, double Ly, double Lz, double phase);
    VOF(const std::string& filename, double ds);
    void updateRHS(const BoundaryContainer& BCu, const std::vector<std::vector<std::vector<Point>>> u, double dt);
    const std::vector<double>& getRHS() { return RHS; };
    void timeStep(std::vector<double> phiVetor);
    void timeStepImplcit(const std::vector<std::vector<std::vector<Point>>>& u, const std::vector<std::vector<std::vector<Point>>>& v, double dt, double time);
    void innerLoopStep(std::vector<double> phiVetor);
    void writecomputebubble3D(double time, double comX, double comY, double comZ, double meanVelocity, double avgVelocity, double massLoss);
    std::vector<double> solveVOF(int maxIterations, double tolerance, std::vector<double>& x, int outputLevel);
    std::vector<double> explicitVOF(BoundaryContainer& boundaryContainerU, const std::vector<std::vector<std::vector<Point>>>& u, const std::vector<std::vector<std::vector<Point>>>& v, const std::vector<std::vector<std::vector<Point>>>& w, double dt);
    std::tuple<double, double, double, double, double, double> computeBubbleProperties(double time, const std::vector<std::vector<std::vector<Point>>>& u, const std::vector<std::vector<std::vector<Point>>>& v, const std::vector<std::vector<std::vector<Point>>>& w, double dt, double phase);
    void updateLHS(BoundaryContainer& boundaryContainerU, const std::vector<std::vector<std::vector<Point>>>& u, const std::vector<std::vector<std::vector<Point>>>& v, const std::vector<std::vector<std::vector<Point>>>& w, double dt);
    void updateLHS(BoundaryContainer& boundaryContainerU, const std::vector<std::vector<std::vector<Point>>>& u, const std::vector<std::vector<std::vector<Point>>>& v, const std::vector<std::vector<std::vector<Point>>>& w);
    std::vector<double> productLHS(const BoundaryContainer& boundaryContainerU, const std::vector<std::vector<std::vector<Point>>>& u, const std::vector<std::vector<std::vector<Point>>>& v, const std::vector<std::vector<std::vector<Point>>>& w, const std::vector<double>& x) const;
    Matrix const getMatrix() { return LHS; };
    void saveMatrix();
    void savePhi(double time, MKL_INT timeStep);
    void saveTecplotSnapshot(double time, MKL_INT timeStep);
    void setPeriodicX(bool periodic);
    void SelectiveSmoothPhi(int iterations, double epsilon);
    void updateRho(
        double rhoRelation,
        double muRelation,
        bool symetry,
        MaterialFields& fields);
    double findInterfaceY_cubicSpline(int k, int i, double eps_band = 0.1, int maxPts = 12) const;

};
#endif // IBM_H

