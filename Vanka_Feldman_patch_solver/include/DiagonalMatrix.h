// DiagonalMatrix.h

#ifndef DIAGONALMATRIX_H
#define DIAGONALMATRIX_H

#include <vector>
#include <stdexcept>
#include <string>
#include <fstream>
#include <iostream>
#include "Point.h"
#include "BoundaryContainer.h" // Assuming you have this header for BoundaryContainer

class DiagonalMatrix {
public:
    // Constructor for Diag Object
    DiagonalMatrix(const std::vector < std::vector<std::vector<Point>>>& u,
        const std::vector < std::vector<std::vector<Point>>>& v, const std::vector < std::vector<std::vector<Point>>>& w,
        const BoundaryContainer& BCu, const  BoundaryContainer& BCv, const  BoundaryContainer& BCw, double dt, double Re,
        const std::vector < std::vector<std::vector<double>>>& rhoX, const std::vector < std::vector<std::vector<double>>>& muX,
        const std::vector < std::vector<std::vector<double>>>& rhoY, const std::vector < std::vector<std::vector<double>>>& muY,
        const std::vector < std::vector<std::vector<double>>>& rhoZ, const std::vector < std::vector<std::vector<double>>>& muZ);

    // Method to update the diagonal matrix
    void update(const std::vector < std::vector<std::vector<Point>>>& u,
        const std::vector < std::vector<std::vector<Point>>>& v, const std::vector < std::vector<std::vector<Point>>>& w,
        const BoundaryContainer& BCu, const  BoundaryContainer& BCv, const  BoundaryContainer& BCw, double dt, double Re,
        const std::vector < std::vector<std::vector<double>>>& rhoX, const std::vector < std::vector<std::vector<double>>>& muX,
        const std::vector < std::vector<std::vector<double>>>& rhoY, const std::vector < std::vector<std::vector<double>>>& muY,
        const std::vector < std::vector<std::vector<double>>>& rhoZ, const std::vector < std::vector<std::vector<double>>>& muZ);

    // Method for vector multiplication
    std::vector<double> multiply_with_vector(const std::vector<double>& vec) const;
    void multiply_with_vector(
        const std::vector<double>& vec,
        std::vector<double>& result) const;

    // Method for Direchlet points adition
    void addPressureDirichlet();
    void saveToFile(const std::string& filename) const {
        std::ofstream outFile(filename);
        if (!outFile.is_open()) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return;
        }

        for (const auto& value : diag) {
            outFile << value << std::endl;
        }

        outFile.close();
    }

    double getValue(int i);
    const std::vector<double>& getDiag() const;
private:
    std::vector<double> diag;

    double calculateValue(size_t k, size_t j, size_t i,
        size_t zSize, size_t ySize, size_t xSize,
        double dt, double Re,const BoundaryContainer& BC,const std::vector < std::vector<std::vector<Point>>>& component, const double& rho, const double& mu, char indexVar);
};

#endif // DIAGONALMATRIX_H
