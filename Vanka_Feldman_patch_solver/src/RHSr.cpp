#include <algorithm>
#include "RHSr.h"
#include "BoundaryContainer.h"
#include <iostream>
#include <mkl.h>

RHSr::RHSr(int BDBTsize) : RHScurrent(BDBTsize, 0.0) {}

void RHSr::updateCurrent(const std::vector<std::vector<std::vector<Point>>>& p,
    const std::vector<std::vector<std::vector<Point>>>& u,
    const std::vector<std::vector<std::vector<Point>>>& v,
    const std::vector<std::vector<std::vector<Point>>>& w,
    const BoundaryContainer& boundaryContainer, int numDirichletPoints) {

    // Determine if the system is 3D
    bool is3D = (w.size() > 1);

    size_t n = p.size();
    size_t m = p[0].size();
    size_t l = p[0][0].size();

    
    // Temporary vector to store the sums
    std::vector<double> tempSums(n * m * l- numDirichletPoints, 0.0);
    #pragma omp parallel for collapse(3)
    for (size_t k = 0; k < n; ++k) {
        for (size_t j = 0; j < m; ++j) {
            for (size_t i = 0; i < l; ++i) {
                 
                // Equation number
                int index = (p[k][j][i].getUniqueNumber() - numDirichletPoints);
                if (index < 0) continue;
                // Calculate derivatives
                double dudx = 0.0, dvdy = 0.0, dwdz = 0.0;

                
                    dudx = (u[is3D ? k + 1 : 0 ][j + 1][i + 1].getValue() - u[is3D ? k + 1 : 0 ][j + 1][i].getValue()) * p[k][j][i].getdy() * (is3D ? p[k][j][i].getdz() : 1);
                

               
                    dvdy = (v[is3D ? k + 1 : 0 ][j + 1][i+1].getValue() - v[is3D ? k + 1 : 0 ][j][i+1].getValue()) * p[k][j][i].getdx() * (is3D ? p[k][j][i].getdz() : 1);
               

                if (is3D) {
                    dwdz = (w[k + 1][j+1][i+1].getValue() - w[k][j+1][i+1].getValue()) * p[k][j][i].getdx() * p[k][j][i].getdy();
                }

                // Sum them up
                double sum = +dudx + dvdy + dwdz;
                tempSums[index] = sum;
            }
        }
    }

    // Copy the sums to the RHScurrent
    int nSum = static_cast<int>(RHScurrent.size());
    cblas_dcopy(nSum, &tempSums[0], 1, &RHScurrent[0], 1);
}

const std::vector<double>& RHSr::getRHSr() const {
    return RHScurrent;
}