#ifndef RHSR_H
#define RHSR_H

#include <vector>
#include "Point.h"
#include "BoundaryContainer.h"

class RHSr {
public:
    // Constructor
    RHSr(int BDBTsize);

    // Method to update the current RHSr
    void updateCurrent(const std::vector<std::vector<std::vector<Point>>>& p,
        const std::vector<std::vector<std::vector<Point>>>& u,
        const std::vector<std::vector<std::vector<Point>>>& v,
        const std::vector<std::vector<std::vector<Point>>>& w,
        const BoundaryContainer& boundaryContainer, int numDirichletPoints);

    // Getter method for the RHSrVector
    const std::vector<double>& getRHSr() const;

private:
    std::vector<double> RHScurrent;
};

#endif //RHSR_H