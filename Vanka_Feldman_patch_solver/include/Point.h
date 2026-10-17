#pragma once
#ifndef POINT_H
#define POINT_H

#include <vector>
#include <cstdint>
#include <limits>
#include "BoundaryCodes.h"

class Point {
private:
    // Pack metadata together to avoid padding between the doubles below.
    // A linear ID can exceed 65535; keep its signed 32-bit range and -1 sentinel.
    std::int32_t unique_number;
    std::uint16_t i, j, k; // Per-axis indices: checked range 0..65535, including ghosts.
    PointBoundary boundaryType;
    double value;  // Value of the point
    double prev_value;  // Value at the previous timestep
    double prev_prev_value;  // Value at the timestep before the previous one
    double correction;  // Correction value

    // Distances to neighboring points
    double dx, dy, dz;
    // Coordinates
    double x, y, z;

    static std::uint16_t checkedIndex(int index) {
        if (index < 0 || index > std::numeric_limits<std::uint16_t>::max()) {
            throw std::out_of_range("Point axis index must be in 0..65535 (including ghost cells)");
        }
        return static_cast<std::uint16_t>(index);
    }

public:
    // Call before allocating/entering OpenMP initialization loops. Exceptions
    // must not escape a parallel region if a requested mesh is too wide.
    static void validateGridIndices(int maxI, int maxJ, int maxK) {
        (void)checkedIndex(maxI);
        (void)checkedIndex(maxJ);
        (void)checkedIndex(maxK);
    }

    // Constructor with index only, all other properties set to zero or equivalent
    Point(int i, int j, int k, int unique_number, double ds)
        : unique_number(unique_number), i(checkedIndex(i)), j(checkedIndex(j)), k(checkedIndex(k)),
        boundaryType(PointBoundary::Interior), value(0), prev_value(0), prev_prev_value(0), correction(0),
        dx(ds), dy(ds), dz(ds),x(0),y(0),z(0) {} 

    // Constructor with given dx and dy, others set to zero or equivalent
    Point(int i, int j, int k, int unique_number, double dx, double dy)
        : unique_number(unique_number), i(checkedIndex(i)), j(checkedIndex(j)), k(checkedIndex(k)),
        boundaryType(PointBoundary::Interior), value(0), prev_value(0), prev_prev_value(0), correction(0),
        dx(dx), dy(dy), dz((dx + dy)/2.0), x(0), y(0), z(0) {}

    // Constructor with given dx, dy, and dz
    Point(int i, int j, int k, int unique_number, double dx, double dy, double dz)
        : unique_number(unique_number), i(checkedIndex(i)), j(checkedIndex(j)), k(checkedIndex(k)),
        boundaryType(PointBoundary::Interior), value(0), prev_value(0), prev_prev_value(0), correction(0),
        dx(dx), dy(dy), dz(dz), x(0), y(0), z(0) {}
    //Empty constructor
    Point() 
    : unique_number(-1), i(0), j(0), k(0), boundaryType(PointBoundary::Interior),
      value(0), prev_value(0), prev_prev_value(0), correction(0),
      dx(0), dy(0), dz(0),
      x(0), y(0), z(0) {}

    // Getter methods
    int getI() const { return i; }
    int getJ() const { return j; }
    int getK() const { return k; }
    int getUniqueNumber() const { return unique_number; }
    double getValue() const { return value; }
    double getPrevValue() const { return prev_value; }
    double getPrevPrevValue() const { return prev_prev_value; }
    double getCorrection() const { return correction; }
    PointBoundary getBoundaryType() const noexcept { return boundaryType; }
    double getdx() const { return dx; }
    double getdy() const { return dy; }
    double getdz() const { return dz; }
    double getX() const { return x; }
    double getY() const { return y; }
    double getZ() const { return z; }

    // Setter methods
    void setX(double newX) { x = newX; }
    void setY(double newY) { y = newY; }
    void setZ(double newZ) { z = newZ; }
    void setValue(double newValue) { value = newValue; }
    void setPrevValue(double newValue) { prev_value = newValue; }
    void setPrevPrevValue(double newValue) { prev_prev_value = newValue; }
    void setCorrection(double newCorrection) { correction = newCorrection; }
    void setdx(double newDist) { dx = newDist; }
    void setdy(double newDist) { dy = newDist; }
    void setdz(double newDist) { dz = newDist; }
    void setBoundaryType(PointBoundary type) noexcept { boundaryType = type; }
    // Text is decoded at input/setup only; a Point never stores a string.
    void setBoundaryType(std::string_view type) { boundaryType = pointBoundaryFromText(type); }
    void switchUniqueNumber(int newUniqueNumber, std::vector<std::vector<std::vector<Point>>>& grid) {
        int oldUniqueNumber = unique_number;
        
        bool stop = false; // Flag to indicate when to stop iteration

        // Check if the newUniqueNumber is less than oldUniqueNumber
        if (newUniqueNumber < oldUniqueNumber) {
            // Iterate over the grid of points
            for (auto& kLayer : grid) {
                for (auto& jLayer : kLayer) {
                    for (Point& pt : jLayer) {
                        // If we find the old unique number, stop iteration
                        if (pt.getUniqueNumber() == oldUniqueNumber) {
                            stop = true;
                            break;
                        }

                        // For every point whose unique number lies between the new and old unique numbers (exclusive)
                        if (pt.getUniqueNumber() >= newUniqueNumber && pt.getUniqueNumber() < oldUniqueNumber) {
                            // Increment its unique number
                            pt.unique_number++;
                        }
                    }
                    if (stop) break;
                }
                if (stop) break;
            }
        }
        
        unique_number = newUniqueNumber;  // set the new unique number
    }
    

};

#endif //POINT_H
