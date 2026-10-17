#ifndef BOUNDARY_CONTAINER_H
#define BOUNDARY_CONTAINER_H

#include <string>
#include <array>
#include <cstddef>
#include "BoundaryCodes.h"

struct Boundary {
    double value = 0.0;
    BoundaryKind type = BoundaryKind::FixedValue;
    Boundary() = default;
    Boundary(BoundaryKind kind, double boundaryValue) : value(boundaryValue), type(kind) {}
    Boundary(std::string_view kind, double boundaryValue)
        : Boundary(boundaryKindFromText(kind), boundaryValue) {}
};

class BoundaryContainer {
private:
    // E,W,N,S,F,B: direct lookup, no strings or hash nodes at runtime.
    std::array<Boundary, 6> boundaries;
    static constexpr std::size_t faceIndex(PointBoundary location) {
        switch (location) {
        case PointBoundary::E: return 0;
        case PointBoundary::W: return 1;
        case PointBoundary::N: return 2;
        case PointBoundary::S: return 3;
        case PointBoundary::F: return 4;
        case PointBoundary::B: return 5;
        default: throw std::invalid_argument("BoundaryContainer requires a face: E,W,N,S,F,B");
        }
    }

public:
    BoundaryContainer();  // Default constructor declaration

    void addBoundary(PointBoundary location, const Boundary& boundary);
    void addBoundary(std::string_view location, const Boundary& boundary) {
        addBoundary(pointBoundaryFromText(location), boundary);
    }

    const Boundary& getBoundary(PointBoundary location) const {
        return boundaries[faceIndex(location)];
    }
    const Boundary& getBoundary(std::string_view location) const {
        return getBoundary(pointBoundaryFromText(location));
    }

    void loadFromFile(const std::string& filename);
};

#endif  // BOUNDARY_CONTAINER_H
