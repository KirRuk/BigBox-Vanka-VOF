#include <fstream>
#include <iostream>
#include "BoundaryContainer.h"

    // Default constructor
    BoundaryContainer::BoundaryContainer() {
        // Initialize boundaries with default values
        addBoundary("E", { "fixed_value", 0.0 });
        addBoundary("W", { "fixed_value", 0.0 });
        addBoundary("N", { "fixed_value", 1.0 });
        addBoundary("S", { "fixed_value", 0.0 });
        addBoundary("F", { "fixed_value", 0.0 });
        addBoundary("B", { "fixed_value", 0.0 });
    }

    void BoundaryContainer::addBoundary(PointBoundary location, const Boundary& boundary) {
        boundaries[faceIndex(location)] = boundary;
    }

    void BoundaryContainer::loadFromFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file) {
            std::cerr << "Failed to open file\n";
            return;
        }

        // Discard the first line (comment line)
        std::string comment;
        std::getline(file, comment);

        std::string location, type;
        double value;

        // Read exactly 6 boundaries
        for (int i = 0; i < 6; ++i) {
            if (file >> location >> type >> value) {
                addBoundary(location, { type, value });
            }
            else {
                std::cerr << "Failed to read boundary " << i << "\n";
                break;
            }
        }
    }
