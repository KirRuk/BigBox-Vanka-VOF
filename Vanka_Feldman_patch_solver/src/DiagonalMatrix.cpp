#include "DiagonalMatrix.h"
#include <vector>
#include <stdexcept>
#include <iostream>
#include "BoundaryContainer.h"


    double DiagonalMatrix::calculateValue(size_t k, size_t j, size_t i,
        size_t zSize, size_t ySize, size_t xSize,
        double dt,double Re,const BoundaryContainer& BC, const std::vector < std::vector<std::vector<Point>>>& component, const double& rhoLocal, const double& muLocal, char indexVar)
    {   
        const Point& point = component[k][j][i];


        // Check if dx, dy and dz are valid
        if (point.getdx() <= 0 || point.getdy() <= 0 || (point.getdz() <= 0 && point.getdz() != 0)) {
            throw std::runtime_error("Invalid point: dx, dy, and dz must be positive real numbers or dz can be zero.");
        }

        double value;
        const PointBoundary boundaryType = point.getBoundaryType();

        if (boundaryType == PointBoundary::N ||
            boundaryType == PointBoundary::S ||
            boundaryType == PointBoundary::E ||
            boundaryType == PointBoundary::W ||
            boundaryType == PointBoundary::F ||
            boundaryType == PointBoundary::B)
        {

            if (BC.getBoundary(boundaryType).type == BoundaryKind::FixedValue || BC.getBoundary(boundaryType).type == BoundaryKind::Slip) {
                value = 1.0;
            }
            else if (BC.getBoundary(boundaryType).type == BoundaryKind::Convective) {
                double dx1, dx2, dx3, a;
                double dz_dt = point.getdz() / dt;

                switch (boundaryLetter(boundaryType)) {
                case 'E':
                    dx1 = (point.getX() - component[0][j][i - 1].getX());
                    dx2 = (point.getX() - component[0][j][i - 2].getX());
                    a = (dx1 + dx2) / (dx1 * dx2 );
                    value = (a + 3.0 / 2.0 /dt);
                    break;
                case 'W':
                    dx1 = (point.getX() - component[0][j][i + 1].getX());
                    dx2 = (point.getX() - component[0][j][i + 2].getX());
                    a = -(dx1 + dx2 ) / (dx1 * dx2 );
                    value = (a + 3.0 / 2.0/dt);
                    break;

                case 'N':
                    dx1 = (point.getY() - component[0][j - 1][i].getY());
                    dx2 = (point.getY() - component[0][j - 2][i].getY());
                    a = (dx1 + dx2 ) / (dx1 * dx2);
                    value =  (a + 3.0 / 2.0/dt);
                    break;
                case 'S':
                    dx1 = (point.getY() - component[0][j + 1][i].getY());
                    dx2 = (point.getY() - component[0][j + 2][i].getY());
                    a = -(dx1  + dx2 ) / (dx1 * dx2);
                    value = (a + 3.0 / 2.0/dt);
                    break;

                case 'F':  
                    if (zSize <= 1) {
                        throw std::runtime_error("Face boundaries are not determined in 2D case.");
                    }
                    value = 3.0 / 2.0 * (1.0 + dz_dt);
                    break;
                case 'B':  
                    if (zSize <= 1) {
                        throw std::runtime_error("Face boundaries are not determined in 2D case.");
                    }
                    value = 3.0 / 2.0 * (-1.0 + dz_dt);
                    break;

                default:
                    throw std::runtime_error("Unknown boundary type encountered.");
                }
            }
            else if (BC.getBoundary(boundaryType).type == BoundaryKind::Symmetry) {
                double dx1, dx2, dx3, a;
                double dz_dt = point.getdz() / dt;

                switch (boundaryLetter(boundaryType)) {
                case 'E':
                    value = 1.0;
                    break;
                case 'W': {
                    if (indexVar == 'U' || indexVar == 'u') {
                        auto& below = (j > 0) ? component[k][j - 1][i] : point;
                        auto& above = (j < ySize - 1) ? component[k][j + 1][i] : point;
                        auto& left = component[k][j][xSize - 2];
                        auto& right = (i < xSize - 1) ? component[k][j][i + 1] : point;
                        const Point* lowerZ = nullptr;
                        const Point* upperZ = nullptr;
                        double dxp = right.getX() - point.getX();
                        double dxm = component[k][j][xSize - 1].getX() - left.getX();
                        double dyp = above.getY() - point.getY();
                        double dym = point.getY() - below.getY();
                        double dx = (dxp + dxm) / 2;
                        double dy = point.getdy();
                        double dz,dzm,dzp = 0.0;
                        if (zSize > 1) {
                            lowerZ = (k > 0) ? &component[k - 1][j][i] : &point;
                            upperZ = (k < zSize - 1) ? &component[k + 1][j][i] : &point;
                            dzp = upperZ->getZ() - point.getZ();
                            dzm = point.getZ() - lowerZ->getZ();
                            dz = point.getdz();
                        }
                        
                        // Default case
                        if (zSize <= 1) {
                            value = 3.0 / (2.0 * dt) * dx * dy * rhoLocal + muLocal / Re * (dx / dym + dx / dyp + dy / dxp + dy / dxm);
                        }
                        else if (zSize > 1) {
                            value = 3.0 / (2.0 * dt) * dx * dy * dz * rhoLocal + muLocal / Re * (dx * dz / dym + dx * dz / dyp +
                                dy * dz / dxp + dy * dz / dxm + dy * dx / dzp + dy * dx / dzm);
                        }
                    }else value = 1.0;

                    break;
                }
                default:
                    throw std::runtime_error("Unknown boundary type encountered.");
                }
            }
            else {
                throw std::runtime_error("Unsupported BC type for velocity on boundary " + std::string(1, boundaryLetter(boundaryType)) +
                    ": " + boundaryKindName(BC.getBoundary(boundaryType).type));
            }
        }
        else {
            // Default case
            // Assuming you have nearby points to access
            auto& below = (j > 0) ? component[k][j - 1][i] : point;
            auto& above = (j < ySize - 1) ? component[k][j + 1][i] : point;
            auto& left = (i > 0) ? component[k][j][i - 1] : point;
            auto& right = (i < xSize - 1) ? component[k][j][i + 1] : point;
            const Point* lowerZ = nullptr;
            const Point* upperZ = nullptr;
            if (zSize > 1) {
                    lowerZ = (k > 0) ? &component[k - 1][j][i] : &point;
                    upperZ = (k < zSize - 1) ? &component[k + 1][j][i] : &point;
            }
            // Default case
            if (zSize <= 1) {
                value = 3.0 / (2.0 * dt)* point.getdx()* point.getdy()*rhoLocal + muLocal / Re * (point.getdx() / (point.getY() - below.getY()) + point.getdx() / (-point.getY() + above.getY()) + point.getdy() / (-point.getX() + right.getX()) + point.getdy() / (point.getX() - left.getX()));
            }
            else if (zSize > 1) {
                value = 3.0 / (2.0 * dt) * point.getdx() * point.getdy() * point.getdz() * rhoLocal + muLocal / Re * (point.getdx() * point.getdz() / (point.getY() - below.getY()) +
                    point.getdx() * point.getdz() / (above.getY() - point.getY()) +
                    point.getdy() * point.getdz() / (right.getX() - point.getX()) +
                    point.getdy() * point.getdz() / (point.getX() - left.getX()) +
                    point.getdy() * point.getdx() / (upperZ->getZ() - point.getZ()) +
                    point.getdy() * point.getdx() / (point.getZ() - lowerZ->getZ()));
            }
        }

        return value;
    }


// Constructor for Diag Object
    DiagonalMatrix::DiagonalMatrix(const std::vector < std::vector<std::vector<Point>>>& u,
        const std::vector < std::vector<std::vector<Point>>>& v, const std::vector < std::vector<std::vector<Point>>>& w,
        const BoundaryContainer& BCu, const  BoundaryContainer& BCv, const  BoundaryContainer& BCw, double dt, double Re,
        const std::vector < std::vector<std::vector<double>>>& rhoX, const std::vector < std::vector<std::vector<double>>>& muX,
        const std::vector < std::vector<std::vector<double>>>& rhoY, const std::vector < std::vector<std::vector<double>>>& muY,
        const std::vector < std::vector<std::vector<double>>>& rhoZ, const std::vector < std::vector<std::vector<double>>>& muZ) {

        size_t xSize = u[0][0].size(); // X-dimension size
        size_t ySize = v[0].size(); // Y-dimension size
        size_t zSize = w.size(); // Z-dimension size
        bool is3D = (zSize > 1);
        int lengthU = (xSize) * (ySize + 1) * (zSize + 1);
        int lengthV = (xSize + 1) * ySize * (zSize + 1);
        int lengthW = is3D ? ((xSize + 1) * (ySize + 1) * zSize) : 0;

        size_t totalSize = lengthU + lengthV + (is3D ? lengthW : 0);
        diag.resize(totalSize);
        // Process u
        #pragma omp parallel for collapse(3)
        for (size_t k = 0; k < u.size(); ++k) {
            for (size_t j = 0; j < u[0].size(); ++j) {
                for (size_t i = 0; i < xSize; ++i) {
                    //const auto& point = u[k][j][i];
                    int adress = u[k][j][i].getUniqueNumber();
                    double value = calculateValue(k, j, i, is3D ? zSize + 1 : zSize, ySize + 1, xSize, dt, Re, BCu, u,rhoX[k][j][i], muX[k][j][i], 'U');
                    diag[adress] = value;
                }
            }
        }

        // Process v
        #pragma omp parallel for collapse(3)
        for (size_t k = 0; k < v.size(); ++k) {
            for (size_t j = 0; j < ySize; ++j) {
                for (size_t i = 0; i < v[0][0].size(); ++i) {
                    //const auto& point = v[k][j][i];
                    int adress = lengthU + v[k][j][i].getUniqueNumber();
                    double value = calculateValue(k, j, i, is3D ? zSize + 1 : zSize, ySize, xSize + 1, dt, Re, BCv, v,rhoY[k][j][i],muY[k][j][i],'V');
                    diag[adress] = value;
                    if (value == 0.0) {
                        std::cout << "calculateValue returned ZERO at diag[" << adress << "] "
                            << "for (k,j,i) = (" << k << "," << j << "," << i << ")\n";
                    }
                }
            }
        }
        // Process w (only if 3D)
        if (zSize > 1) {
            #pragma omp parallel for collapse(3)
            for (size_t k = 0; k < zSize; ++k) {
                for (size_t j = 0; j < w[0].size(); ++j) {
                    for (size_t i = 0; i < w[0][0].size(); ++i) {
                        //const auto& point = w[k][j][i];
                        int adress = lengthU + lengthV + w[k][j][i].getUniqueNumber();
                        double value = calculateValue(k, j, i, zSize, ySize + 1, xSize + 1, dt, Re, BCw, w,rhoZ[k][j][i],muZ[k][j][i],'W');
                        diag[adress] = value;
                    }
                }
            }
        }

    }
    
    // Method to update the diagonal matrix
    void DiagonalMatrix::update(const std::vector < std::vector<std::vector<Point>>>& u,
        const std::vector < std::vector<std::vector<Point>>>& v, const std::vector < std::vector<std::vector<Point>>>& w,
        const BoundaryContainer& BCu, const  BoundaryContainer& BCv, const  BoundaryContainer& BCw, double dt, double Re,
        const std::vector < std::vector<std::vector<double>>>& rhoX, const std::vector < std::vector<std::vector<double>>>& muX,
        const std::vector < std::vector<std::vector<double>>>& rhoY, const std::vector < std::vector<std::vector<double>>>& muY,
        const std::vector < std::vector<std::vector<double>>>& rhoZ, const std::vector < std::vector<std::vector<double>>>& muZ) {
        size_t xSize = u[0][0].size(); // X-dimension size
        size_t ySize = v[0].size(); // Y-dimension size
        size_t zSize = w.size(); // Z-dimension size
        bool is3D = (zSize > 1);
        int lengthU = (xSize) * (ySize + 1) * (zSize + 1);
        int lengthV = (xSize + 1) * ySize * (zSize + 1);
        int lengthW = is3D ? ((xSize + 1) * (ySize + 1) * zSize) : 0;

        size_t totalSize = lengthU + lengthV + (is3D ? lengthW : 0);
        // Process u
        #pragma omp parallel for collapse(3)
        for (size_t k = 0; k < u.size(); ++k) {
            for (size_t j = 0; j < u[0].size(); ++j) {
                for (size_t i = 0; i < xSize; ++i) {
                    //const auto& point = u[k][j][i];
                    int adress = u[k][j][i].getUniqueNumber();
                    double value = calculateValue(k, j, i, is3D ? zSize + 1 : zSize, ySize + 1, xSize, dt, Re, BCu, u, rhoX[k][j][i], muX[k][j][i],'U');
                    diag[adress] = value;
                }
            }
        }
        // Process v
        #pragma omp parallel for collapse(3)
        for (size_t k = 0; k < v.size(); ++k) {
            for (size_t j = 0; j < ySize; ++j) {
                for (size_t i = 0; i < v[0][0].size(); ++i) {
                    //const auto& point = v[k][j][i];
                    int adress = lengthU + v[k][j][i].getUniqueNumber();
                    double value = calculateValue(k, j, i, is3D ? zSize + 1 : zSize, ySize, xSize + 1, dt, Re, BCv, v, rhoY[k][j][i], muY[k][j][i],'V');
                    diag[adress] = value;
                }
            }
        }
        // Process w (only if 3D)
        if (zSize > 1) {
        #pragma omp parallel for collapse(3)
            for (size_t k = 0; k < zSize; ++k) {
                for (size_t j = 0; j < w[0].size(); ++j) {
                    for (size_t i = 0; i < w[0][0].size(); ++i) {
                        //const auto& point = w[k][j][i];
                        int adress = lengthU + lengthV + w[k][j][i].getUniqueNumber();
                        double value = calculateValue(k, j, i, zSize, ySize + 1, xSize + 1, dt, Re, BCw, w, rhoZ[k][j][i], muZ[k][j][i],'W');
                         diag[adress] = value;
                    }
                }
            }
        }
    }
	// Method for vector multiplication
	std::vector<double> DiagonalMatrix::multiply_with_vector(const std::vector<double>& vec) const {
		std::vector<double> result;
		multiply_with_vector(vec, result);
		return result;
	}

	void DiagonalMatrix::multiply_with_vector(
		const std::vector<double>& vec,
		std::vector<double>& result) const {
        if (vec.size() != diag.size()) {
            throw std::invalid_argument("Vector size must match the size of the diagonal.");
        }

        result.resize(vec.size());
        #pragma omp parallel for
        for (size_t i = 0; i < vec.size(); i++) {
            result[i] = vec[i] / diag[i];  // Division because we're working with inverse of the diagonal matrix.
        }
    }
    //Getter
    double DiagonalMatrix::getValue(int i) {
        try {
            return diag.at(i); 
        }
        catch (const std::out_of_range&) {
            // Handle the case when index is out of range
            std::cerr << "Index out of range" << std::endl;
            return 0.0; // or some other default value
        }
        catch (const std::exception&) {
            // Handle other exceptions
            std::cerr << "An error occurred" << std::endl;
            return 0.0; // or some other default value
        }
    }
    const std::vector<double>& DiagonalMatrix::getDiag() const {
        return diag;
    }

    //Adding Dirichlet from pressure
    void DiagonalMatrix::addPressureDirichlet() {
        diag.push_back(1);
    }
