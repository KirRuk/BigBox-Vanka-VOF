#include <tuple>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include "Point.h"
#include "Initialize.h"
#include "BoundaryContainer.h"

std::tuple<
    std::vector<std::vector<std::vector<Point>>>,
    std::vector<std::vector<std::vector<Point>>>,
    std::vector<std::vector<std::vector<Point>>>,
    std::vector<std::vector<std::vector<Point>>>
> initialize(const BoundaryContainer& BCu, const BoundaryContainer& BCv, const BoundaryContainer& BCw, double Lx, double Ly, double Lz, double ds, double initialValueU, double initialValueV, double initialValueW,const std::vector<std::vector<std::vector<Point>>>& phi, double rho_ratio, double Fr, bool developed) {
    int nx = static_cast<int>(std::round(Lx / ds)) + 1,
        ny = static_cast<int>(std::round(Ly / ds)) + 1,
        nz = static_cast<int>(std::round(Lz / ds)) + 1;  // Number of grid points in each direction
    Point::validateGridIndices(nx, ny, nz > 1 ? nz : 0);
    if (Lz == 0) nz = 0;
    // Create the 3D grids 
    std::vector<std::vector<std::vector<Point>>> u;
    std::vector<std::vector<std::vector<Point>>> v;
    std::vector<std::vector<std::vector<Point>>> w;
    std::vector<std::vector<std::vector<Point>>> p;
    {
        u.resize(nz + 1);
        for (int k = 0; k <= nz; ++k) {
            u[k].resize(ny + 1);
            for (int j = 0; j <= ny; ++j) {
                u[k][j].resize(nx);  // all elements default-constructed
            }
        }

        // Create the grid for Vx(u)
        #pragma omp parallel for collapse(3)
        for (int k = 0; k <= nz; ++k) {
            for (int j = 0; j <= ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    int unique_number = (k)*(ny+1)*nx + (j)*nx + i;
                    // Create point object and increment unique number
                    Point point(i, j, k, unique_number, ds);
                    double x_coord = i * ds;
                    double y_coord = (j == 0) ? 0 : ((j - 1) * ds + ds / 2);
                    double z_coord = (k == 0) ? 0 : ((k - 1) * ds + ds / 2);
                    (j == ny) ? y_coord = Ly : ((j - 1) * ds + ds / 2);
                    (k == nz) ? z_coord = Lz : ((k - 1) * ds + ds / 2);
                    point.setX(x_coord);
                    point.setY(y_coord);
                    point.setZ(z_coord);

                    double boundaryValue;
                        // Set initial conditions
                    if (developed) {
                        initialValueU = 2.0 * y_coord - 1.0;
                    }
                    
                    point.setValue(initialValueU);
                    point.setPrevValue(initialValueU);
                    point.setPrevPrevValue(initialValueU);// Initial value for all the volume. To change with boundaries
                 
                    // Set the boundaryType for boundary points
                    if (i == 0) {
                        point.setBoundaryType(PointBoundary::W);
                        boundaryValue = BCu.getBoundary(PointBoundary::W).value;
                        point.setdx(ds / 2);
                        point.setValue(boundaryValue);
                        point.setPrevValue(boundaryValue);
                        point.setPrevPrevValue(boundaryValue);
                    }
                    if (i == nx - 1) {
                        point.setBoundaryType(PointBoundary::E);
                        point.setdx(ds / 2.0);
                        boundaryValue = BCu.getBoundary(PointBoundary::E).value;
                        point.setValue(boundaryValue);
                        point.setPrevValue(boundaryValue);
                        point.setPrevPrevValue(boundaryValue);
                    }
                    if (j == 0) {
                        point.setBoundaryType(PointBoundary::S);
                        point.setdy(ds / 2.0);
                        boundaryValue = BCu.getBoundary(PointBoundary::S).value;
                        point.setValue(boundaryValue);
                        point.setPrevValue(boundaryValue);
                        point.setPrevPrevValue(boundaryValue);
                    }
                    if (j == ny) {
                        point.setBoundaryType(PointBoundary::N);
                        point.setdy(ds / 2.0);
                        boundaryValue = BCu.getBoundary(PointBoundary::N).value;
                        point.setValue(boundaryValue);
                        point.setPrevValue(boundaryValue);
                        point.setPrevPrevValue(boundaryValue);
                    }
                    if (nz > 1) {
                        if (k == 0) {
                            point.setBoundaryType(PointBoundary::B);
                            point.setdz(ds / 2.0);
                            boundaryValue = BCu.getBoundary(PointBoundary::B).value;
                            point.setValue(boundaryValue);
                            point.setPrevValue(boundaryValue);
                            point.setPrevPrevValue(boundaryValue);
                        }
                        if (k == nz) {
                            point.setBoundaryType(PointBoundary::F);
                            point.setdz(ds / 2.0);
                            boundaryValue = BCu.getBoundary(PointBoundary::F).value;
                            point.setValue(boundaryValue);
                            point.setPrevValue(boundaryValue);
                            point.setPrevPrevValue(boundaryValue);
                        }
                    }

                    u[k][j][i] = point;
                }
            }
        }
    }
    {
        v.resize(nz + 1);
        for (int k = 0; k <= nz; ++k) {
            v[k].resize(ny);
            for (int j = 0; j < ny; ++j) {
                v[k][j].resize(nx + 1);  // all elements default-constructed
            }
        }
        // Create the grid for Vy(v)        
        #pragma omp parallel for collapse(3)
        for (int k = 0; k <= nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i <= nx; ++i) {
                    int unique_number = (k)*(ny)*(nx + 1) + (j)*(nx + 1) + i;
                    // Create point object and increment unique number
                    Point point(i, j, k, unique_number, ds);

                    // Set initial conditions
                    //point.setValue(j == ny + 1 ? 1.0 : 0.0);  // Initial value if needed
                    double x_coord = (i == 0) ? 0 : ((i-1) * ds + ds / 2);
                    double y_coord =  j * ds;
                    double z_coord = (k == 0) ? 0 : ((k-1) * ds + ds / 2);
                    (i == nx) ? x_coord = Lx : ((i - 1) * ds + ds / 2);
                    (k == nz) ? z_coord = Lz : ((k - 1) * ds + ds / 2);
                    point.setX(x_coord);
                    point.setY(y_coord);
                    point.setZ(z_coord);

                    double boundaryValue;
                    // Set initial conditions
                    point.setValue(initialValueV);
                    point.setPrevValue(initialValueV);
                    point.setPrevPrevValue(initialValueV);
                    if (i == 0) {
                        point.setBoundaryType(PointBoundary::W);
                        point.setdx(ds / 2.0);
                        boundaryValue = BCv.getBoundary(PointBoundary::W).value;
                        point.setValue(boundaryValue);
                        point.setPrevValue(boundaryValue);
                        point.setPrevPrevValue(boundaryValue);
                    }
                    if (i == nx) {
                        point.setBoundaryType(PointBoundary::E);
                        point.setdx(ds / 2.0);
                        boundaryValue = BCv.getBoundary(PointBoundary::E).value;
                        point.setValue(boundaryValue);
                        point.setPrevValue(boundaryValue);
                        point.setPrevPrevValue(boundaryValue);
                    }
                    if (j == 0) {
                        point.setBoundaryType(PointBoundary::S);
                        point.setdy(ds / 2.0);
                        boundaryValue = BCv.getBoundary(PointBoundary::S).value;
                        point.setValue(boundaryValue);
                        point.setPrevValue(boundaryValue);
                        point.setPrevPrevValue(boundaryValue);
                    }
                    if (j == ny - 1) {
                        point.setBoundaryType(PointBoundary::N);
                        point.setdy(ds / 2.0);
                        boundaryValue = BCv.getBoundary(PointBoundary::N).value;
                        point.setValue(boundaryValue);
                        point.setPrevValue(boundaryValue);
                        point.setPrevPrevValue(boundaryValue);
                    }
                    if (nz > 1) {
                        if (k == 0) {
                            point.setBoundaryType(PointBoundary::B);
                            point.setdz(ds / 2.0);
                            boundaryValue = BCv.getBoundary(PointBoundary::B).value;
                            point.setValue(boundaryValue);
                            point.setPrevValue(boundaryValue);
                            point.setPrevPrevValue(boundaryValue);
                        }
                        if (k == nz) {
                            point.setBoundaryType(PointBoundary::F);
                            point.setdz(ds / 2.0);
                            boundaryValue = BCv.getBoundary(PointBoundary::F).value;
                            point.setValue(boundaryValue);
                            point.setPrevValue(boundaryValue);
                            point.setPrevPrevValue(boundaryValue);
                        }
                    }

                    v[k][j][i] = point;
                }
            }
        }
    }

    if (nz > 1) {
        w.resize(nz);
        for (int k = 0; k < nz; ++k) {
            w[k].resize(ny + 1);
            for (int j = 0; j < ny + 1; ++j) {
                w[k][j].resize(nx + 1);  // all elements default-constructed
            }
        }
        // Create the grid for Vz(w)
        #pragma omp parallel for collapse(3)
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j <= ny; ++j) {
                for (int i = 0; i <= nx; ++i) {
                    int unique_number = (k)*(ny + 1)*(nx + 1) + (j)*(nx + 1) + i;
                    // Create point object and increment unique number
                    Point point(i, j, k, unique_number, ds);
                    // Set initial conditions
                    //point.setValue(j == ny - 1 ? 1.0 : 0.0);  // Initial value if needed
                    double x_coord = (i == 0) ? 0 : ((i-1) * ds + ds / 2);
                    double y_coord = (j == 0) ? 0 : ((j-1) * ds + ds / 2); 
                    double z_coord = k * ds;
                    (i == nx) ? x_coord = Lx : ((i - 1) * ds + ds / 2);
                    (j == ny) ? y_coord = Ly : ((j - 1) * ds + ds / 2);
                    point.setX(x_coord);
                    point.setY(y_coord);
                    point.setZ(z_coord);

                    double boundaryValue;
                    // Set initial conditions
                    point.setValue(initialValueW);
                    point.setPrevValue(initialValueW);
                    point.setPrevPrevValue(initialValueW);
                    // Set the boundaryType for boundary points
                    if (i == 0) {
                        point.setBoundaryType(PointBoundary::W);
                        point.setdx(ds / 2);
                        boundaryValue = BCw.getBoundary(PointBoundary::W).value;
                        point.setValue(boundaryValue);
                        point.setPrevValue(boundaryValue);
                        point.setPrevPrevValue(boundaryValue);
                    }
                    if (i == nx) {
                        point.setBoundaryType(PointBoundary::E);
                        point.setdx(ds / 2);
                        boundaryValue = BCw.getBoundary(PointBoundary::E).value;
                        point.setValue(boundaryValue);
                        point.setPrevValue(boundaryValue);
                        point.setPrevPrevValue(boundaryValue);
                    }
                    if (j == 0) {
                        point.setBoundaryType(PointBoundary::S);
                        point.setdy(ds / 2);
                        boundaryValue = BCw.getBoundary(PointBoundary::S).value;
                        point.setValue(boundaryValue);
                        point.setPrevValue(boundaryValue);
                        point.setPrevPrevValue(boundaryValue);
                    }
                    if (j == ny) {
                        point.setBoundaryType(PointBoundary::N);
                        point.setdy(ds / 2);
                        boundaryValue = BCw.getBoundary(PointBoundary::N).value;
                        point.setValue(boundaryValue);
                        point.setPrevValue(boundaryValue);
                        point.setPrevPrevValue(boundaryValue);
                    }
                    
                    if (k == 0) {
                        point.setBoundaryType(PointBoundary::B);
                        point.setdz(ds/2);
                        boundaryValue = BCw.getBoundary(PointBoundary::B).value;
                        point.setValue(boundaryValue);
                        point.setPrevValue(boundaryValue);
                        point.setPrevPrevValue(boundaryValue);
                    }
                    if (k == nz - 1) {
                        point.setBoundaryType(PointBoundary::F);
                        point.setdz(ds/2);
                        boundaryValue = BCw.getBoundary(PointBoundary::F).value;
                        point.setValue(boundaryValue);
                        point.setPrevValue(boundaryValue);
                        point.setPrevPrevValue(boundaryValue);
                    }
                    
                    w[k][j][i] = point;
                }
            }
        }
    }
    {
        int sizeZ = (nz == 0 ? 1 : nz - 1);
        p.resize(sizeZ);
        for (int k = 0; k < sizeZ; ++k) {
            p[k].resize(ny - 1);
            for (int j = 0; j < ny - 1; ++j) {
                p[k][j].resize(nx - 1);  // all elements default-constructed
            }
        }
        // Create the grid for pressure(p)
        #pragma omp parallel for collapse(3)
        for (int k = 0; k < sizeZ; ++k) {
            for (int j = 0; j < ny-1; ++j) {
                for (int i = 0; i < nx-1; ++i) {
                    int unique_number = (k)*(ny - 1)*(nx - 1) + (j)*(nx - 1) + i;
                    // Create point object and increment unique number
                    Point point(i, j, k, unique_number, ds);
                    
                    // Set initial conditions
                    double x_coord = (i * ds + ds / 2);
                    double y_coord = (j * ds + ds / 2);
                    double z_coord = (k * ds + ds / 2);

                    point.setX(x_coord);
                    point.setY(y_coord);
                    point.setZ(z_coord);


                    p[k][j][i] = point;
                }
            }
        }
        for (int k = 0; k < sizeZ; ++k) {
            for (int j = ny - 2; j >= 0; --j) {
                for (int i = 0; i < nx - 1; ++i) {
                    // Create point object and increment unique number
                    double rhoRef = rho_ratio;
                    double rho = (1.0 + (rho_ratio - 1.0) * phi[k][j][i].getValue()) - rhoRef;
                    double pressure = (j == ny - 2) ? (rho * (Ly - p[k][j][i].getY()) / Fr / Fr) : (p[k][j + 1][i].getValue() + rho * ds / Fr / Fr);
                    p[k][j][i].setValue(pressure);
                }
            }
        }

    }

    return std::make_tuple(u, v, w, p);
}

bool extractIJ(const std::string& line, int& I, int& J) {
    std::size_t i_pos = line.find("I=");
    std::size_t j_pos = line.find("J=");

    if (i_pos == std::string::npos || j_pos == std::string::npos) return false;

    std::stringstream ssI(line.substr(i_pos + 2));
    ssI >> I;

    std::stringstream ssJ(line.substr(j_pos + 2));
    ssJ >> J;

    return true;
}


bool extractIJK(const std::string& line, int& I, int& J, int& K)
{
    bool is3D = true;                        // 1) legal identifier

    std::size_t i_pos = line.find("I=");
    std::size_t j_pos = line.find("J=");
    std::size_t k_pos = line.find("K=");

    // I and J are mandatory
    if (i_pos == std::string::npos || j_pos == std::string::npos)
        throw std::runtime_error(
            "extractIJK(): Tecplot zone header is missing mandatory I= or J=.");

    if (k_pos == std::string::npos) {
        is3D = false;
        K = 1;
    }

    std::stringstream ssI(line.substr(i_pos + 2));
    ssI >> I;

    std::stringstream ssJ(line.substr(j_pos + 2));
    ssJ >> J;

    if (is3D) {
        std::stringstream ssK(line.substr(k_pos + 2));
        ssK >> K;
    }

    return is3D;                             // 2) return the flag
}


double calculate_dy(std::ifstream& file, int j, int ny, double& y_down, double& y) {
    double dy = 0.0;
    double y_up; // the y-value of the next line where y changes
    std::string line;
    // Assuming file is already open and at the correct position.

    // Remember the current position.
    long long currentPos = file.tellg();

    // Attempt to read the next line to find y_up
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        double temp_x, temp_y, temp_u_val;
        ss >> temp_x >> temp_y >> temp_u_val;
        // Check if the y-value has changed, indicating the next y-value (y_up)
        if (temp_y != y) {
            y_up = temp_y;

            // For the first row, calculate dy using only y_up and y
            if (j == 0) {
                dy = y_up - y;
            }
            else {
                // For other rows, dy is the average change in y
                dy = (y_up - y_down) / 2;
            }
            break; // y_up found, break out of the loop
        }
    }

    if (j == ny) {
        // If we're at the boundary, calculate dy based on the last change in y
        dy = (y - y_down)/2;
    }

    file.clear(); // Clear any potential EOF flag to ensure seeking works
    file.seekg(currentPos); // Return to saved file position
    return dy;
}
double calculate_dz(std::ifstream& file, int k, int nz, double& z_back, double& z) {
    double dz = 0.0;
    double z_face; // the y-value of the next line where y changes
    std::string line;
    // Assuming file is already open and at the correct position.

    // Remember the current position.
    long long currentPos = file.tellg();

    // Attempt to read the next line to find y_up
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        double temp_x, temp_y, temp_z, temp_u_val;
        ss >> temp_x >> temp_y >> temp_z >> temp_u_val;
        // Check if the y-value has changed, indicating the next y-value (y_up)
        if (temp_z != z) {
            z_face = temp_z;

            // For the first row, calculate dy using only y_up and y
            if (k == 0) {
                dz = z_face - z;
            }
            else {
                // For other rows, dy is the average change in y
                dz = (z_face - z_back) / 2;
            }
            break; // y_up found, break out of the loop
        }
    }

    if (k == nz) {
        // If we're at the boundary, calculate dy based on the last change in y
        dz = (z - z_back)/2;
    }

    file.clear(); // Clear any potential EOF flag to ensure seeking works
    file.seekg(currentPos); // Return to saved file position
    return dz;
}

std::vector<std::vector<std::vector<Point>>> loadU(const std::string& filename, double ds) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(1);  // Exiting if the file fails to open.
    }

    std::string line;

    // Skip the first line, it's just a header.
    std::getline(file, line);

    // Extract I and J values from the second line.
    std::getline(file, line);
    int I, J, K;
    bool is3D = extractIJK(line, I, J, K);

    // Derive nx and ny based on the extracted I and J.
    int nx = I - 1;
    int ny = J - 2;
    int nz = K - 2;// fir 2D = -1

    int unique_number = 0; // Counter for unique numbers.
    // Variables to hold the two previous line's x and y values.
    std::getline(file, line);
    std::stringstream ss(line);
    double x,y,z,u_val;
    ss >> x >> y >> z >> u_val;
    double prev_x = 0, prev_y = 0, prev_z = 0,  y_down = 0, z_back = 0;

    // Initialize an empty 3D vector structure.
    std::vector<std::vector<std::vector<Point>>> u;
    for (int k = 0; k < nz + 2; ++k) {
        // Loop through the file's data lines and populate the vector.
        std::vector<std::vector<Point>> zLayer;
        double dz = calculate_dz(file, k, nz + 1, z_back, z);
        for (int j = 0; j < ny + 2; ++j) {
            std::vector<Point> yLayer;
            double dy = calculate_dy(file, j, ny + 1, y_down, y);
            for (int i = 0; i < nx + 1; ++i) {

                if (std::getline(file, line)) {
                    double dx, x_next, y_next,z_next, u_val_next;
                    std::stringstream ss_new(line);
                    ss_new >> x_next >> y_next >> z_next >> u_val_next;

                    // Calculate dx and dy using the average distance.

                    dx = i == 0 ? x_next - x : (x_next - prev_x) / 2;
                    if (i == nx) dx = x - prev_x;

                    Point point(i, j, k, unique_number++, dx, dy, dz);
                    point.setX(x);
                    point.setY(y);
                    point.setZ(z);  // Setting Z value to 0, assuming 2D.

                    // Set boundary conditions based on position.
                    if (i == 0) {
                        point.setBoundaryType(PointBoundary::W);
                    }
                    if (i == nx) {
                        point.setBoundaryType(PointBoundary::E);
                    }
                    if (j == 0) {
                        point.setBoundaryType(PointBoundary::S);
                    }
                    if (j == ny + 1) {
                        point.setBoundaryType(PointBoundary::N);
                    }
                    if (k == 0 && is3D) {
                        point.setBoundaryType(PointBoundary::B);
                    }
                    if (k == nz + 1 && is3D) {
                        point.setBoundaryType(PointBoundary::F);
                    }

                    point.setValue(u_val);         // Set current value.
                    point.setPrevValue(u_val);     // Assuming current value is also the previous value.
                    point.setPrevPrevValue(u_val); // Assuming current value is the previous-previous value.

                    yLayer.push_back(point);  // Add point to yLayer.
                    u_val = u_val_next;
                    if (y_next != y) y_down = y;
                    if (z_next != z) z_back = z;
                    prev_x = x; prev_y = y; prev_z = z;
                    x = x_next; y = y_next; z = z_next;

                }
                else {
                    double dx;
                    dx = x - prev_x;


                    Point point(i, j, k, unique_number++, dx, dy, dz);
                    point.setX(x);
                    point.setY(y);
                    point.setZ(z);  // Setting Z value to 0, assuming 2D.

                    // Set boundary conditions based on position.
                    if (i == 0) {
                        point.setBoundaryType(PointBoundary::W);
                    }
                    if (i == nx) {
                        point.setBoundaryType(PointBoundary::E);
                    }
                    if (j == 0) {
                        point.setBoundaryType(PointBoundary::S);
                    }
                    if (j == ny + 1) {
                        point.setBoundaryType(PointBoundary::N);
                    }
                    if (k == 0 && is3D) {
                        point.setBoundaryType(PointBoundary::B);
                    }
                    if (k == nz + 1 && is3D) {
                        point.setBoundaryType(PointBoundary::F);
                    }

                    point.setValue(u_val);         // Set current value.
                    point.setPrevValue(u_val);     // Assuming current value is also the previous value.
                    point.setPrevPrevValue(u_val); // Assuming current value is the previous-previous value.

                    yLayer.push_back(point);  // Add point to yLayer.
                    prev_x = x; prev_y = y; prev_z = z;
                }
            }
            zLayer.push_back(yLayer);//Add line to zLayer
        }
        u.push_back(zLayer);  // Add yLayer to the 3D vector.
    }
    file.close();  // Close the file after reading.
    return u;      // Return the populated 3D vector.
}
std::tuple<
    std::vector<std::vector<std::vector<Point>>>,
    std::vector<std::vector<std::vector<Point>>>,
    std::vector<std::vector<std::vector<Point>>>,
    std::vector<std::vector<std::vector<Point>>>
> initialize(const BoundaryContainer& BCu, const BoundaryContainer& BCv, const BoundaryContainer& BCw, double Lx, double Ly, double Lz, double ds) {
    int nx = static_cast<int>(std::round(Lx / ds)) + 1,
        ny = static_cast<int>(std::round(Ly / ds)) + 1,
        nz = static_cast<int>(std::round(Lz / ds)) + 1;  // Number of grid points in each direction
    Point::validateGridIndices(nx, ny, nz > 1 ? nz : 0);
    if (Lz == 0) nz = 0;
    // Create the 3D grids 
    std::vector<std::vector<std::vector<Point>>> u;
    std::vector<std::vector<std::vector<Point>>> v;
    std::vector<std::vector<std::vector<Point>>> w;
    std::vector<std::vector<std::vector<Point>>> p;
    {
        u.resize(nz + 1);
        for (int k = 0; k <= nz; ++k) {
            u[k].resize(ny + 1);
            for (int j = 0; j <= ny; ++j) {
                u[k][j].resize(nx);  // all elements default-constructed
            }
        }

        // Create the grid for Vx(u)
#pragma omp parallel for collapse(3)
        for (int k = 0; k <= nz; ++k) {
            for (int j = 0; j <= ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    int unique_number = (k) * (ny + 1) * nx + (j)*nx + i;
                    // Create point object and increment unique number
                    Point point(i, j, k, unique_number, ds);
                    double x_coord = i * ds;
                    double y_coord = (j == 0) ? 0 : ((j - 1) * ds + ds / 2);
                    double z_coord = (k == 0) ? 0 : ((k - 1) * ds + ds / 2);
                    (j == ny) ? y_coord = Ly : ((j - 1) * ds + ds / 2);
                    (k == nz) ? z_coord = Lz : ((k - 1) * ds + ds / 2);
                    point.setX(x_coord);
                    point.setY(y_coord);
                    point.setZ(z_coord);
                    double initialValueU = -1 + y_coord;
                    double boundaryValue;
                    // Set initial conditions
                    point.setValue(initialValueU);
                    point.setPrevValue(initialValueU);
                    point.setPrevPrevValue(initialValueU);// Initial value for all the volume. To change with boundaries
                    // Set the boundaryType for boundary points
                    if (i == 0) {
                        point.setBoundaryType(PointBoundary::W);
                        boundaryValue = BCu.getBoundary(PointBoundary::W).value;
                        point.setdx(ds / 2);
                        point.setValue(boundaryValue);
                        point.setPrevValue(boundaryValue);
                        point.setPrevPrevValue(boundaryValue);
                    }
                    if (i == nx - 1) {
                        point.setBoundaryType(PointBoundary::E);
                        point.setdx(ds / 2.0);
                        boundaryValue = BCu.getBoundary(PointBoundary::E).value;
                        point.setValue(boundaryValue);
                        point.setPrevValue(boundaryValue);
                        point.setPrevPrevValue(boundaryValue);
                    }
                    if (j == 0) {
                        point.setBoundaryType(PointBoundary::S);
                        point.setdy(ds / 2.0);
                        boundaryValue = BCu.getBoundary(PointBoundary::S).value;
                        point.setValue(boundaryValue);
                        point.setPrevValue(boundaryValue);
                        point.setPrevPrevValue(boundaryValue);
                    }
                    if (j == ny) {
                        point.setBoundaryType(PointBoundary::N);
                        point.setdy(ds / 2.0);
                        boundaryValue = BCu.getBoundary(PointBoundary::N).value;
                        point.setValue(boundaryValue);
                        point.setPrevValue(boundaryValue);
                        point.setPrevPrevValue(boundaryValue);
                    }
                    if (nz > 1) {
                        if (k == 0) {
                            point.setBoundaryType(PointBoundary::B);
                            point.setdz(ds / 2.0);
                            boundaryValue = BCu.getBoundary(PointBoundary::B).value;
                            point.setValue(boundaryValue);
                            point.setPrevValue(boundaryValue);
                            point.setPrevPrevValue(boundaryValue);
                        }
                        if (k == nz) {
                            point.setBoundaryType(PointBoundary::F);
                            point.setdz(ds / 2.0);
                            boundaryValue = BCu.getBoundary(PointBoundary::F).value;
                            point.setValue(boundaryValue);
                            point.setPrevValue(boundaryValue);
                            point.setPrevPrevValue(boundaryValue);
                        }
                    }

                    u[k][j][i] = point;
                }
            }
        }
    }
    {
        v.resize(nz + 1);
        for (int k = 0; k <= nz; ++k) {
            v[k].resize(ny);
            for (int j = 0; j < ny; ++j) {
                v[k][j].resize(nx + 1);  // all elements default-constructed
            }
        }
        // Create the grid for Vy(v)        
#pragma omp parallel for collapse(3)
        for (int k = 0; k <= nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i <= nx; ++i) {
                    int unique_number = (k) * (ny) * (nx + 1) + (j) * (nx + 1) + i;
                    // Create point object and increment unique number
                    Point point(i, j, k, unique_number, ds);

                    // Set initial conditions
                    //point.setValue(j == ny + 1 ? 1.0 : 0.0);  // Initial value if needed
                    double x_coord = (i == 0) ? 0 : ((i - 1) * ds + ds / 2);
                    double y_coord = j * ds;
                    double z_coord = (k == 0) ? 0 : ((k - 1) * ds + ds / 2);
                    (i == nx) ? x_coord = Lx : ((i - 1) * ds + ds / 2);
                    (k == nz) ? z_coord = Lz : ((k - 1) * ds + ds / 2);
                    point.setX(x_coord);
                    point.setY(y_coord);
                    point.setZ(z_coord);
                    double initialValueV = 0;
                    double boundaryValue;
                    // Set initial conditions
                    point.setValue(initialValueV);
                    point.setPrevValue(initialValueV);
                    point.setPrevPrevValue(initialValueV);
                    if (i == 0) {
                        point.setBoundaryType(PointBoundary::W);
                        point.setdx(ds / 2.0);
                        boundaryValue = BCv.getBoundary(PointBoundary::W).value;
                        point.setValue(boundaryValue);
                        point.setPrevValue(boundaryValue);
                        point.setPrevPrevValue(boundaryValue);
                    }
                    if (i == nx) {
                        point.setBoundaryType(PointBoundary::E);
                        point.setdx(ds / 2.0);
                        boundaryValue = BCv.getBoundary(PointBoundary::E).value;
                        point.setValue(boundaryValue);
                        point.setPrevValue(boundaryValue);
                        point.setPrevPrevValue(boundaryValue);
                    }
                    if (j == 0) {
                        point.setBoundaryType(PointBoundary::S);
                        point.setdy(ds / 2.0);
                        boundaryValue = BCv.getBoundary(PointBoundary::S).value;
                        point.setValue(boundaryValue);
                        point.setPrevValue(boundaryValue);
                        point.setPrevPrevValue(boundaryValue);
                    }
                    if (j == ny - 1) {
                        point.setBoundaryType(PointBoundary::N);
                        point.setdy(ds / 2.0);
                        boundaryValue = BCv.getBoundary(PointBoundary::N).value;
                        point.setValue(boundaryValue);
                        point.setPrevValue(boundaryValue);
                        point.setPrevPrevValue(boundaryValue);
                    }
                    if (nz > 1) {
                        if (k == 0) {
                            point.setBoundaryType(PointBoundary::B);
                            point.setdz(ds / 2.0);
                            boundaryValue = BCv.getBoundary(PointBoundary::B).value;
                            point.setValue(boundaryValue);
                            point.setPrevValue(boundaryValue);
                            point.setPrevPrevValue(boundaryValue);
                        }
                        if (k == nz) {
                            point.setBoundaryType(PointBoundary::F);
                            point.setdz(ds / 2.0);
                            boundaryValue = BCv.getBoundary(PointBoundary::F).value;
                            point.setValue(boundaryValue);
                            point.setPrevValue(boundaryValue);
                            point.setPrevPrevValue(boundaryValue);
                        }
                    }

                    v[k][j][i] = point;
                }
            }
        }
    }

    if (nz > 1) {
        w.resize(nz);
        for (int k = 0; k < nz; ++k) {
            w[k].resize(ny + 1);
            for (int j = 0; j < ny + 1; ++j) {
                w[k][j].resize(nx + 1);  // all elements default-constructed
            }
        }
        // Create the grid for Vz(w)
#pragma omp parallel for collapse(3)
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j <= ny; ++j) {
                for (int i = 0; i <= nx; ++i) {
                    int unique_number = (k) * (ny + 1) * (nx + 1) + (j) * (nx + 1) + i;
                    // Create point object and increment unique number
                    Point point(i, j, k, unique_number, ds);
                    // Set initial conditions
                    //point.setValue(j == ny - 1 ? 1.0 : 0.0);  // Initial value if needed
                    double x_coord = (i == 0) ? 0 : ((i - 1) * ds + ds / 2);
                    double y_coord = (j == 0) ? 0 : ((j - 1) * ds + ds / 2);
                    double z_coord = k * ds;
                    (i == nx) ? x_coord = Lx : ((i - 1) * ds + ds / 2);
                    (j == ny) ? y_coord = Ly : ((j - 1) * ds + ds / 2);
                    point.setX(x_coord);
                    point.setY(y_coord);
                    point.setZ(z_coord);
                    double initialValueW = -1 + y_coord;
                    double boundaryValue;
                    // Set initial conditions
                    point.setValue(initialValueW);
                    point.setPrevValue(initialValueW);
                    point.setPrevPrevValue(initialValueW);
                    // Set the boundaryType for boundary points
                    if (i == 0) {
                        point.setBoundaryType(PointBoundary::W);
                        point.setdx(ds / 2);
                        boundaryValue = BCw.getBoundary(PointBoundary::W).value;
                        point.setValue(boundaryValue);
                        point.setPrevValue(boundaryValue);
                        point.setPrevPrevValue(boundaryValue);
                    }
                    if (i == nx) {
                        point.setBoundaryType(PointBoundary::E);
                        point.setdx(ds / 2);
                        boundaryValue = BCw.getBoundary(PointBoundary::E).value;
                        point.setValue(boundaryValue);
                        point.setPrevValue(boundaryValue);
                        point.setPrevPrevValue(boundaryValue);
                    }
                    if (j == 0) {
                        point.setBoundaryType(PointBoundary::S);
                        point.setdy(ds / 2);
                        boundaryValue = BCw.getBoundary(PointBoundary::S).value;
                        point.setValue(boundaryValue);
                        point.setPrevValue(boundaryValue);
                        point.setPrevPrevValue(boundaryValue);
                    }
                    if (j == ny) {
                        point.setBoundaryType(PointBoundary::N);
                        point.setdy(ds / 2);
                        boundaryValue = BCw.getBoundary(PointBoundary::N).value;
                        point.setValue(boundaryValue);
                        point.setPrevValue(boundaryValue);
                        point.setPrevPrevValue(boundaryValue);
                    }

                    if (k == 0) {
                        point.setBoundaryType(PointBoundary::B);
                        point.setdz(ds / 2);
                        boundaryValue = BCw.getBoundary(PointBoundary::B).value;
                        point.setValue(boundaryValue);
                        point.setPrevValue(boundaryValue);
                        point.setPrevPrevValue(boundaryValue);
                    }
                    if (k == nz - 1) {
                        point.setBoundaryType(PointBoundary::F);
                        point.setdz(ds / 2);
                        boundaryValue = BCw.getBoundary(PointBoundary::F).value;
                        point.setValue(boundaryValue);
                        point.setPrevValue(boundaryValue);
                        point.setPrevPrevValue(boundaryValue);
                    }

                    w[k][j][i] = point;
                }
            }
        }
    }
    {
        int sizeZ = (nz == 0 ? 1 : nz - 1);
        p.resize(sizeZ);
        for (int k = 0; k < sizeZ; ++k) {
            p[k].resize(ny - 1);
            for (int j = 0; j < ny - 1; ++j) {
                p[k][j].resize(nx - 1);  // all elements default-constructed
            }
        }
        // Create the grid for pressure(p)
#pragma omp parallel for collapse(3)
        for (int k = 0; k < sizeZ; ++k) {
            for (int j = 0; j < ny - 1; ++j) {
                for (int i = 0; i < nx - 1; ++i) {
                    int unique_number = (k) * (ny - 1) * (nx - 1) + (j) * (nx - 1) + i;
                    // Create point object and increment unique number
                    Point point(i, j, k, unique_number, ds);

                    // Set initial conditions
                    double x_coord = (i * ds + ds / 2);
                    double y_coord = (j * ds + ds / 2);
                    double z_coord = (k * ds + ds / 2);

                    point.setX(x_coord);
                    point.setY(y_coord);
                    point.setZ(z_coord);


                    p[k][j][i] = point;
                }
            }
        }
        for (int k = 0; k < sizeZ; ++k) {
            for (int j = ny - 2; j >= 0; --j) {
                for (int i = 0; i < nx - 1; ++i) {
                    // Create point object and increment unique number
                    double pressure = 0.0;
                    p[k][j][i].setValue(pressure);
                }
            }
        }

    }

    return std::make_tuple(u, v, w, p);
}


std::vector<std::vector<std::vector<Point>>> loadV(const std::string& filename, double ds) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(1); // Exiting if the file fails to open.
    }

    std::string line;

    // Skip the header.
    std::getline(file, line);

    // Extract I and J values from the second line.
    std::getline(file, line);
    int I, J, K;
    bool is3D = extractIJK(line, I, J, K);

    int nx = I - 2;
    int ny = J - 1;
    int nz = K - 2; // fir 2D = -1
    int unique_number = 0; // Counter for unique numbers.
    // Variables to hold the two previous line's x and y values.
    std::getline(file, line);
    std::stringstream ss(line);
    double x, y, z, v_val;
    ss >> x >> y >> z >> v_val;
    double prev_x = 0, prev_y = 0, prev_z = 0, y_down = 0, z_back = 0;
    // Initialize an empty 3D vector structure.
    std::vector<std::vector<std::vector<Point>>> v;
    for (int k = 0; k < nz + 2; ++k) {
        // Loop through the file's data lines and populate the vector.
        std::vector<std::vector<Point>> zLayer;
        double dz = calculate_dz(file, k, nz + 1, z_back, z);
        for (int j = 0; j < ny + 1; ++j) {
            std::vector<Point> yLayer;
            double dy = calculate_dy(file, j, ny, y_down, y);
            for (int i = 0; i < nx + 2; ++i) {

                if (std::getline(file, line)) {
                    double dx, x_next, y_next, z_next, v_val_next;
                    std::stringstream ss_new(line);
                    ss_new >> x_next >> y_next >> z_next >> v_val_next;

                    // Calculate dx and dy using the average distance.

                    dx = i == 0 ? x_next - x : (x_next - prev_x) / 2;
                    if (i == nx + 1) dx = x - prev_x;

                    Point point(i, j, k, unique_number++, dx, dy, dz);
                    point.setX(x);
                    point.setY(y);
                    point.setZ(z); // Setting Z value to 0, assuming 2D.

                    // Set boundary conditions based on position.
                    if (i == 0) {
                        point.setBoundaryType(PointBoundary::W);
                    }
                    if (i == nx + 1) {
                        point.setBoundaryType(PointBoundary::E);
                    }
                    if (j == 0) {
                        point.setBoundaryType(PointBoundary::S);
                    }
                    if (j == ny) {
                        point.setBoundaryType(PointBoundary::N);
                    }
                    if (k == 0 && is3D) {
                        point.setBoundaryType(PointBoundary::B);
                    }
                    if (k == nz + 1 && is3D) {
                        point.setBoundaryType(PointBoundary::F);
                    }
                    point.setValue(v_val);
                    point.setPrevValue(v_val);
                    point.setPrevPrevValue(v_val);
                    yLayer.push_back(point); // Add point to yLayer.
                    v_val = v_val_next;
                    if (y_next != y) y_down = y;
                    if (z_next != z) z_back = z;
                    prev_x = x; prev_y = y; prev_z = z;
                    x = x_next; y = y_next; z = z_next;
                }
                else {
                    double dx;
                    dx = x - prev_x;

                    Point point(i, j, k, unique_number++, dx, dy, dz);
                    point.setX(x);
                    point.setY(y);
                    point.setZ(z);  // Setting Z value to 0, assuming 2D.

                    // Set boundary conditions based on position.
                    if (i == 0) {
                        point.setBoundaryType(PointBoundary::W);
                    }
                    if (i == nx + 1) {
                        point.setBoundaryType(PointBoundary::E);
                    }
                    if (j == 0) {
                        point.setBoundaryType(PointBoundary::S);
                    }
                    if (j == ny) {
                        point.setBoundaryType(PointBoundary::N);
                    }
                    if (k == 0 && is3D) {
                        point.setBoundaryType(PointBoundary::B);
                    }
                    if (k == nz + 1 && is3D) {
                        point.setBoundaryType(PointBoundary::F);
                    }
                    point.setValue(v_val);         // Set current value.
                    point.setPrevValue(v_val);     // Assuming current value is also the previous value.
                    point.setPrevPrevValue(v_val); // Assuming current value is the previous-previous value.

                    yLayer.push_back(point);  // Add point to yLayer.
                    prev_x = x; prev_y = y; prev_z = z;
                }
            }
            zLayer.push_back(yLayer); //Add line to zLayer
        }
        v.push_back(zLayer); // Add yLayer to the 3D vector.
    }
    file.close(); // Close the file after reading.
    return v; // Return the populated 3D vector.
}


std::vector<std::vector<std::vector<Point>>> loadW(const std::string& filename, double ds) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(1); // Exiting if the file fails to open.
    }

    std::string line;

    // Skip the header.
    std::getline(file, line);

    // Extract I and J values from the second line.
    std::getline(file, line);
    int I, J, K;
    bool is3D = extractIJK(line, I, J, K);
    if (!is3D) {
        std::cerr << "Error reading K values from file: " << filename << std::endl;
        exit(1); // Exiting if I, J extraction fails.
    }

    int nx = I - 2;
    int ny = J - 2;
    int nz = K - 1;
    int unique_number = 0; // Counter for unique numbers.
    // Variables to hold the two previous line's x and y values.
    std::getline(file, line);
    std::stringstream ss(line);
    double x, y, z, w_val;
    ss >> x >> y >> z >> w_val;
    double prev_x = 0, prev_y = 0, prev_z = 0, y_down = 0, z_back = 0;
    // Initialize an empty 3D vector structure.
    std::vector<std::vector<std::vector<Point>>> w;
    for (int k = 0; k < nz + 1; ++k) {
        // Loop through the file's data lines and populate the vector.
        std::vector<std::vector<Point>> zLayer;
        double dz = calculate_dz(file, k, nz, z_back, z);
        for (int j = 0; j < ny + 2; ++j) {
            std::vector<Point> yLayer;
            double dy = calculate_dy(file, j, ny + 1, y_down, y);
            for (int i = 0; i < nx + 2; ++i) {

                if (std::getline(file, line)) {
                    double dx, x_next, y_next, z_next, w_val_next;
                    std::stringstream ss_new(line);
                    ss_new >> x_next >> y_next >> z_next >> w_val_next;

                    // Calculate dx and dy using the average distance.

                    dx = i == 0 ? x_next - x : (x_next - prev_x) / 2;
                    if (i == nx + 1) dx = x - prev_x;

                    Point point(i, j, k, unique_number++, dx, dy, dz);
                    point.setX(x);
                    point.setY(y);
                    point.setZ(z); // Setting Z value to 0, assuming 2D.

                    // Set boundary conditions based on position.
                    if (i == 0) {
                        point.setBoundaryType(PointBoundary::W);
                    }
                    if (i == nx + 1) {
                        point.setBoundaryType(PointBoundary::E);
                    }
                    if (j == 0) {
                        point.setBoundaryType(PointBoundary::S);
                    }
                    if (j == ny + 1) {
                        point.setBoundaryType(PointBoundary::N);
                    }
                    if (k == 0) {
                        point.setBoundaryType(PointBoundary::B);
                    }
                    if (k == nz) {
                        point.setBoundaryType(PointBoundary::F);
                    }
                    point.setValue(w_val);
                    point.setPrevValue(w_val);
                    point.setPrevPrevValue(w_val);
                    yLayer.push_back(point); // Add point to yLayer.
                    w_val = w_val_next;
                    if (y_next != y) y_down = y;
                    if (z_next != z) z_back = z;
                    prev_x = x; prev_y = y; prev_z = z;
                    x = x_next; y = y_next; z = z_next;
                }
                else {
                    double dx;
                    dx = x - prev_x;

                    Point point(i, j, k, unique_number++, dx, dy, dz);
                    point.setX(x);
                    point.setY(y);
                    point.setZ(z);  // Setting Z value to 0, assuming 2D.

                    // Set boundary conditions based on position.
                    if (i == 0) {
                        point.setBoundaryType(PointBoundary::W);
                    }
                    if (i == nx + 1) {
                        point.setBoundaryType(PointBoundary::E);
                    }
                    if (j == 0) {
                        point.setBoundaryType(PointBoundary::S);
                    }
                    if (j == ny + 1) {
                        point.setBoundaryType(PointBoundary::N);
                    }
                    if (k == 0) {
                        point.setBoundaryType(PointBoundary::B);
                    }
                    if (k == nz) {
                        point.setBoundaryType(PointBoundary::F);
                    }
                    point.setValue(w_val);         // Set current value.
                    point.setPrevValue(w_val);     // Assuming current value is also the previous value.
                    point.setPrevPrevValue(w_val); // Assuming current value is the previous-previous value.

                    yLayer.push_back(point);  // Add point to yLayer.
                    prev_x = x; prev_y = y; prev_z = z;
                }
            }
            zLayer.push_back(yLayer); //Add line to zLayer
        }
        w.push_back(zLayer); // Add yLayer to the 3D vector.
    }
    file.close(); // Close the file after reading.
    return w; // Return the populated 3D vector.
}

std::vector<std::vector<std::vector<Point>>>
loadP(const std::string& filename,
    double ds,
    double Lx, double Ly, double Lz,
    const std::vector<std::vector<std::vector<Point>>>& u,
    const std::vector<std::vector<std::vector<Point>>>& v,
    const std::vector<std::vector<std::vector<Point>>>& w)
{
    //---------------------------------------------------------------------
    //  open file and skip the first (comment) line
    //---------------------------------------------------------------------
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("cannot open " + filename);

    std::string line;
    std::getline(file, line);                     // header #1
    std::getline(file, line);                     // header #2
    int I = 0, J = 0, K = 0;
    const bool is3D = extractIJK(line, I, J, K);  // <- TRUE => 3D
    if (!is3D) K = 1;                             // force single layer

    const int nx = I;               // cell centres, not u / v counts
    const int ny = J;
    const int nz = K;               // will be 1 in purely 2D data

    //---------------------------------------------------------------------
    //  simple sanity check against nominal domain size
    //---------------------------------------------------------------------
    auto ncell = [ds](double L) { return static_cast<int>(std::round(L / ds)); };
    if (nx != ncell(Lx) || ny != ncell(Ly) || nz != (is3D ? ncell(Lz) : 1)) {
        std::ostringstream msg;
        msg << "grid mismatch in " << filename << "\n"
            << "  Tecplot I,J,K = " << nx << "," << ny << "," << nz
            << "   expected " << ncell(Lx) << "," << ncell(Ly) << ","
            << (is3D ? ncell(Lz) : 1);
        throw std::runtime_error(msg.str());
    }

    //---------------------------------------------------------------------
    //  read first data quadruple (x y [z] p) – we stream the rest later
    //---------------------------------------------------------------------
    std::getline(file, line);
    std::stringstream ss0(line);
    double x{}, y{}, z{}, pVal{};
    ss0 >> x >> y;
    if (is3D) ss0 >> z;
    ss0 >> pVal;

    //---------------------------------------------------------------------
    //  allocate outer container
    //---------------------------------------------------------------------
    std::vector<std::vector<std::vector<Point>>> P(nz);

    long long uniq = 0;                            // running unique id
    double prevX = 0.0, prevY = 0.0, prevZ = 0.0;
    double yLow = 0.0, zBack = 0.0;               // “previous row / slab”

    //---------------------------------------------------------------------
    //  main loop:  k-layers  ->  j-rows  ->  i-columns
    //---------------------------------------------------------------------
    for (int k = 0; k < nz; ++k) {

        //------------------- dz from W-grid (or dummy 1.0) ---------------
        double dz = 1.0;
        if (is3D) {
            const double zW = w[k][0][0].getZ();
            const double zWnext = w[k + 1][0][0].getZ();
            dz = zWnext - zW;
        }
        else {
            z = 0.0;                              // single plane
        }

        std::vector<std::vector<Point>> zLayer;
        zLayer.reserve(ny);

        for (int j = 0; j < ny; ++j) {

            //---------------- dy from V-grid -----------------------------
            const double yV = v[k][j][0].getY();
            const double yVnext = v[k][j + 1][0].getY();
            const double dy = yVnext - yV;

            std::vector<Point> yLayer;
            yLayer.reserve(nx);

            for (int i = 0; i < nx; ++i)
            {
                //------------------------------------------------------------------
                // 1.  Try to fetch the *next* record.  Works for 2-D (x y p) and
                //     3-D (x y z p) Tecplot node format.
                //------------------------------------------------------------------
                double x_next = x, y_next = y, z_next = z;   // fall-back = “repeat”
                double p_next = pVal;
                bool   have_next = false;

                std::string nextLine;
                if (std::getline(file, nextLine))
                {
                    std::stringstream ssn(nextLine);
                    if (is3D) {
                        ssn >> x_next >> y_next >> z_next >> p_next;   // x y z p
                    }
                    else {
                        ssn >> x_next >> y_next >> p_next;             // x y p
                        z_next = 0.0;                                  // 2-D plane
                    }
                    have_next = true;
                }
                //------------------------------------------------------------------
                // 2.  Cell widths from staggered grids (u, v, w)
                //------------------------------------------------------------------
                // dx     – from U grid
                const double xU = u[k][j + 1][i].getX();
                const double xUnext = u[k][j + 1][i + 1].getX();
                const double dx = (i == nx - 1) ? (Lx - x) * 2.0
                    : (xUnext - xU);

                // dy     – from V grid
                const double yV = v[k][j][0].getY();
                const double yVnext = v[k][j + 1][0].getY();
                const double dy = yVnext - yV;

                // dz     – from W grid if 3D, else constant 1.0
                double dz = 1.0;
                if (is3D) {
                    const double zW = w[k][0][0].getZ();
                    const double zWnext = w[k + 1][0][0].getZ();
                    dz = zWnext - zW;
                }

                //------------------------------------------------------------------
                // 3.  Create the Point
                //------------------------------------------------------------------
                Point pt(i, j, k, uniq++, dx, dy, dz);
                pt.setX(x);  pt.setY(y);  pt.setZ(z);

                /* ----- existing boundary-type logic stays unchanged here ----- */

                pt.setValue(pVal);
                pt.setPrevValue(pVal);
                pt.setPrevPrevValue(pVal);

                yLayer.push_back(std::move(pt));

                //------------------------------------------------------------------
                // 4.  Advance “current” values to those we just peeked
                //------------------------------------------------------------------
                if (have_next) {
                    x = x_next;
                    y = y_next;
                    z = z_next;
                    pVal = p_next;
                }
            }
            zLayer.emplace_back(std::move(yLayer));
        }
        P[k] = std::move(zLayer);
    }

    file.close();
    return P;
}

void update_dxdy(std::vector<std::vector<std::vector<Point>>>& u, std::vector<std::vector<std::vector<Point>>>& v) {
    for (int j = 1; j < v[0].size(); j++) {
        for (size_t i = 1; i < u[0][0].size(); i++)
        {
            v[0][j][i].setdx(u[0][j][i].getX() - u[0][j][i-1].getX());
            u[0][j][i].setdy(v[0][j][i].getY() - v[0][j-1][i].getY());
        }
    }
}
