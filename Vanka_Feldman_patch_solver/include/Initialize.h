#pragma once
#ifndef INITIALIZE_H
#define INITIALIZE_H

#include "Point.h"
#include <tuple>
#include <vector>
#include "BoundaryContainer.h"

std::tuple<std::vector<std::vector<std::vector<Point>>>, std::vector<std::vector<std::vector<Point>>>, std::vector<std::vector<std::vector<Point>>>, std::vector<std::vector<std::vector<Point>>>> initialize(const BoundaryContainer& BCu, const BoundaryContainer& BCv, const BoundaryContainer& BCw, double Lx, double Ly, double Lz, double ds, double initialValueU,double initialValueV,double initialValueW, const std::vector<std::vector<std::vector<Point>>>& phi, double rho_ratio, double Fr, bool developed);

std::tuple<std::vector<std::vector<std::vector<Point>>>, std::vector<std::vector<std::vector<Point>>>, std::vector<std::vector<std::vector<Point>>>, std::vector<std::vector<std::vector<Point>>>> initialize(const BoundaryContainer& BCu, const BoundaryContainer& BCv, const BoundaryContainer& BCw, double Lx, double Ly, double Lz, double ds);

bool extractIJ(const std::string& line, int& I, int& J);
bool extractIJK(const std::string& line, int& I, int& J, int& K);
double calculate_dy(std::ifstream& file, int j, int ny, double& y_down, double& y);
double calculate_dy(std::ifstream& file, int j, int ny, double& y_down, double& y, double Ly);
std::vector<std::vector<std::vector<Point>>> loadU(const std::string& filename, double ds);
std::vector<std::vector<std::vector<Point>>> loadV(const std::string& filename, double ds);
std::vector<std::vector<std::vector<Point>>> loadW(const std::string& filename, double ds);
std::vector<std::vector<std::vector<Point>>> loadP(const std::string& filename, double ds, double Lx, double Ly, double Lz,const std::vector<std::vector<std::vector<Point>>>& u, const std::vector<std::vector<std::vector<Point>>>& v, const std::vector<std::vector<std::vector<Point>>>& w);

void update_dxdy(std::vector<std::vector<std::vector<Point>>>& u, std::vector<std::vector<std::vector<Point>>>& v);

#endif //INITIALIZE_H