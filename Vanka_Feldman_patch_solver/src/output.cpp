#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include "Point.h"
#include "output.h"
#include <numeric>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <system_error>

#ifdef VANKA_HAVE_TECIO
#include <TECIO.h>
#endif

bool writeTecplotBinarySnapshot(
    const std::vector<std::vector<std::vector<Point>>>& field,
    const std::string& varName,
    const std::string& fileName,
    double time,
    int timeStep
) {
#ifndef VANKA_HAVE_TECIO
    static bool warned = false;
    if (!warned) {
        std::cerr
            << "Tecplot SZL snapshot skipped: this executable was built without TecIO. "
            << "Configure with -DTECIO_ROOT=<TecIO installation> to enable it."
            << std::endl;
        warned = true;
    }
    (void)field;
    (void)varName;
    (void)fileName;
    (void)time;
    (void)timeStep;
    return false;
#else
    namespace fs = std::filesystem;

    if (field.empty() || field.front().empty() || field.front().front().empty()) {
        std::cerr << "Tecplot SZL snapshot failed: field is empty." << std::endl;
        return false;
    }

    const size_t kMax = field.size();
    const size_t jMax = field.front().size();
    const size_t iMax = field.front().front().size();
    for (const auto& layer : field) {
        if (layer.size() != jMax) {
            std::cerr << "Tecplot SZL snapshot failed: field is not rectangular." << std::endl;
            return false;
        }
        for (const auto& row : layer) {
            if (row.size() != iMax) {
                std::cerr << "Tecplot SZL snapshot failed: field is not rectangular." << std::endl;
                return false;
            }
        }
    }

    if (jMax > std::numeric_limits<size_t>::max() / iMax ||
        kMax > std::numeric_limits<size_t>::max() / (iMax * jMax)) {
        std::cerr << "Tecplot SZL snapshot failed: field size overflow." << std::endl;
        return false;
    }
    const size_t valueCount = iMax * jMax * kMax;
    if (valueCount > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
        std::cerr << "Tecplot SZL snapshot failed: field is too large for TecIO." << std::endl;
        return false;
    }

    const bool is3D = kMax > 1;
    const std::string variables = is3D
        ? "X Y Z " + varName
        : "X Y " + varName;
    const std::string zoneTitle = "Frame " + std::to_string(timeStep);

    const fs::path targetPath(fileName);
    const fs::path temporaryPath =
        targetPath.parent_path() /
        (targetPath.stem().string() + ".tmp" + targetPath.extension().string());
    std::error_code fileError;
    fs::remove(temporaryPath, fileError);

    void* fileHandle = nullptr;
    auto fail = [&](const char* operation, int32_t status) {
        std::cerr << "Tecplot SZL snapshot failed during " << operation
            << " (TecIO status " << status << ")." << std::endl;
        if (fileHandle != nullptr) {
            tecFileWriterClose(&fileHandle);
        }
        std::error_code cleanupError;
        fs::remove(temporaryPath, cleanupError);
        return false;
        };

    int32_t status = tecFileWriterOpen(
        temporaryPath.string().c_str(),
        "Vanka-Feldman VOF",
        variables.c_str(),
        FILEFORMAT_SZL,
        FILETYPE_FULL,
        FieldDataType_Float,
        nullptr,
        &fileHandle);
    if (status != 0) {
        return fail("tecFileWriterOpen", status);
    }

    const int32_t variableCount = is3D ? 4 : 3;
    std::vector<int32_t> variableTypes(variableCount, FieldDataType_Float);
    std::vector<int32_t> sharedVariables(variableCount, 0);
    std::vector<int32_t> valueLocations(variableCount, 1);
    int32_t zone = 0;
    status = tecZoneCreateIJK(
        fileHandle,
        zoneTitle.c_str(),
        static_cast<int64_t>(iMax),
        static_cast<int64_t>(jMax),
        static_cast<int64_t>(kMax),
        variableTypes.data(),
        sharedVariables.data(),
        valueLocations.data(),
        nullptr,
        0,
        0,
        0,
        &zone);
    if (status != 0) {
        return fail("tecZoneCreateIJK", status);
    }

    status = tecZoneSetUnsteadyOptions(fileHandle, zone, time, 1);
    if (status != 0) {
        return fail("tecZoneSetUnsteadyOptions", status);
    }

    std::vector<float> values(valueCount);
    auto writeVariable = [&](int32_t variable, const auto& accessor) {
        size_t index = 0;
        for (const auto& layer : field) {
            for (const auto& row : layer) {
                for (const auto& point : row) {
                    values[index++] = static_cast<float>(accessor(point));
                }
            }
        }
        return tecZoneVarWriteFloatValues(
            fileHandle,
            zone,
            variable,
            0,
            static_cast<int64_t>(valueCount),
            values.data());
        };

    status = writeVariable(1, [](const Point& point) { return point.getX(); });
    if (status != 0) return fail("writing X", status);
    status = writeVariable(2, [](const Point& point) { return point.getY(); });
    if (status != 0) return fail("writing Y", status);
    if (is3D) {
        status = writeVariable(3, [](const Point& point) { return point.getZ(); });
        if (status != 0) return fail("writing Z", status);
    }
    status = writeVariable(
        is3D ? 4 : 3,
        [](const Point& point) { return point.getValue(); });
    if (status != 0) return fail("writing field values", status);

    status = tecFileWriterClose(&fileHandle);
    if (status != 0) {
        return fail("tecFileWriterClose", status);
    }

    fileError.clear();
    if (fs::exists(targetPath, fileError)) {
        fs::remove(targetPath, fileError);
    }
    if (fileError) {
        std::cerr << "Tecplot SZL snapshot failed while replacing "
            << targetPath << ": " << fileError.message() << std::endl;
        std::error_code cleanupError;
        fs::remove(temporaryPath, cleanupError);
        return false;
    }

    fs::rename(temporaryPath, targetPath, fileError);
    if (fileError) {
        std::cerr << "Tecplot SZL snapshot failed while finalizing "
            << targetPath << ": " << fileError.message() << std::endl;
        std::error_code cleanupError;
        fs::remove(temporaryPath, cleanupError);
        return false;
    }
    return true;
#endif
}

void writeTecplotFile(
    const std::vector<std::vector<std::vector<Point>>>& field,
    const std::string& varName,
    const std::string& fileName,
    double time,
    int timeStep
) {
    // Open an output file stream
    std::ofstream outputFile(fileName);

    // Check if the file is open
    if (!outputFile.is_open()) {
        std::cerr << "Could not open the file for writing." << std::endl;
        return;
    }

    // Determine if the data is 3D
    bool is3D = (field.size() > 1);

    // Write header information for Tecplot
    outputFile << "VARIABLES = \"X\", \"Y\"";
    if (is3D) outputFile << ", \"Z\"";
    outputFile << ", \"" << varName << "\" \n";
    outputFile << "ZONE T=\"Frame " << timeStep << "\", I=" << field[0][0].size()
        << ", J=" << field[0].size();
    if (is3D) outputFile << ", K=" << field.size();
    outputFile << ", F=POINT\n";

    // Write data
    for (size_t k = 0; k < field.size(); ++k) {
        // If not 3D, only execute outer loop once
        if (!is3D && k > 0) break;

        for (size_t j = 0; j < field[k].size(); ++j) {
            for (size_t i = 0; i < field[k][j].size(); ++i) {
                double x = field[k][j][i].getX();
                double y = field[k][j][i].getY();
                double z = field[k][j][i].getZ();

                outputFile << std::scientific << std::setprecision(8)
                    << x << " " << y;
                if (is3D) outputFile << " " << z;
                outputFile << std::scientific << std::setprecision(8) << " " << field[k][j][i].getValue() << "\n";
            }
        }
    }

    // Close the output file stream
    outputFile.close();
}

void writeTecplotFile(
    const std::vector<std::vector<std::vector<Point>>>& field,
    const std::vector<std::vector<std::vector<double>>>& fieldValue,
    const std::string& varName,
    const std::string& fileName,
    double time,
    int timeStep
) {
    // Open an output file stream
    std::ofstream outputFile(fileName);

    // Check if the file is open
    if (!outputFile.is_open()) {
        std::cerr << "Could not open the file for writing." << std::endl;
        return;
    }

    // Determine if the data is 3D
    bool is3D = (field.size() > 1);

    // Write header information for Tecplot
    outputFile << "VARIABLES = \"X\", \"Y\"";
    if (is3D) outputFile << ", \"Z\"";
    outputFile << ", \"" << varName << "\" \n";
    outputFile << "ZONE T=\"Frame " << timeStep << "\", I=" << field[0][0].size()
        << ", J=" << field[0].size();
    if (is3D) outputFile << ", K=" << field.size();
    outputFile << ", F=POINT\n";

    // Write data
    for (size_t k = 0; k < field.size(); ++k) {
        // If not 3D, only execute outer loop once
        if (!is3D && k > 0) break;

        for (size_t j = 0; j < field[k].size(); ++j) {
            for (size_t i = 0; i < field[k][j].size(); ++i) {
                double x = field[k][j][i].getX();
                double y = field[k][j][i].getY();
                double z = field[k][j][i].getZ();

                outputFile << std::scientific << std::setprecision(8)
                    << x << " " << y;
                if (is3D) outputFile << " " << z;
                outputFile << std::scientific << std::setprecision(8) << " " << fieldValue[k][j][i] << "\n";
            }
        }
    }

    // Close the output file stream
    outputFile.close();
}
void writeAllTecplotFile(
    const std::vector<std::vector<std::vector<double>>>& uModified,
    const std::vector<std::vector<std::vector<double>>>& vModified,
    const std::vector<std::vector<std::vector<double>>>& wModified,
    const std::vector<std::vector<std::vector<Point>>>& p,
    const std::vector<std::vector<std::vector<Point>>>& phi,
    const std::string& fileName,
    double time,
    int timeStep
) {
    // Open an output file stream
    std::ofstream outputFile(fileName);

    // Check if the file is open
    if (!outputFile.is_open()) {
        std::cerr << "Could not open the file for writing." << std::endl;
        return;
    }

    // Determine if the data is 3D
    bool is3D = (p.size() > 1);

    // Write header information for Tecplot
    outputFile << "VARIABLES = \"X\", \"Y\"";
    if (is3D) outputFile << ", \"Z\"";
    outputFile << ", \"u\", \"v\"";
    if (is3D) outputFile << ", \"w\"";
    outputFile << ", \"p\", \"Phi\" \n";
    outputFile << "ZONE T=\"Frame " << timeStep << ", Time: " << time
        << "\", I=" << p[0][0].size()
        << ", J=" << p[0].size();
    if (is3D) outputFile << ", K=" << p.size();
    outputFile << ", F=POINT\n";
    // Write data
    for (size_t k = 0; k < uModified.size(); ++k) {
        // If not 3D, only execute outer loop once
        if (!is3D && k > 0) break;

        for (size_t j = 0; j < uModified[k].size(); ++j) {

            for (size_t i = 0; i < uModified[k][j].size(); ++i) {
                double x = p[k][j][i].getX();
                double y = p[k][j][i].getY();
                double z = p[k][j][i].getZ();
                outputFile << std::fixed << std::setprecision(5) << x;
                if (is3D) outputFile << " " << z;
                outputFile << " " << y;
                outputFile << std::scientific << std::setprecision(8)
                    << " " << uModified[k][j][i];
                if (is3D) outputFile << " " << wModified[k][j][i];
                outputFile << " " << vModified[k][j][i];
                outputFile << " " << p[k][j][i].getValue() << " " << phi[k][j][i].getValue() << "\n";
            }
        }
    }


    // Close the output file stream
    outputFile.close();
}

void writeAllTecplotFile(
    const std::vector<std::vector<std::vector<double>>>& uModified,
    const std::vector<std::vector<std::vector<double>>>& vModified,
    const std::vector<std::vector<std::vector<double>>>& wModified,
    const std::vector<std::vector<std::vector<Point>>>& p,
    const std::string& fileName,
    double time,
    int timeStep
) {
    // Open an output file stream
    std::ofstream outputFile(fileName);

    // Check if the file is open
    if (!outputFile.is_open()) {
        std::cerr << "Could not open the file for writing." << std::endl;
        return;
    }

    // Determine if the data is 3D
    bool is3D = (p.size() > 1);

    // Write header information for Tecplot
    outputFile << "VARIABLES = \"X\", \"Y\"";
    if (is3D) outputFile << ", \"Z\"";
    outputFile << ", \"u\", \"v\"";
    if (is3D) outputFile << ", \"w\"";
    outputFile << ", \"p\" \n";
    outputFile << "ZONE T=\"Frame " << timeStep << ", Time: " << time
        << "\", I=" << p[0][0].size()
        << ", J=" << p[0].size();
    if (is3D) outputFile << ", K=" << p.size();
    outputFile << ", F=POINT\n";
    // Write data
    for (size_t k = 0; k < uModified.size(); ++k) {
        // If not 3D, only execute outer loop once
        if (!is3D && k > 0) break;

        for (size_t j = 0; j < uModified[k].size(); ++j) {

            for (size_t i = 0; i < uModified[k][j].size(); ++i) {
                double x = p[k][j][i].getX();
                double y = p[k][j][i].getY();
                double z = p[k][j][i].getZ();
                outputFile << std::fixed << std::setprecision(5)
                    << x << " " << y;
                if (is3D) outputFile << " " << z;
                outputFile << std::scientific << std::setprecision(8)
                     << " " << uModified[k][j][i]
                    << " " << vModified[k][j][i];
                if (is3D) outputFile << " " << wModified[k][j][i];
                outputFile << " " << p[k][j][i].getValue() << "\n";
            }
        }
    }


    // Close the output file stream
    outputFile.close();
}
void out(
    const std::vector<std::vector<std::vector<Point>>>& u,
    const std::vector<std::vector<std::vector<Point>>>& v,
    const std::vector<std::vector<std::vector<Point>>>& w,
    const std::vector<std::vector<std::vector<Point>>>& p,
    double timeDerivativeU,
    double timeDerivativeV,
    double timeDerivativeW,
    double timeDerivativeP,
    double CPUtime,
    double time,
    int timeStep,
    double dt
) {
    // Output the data to Tecplot files
    writeTecplotFile(u, "U", "Vx.dat", time, timeStep);
    writeTecplotFile(v, "V", "Vy.dat", time, timeStep);
    if (w.size() > 1) {
        writeTecplotFile(w, "W", "Vz.dat", time, timeStep);
    }
    writeTecplotFile(p, "P", "P.dat", time, timeStep);

    // Output messages to console
    std::cout << "Current calculation time: " << time
        << "; timestep number: " << timeStep
        << "; timestep size: " << time / timeStep << "\n";
    std::cout << "Residual U: " << timeDerivativeU
        << "; Residual V: " << timeDerivativeV;
    if (w.size() > 1) {
        std::cout << "; Residual W: " << timeDerivativeW;
    }
    std::cout << "; Residual P: " << timeDerivativeP << "\n";
    std::cout << "Current CPU time: " << CPUtime << std::endl;

}

void out(
    const std::vector<std::vector<std::vector<Point>>>& u,
    const std::vector<std::vector<std::vector<Point>>>& v,
    const std::vector<std::vector<std::vector<Point>>>& w,
    const std::vector<std::vector<std::vector<Point>>>& p,
    const std::vector<std::vector<std::vector<Point>>>& phi,
    double timeDerivativeU,
    double timeDerivativeV,
    double timeDerivativeW,
    double timeDerivativeP,
    double time,
    double CPUtime,
    double dt,
    int timeStep) {
    // Output the data to Tecplot files
    writeTecplotFile(u, "U", "Vx.dat", time, timeStep);
    writeTecplotFile(v, "V", "Vy.dat", time, timeStep);
    if (w.size() > 1) {
        writeTecplotFile(w, "W", "Vz.dat", time, timeStep);
    }
    writeTecplotFile(p, "P", "P.dat", time, timeStep);
    writeTecplotFile(phi, "Phi", "Phi.dat", time, timeStep);
    //  writeTecplotFile(rho, "Rho", "Rho.dat", time, timeStep);
      // Output messages to console
    std::cout << "Current calculation time: " << time
        << "; timestep number: " << timeStep
        << "; timestep size: " << time / timeStep << " Calculation time " << CPUtime << "\n";
    std::cout << "Residual U: " << timeDerivativeU
        << "; Residual V: " << timeDerivativeV;
    if (w.size() > 1) {
        std::cout << "; Residual W: " << timeDerivativeW;
    }
    std::cout << "; Residual P: " << timeDerivativeP << "\n";
}


void allOut(
    const std::vector<std::vector<std::vector<Point>>>& u,
    const std::vector<std::vector<std::vector<Point>>>& v,
    const std::vector<std::vector<std::vector<Point>>>& w,
    const std::vector<std::vector<std::vector<Point>>>& p,
    double CPUtime,
    double time,
    int timeStep
) {
    // Determine if the data is 3D
    bool is3D = (w.size() > 1);
    // Interpolate u, v, w, remove first and last elements in respective dimensions
    auto uModified = interpolateAndProcess(u, is3D, 'u');
    auto vModified = interpolateAndProcess(v, is3D, 'v');
    auto wModified = is3D ? interpolateAndProcess(w, is3D, 'w') : std::vector<std::vector<std::vector<double>>>();

    // Write the Tecplot files for u, v, w, p
    writeTecplotFile(u, "U", "Vx.dat", time, timeStep);
    writeTecplotFile(v, "V", "Vy.dat", time, timeStep);
    if (is3D) writeTecplotFile(w, "W", "Vz.dat", time, timeStep);
    writeTecplotFile(p, "P", "P.dat", time, timeStep);

    // Write all to a single Tecplot file
    writeAllTecplotFile(uModified, vModified, wModified, p, "all.dat", time, timeStep);

    // Print to command prompt
    std::cout << "Current CPU time: " << CPUtime << std::endl;
}
void allOut(
    const std::vector<std::vector<std::vector<Point>>>& u,
    const std::vector<std::vector<std::vector<Point>>>& v,
    const std::vector<std::vector<std::vector<Point>>>& w,
    const std::vector<std::vector<std::vector<Point>>>& p,
    const std::vector<std::vector<std::vector<Point>>>& phi,
    double CPUtime,
    double time,
    int timeStep
) {
    // Determine if the data is 3D
    bool is3D = (w.size() > 1);
    // Interpolate u, v, w, remove first and last elements in respective dimensions
    auto uModified = interpolateAndProcess(u, is3D, 'u');
    auto vModified = interpolateAndProcess(v, is3D, 'v');
    auto wModified = is3D ? interpolateAndProcess(w, is3D, 'w') : std::vector<std::vector<std::vector<double>>>();

    // Write the Tecplot files for u, v, w, p
    writeTecplotFile(u, "U", "Vx.dat", time, timeStep);
    writeTecplotFile(v, "V", "Vy.dat", time, timeStep);
    if (is3D) writeTecplotFile(w, "W", "Vz.dat", time, timeStep);
    writeTecplotFile(p, "P", "P.dat", time, timeStep);
    writeTecplotFile(phi, "Phi", "Phi.dat", time, timeStep);
    // Write all to a single Tecplot file
    writeAllTecplotFile(uModified, vModified, wModified, p, phi, "all.dat", time, timeStep);

    // Print to command prompt
    std::cout << "Current CPU time: " << CPUtime << "\n";
}

bool writeToleranceStudySnapshot(
    const std::vector<std::vector<std::vector<Point>>>& u,
    const std::vector<std::vector<std::vector<Point>>>& v,
    const std::vector<std::vector<std::vector<Point>>>& w,
    const std::vector<std::vector<std::vector<Point>>>& p,
    const std::vector<std::vector<std::vector<Point>>>& phi,
    const std::string& fileName) {
    if (p.empty() || p[0].empty() || p[0][0].empty() || p.size() != phi.size()) {
        return false;
    }

    std::ofstream output(fileName, std::ios::binary | std::ios::trunc);
    if (!output.is_open()) return false;

    constexpr char magic[8] = {'V', 'A', 'N', 'K', 'A', 'T', 'O', 'L'};
    const std::uint32_t version = 1;
    const std::uint32_t nx = static_cast<std::uint32_t>(p[0][0].size());
    const std::uint32_t ny = static_cast<std::uint32_t>(p[0].size());
    const std::uint32_t nz = static_cast<std::uint32_t>(p.size());
    const std::uint64_t cellCount =
        static_cast<std::uint64_t>(nx) * ny * nz;
    output.write(magic, sizeof(magic));
    output.write(reinterpret_cast<const char*>(&version), sizeof(version));
    output.write(reinterpret_cast<const char*>(&nx), sizeof(nx));
    output.write(reinterpret_cast<const char*>(&ny), sizeof(ny));
    output.write(reinterpret_cast<const char*>(&nz), sizeof(nz));
    output.write(reinterpret_cast<const char*>(&cellCount), sizeof(cellCount));

    for (std::size_t k = 0; k < p.size(); ++k) {
        for (std::size_t j = 0; j < p[k].size(); ++j) {
            for (std::size_t i = 0; i < p[k][j].size(); ++i) {
                const double values[5] = {
                    0.5 * (u[k + 1][j + 1][i].getValue() +
                        u[k + 1][j + 1][i + 1].getValue()),
                    0.5 * (v[k + 1][j][i + 1].getValue() +
                        v[k + 1][j + 1][i + 1].getValue()),
                    0.5 * (w[k][j + 1][i + 1].getValue() +
                        w[k + 1][j + 1][i + 1].getValue()),
                    p[k][j][i].getValue(),
                    phi[k][j][i].getValue()
                };
                output.write(
                    reinterpret_cast<const char*>(values),
                    static_cast<std::streamsize>(sizeof(values)));
            }
        }
    }

    return output.good();
}
std::vector<std::vector<std::vector<double>>> interpolateAndProcess(
    const std::vector<std::vector<std::vector<Point>>>& field,
    bool is3D,
    char dimension)
{
    int width = (dimension == 'u') ? field[0][0].size() - 1 : field[0][0].size() - 2;
    int height = (dimension == 'v') ? field[0].size() - 1 : field[0].size() - 2;
    int depth = (is3D) ? ((dimension == 'w') ? field.size() - 1 : field.size() - 2) : 1;
    // Initialize fieldModified with the same size as the input field
    std::vector<std::vector<std::vector<double>>> fieldModified(depth, std::vector<std::vector<double>>(height, std::vector<double>(width, 0.0)));

    switch (dimension)
    {
    case 'u':
        for (size_t k = is3D ? 1 : 0; k < field.size() - (is3D ? 1 : 0); ++k) {
            for (size_t j = 1; j < field[k].size() - 1; ++j) {
                for (size_t i = 0; i < field[k][j].size() - 1; ++i) {
                    // Compute the average of adjacent points in the x dimension
                    fieldModified[is3D ? k - 1 : 0][j - 1][i] = (field[k][j][i].getValue() + field[k][j][i + 1].getValue()) / 2.0;
                }
            }
        }
        break;
    case 'v':
        for (size_t k = is3D ? 1 : 0; k < field.size() - (is3D ? 1 : 0); ++k) {
            for (size_t j = 0; j < field[k].size() - 1; ++j) {
                for (size_t i = 1; i < field[k][j].size() - 1; ++i) {
                    // Compute the average of adjacent points in the y dimension
                    fieldModified[is3D ? k - 1 : 0][j][i - 1] = (field[k][j][i].getValue() + field[k][j + 1][i].getValue()) / 2.0;
                }
            }
        }
        break;
    case 'w':
        if (is3D) {
            for (size_t k = 0; k < field.size() - 1; ++k) {
                for (size_t j = 1; j < field[k].size() - 1; ++j) {
                    for (size_t i = 1; i < field[k][j].size() - 1; ++i) {
                        // Compute the average of adjacent points in the z dimension
                        fieldModified[k][j - 1][i - 1] = (field[k][j][i].getValue() + field[k + 1][j][i].getValue()) / 2.0;
                    }
                }
            }
        }
        break;
    }

    return fieldModified;
}



/*void VideoOut(
    const std::vector<std::vector<std::vector<Point>>>& u,
    const std::vector<std::vector<std::vector<Point>>>& v,
    const std::vector<std::vector<std::vector<Point>>>& w,
    const std::vector<std::vector<std::vector<Point>>>& p,
    const std::vector<std::vector<std::vector<Point>>>& phi,
    const std::vector<std::vector<std::vector<Point>>>& rho,
    double time,
    int timeStep
) {
    // Determine if the data is 3D
    bool is3D = (w.size() > 1);
    // Interpolate u, v, w, remove first and last elements in respective dimensions
    auto uModified = interpolateAndProcess(u, is3D, 'u');
    auto vModified = interpolateAndProcess(v, is3D, 'v');
    auto wModified = is3D ? interpolateAndProcess(w, is3D, 'w') : std::vector<std::vector<std::vector<double>>>();
    std::ostringstream filename;
    filename << "video" << timeStep << ".dat";
    std::string filenameStr = filename.str();

    // Write all to a single Tecplot file
    writeAllTecplotFile(uModified, vModified, wModified, p, phi, filenameStr, time, timeStep);

}*/





