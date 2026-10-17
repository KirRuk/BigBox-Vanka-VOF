#include "read.h"
#include <fstream>
#include <sstream>
#include <exception>

inline void trim(std::string &s) {
    auto not_space = [](unsigned char ch) {
        return !std::isspace(ch);
    };

    // left trim
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    // right trim
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
}

UserChoices read(const std::string& filename) {
    std::map<std::string, std::string> userChoices;
    UserChoices result;

    try {
        std::ifstream userChoiceFile(filename);

        if (!userChoiceFile) {
            throw std::runtime_error("File " + filename + " cannot be found\n");
        }
        // Set locale to ensure consistent behavior across platforms
        std::locale::global(std::locale("C"));


        std::string line;
        while (std::getline(userChoiceFile, line)) {
            // skip empty lines and comments if you want
            if (line.empty() || line[0] == '#') {
                continue;
            }
        
            std::istringstream lineStream(line);
            std::string key;
            std::string value;
        
            // this splits on any whitespace, so multiple spaces are fine
            if (lineStream >> key >> value) {
                trim(key);
                trim(value);
                userChoices[key] = value;
            }
        }
        

        userChoiceFile.close();

        result.Lx = std::stod(userChoices["X_direction"]);
        result.Ly = std::stod(userChoices["Y_direction"]);
        result.Lz = std::stod(userChoices["Z_direction"]);
        result.ds = std::stod(userChoices["Grid_step"]);
        result.twoDimensional = (userChoices["Two_dimensional"] == "true");
        result.developed = (userChoices["Developed_flow"] == "true");
        result.writeFreq = std::stoi(userChoices["output_frequency"]);
        result.dt = std::stod(userChoices["time_step"]);
        result.terminationTime = std::stod(userChoices["Termination_Time"]);
        result.Re = std::stod(userChoices["Reynold's_number"]);
        result.nSubdomainsX = std::stoi(userChoices["n_subdomains_X"]);
        result.nSubdomainsY = std::stoi(userChoices["n_subdomains_Y"]);
        result.UR = std::stod(userChoices["UR"]);
        result.convgCrit = std::stod(userChoices["ConvgCrit"]);
        result.convgCritSS = std::stod(userChoices["ConvgCrit_S_S"]);
        result.maxRelaxNumber = std::stoi(userChoices["Max_Relax_Number"]);
        result.Iprint = std::stoi(userChoices["Iprint"]);
        result.supportCellsRad = std::stod(userChoices["support_cells_Rad"]);
        result.read_data = (userChoices["read_data"] == "true");
        result.dynamics = (userChoices["dynamics"] == "true");
        result.DynCoord.resize(3);
        result.DynCoord[0] = std::stoi(userChoices["X_Dynamics"]);
        result.DynCoord[1] = std::stoi(userChoices["Y_Dynamics"]);
        result.DynCoord[2] = std::stoi(userChoices["Z_Dynamics"]);
        result.mooving = (userChoices["Mooving"] == "true");
        result.skipfluid = (userChoices["SkipFluid"] == "true");
        result.readBody = (userChoices["ReadBody"] == "true");
        result.Multigrid = std::stoi(userChoices["Multigrid"]);
        result.threads = std::stoi(userChoices["Threads"]);
        result.CG = std::stod(userChoices["CG"]);
        result.Rho = std::stod(userChoices["RhoRelation"]);
        result.Mu = std::stod(userChoices["MuRelation"]);
        result.Fr = std::stod(userChoices["Fr"]);
        result.We = std::stod(userChoices["We"]);
        return result;
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << '\n';
        exit(1);
    }
}

/*
#include <iostream>
#include <fstream>
#include <string>
#include <exception>

void read() {

    std::string str_name;

    bool two_dimensional;
    int write_freq;
    double Lx, Ly, Lz, ds, dt, Termination_Time, Re;
    int n_subdomains_X, n_subdomains_Y;
    double UR, ConvgCrit, ConvgCrit_S_S;
    int Max_Relax_Number, Iprint, support;

    try {
        std::ifstream user_choice("user.choice");

        if(!user_choice) {
            throw std::runtime_error("File user.choice cannot be found\n");
        }

        user_choice >> str_name >> Lx;
        user_choice >> str_name >> Ly;
        user_choice >> str_name >> Lz;
        user_choice >> str_name >> ds;
        user_choice >> str_name >> two_dimensional;
        user_choice >> str_name >> write_freq;
        user_choice >> str_name >> dt;
        user_choice >> str_name >> Termination_Time;
        user_choice >> str_name >> Re;
        user_choice >> str_name >> n_subdomains_X;
        user_choice >> str_name >> n_subdomains_Y;
        user_choice >> str_name >> UR;
        user_choice >> str_name >> ConvgCrit;
        user_choice >> str_name >> ConvgCrit_S_S;
        user_choice >> str_name >> Max_Relax_Number;
        user_choice >> str_name >> Iprint;
        user_choice >> str_name >> support;

        user_choice.close();

        //int Nx1 = Nx + 1;
        //int Nx2 = Nx + 2;
        //int Nx3 = Nx + 3;
        //int Nx4 = Nx + 4;
        //int Ny1 = Ny + 1;
        //int Ny2 = Ny + 2;

        // std::cout << "Nx1=" << Nx1 << "\n";
    }
    catch(const std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << '\n';
        exit(1);
    }
}
*/