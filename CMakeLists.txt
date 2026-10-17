# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.21)

# CMP0141 must be selected before project() initializes the toolchain.
if (POLICY CMP0141)
    cmake_policy(SET CMP0141 NEW)
endif()

project ("Vanka_Feldman_patch_solver" LANGUAGES C CXX)

# Include sub-projects.
add_subdirectory ("Vanka_Feldman_patch_solver")
