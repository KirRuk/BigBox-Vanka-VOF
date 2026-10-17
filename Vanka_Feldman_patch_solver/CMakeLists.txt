cmake_minimum_required (VERSION 3.20)          # 3.8 is 8-years old – bump a bit
project (Vanka_Feldman_patch_solver LANGUAGES C CXX)

# -----------------------------------------------------------------------------
# 1.  Source tree
# -----------------------------------------------------------------------------
set(SRC_DIR     ${CMAKE_SOURCE_DIR}/Vanka_Feldman_patch_solver/src)
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/Vanka_Feldman_patch_solver/include)

file (GLOB_RECURSE  SOURCE_FILES  CONFIGURE_DEPENDS  "${SRC_DIR}/*.cpp")
file (GLOB_RECURSE  HEADER_FILES  CONFIGURE_DEPENDS  "${INCLUDE_DIR}/*.h")

# MUMPS is intentionally retained in the source tree for possible future use,
# but the current executable is a CG-only build.
list (FILTER SOURCE_FILES EXCLUDE REGEX "Mumps_Solver\\.cpp$")

add_executable (Vanka_Feldman_patch_solver  ${SOURCE_FILES})

target_include_directories (Vanka_Feldman_patch_solver PRIVATE
    ${INCLUDE_DIR} )

# -----------------------------------------------------------------------------
# 2.  Compiler options / language level
# -----------------------------------------------------------------------------
set_target_properties (Vanka_Feldman_patch_solver PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON)

# -----------------------------------------------------------------------------
# 3.  Third-party packages  ---------------------------------------------------
# -----------------------------------------------------------------------------
find_package (OpenMP REQUIRED COMPONENTS CXX) # brings OpenMP::OpenMP_CXX
find_package (MKL REQUIRED)               # brings MKL::MKL (threaded / ILP64)
# If you ever switch to an MPI build, just add:
# find_package (MPI REQUIRED)

target_link_libraries (Vanka_Feldman_patch_solver PRIVATE
    OpenMP::OpenMP_CXX
    MKL::MKL)

# -----------------------------------------------------------------------------
# 4.  Optional TecIO SZL output  -----------------------------------------------
# -----------------------------------------------------------------------------
option (ENABLE_TECIO_OUTPUT "Enable Tecplot SZL binary snapshots when TecIO is available" ON)
set (TECIO_ROOT "" CACHE PATH "Tecplot/TecIO installation root")

set (TECIO_OUTPUT_AVAILABLE OFF)
if (ENABLE_TECIO_OUTPUT)
    set (_tecio_hints)
    if (TECIO_ROOT)
        list (APPEND _tecio_hints "${TECIO_ROOT}")
    endif()
    if (DEFINED ENV{TECIO_ROOT})
        list (APPEND _tecio_hints "$ENV{TECIO_ROOT}")
    endif()
    if (WIN32)
        file (GLOB _tecplot_install_roots LIST_DIRECTORIES true
            "C:/Program Files/Tecplot/Tecplot 360 EX *")
        list (SORT _tecplot_install_roots COMPARE NATURAL ORDER DESCENDING)
        list (APPEND _tecio_hints ${_tecplot_install_roots})
    endif()

    find_path (TECIO_INCLUDE_DIR
        NAMES TECIO.h
        HINTS ${_tecio_hints}
        PATH_SUFFIXES include)
    find_library (TECIO_LIBRARY
        NAMES tecio
        HINTS ${_tecio_hints}
        PATH_SUFFIXES lib bin Frameworks)

    if (TECIO_INCLUDE_DIR AND TECIO_LIBRARY)
        if (WIN32)
            find_file (TECIO_RUNTIME_DLL
                NAMES tecio.dll
                HINTS ${_tecio_hints}
                PATH_SUFFIXES bin)
            if (NOT TECIO_RUNTIME_DLL)
                message (FATAL_ERROR
                    "TecIO was found, but tecio.dll was not. "
                    "Set TECIO_ROOT to a complete TecIO installation.")
            endif()
        endif()

        add_library (TecIO::TecIO UNKNOWN IMPORTED)
        set_target_properties (TecIO::TecIO PROPERTIES
            IMPORTED_LOCATION "${TECIO_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${TECIO_INCLUDE_DIR}")
        target_link_libraries (Vanka_Feldman_patch_solver PRIVATE TecIO::TecIO)
        target_compile_definitions (Vanka_Feldman_patch_solver PRIVATE VANKA_HAVE_TECIO=1)
        set (TECIO_OUTPUT_AVAILABLE ON)

        if (WIN32)
            add_custom_command (TARGET Vanka_Feldman_patch_solver POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${TECIO_RUNTIME_DLL}"
                    "$<TARGET_FILE_DIR:Vanka_Feldman_patch_solver>")
        elseif (UNIX)
            find_package (Threads REQUIRED)
            target_link_libraries (Vanka_Feldman_patch_solver PRIVATE
                Threads::Threads
                ${CMAKE_DL_LIBS})
            if (NOT APPLE)
                target_link_libraries (Vanka_Feldman_patch_solver PRIVATE m)
            endif()
        endif()
    else()
        message (FATAL_ERROR
            "ENABLE_TECIO_OUTPUT is ON, but TecIO was not found. "
            "Set TECIO_ROOT to a TecIO installation, or explicitly configure "
            "with -DENABLE_TECIO_OUTPUT=OFF.")
    endif()
endif()

# -----------------------------------------------------------------------------
# 5.  Pre-processor definitions  ----------------------------------------------
# -----------------------------------------------------------------------------
target_compile_definitions (Vanka_Feldman_patch_solver PRIVATE
    RESOURCE_DIR=\"${CMAKE_SOURCE_DIR}/resources/\"
)

# -----------------------------------------------------------------------------
# 6.  Diagnostics
# -----------------------------------------------------------------------------
message (STATUS "\nBuild type             : ${CMAKE_BUILD_TYPE}")
message (STATUS "Pressure solver        : CG (MUMPS disabled)")
message (STATUS "TecIO SZL output       : ${TECIO_OUTPUT_AVAILABLE}\n")
