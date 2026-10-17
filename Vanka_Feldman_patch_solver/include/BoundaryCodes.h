#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

// One-byte integer codes, not floating point. ASCII values retain the existing
// switch labels in VOF. The input-file spelling/letters are unchanged.
enum class PointBoundary : std::uint8_t {
    Interior = '0',
    N = 'N', S = 'S', W = 'W', E = 'E', F = 'F', B = 'B', // N/S/W/E/top/bottom
    D = 'D', U = 'U', T = 'T', Y = 'Y', // W-S, W-N, E-S, E-N edges
    A = 'A', C = 'C', G = 'G', H = 'H', // W-bottom, W-top, E-bottom, E-top
    R = 'R', V = 'V', Z = 'Z', X = 'X', // S-bottom, S-top, N-bottom, N-top
    I = 'I', J = 'J', K = 'K', L = 'L', // W-S-bottom, W-N-bottom, W-S-top, W-N-top
    M = 'M', O = 'O', P = 'P', Q = 'Q'  // E-S-bottom, E-N-bottom, E-S-top, E-N-top
};

constexpr char boundaryLetter(PointBoundary code) noexcept {
    return static_cast<char>(code);
}

inline PointBoundary pointBoundaryFromText(std::string_view text) {
    if (text.size() == 1 && (text[0] == '0' || (text[0] >= 'A' && text[0] <= 'Z'))) {
        return static_cast<PointBoundary>(text[0]);
    }
    throw std::invalid_argument("Unknown point boundary: " + std::string(text));
}

// Translation from the legacy input-file names to compact condition codes.
enum class BoundaryKind : std::uint8_t {
    FixedValue = 0,         // fixed_value
    Slip = 1,               // slip
    Convective = 2,         // convective_boundary
    Symmetry = 3,           // symetry (legacy spelling kept in input files)
    DirichletPoint = 4,     // dirichlet_point
    DirichletRow = 5        // dirichlet_row
};

inline BoundaryKind boundaryKindFromText(std::string_view text) {
    if (text == "fixed_value") return BoundaryKind::FixedValue;
    if (text == "slip") return BoundaryKind::Slip;
    if (text == "convective_boundary") return BoundaryKind::Convective;
    if (text == "symetry") return BoundaryKind::Symmetry;
    if (text == "dirichlet_point") return BoundaryKind::DirichletPoint;
    if (text == "dirichlet_row") return BoundaryKind::DirichletRow;
    throw std::invalid_argument("Unknown boundary condition: " + std::string(text));
}

constexpr const char* boundaryKindName(BoundaryKind kind) noexcept {
    switch (kind) {
    case BoundaryKind::FixedValue: return "fixed_value";
    case BoundaryKind::Slip: return "slip";
    case BoundaryKind::Convective: return "convective_boundary";
    case BoundaryKind::Symmetry: return "symetry";
    case BoundaryKind::DirichletPoint: return "dirichlet_point";
    case BoundaryKind::DirichletRow: return "dirichlet_row";
    }
    return "unknown";
}

static_assert(sizeof(PointBoundary) == 1 && sizeof(BoundaryKind) == 1);
