// ===========================================================================
// qrcp_sketch.hpp -- the Gaussian sketch matrix G shared by the HQRRP
//                    implementations (HQRRP_MPI, TileHQRRP-MPI-OpenMP and
//                    TileHQRRP-MPI-OpenMP-UNB).
//
// HQRRP draws a (b+p) x m Gaussian G and works with the sketch Y = G*A.  Which
// columns the algorithm picks depends on G, so two implementations that draw
// different G matrices select different pivots and their pivot-quality numbers
// are not comparable -- the difference would be partly luck of the draw.  This
// header therefore holds the one definition all of them use.
//
// G is a pure function of (seed, i, j): every process can compute exactly the
// part it stores, with no communication and no draw-and-discard, and the values
// do not depend on the process grid or on the storage layout.
//
//   G(i, j)   with i in [0, b+p), j in [0, m)   ==   normal_at(seed, i+1, j+1)
//
// The 1-based arguments are historical: HQRRP_MPI stores G transposed in a
// ScaLAPACK array and passes ScaLAPACK's 1-based global indices, so keeping the
// convention leaves that implementation's matrix bit-identical.
// ===========================================================================
#pragma once

#include <cmath>
#include <cstdint>

namespace qrcp {
namespace sketch {

// Default seed.  Shared, so that changing it changes every implementation.
constexpr unsigned kSeed = 12u;

// splitmix64: a well-mixed 64-bit finalizer, used as a counter-based RNG.
inline std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// Uniform on (0, 1).
inline double uniform_at(std::uint64_t key) {
    const std::uint64_t r = splitmix64(key);
    // 53 significant bits, shifted off zero.
    return (static_cast<double>(r >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}

// Standard normal at the (1-based) coordinates (i, j).
inline double normal_at(std::uint64_t seed, std::uint64_t i, std::uint64_t j) {
    const std::uint64_t key =
        splitmix64(seed ^ splitmix64(i * 0xD1342543DE82EF95ull + j));
    const double u1 = uniform_at(key);
    const double u2 = uniform_at(key + 0x9E3779B97F4A7C15ull);
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586477 * u2);
}

// Entry of the sketch matrix at the 0-based (row, column) of G itself, i.e.
// row in [0, b+p) and column in [0, m).
inline double g_entry(int row, int col, unsigned seed = kSeed) {
    return normal_at(seed, static_cast<std::uint64_t>(row) + 1,
                     static_cast<std::uint64_t>(col) + 1);
}

}  // namespace sketch
}  // namespace qrcp
