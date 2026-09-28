// ===========================================================================
// qrcp_bench.hpp -- command line options and test matrices shared by the four
//                   distributed QRCP benchmarks
//                   (SCALAPACK_DGEQPF, SCALAPACK_DGEQP3, HQRRP_MPI,
//                    TileHQRRP-MPI-OpenMP).
//
// Why this file is shared rather than copied
// ------------------------------------------
// The point of the test matrices below is to compare PIVOT QUALITY across the
// four implementations.  That comparison is only meaningful if every
// implementation factorizes a bit-identical matrix, so the generators must
// live in exactly one place.  Every generator is a pure function of the GLOBAL
// indices, so the matrix never depends on the process grid or the block size.
//
// The "legacy" matrix is the exception: it means "whatever generator this
// implementation has always used", and it is NOT the same matrix in the four
// codes -- TileHQRRP's, for instance, grades the column norms over 10^0..10^-3
// on purpose, while the others are flat uniform.  It is kept as the default so
// that existing timing measurements stay reproducible, and it must never be
// used for a cross-implementation quality comparison.
//
// Header-only, C++17.  Depends on MPI only for the checksum reduction; every
// consumer of this header is an MPI benchmark.
// ===========================================================================
#pragma once

#include "qrcp_int.hpp"

#include <mpi.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#ifndef QRCP_NO_LAPACK_DECLS
extern "C" {
// These are the SEQUENTIAL BLAS/LAPACK, but their integer width still follows
// the interface layer we link (see qrcp_int.hpp): libmkl_intel_ilp64 widens
// them exactly as it widens the ScaLAPACK entry points.
void dgeqrf_(const qrcp::blas_int* m, const qrcp::blas_int* n, double* a,
             const qrcp::blas_int* lda, double* tau, double* work,
             const qrcp::blas_int* lwork, qrcp::blas_int* info);
void dorgqr_(const qrcp::blas_int* m, const qrcp::blas_int* n,
             const qrcp::blas_int* k, double* a, const qrcp::blas_int* lda,
             const double* tau, double* work, const qrcp::blas_int* lwork,
             qrcp::blas_int* info);
void dgemm_(const char* transa, const char* transb, const qrcp::blas_int* m,
            const qrcp::blas_int* n, const qrcp::blas_int* k,
            const double* alpha, const double* a, const qrcp::blas_int* lda,
            const double* b, const qrcp::blas_int* ldb, const double* beta,
            double* c, const qrcp::blas_int* ldc);
}
#endif

namespace qrcp {

// ---------------------------------------------------------------------------
// Test matrices.  Matrix1..Matrix4 are the four matrices of section 4.2 of
//   P.-G. Martinsson, G. Quintana-Orti, N. Heavner, R. van de Geijn,
//   "Householder QR Factorization With Randomization for Column Pivoting",
//   SIAM J. Sci. Comput. 39(2), C96-C115, 2017.
// ---------------------------------------------------------------------------
enum class Matrix {
    Legacy,     // per-implementation historical generator (default)
    Random,     // shared uniform [-1, 1)
    FastDecay,  // paper Matrix 1: singular values beta^((j-1)/(n-1)), beta=1e-5
    SDecay,     // paper Matrix 2: flat near 1, sharp drop, flat at 1e-6
    Bie,        // paper Matrix 3: single layer Laplace BIE on a smooth curve
    Kahan       // paper Matrix 4: Kahan counterexample A = S*K
};

namespace detail {

struct MatrixEntry {
    Matrix      kind;
    const char* canonical;
    const char* aliases[3];   // nullptr-terminated
    bool        shared;       // identical in all four implementations?
    bool        implemented;  // is element() able to produce it yet?
    const char* description;
};

inline const MatrixEntry* table(std::size_t& count) {
    static const MatrixEntry t[] = {
        { Matrix::Legacy, "legacy", { nullptr, nullptr, nullptr }, false, true,
          "this implementation's historical generator (default; NOT shared)" },
        { Matrix::Random, "random", { "rand", nullptr, nullptr }, true, true,
          "shared uniform [-1,1)" },
        { Matrix::FastDecay, "fast-decay", { "1", "matrix1", nullptr }, true, true,
          "paper Matrix 1: A = U D V*, singular values 1 .. 1e-5" },
        { Matrix::SDecay, "s-decay", { "2", "matrix2", nullptr }, true, true,
          "paper Matrix 2: A = U D V*, S shaped decay 1 .. 1e-6" },
        { Matrix::Bie, "bie", { "3", "matrix3", nullptr }, true, true,
          "paper Matrix 3: Laplace single layer BIE, ill conditioned (square n only)" },
        { Matrix::Kahan, "kahan", { "4", "matrix4", nullptr }, true, true,
          "paper Matrix 4: Kahan counterexample, every column of norm 1" },
    };
    count = sizeof(t) / sizeof(t[0]);
    return t;
}

inline const MatrixEntry* find(Matrix k) {
    std::size_t n = 0;
    const MatrixEntry* t = table(n);
    for (std::size_t i = 0; i < n; ++i)
        if (t[i].kind == k) return &t[i];
    return &t[0];
}

// SplitMix64-based uniform value in [-1, 1) from the global indices.
inline double uniform(std::int64_t i, std::int64_t j) {
    auto splitmix64 = [](std::uint64_t x) -> std::uint64_t {
        x += 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        x = x ^ (x >> 31);
        return x;
    };
    std::uint64_t key = 0x9e3779b97f4a7c15ULL + static_cast<std::uint64_t>(i);
    key ^= 0x9e3779b97f4a7c15ULL + static_cast<std::uint64_t>(j) + (key << 6) + (key >> 2);
    key = splitmix64(key);
    const std::uint64_t mant = key & ((1ULL << 52) - 1);
    return 2.0 * (static_cast<double>(mant) / static_cast<double>(1ULL << 52)) - 1.0;
}

// Deterministic uniform in (0, 1) from three mixed keys.
inline double u01(std::uint64_t a, std::uint64_t b, std::uint64_t stream) {
    auto mix = [](std::uint64_t x) -> std::uint64_t {
        x += 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        return x ^ (x >> 31);
    };
    std::uint64_t k = mix(a + 0x1000193ULL * stream);
    k = mix(k ^ (b + 0x9e3779b97f4a7c15ULL));
    const std::uint64_t mant = (k & ((1ULL << 52) - 1)) + 1;   // never 0
    return static_cast<double>(mant) / static_cast<double>(1ULL << 52);
}

// x^k by binary exponentiation.  std::pow is not bit-identical across libm
// implementations, and the Kahan matrix must be the same in all four
// benchmarks; this depends only on IEEE double multiplication.
inline double ipow(double x, int k) {
    double r = 1.0;
    while (k > 0) {
        if (k & 1) r *= x;
        x *= x;
        k >>= 1;
    }
    return r;
}

// Deterministic standard normal (Box-Muller).
inline double gaussian(std::uint64_t a, std::uint64_t b, std::uint64_t stream) {
    const double u1 = u01(a, b, 2 * stream);
    const double u2 = u01(a, b, 2 * stream + 1);
    return std::sqrt(-2.0 * std::log(u1)) *
           std::cos(6.283185307179586476925286766559 * u2);
}

}  // namespace detail

// Can this matrix be evaluated one entry at a time from the global indices?
// The paper's Matrix 1 and 2 are defined as A = U D V* with RANDOM orthonormal
// U and V, so they cannot; they are built once and replicated instead.
inline bool matrix_is_elementwise(Matrix k) {
    return k != Matrix::FastDecay && k != Matrix::SDecay && k != Matrix::Bie;
}

// Is the exact spectrum known in closed form?  Only the two U D V* matrices
// are built from a prescribed D; the BIE matrix is assembled from a kernel and
// its singular values would have to be computed numerically.
inline bool matrix_has_known_spectrum(Matrix k) {
    return k == Matrix::FastDecay || k == Matrix::SDecay;
}

// Singular values of the U D V* matrices, d_1 >= ... >= d_n.  Exposed so that a
// plot of |R(k,k)| can be compared against the exact spectrum (the black
// reference line of the paper's Figure 10).
inline double singular_value(Matrix kind, int j, int n) {
    if (n <= 1) return 1.0;
    const double t = static_cast<double>(j) / static_cast<double>(n - 1);

    if (kind == Matrix::FastDecay) {
        // Paper section 4.2, Matrix 1: d_j = beta^((j-1)/(n-1)), beta = 1e-5.
        return std::pow(1e-5, t);
    }

    if (kind == Matrix::SDecay) {
        // Paper section 4.2, Matrix 2, is described only in words -- the
        // diagonal "hovers around 1, then decays rapidly, then levels out at
        // 1e-6" -- and shown in Figures 7 and 10.  No formula is given, so this
        // is our reading of Figure 10 (n = 4000):
        //
        //     flat near 1 up to  k ~ 1000
        //     steep drop over    k ~ 1000 .. 1700
        //     flat at 1e-6 from  k ~ 1700
        //
        // A logistic in log10 space reproduces that shape with two parameters:
        // the centre of the drop and its width.  For a logistic the 10%-90%
        // transition spans 2*ln(9)*W = 4.394*W, so W = 0.04 puts it at
        // 0.176*(n-1) ~ 700 columns wide, centred at 0.3125*(n-1) ~ 1250.
        const double centre = 0.3125;   // fraction of (n-1) where sigma = 1e-3
        const double width  = 0.04;     // fraction of (n-1), logistic scale
        const double x = (t - centre) / width;
        const double step = 1.0 / (1.0 + std::exp(-x));   // 0 -> 1
        return std::pow(10.0, -6.0 * step);
    }

    return 0.0;
}

inline const char* matrix_name(Matrix k)        { return detail::find(k)->canonical; }
inline const char* matrix_description(Matrix k) { return detail::find(k)->description; }
inline bool        matrix_is_shared(Matrix k)   { return detail::find(k)->shared; }
inline bool        matrix_implemented(Matrix k) { return detail::find(k)->implemented; }

// Resolve a --matrix= value.  Accepts the canonical name or any alias.
inline bool parse_matrix(const std::string& s, Matrix& out) {
    std::size_t n = 0;
    const detail::MatrixEntry* t = detail::table(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (s == t[i].canonical) { out = t[i].kind; return true; }
        for (int a = 0; a < 3 && t[i].aliases[a]; ++a)
            if (s == t[i].aliases[a]) { out = t[i].kind; return true; }
    }
    return false;
}

// Multi-line list of the accepted values, for usage text and error messages.
inline std::string matrix_help() {
    std::size_t n = 0;
    const detail::MatrixEntry* t = detail::table(n);
    std::string s;
    for (std::size_t i = 0; i < n; ++i) {
        std::string line = "      ";
        line += t[i].canonical;
        std::string al;
        for (int a = 0; a < 3 && t[i].aliases[a]; ++a) {
            al += al.empty() ? " (= " : ", ";
            al += t[i].aliases[a];
        }
        if (!al.empty()) al += ")";
        line += al;
        while (line.size() < 34) line += ' ';
        line += t[i].description;
        if (!t[i].implemented) line += "  [not implemented yet]";
        s += line + "\n";
    }
    return s;
}

// ---------------------------------------------------------------------------
// Shared options
// ---------------------------------------------------------------------------
struct Options {
    bool   res      = false;           // --res              compute the residual
    bool   dry_run  = false;           // --dry-run          report the configuration and stop
    bool   checksum = false;           // --matrix-checksum  hash the matrix and stop
    Matrix matrix   = Matrix::Legacy;  // --matrix=<name>
    std::string rdiag;                 // --rdiag=<path>     write |diag(R)| there
    std::string ek;                    // --ek=<path>        write e_k there
};

inline const char* options_help() {
    return
        "  --res            compute the ABSOLUTE residual ||A*P - Q*R||_F (not divided by\n"
"                   ||A||_F) and append it to the output\n"
        "  --matrix=<name>  test matrix (default: legacy)\n"
        "  --dry-run        print the resolved configuration and exit without factorizing\n"
        "  --matrix-checksum  generate the matrix, print a checksum of it and exit.  All four\n"
        "                   benchmarks must print the same value for the same shared matrix\n"
        "  --rdiag=<path>   after factorizing, append |R(k,k)| for every k to this CSV.  This\n"
        "                   is the quantity plotted in Figure 10 of the HQRRP paper\n"
        "  --ek=<path>      append the rank-k truncation error e_k (Frobenius) to this CSV.\n"
        "                   This is equation (4.2) of the paper, Figures 6-9\n";
}

// Parse the shared flags.  Tokens that do not start with \"--\" are appended to
// `positional` untouched (argv[0] is skipped), so a driver may combine this
// with positional arguments or with its own -key value scanning.
//
// `extra_flags` and `extra_valued` name driver-specific options that this
// parser must skip rather than reject; `extra_valued` options also consume the
// following token.  The driver reads those from argv itself.
//
// Returns false and fills `err` on an unknown --option or an unknown matrix.
inline bool parse(int argc, char** argv, Options& opt,
                  std::vector<const char*>& positional, std::string& err,
                  const std::vector<std::string>& extra_flags = {},
                  const std::vector<std::string>& extra_valued = {}) {
    auto contains = [](const std::vector<std::string>& v, const std::string& s) {
        for (const std::string& e : v) if (e == s) return true;
        return false;
    };

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a.rfind("--", 0) != 0) { positional.push_back(argv[i]); continue; }

        if (a == "--res")             { opt.res = true; continue; }
        if (a == "--dry-run")         { opt.dry_run = true; continue; }
        if (a == "--matrix-checksum") { opt.checksum = true; continue; }

        std::string value;
        bool have_value = false;
        if (a.rfind("--ek=", 0) == 0) {
            opt.ek = a.substr(std::strlen("--ek="));
            if (opt.ek.empty()) { err = "--ek needs a path"; return false; }
            continue;
        }
        if (a == "--ek") {
            if (i + 1 >= argc) { err = "--ek needs a path"; return false; }
            opt.ek = argv[++i];
            continue;
        }

        if (a.rfind("--rdiag=", 0) == 0) {
            opt.rdiag = a.substr(std::strlen("--rdiag="));
            if (opt.rdiag.empty()) { err = "--rdiag needs a path"; return false; }
            continue;
        }
        if (a == "--rdiag") {
            if (i + 1 >= argc) { err = "--rdiag needs a path"; return false; }
            opt.rdiag = argv[++i];
            continue;
        }

        if (a.rfind("--matrix=", 0) == 0) {
            value = a.substr(std::strlen("--matrix="));
            have_value = true;
        } else if (a == "--matrix") {
            if (i + 1 >= argc) { err = "--matrix needs a value"; return false; }
            value = argv[++i];
            have_value = true;
        }
        if (have_value) {
            if (!parse_matrix(value, opt.matrix)) {
                err = "unknown --matrix value '" + value + "'. Accepted values:\n" + matrix_help();
                return false;
            }
            continue;
        }

        if (contains(extra_flags, a)) continue;
        if (contains(extra_valued, a)) { if (i + 1 < argc) ++i; continue; }

        err = "unknown option '" + a + "'";
        return false;
    }
    return true;
}

// One line per setting, for --dry-run.  `extra` is appended verbatim.
inline std::string describe(const char* impl, const Options& opt,
                            const std::string& extra) {
    std::string s;
    s += "[dry-run] implementation : ";  s += impl;        s += "\n";
    s += extra;
    s += "[dry-run] matrix         : "; s += matrix_name(opt.matrix);
    s += matrix_is_shared(opt.matrix) ? "  (shared across implementations)"
                                      : "  (per-implementation, NOT shared)";
    s += "\n";
    s += "[dry-run]                  "; s += matrix_description(opt.matrix); s += "\n";
    s += "[dry-run] residual       : "; s += opt.res ? "on (--res)" : "off"; s += "\n";
    if (opt.checksum)
        s += "[dry-run] checksum mode  : on (--matrix-checksum)\n";
    if (!opt.rdiag.empty())
        s += "[dry-run] |diag(R)| CSV  : " + opt.rdiag + "\n";
    if (!opt.ek.empty())
        s += "[dry-run] e_k CSV        : " + opt.ek + "\n";
    if (!matrix_implemented(opt.matrix))
        s += "[dry-run] NOTE           : this matrix is not implemented yet; a real run would abort\n";
    return s;
}

// ---------------------------------------------------------------------------
// Shared data layout and fill.
//
// All four benchmarks distribute A the same way -- 2D block-cyclic with block
// (mb, nb) over an nprow x npcol process grid sourced at (0,0) -- and differ
// only in how the local pieces are stored:
//
//   SCALAPACK_DGEQPF / SCALAPACK_DGEQP3 / HQRRP_MPI
//       one local array, column-major, single leading dimension  -> fill_local
//   TileHQRRP-MPI-OpenMP
//       every ts x ts tile stored separately with its own ld     -> fill_tile
//
// Both entry points route through the same Entries object, so the four codes
// generate a bit-identical matrix for a given shared kind.  Verify it with
// --matrix-checksum, which must print the same value from all four.
// ---------------------------------------------------------------------------

// Local row/column count of a block-cyclic dimension (ScaLAPACK NUMROC).
inline int numroc(int n, int nb, int iproc, int nprocs) {
    const int nblocks = n / nb;
    int nloc = (nblocks / nprocs) * nb;
    const int extra = nblocks % nprocs;
    if (iproc < extra)       nloc += nb;
    else if (iproc == extra) nloc += n % nb;
    return nloc;
}

// Global 0-based index of a local 0-based index.
inline int local_to_global(int loc, int nb, int iproc, int nprocs) {
    return (loc / nb * nprocs + iproc) * nb + loc % nb;
}

struct BlockCyclic {
    int m = 0, n = 0;            // global dimensions
    int mb = 0, nb = 0;          // block sizes
    int nprow = 1, npcol = 1;    // process grid
    int myrow = 0, mycol = 0;    // this process
};

// One source of matrix entries.  Matrix::Legacy delegates to the driver's own
// historical generator (which is deliberately NOT shared); every other kind
// comes from element() below.
struct Entries {
    Matrix kind = Matrix::Legacy;
    int m = 0, n = 0;
    double (*legacy)(int, int) = nullptr;
    // Non-elementwise kinds only: the whole m x n matrix, column-major,
    // replicated on every process by build_global().
    const double* global = nullptr;

    double operator()(int i, int j) const;
};

// Fill a driver's local array: column-major, leading dimension `lld`, entry
// (lr, lc) being global (local_to_global(lr,...), local_to_global(lc,...)).
inline void fill_local(const BlockCyclic& L, const Entries& e,
                       double* a, int lld) {
    const int mloc = numroc(L.m, L.mb, L.myrow, L.nprow);
    const int nloc = numroc(L.n, L.nb, L.mycol, L.npcol);
    for (int lc = 0; lc < nloc; ++lc) {
        const int gj = local_to_global(lc, L.nb, L.mycol, L.npcol);
        double* col = a + static_cast<std::size_t>(lc) * lld;
        for (int lr = 0; lr < mloc; ++lr)
            col[lr] = e(local_to_global(lr, L.mb, L.myrow, L.nprow), gj);
    }
}

// Fill one separately stored tile whose top-left global entry is (i0, j0).
// `rows` x `cols` is the tile's real extent; the allocated tile is
// `ld` x `cols_alloc` and everything outside the real extent is zeroed, so that
// communication and column swaps that move whole tiles never carry
// uninitialised values.
inline void fill_tile(const Entries& e, int i0, int j0, int rows, int cols,
                      double* t, int ld, int cols_alloc) {
    for (std::size_t k = 0; k < static_cast<std::size_t>(ld) * cols_alloc; ++k)
        t[k] = 0.0;
    for (int c = 0; c < cols; ++c) {
        double* col = t + static_cast<std::size_t>(c) * ld;
        for (int r = 0; r < rows; ++r) col[r] = e(i0 + r, j0 + c);
    }
}

// ---------------------------------------------------------------------------
// Checksum of the generated matrix.
//
// Accumulated with XOR so that it does not depend on the order in which the
// processes contribute, and mixed with the global indices so that a value
// landing in the wrong place is caught.  Combine across ranks with a bitwise-OR
// reduction of the XOR (MPI_BXOR).
// ---------------------------------------------------------------------------
inline std::uint64_t checksum_entry(int i, int j, double v) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    std::uint64_t h = 0xcbf29ce484222325ULL;
    const std::uint64_t pos = (static_cast<std::uint64_t>(i) << 32) ^
                              static_cast<std::uint64_t>(j);
    for (std::uint64_t word : { bits, pos }) {
        for (int b = 0; b < 8; ++b) {
            h ^= (word >> (8 * b)) & 0xffULL;
            h *= 0x100000001b3ULL;
        }
    }
    return h;
}

inline std::uint64_t checksum_local(const BlockCyclic& L, const Entries& e) {
    std::uint64_t h = 0;
    const int mloc = numroc(L.m, L.mb, L.myrow, L.nprow);
    const int nloc = numroc(L.n, L.nb, L.mycol, L.npcol);
    for (int lc = 0; lc < nloc; ++lc) {
        const int gj = local_to_global(lc, L.nb, L.mycol, L.npcol);
        for (int lr = 0; lr < mloc; ++lr) {
            const int gi = local_to_global(lr, L.mb, L.myrow, L.nprow);
            h ^= checksum_entry(gi, gj, e(gi, gj));
        }
    }
    return h;
}

// Combine the per-process checksums and report from rank 0.  The value covers
// every global entry exactly once, so it must be identical across the four
// implementations AND across process grids for a given shared matrix.
inline std::uint64_t checksum_report(const char* impl, const BlockCyclic& L,
                                     std::uint64_t local, MPI_Comm comm) {
    unsigned long long h = local, g = 0;
    MPI_Allreduce(&h, &g, 1, MPI_UNSIGNED_LONG_LONG, MPI_BXOR, comm);
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0)
        std::printf("matrix-checksum %016llx  %-24s m=%d n=%d block=%dx%d grid=%dx%d\n",
                    g, impl, L.m, L.n, L.mb, L.nb, L.nprow, L.npcol);
    return g;
}

// ---------------------------------------------------------------------------
// |R(k,k)| output.
//
// This is the quantity of Figure 10 of the paper: for a column pivoted QR the
// diagonal of R decays monotonically and tracks the singular values, and how
// closely it does so is the measure of pivot quality.
//
// The file is a tidy CSV, one row per k, with the run's metadata repeated on
// every row so that files from several runs can simply be concatenated before
// plotting.  Rank 0 writes it; the file is appended to if it already exists,
// and the header is written only when the file is new.
// ---------------------------------------------------------------------------
inline void write_rdiag_values(const std::string& path, const char* impl,
                               Matrix kind, const BlockCyclic& L,
                               std::vector<double> local, MPI_Comm comm) {
    const int mn = std::min(L.m, L.n);
    std::vector<double> all(static_cast<std::size_t>(std::max(1, mn)), 0.0);
    local.resize(all.size(), 0.0);
    // Every diagonal entry is owned by exactly one process, so a sum collects
    // them without any process needing to know who owns what.
    MPI_Reduce(local.data(), all.data(), mn, MPI_DOUBLE, MPI_SUM, 0, comm);

    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    if (rank != 0) return;

    bool exists = false;
    { std::ifstream probe(path); exists = probe.good() && probe.peek() != std::ifstream::traits_type::eof(); }
    std::ofstream f(path, std::ios::app);
    if (!f) {
        std::fprintf(stderr, "error: cannot write %s\n", path.c_str());
        return;
    }
    if (!exists)
        f << "implementation,matrix,m,n,mb,nb,P,Q,k,abs_rkk,sigma\n";
    f.setf(std::ios::scientific);
    f.precision(12);
    const bool have_sigma = matrix_has_known_spectrum(kind);
    for (int k = 0; k < mn; ++k) {
        f << impl << ',' << matrix_name(kind) << ',' << L.m << ',' << L.n << ','
          << L.mb << ',' << L.nb << ',' << L.nprow << ',' << L.npcol << ','
          << k << ',' << std::fabs(all[static_cast<std::size_t>(k)]) << ',';
        if (have_sigma) f << singular_value(kind, k, L.n);
        else            f << "";
        f << '\n';
    }
    std::printf("rdiag: wrote %d rows to %s\n", mn, path.c_str());
}

// Collect |R(k,k)| straight from a block-cyclic local array (the three
// ScaLAPACK-layout drivers) and write the CSV.
inline void write_rdiag(const std::string& path, const char* impl, Matrix kind,
                        const BlockCyclic& L, const double* a, int lld,
                        MPI_Comm comm) {
    const int mn = std::min(L.m, L.n);
    std::vector<double> local(static_cast<std::size_t>(std::max(1, mn)), 0.0);
    const int mloc = numroc(L.m, L.mb, L.myrow, L.nprow);
    const int nloc = numroc(L.n, L.nb, L.mycol, L.npcol);
    for (int lc = 0; lc < nloc; ++lc) {
        const int gj = local_to_global(lc, L.nb, L.mycol, L.npcol);
        if (gj >= mn) continue;
        for (int lr = 0; lr < mloc; ++lr) {
            if (local_to_global(lr, L.mb, L.myrow, L.nprow) != gj) continue;
            local[static_cast<std::size_t>(gj)] =
                a[lr + static_cast<std::size_t>(lc) * lld];
            break;
        }
    }
    write_rdiag_values(path, impl, kind, L, std::move(local), comm);
}

// ---------------------------------------------------------------------------
// Rank-k truncation error e_k.
//
// Equation (4.2) of the paper:
//
//     e_k = || A P - Q(:,1:k) R(1:k,:) || = || R((k+1):n, (k+1):n) ||
//
// which is what Figures 6-9 plot.  Only the Frobenius norm is computed here,
// and it is exact rather than estimated: R is upper triangular, so the trailing
// block for k differs from the one for k+1 by row k alone, giving
//
//     e_n = 0,     e_k^2 = e_{k+1}^2 + sum_{j >= k} R(k,j)^2.
//
// So all n+1 values come from one length-n reduction of the squared row norms
// of R.  (The operator norm of Figures 6-9's left panels would need a singular
// value of every trailing block, i.e. an SVD or a power iteration per k; it is
// not computed here.)
//
// The reference line is the Eckart-Young bound, exact for the matrices whose
// spectrum we prescribed:  e_k >= ( sum_{j>k} sigma_j^2 )^(1/2).
// ---------------------------------------------------------------------------

// Squared norms of the rows of R, restricted to the upper triangle, summed over
// the columns this process owns.
inline std::vector<double> ek_rowsums_local(const BlockCyclic& L,
                                            const double* a, int lld) {
    const int mn = std::min(L.m, L.n);
    std::vector<double> rs(static_cast<std::size_t>(std::max(1, mn)), 0.0);
    const int mloc = numroc(L.m, L.mb, L.myrow, L.nprow);
    const int nloc = numroc(L.n, L.nb, L.mycol, L.npcol);
    for (int lc = 0; lc < nloc; ++lc) {
        const int gj = local_to_global(lc, L.nb, L.mycol, L.npcol);
        const double* col = a + static_cast<std::size_t>(lc) * lld;
        for (int lr = 0; lr < mloc; ++lr) {
            const int gi = local_to_global(lr, L.mb, L.myrow, L.nprow);
            if (gi > gj || gi >= mn) continue;   // strict lower part holds V, not R
            rs[static_cast<std::size_t>(gi)] += col[lr] * col[lr];
        }
    }
    return rs;
}

inline void write_ek_values(const std::string& path, const char* impl,
                            Matrix kind, const BlockCyclic& L,
                            std::vector<double> rowsums, MPI_Comm comm) {
    const int mn = std::min(L.m, L.n);
    std::vector<double> all(static_cast<std::size_t>(std::max(1, mn)), 0.0);
    rowsums.resize(all.size(), 0.0);
    MPI_Reduce(rowsums.data(), all.data(), mn, MPI_DOUBLE, MPI_SUM, 0, comm);

    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    if (rank != 0) return;

    // e_k^2 by backward accumulation; e[mn] = 0.
    std::vector<double> e2(static_cast<std::size_t>(mn) + 1, 0.0);
    for (int k = mn - 1; k >= 0; --k)
        e2[static_cast<std::size_t>(k)] =
            e2[static_cast<std::size_t>(k) + 1] + all[static_cast<std::size_t>(k)];

    const bool have_ref = matrix_has_known_spectrum(kind);
    std::vector<double> ref;
    if (have_ref) {
        ref.assign(static_cast<std::size_t>(mn) + 1, 0.0);
        for (int k = mn - 1; k >= 0; --k) {
            const double sg = singular_value(kind, k, L.n);
            ref[static_cast<std::size_t>(k)] =
                ref[static_cast<std::size_t>(k) + 1] + sg * sg;
        }
    }

    bool exists = false;
    { std::ifstream probe(path);
      exists = probe.good() && probe.peek() != std::ifstream::traits_type::eof(); }
    std::ofstream f(path, std::ios::app);
    if (!f) {
        std::fprintf(stderr, "error: cannot write %s\n", path.c_str());
        return;
    }
    if (!exists)
        f << "implementation,matrix,m,n,mb,nb,P,Q,k,ek_fro,eckart_young\n";
    f.setf(std::ios::scientific);
    f.precision(12);
    for (int k = 0; k <= mn; ++k) {
        f << impl << ',' << matrix_name(kind) << ',' << L.m << ',' << L.n << ','
          << L.mb << ',' << L.nb << ',' << L.nprow << ',' << L.npcol << ','
          << k << ',' << std::sqrt(std::max(0.0, e2[static_cast<std::size_t>(k)])) << ',';
        if (have_ref) f << std::sqrt(std::max(0.0, ref[static_cast<std::size_t>(k)]));
        f << '\n';
    }
    std::printf("ek: wrote %d rows to %s\n", mn + 1, path.c_str());
}

inline void write_ek(const std::string& path, const char* impl, Matrix kind,
                     const BlockCyclic& L, const double* a, int lld,
                     MPI_Comm comm) {
    write_ek_values(path, impl, kind, L, ek_rowsums_local(L, a, lld), comm);
}

// ---------------------------------------------------------------------------
// Matrix elements.  Pure functions of the global 0-based indices (i, j) of an
// m x n matrix.  Matrix::Legacy is never produced here -- each driver keeps its
// own generator for that -- and the paper matrices are not implemented yet.
// ---------------------------------------------------------------------------
inline double element(Matrix kind, int i, int j, int m, int n) {
    (void)m; (void)n;
    switch (kind) {
        case Matrix::Random:
            return detail::uniform(i, j);

        case Matrix::Kahan: {
            // Paper section 4.2, Matrix 4, with zeta = 0.99999 from p. C110:
            //
            //     A = S K,   S = diag(1, zeta, zeta^2, ...),
            //     K unit upper triangular with -phi off the diagonal,
            //     zeta^2 + phi^2 = 1.
            //
            // Every column then has norm exactly 1:
            //     ||A(:,j)||^2 = phi^2 (1-zeta^(2j))/(1-zeta^2) + zeta^(2j) = 1,
            // so classical column pivoting sees an exact tie at every step and
            // never swaps.  That is the point of the counterexample.
            if (j < i) return 0.0;
            constexpr double zeta = 0.99999;
            const double phi = std::sqrt(1.0 - zeta * zeta);
            const double s = detail::ipow(zeta, i);
            return (j == i) ? s : -phi * s;
        }

        case Matrix::Legacy:
        case Matrix::FastDecay:
        case Matrix::SDecay:
        case Matrix::Bie:
        default:
            return 0.0;  // guarded by matrix_implemented(); drivers must not get here
    }
}

inline double Entries::operator()(int i, int j) const {
    if (global) return global[static_cast<std::size_t>(j) * m + i];
    return (kind == Matrix::Legacy) ? legacy(i, j) : element(kind, i, j, m, n);
}

// ---------------------------------------------------------------------------
// Build a non-elementwise matrix (A = U D V*) once on rank 0 and replicate it.
//
// U and V are drawn exactly as the paper describes: QR factorizations of
// Gaussian random matrices.  The Gaussian entries come from a fixed
// deterministic stream, so the matrix depends only on (m, n) and the kind.
//
// The whole matrix is broadcast, which is why this is limited to the sizes the
// quality experiments actually use (the paper runs them at 4000 x 4000).  It is
// not a path for the timing benchmark sizes.
//
// Returns 0 on success, or a negative code: -1 too large, -2 LAPACK failed.
inline int build_global(Matrix kind, int m, int n, std::vector<double>& A,
                        MPI_Comm comm) {
    const std::size_t nelem = static_cast<std::size_t>(m) * n;
    if (nelem > static_cast<std::size_t>(8192) * 8192) return -1;
    // The BIE matrix is a square Nystrom system on an even number of
    // quadrature points; it has no meaning for other shapes.
    if (kind == Matrix::Bie && (m != n || n % 2 != 0)) return -3;

    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    A.assign(nelem, 0.0);
    int info = 0;

    if (rank == 0 && kind == Matrix::Bie) {
        // Fused multiply-add would make the assembly depend on the optimization
        // level: with contraction on, "R*m1 + h*m2" rounds once instead of
        // twice, and the four benchmarks would then only agree if built with
        // identical flags.  (--matrix-checksum caught exactly that.)
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
        // ------------------------------------------------------------------
        // Laplace single layer operator on a smooth closed curve.
        //
        //   (S phi)(x) = -1/(2 pi) \int_Gamma log|x - y| phi(y) ds(y)
        //
        // Nystrom discretization on n equidistant points in the parameter t,
        // with the logarithmic singularity handled by the Kress /
        // Martensen-Kussmaul splitting
        //
        //   log|x(t)-x(tau)| = 1/2 log(4 sin^2((t-tau)/2))
        //                    + 1/2 log( |x(t)-x(tau)|^2 / (4 sin^2((t-tau)/2)) )
        //
        // where the second term is analytic (it tends to log|x'(t)|^2 as
        // tau -> t).  The first term is integrated with the exact weights
        //
        //   R_d = -(2 pi / M) sum_{k=1..M-1} cos(k pi d / M)/k - (pi / M^2)(-1)^d,
        //   M = n/2,
        //
        // and the second with the trapezoidal rule.  See Kress, "Linear
        // Integral Equations", section 12.3.
        //
        // DEVIATION FROM THE PAPER: section 4.2 specifies Alpert's sixth order
        // hybrid Gauss-trapezoidal rule.  Kress' rule is the other standard
        // choice for this operator, is spectrally accurate rather than sixth
        // order, and needs no tabulated auxiliary nodes.  The operator, the
        // curve class and the ill conditioning -- which is what the pivot
        // quality experiment probes -- are the same.
        //
        // The curve is not specified by the paper either; we use the smooth
        // star  r(t) = 1 + 0.3 cos(3t),  x(t) = (r cos t, r sin t).
        // ------------------------------------------------------------------
        const int M = n / 2;
        const double two_pi = 6.283185307179586476925286766559;
        const double h = two_pi / n;

        std::vector<double> tv(n), xx(n), xy(n), speed(n);
        for (int j = 0; j < n; ++j) {
            const double t = h * j;
            const double r  = 1.0 + 0.3 * std::cos(3.0 * t);
            const double rp = -0.9 * std::sin(3.0 * t);
            tv[j] = t;
            xx[j] = r * std::cos(t);
            xy[j] = r * std::sin(t);
            const double dx = rp * std::cos(t) - r * std::sin(t);
            const double dy = rp * std::sin(t) + r * std::cos(t);
            speed[j] = std::sqrt(dx * dx + dy * dy);
        }

        // R_d, d = 0..n-1.  O(n*M) once, then every entry is O(1).
        std::vector<double> R(n, 0.0);
        for (int d = 0; d < n; ++d) {
            double acc = 0.0;
            for (int k = 1; k < M; ++k)
                acc += std::cos(k * 3.141592653589793238462643383279 * d / M) / k;
            R[d] = -(two_pi / M) * acc
                   - (3.141592653589793238462643383279 / (double(M) * M)) *
                         ((d % 2 == 0) ? 1.0 : -1.0);
        }

        const double quarter_inv_pi = 1.0 / (4.0 * 3.141592653589793238462643383279);
        for (int j = 0; j < n; ++j) {
            const double m1 = -quarter_inv_pi * speed[j];
            for (int i = 0; i < n; ++i) {
                double m2;
                if (i == j) {
                    // ratio -> |x'(t)|^2
                    m2 = m1 * 2.0 * std::log(speed[i]);
                } else {
                    const double dx = xx[i] - xx[j], dy = xy[i] - xy[j];
                    const double sn = std::sin(0.5 * (tv[i] - tv[j]));
                    m2 = m1 * std::log((dx * dx + dy * dy) / (4.0 * sn * sn));
                }
                const int d = (i >= j) ? (i - j) : (i - j + n);
                A[static_cast<std::size_t>(j) * n + i] = R[d] * m1 + h * m2;
            }
        }
    } else if (rank == 0) {
        const int mn = std::min(m, n);
        std::vector<double> U(nelem), V(static_cast<std::size_t>(n) * n);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i)
                U[static_cast<std::size_t>(j) * m + i] = detail::gaussian(i, j, 1);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                V[static_cast<std::size_t>(j) * n + i] = detail::gaussian(i, j, 2);

        // rows/cols are taken by value as blas_int so that &rows and &cols have
        // the width the Fortran side expects.
        auto orth = [&](double* X, qrcp::blas_int rows, qrcp::blas_int cols) -> int {
            std::vector<double> tau(static_cast<std::size_t>(std::min(rows, cols)));
            qrcp::blas_int lw = -1, inf = 0;
            double wq = 0.0;
            dgeqrf_(&rows, &cols, X, &rows, tau.data(), &wq, &lw, &inf);
            if (inf != 0) return static_cast<int>(inf);
            lw = std::max<qrcp::blas_int>(1, static_cast<qrcp::blas_int>(wq));
            std::vector<double> work(static_cast<std::size_t>(lw));
            dgeqrf_(&rows, &cols, X, &rows, tau.data(), work.data(), &lw, &inf);
            if (inf != 0) return static_cast<int>(inf);
            const qrcp::blas_int k = std::min(rows, cols);
            lw = -1;
            dorgqr_(&rows, &cols, &k, X, &rows, tau.data(), &wq, &lw, &inf);
            if (inf != 0) return static_cast<int>(inf);
            lw = std::max<qrcp::blas_int>(1, static_cast<qrcp::blas_int>(wq));
            work.assign(static_cast<std::size_t>(lw), 0.0);
            dorgqr_(&rows, &cols, &k, X, &rows, tau.data(), work.data(), &lw, &inf);
            return static_cast<int>(inf);
        };
        info = orth(U.data(), m, n);
        if (info == 0) info = orth(V.data(), n, n);

        if (info == 0) {
            // U := U * D   (scale column j by the j-th singular value)
            for (int j = 0; j < n; ++j) {
                const double d = singular_value(kind, j, n);
                double* col = U.data() + static_cast<std::size_t>(j) * m;
                for (int i = 0; i < m; ++i) col[i] *= d;
            }
            // A := (U D) * V^T
            const double one = 1.0, zero = 0.0;
            // m and n are this function's `int` parameters; the Fortran side
            // needs blas_int-wide lvalues, so copy before taking addresses.
            const qrcp::blas_int bm = m, bn = n;
            dgemm_("N", "T", &bm, &bn, &bn, &one, U.data(), &bm, V.data(), &bn,
                   &zero, A.data(), &bm);
            (void)mn;
        }
    }

    MPI_Bcast(&info, 1, MPI_INT, 0, comm);
    if (info != 0) return -2;
    // MPI_Bcast takes an int count; send in chunks so that large matrices work.
    const std::size_t chunk = 1u << 26;   // 64 Mi doubles
    for (std::size_t off = 0; off < nelem; off += chunk) {
        const int cnt = static_cast<int>(std::min(chunk, nelem - off));
        MPI_Bcast(A.data() + off, cnt, MPI_DOUBLE, 0, comm);
    }
    return 0;
}

}  // namespace qrcp
