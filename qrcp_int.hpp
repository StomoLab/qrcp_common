// ===========================================================================
// qrcp_int.hpp -- the integer type of the BLAS / LAPACK / ScaLAPACK interface.
//
// Every Fortran routine we call takes its integer arguments by reference, so
// the width of that integer is part of the ABI.  Two widths exist:
//
//   LP64   (default)   32-bit integers; MKL's *_lp64 libraries, Netlib.
//   ILP64  (opt in)    64-bit integers; MKL's *_ilp64 libraries.
//
// Why this matters here
// ---------------------
// ScaLAPACK addresses a local array as  a[i + j*lld].  Under LP64 that
// expression is evaluated in 32-bit arithmetic INSIDE the library, so it wraps
// once j*lld passes 2^31 and the routine reads a wild address.  A square
// matrix on one rank therefore cannot exceed
//
//     n = floor(sqrt(2^31 - 1)) = 46340,
//
// which is exactly why the N = 40960 runs finished and the N = 81920 runs died
// with SIGSEGV (jobs 633118 / 633119).  There is no way around it other than a
// 64-bit interface: it is not a shortage of memory and not a bug in our code.
//
// How to get it wrong
// -------------------
// Passing a 32-bit int where the library expects 64 bits does NOT crash.  The
// callee reads four bytes of the neighbouring variable as the high half, so
// the routine quietly computes with a nonsense dimension and returns numbers
// that look plausible.  Every declaration below and every variable whose
// address reaches a Fortran routine must therefore use `blas_int`, never
// `int`.  The `--check` mode (comparison of the pivot sequence against
// sequential LAPACK dgeqp3) is what actually proves this was done right.
//
// Note that the sequential BLAS and LAPACK are affected too, not just
// ScaLAPACK: linking libmkl_intel_ilp64 widens dgeqrf, dorgqr, dgemm, dnrm2
// and dlamch along with pdgemm and friends.
// ===========================================================================
#pragma once

#include <climits>
#include <type_traits>

namespace qrcp {

#if defined(QRCP_ILP64)
// Matches MKL_INT when MKL is compiled with -DMKL_ILP64 (mkl_types.h then
// defines MKL_INT as long long int).
using blas_int = long long;
#else
using blas_int = int;
#endif

#if defined(QRCP_ILP64)
static_assert(sizeof(blas_int) == 8,
              "QRCP_ILP64 is defined but blas_int is not 64 bits; the link "
              "line must use MKL's *_ilp64 libraries and the compile line "
              "must define MKL_ILP64.");
#else
static_assert(sizeof(blas_int) == 4,
              "blas_int must be 32 bits for the LP64 interface.");
#endif

// Largest square matrix one rank can hold before a Fortran routine's internal
// i + j*lld overflows.  Drivers use it to fail with a clear message instead of
// a segmentation fault.
constexpr long long max_square_local_dim() {
    return (sizeof(blas_int) == 8) ? 3037000499LL : 46340LL;
}

// Narrowing guard for the reverse direction: a blas_int fed to MPI or to any
// of our own `int` interfaces.  Dimensions always fit; this documents that and
// catches the case where they would not.
inline int to_int(blas_int v) {
    return static_cast<int>(v);
}
static_assert(std::is_signed<blas_int>::value, "blas_int must be signed");

}  // namespace qrcp
