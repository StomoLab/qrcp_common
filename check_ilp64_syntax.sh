#!/bin/bash
# ===========================================================================
# Type-check every translation unit as if QRCP_ILP64 were on.
#
# Why this exists: locally qrcp::blas_int is `int`, so `std::max(x, 0)` with a
# blas_int x compiles fine.  Under ILP64 blas_int becomes `long long` and the
# same call is ambiguous.  Job 633323 found two such lines only after a full
# HUCC build -- a 15 minute queue wait for an error the compiler can see in a
# second.  -fsyntax-only needs no ILP64 MKL, only the headers, so this catches
# the whole class of error before anything is submitted.
#
# Usage:  bash qrcp_common/check_ilp64_syntax.sh
# Run it from build_space/.
# ===========================================================================
set -u
CXX=${CXX:-mpicxx}
COMMON=$(cd "$(dirname "$0")" && pwd)
ROOT=$(dirname "$COMMON")
fail=0
missing=0

if ! command -v "$CXX" >/dev/null 2>&1; then
    echo "ILP64 CHECK UNAVAILABLE: compiler not found: $CXX" >&2
    exit 2
fi

check() {   # check <label> <file> <extra include dirs...>
    label=$1; file=$2; shift 2
    if [ ! -f "$file" ]; then
        printf '  MISSING SOURCE  %s\n' "$file" >&2
        missing=1
        return
    fi
    inc=()
    for d in "$@"; do inc+=("-I$d"); done
    if out=$("$CXX" -std=c++17 -fsyntax-only -Wall -Wextra \
                    -DQRCP_ILP64 -I"$COMMON" "${inc[@]}" "$file" 2>&1); then
        printf "  OK    %s\n" "$label"
    else
        printf "  FAIL  %s\n" "$label"
        printf '%s\n' "$out" | head -12 | sed 's/^/          /'
        fail=1
    fi
}

echo "=== SCALAPACK_DGEQP3 (QRCP_ILP64) ==="
D="$ROOT/SCALAPACK_DGEQP3"
check "main.cpp"       "$D/main.cpp"       "$D" "$D/src"
check "src/pdgeqp3.cpp" "$D/src/pdgeqp3.cpp" "$D/src"

echo "=== HQRRP_MPI (QRCP_ILP64) ==="
H="$ROOT/HQRRP_MPI"
for f in src/internal.cpp src/qrp_unb.cpp src/hqrrp.cpp src/pdgeqp4.cpp src/utils.cpp; do
    check "$f" "$H/$f" "$H/include" "$H/src"
done
check "test/benchmark.cpp" "$H/test/benchmark.cpp" "$H/include" "$H/src" "$H/test"

echo
if [ $missing -ne 0 ]; then
    echo "ILP64 CHECK INCOMPLETE: required sources missing -- do not submit"
    exit 2
elif [ $fail -eq 0 ]; then
    echo "ALL TRANSLATION UNITS TYPE-CHECK UNDER ILP64"
else
    echo "ILP64 TYPE ERRORS PRESENT -- do not submit"
fi
exit $fail
