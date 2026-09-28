#!/usr/bin/env python3
"""Plot the pivot-quality curves written by --rdiag and --ek.

    --rdiag CSV  ->  |R(k,k)|            (Figure 10 of the paper)
    --ek    CSV  ->  e_k, Frobenius      (Figures 6-9, right panels)

The column layout tells the two apart, so the same command plots either.

This reproduces the corresponding figures of

    P.-G. Martinsson, G. Quintana-Orti, N. Heavner, R. van de Geijn,
    "Householder QR Factorization With Randomization for Column Pivoting",
    SIAM J. Sci. Comput. 39(2), C96-C115, 2017.

For a column pivoted QR the diagonal of R decays monotonically and tracks the
singular values; how closely it does so is the measure of pivot quality.  The
singular values are plotted as a black reference line for the matrices whose
spectrum is known exactly (fast-decay, s-decay).

Standard library only; the plot itself is drawn by gnuplot.

Usage:
    plot_quality.py [-o out.png] [--title T] [--matrix NAME] file.csv [more.csv ...]

Several runs can be concatenated into one CSV, or passed as several files; rows
are grouped by (matrix, implementation, m, n, grid).
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile

# colour, dash type, draw order.
#
# Drawing order matters more than colour here: the curves overlap almost
# everywhere, so whatever is plotted last hides the rest.  The noisiest series
# goes down first and the exact QRCPs on top, and because DGEQPF and DGEQP3
# agree to twelve digits one of them is dashed -- otherwise it is simply
# invisible under the other.
#
# dashtype 1 is solid; gnuplot's dt 2 / dt 4 are dashed / dash-dot.
STYLE = {
    "TileHQRRP-MPI-OpenMP":     ("#2ca02c", 1, 0),
    "TileHQRRP-MPI-OpenMP-UNB": ("#9467bd", 1, 1),
    "HQRRP_MPI":                ("#1f77b4", 1, 2),
    "SCALAPACK_DGEQPF":         ("#d62728", 1, 3),
    "SCALAPACK_DGEQP3":         ("#ff7f0e", 2, 4),
}
DEFAULT_STYLE = ("#7f7f7f", 1, 9)


def gp_quote(text):
    """Quote a string for gnuplot (double quotes, backslash escapes)."""
    return '"%s"' % text.replace("\\", "\\\\").replace('"', '\\"')


# value column -> (y axis label, reference column, reference label)
SCHEMA = {
    "abs_rkk": ("|R(k,k)|", "sigma", "singular values"),
    "ek_fro":  ("e_k  (Frobenius)", "eckart_young", "Eckart-Young bound"),
}
RATIO_LABEL = {"abs_rkk": "|R(k,k)| ratio", "ek_fro": "e_k ratio"}


def read(paths):
    """Return (series, schema) where series maps
    (matrix, impl, m, n, P, Q) -> [(k, value, reference_or_None)]."""
    series, schema = {}, None
    for p in paths:
        with open(p, newline="") as fh:
            rd = csv.DictReader(fh)
            here = [c for c in SCHEMA if c in (rd.fieldnames or [])]
            if len(here) != 1:
                sys.exit("error: %s has no recognised value column "
                         "(expected one of %s)" % (p, ", ".join(SCHEMA)))
            if schema and schema != here[0]:
                sys.exit("error: %s mixes %s with %s" % (p, here[0], schema))
            schema = here[0]
            refcol = SCHEMA[schema][1]
            for row in rd:
                key = (row["matrix"], row["implementation"],
                       int(row["m"]), int(row["n"]), int(row["P"]), int(row["Q"]))
                ref = row.get(refcol) or ""
                series.setdefault(key, []).append(
                    (int(row["k"]), float(row[schema]),
                     float(ref) if ref else None))
    for v in series.values():
        v.sort()
    return series, schema


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("-o", "--out", default="rdiag.png")
    ap.add_argument("--title")
    ap.add_argument("--matrix", help="plot only this matrix kind")
    ap.add_argument("--only", help="comma separated list of implementations to "
                                   "plot; the others are dropped")
    ap.add_argument("--ratio", nargs="?", const="SCALAPACK_DGEQP3", default=None,
                    metavar="IMPL",
                    help="plot every series divided by IMPL (default "
                         "SCALAPACK_DGEQP3).  The curves differ by only a few "
                         "percent over several decades, so this is the only way "
                         "to actually see the differences between them.")
    ap.add_argument("--kmax", type=int, default=None,
                    help="plot only k <= KMAX (zoom into the leading columns)")
    ap.add_argument("--yscale", choices=("auto", "log", "linear"), default="auto",
                    help="y axis scale (default: auto, log unless the values "
                         "span less than a decade)")
    ap.add_argument("--keep", action="store_true",
                    help="keep the generated gnuplot script and data files")
    a = ap.parse_args()

    if not shutil.which("gnuplot"):
        sys.exit("error: gnuplot not found in PATH")

    series, schema = read(a.csv)
    if a.matrix:
        series = {k: v for k, v in series.items() if k[0] == a.matrix}
    if a.only:
        want = [x.strip() for x in a.only.split(",") if x.strip()]
        have = sorted({k[1] for k in series})
        for w in want:
            if w not in have:
                sys.exit("error: --only %s not in the data (have: %s)"
                         % (w, ", ".join(have)))
        series = {k: v for k, v in series.items() if k[1] in want}
    if not series:
        sys.exit("error: no rows to plot")

    # --ratio: divide every series, and the reference curve, by one
    # implementation.  Points where the divisor is zero (k = n for e_k) drop out.
    if a.ratio:
        base = {k: v for k, v in series.items() if k[1] == a.ratio}
        if not base:
            sys.exit("error: --ratio %s not present in the data (have: %s)"
                     % (a.ratio, ", ".join(sorted({k[1] for k in series}))))
        # one divisor per (matrix, m, n, P, Q)
        div = {}
        for k, v in base.items():
            div[(k[0], k[2], k[3], k[4], k[5])] = {kk: val for kk, val, _ in v}
        scaled = {}
        for k, v in series.items():
            dv = div.get((k[0], k[2], k[3], k[4], k[5]))
            if dv is None:
                continue
            rows = []
            for kk, val, ref in v:
                d = dv.get(kk)
                if not d:
                    continue
                rows.append((kk, val / d, (ref / d) if ref else None))
            if rows:
                scaled[k] = rows
        series = scaled

    if a.kmax is not None:
        series = {k: [r for r in v if r[0] <= a.kmax] for k, v in series.items()}
        series = {k: v for k, v in series.items() if v}
        if not series:
            sys.exit("error: --kmax %d leaves no rows" % a.kmax)

    kinds = sorted({k[0] for k in series})
    if len(kinds) > 1:
        sys.exit("error: several matrix kinds present (%s); "
                 "use --matrix to pick one" % ", ".join(kinds))
    kind = kinds[0]

    tmp = tempfile.mkdtemp(prefix="rdiag-")
    plots, sigma_file = [], None
    for key in sorted(series, key=lambda k: STYLE.get(k[1], DEFAULT_STYLE)[2]):
        _, impl, m, n, P, Q = key
        rows = series[key]
        path = os.path.join(tmp, "%s.dat" % impl.replace("/", "_"))
        with open(path, "w") as fh:
            for k, r, _ in rows:
                # gnuplot's log scale drops non-positive values; skip them
                # rather than letting the line break silently.
                if r > 0.0:
                    fh.write("%d %.12e\n" % (k, r))
        colour, dash, _ = STYLE.get(impl, DEFAULT_STYLE)
        plots.append('"%s" using 1:2 with lines lw 1 dt %d lc rgb "%s" title %s'
                     % (path, dash, colour,
                        gp_quote("%s (%dx%d)" % (impl, P, Q))))
        if sigma_file is None and rows[0][2] is not None:
            sigma_file = os.path.join(tmp, "sigma.dat")
            with open(sigma_file, "w") as fh:
                for k, _, s in rows:
                    if s and s > 0.0:
                        fh.write("%d %.12e\n" % (k, s))

    if sigma_file:
        plots.insert(0, '"%s" using 1:2 with lines lw 1.5 lc rgb "black" title %s'
                        % (sigma_file, gp_quote(SCHEMA[schema][2])))

    m, n = next(iter(series))[2], next(iter(series))[3]
    kmax = max(r[0] for rows in series.values() for r in rows)

    # A log y axis needs about a decade of range to be readable.  The Kahan
    # matrix has every column of norm 1, so its |R(k,k)| barely moves and the
    # paper plots it linearly (Figure 10, bottom right); everything else spans
    # enough to warrant a log axis.
    vals = [r[1] for rows in series.values() for r in rows if r[1] > 0.0]
    vals += [r[2] for rows in series.values() for r in rows if r[2] and r[2] > 0.0]
    span = (max(vals) / min(vals)) if vals and min(vals) > 0 else 1.0
    logscale = (a.yscale == "log") or (a.yscale == "auto" and span >= 10.0)
    if a.ratio and a.yscale == "auto":
        logscale = False
    title = a.title or "matrix %s  (%d x %d)" % (kind, m, n)
    if a.ratio:
        title += "   relative to %s" % a.ratio
    if a.kmax is not None:
        title += "   k <= %d" % a.kmax
    out = os.path.abspath(a.out)
    term = "pdfcairo size 7in,5in" if out.endswith(".pdf") else "pngcairo size 1000,700"

    script = os.path.join(tmp, "plot.gp")
    with open(script, "w") as fh:
        fh.write(
            'set terminal %s\n'
            'set output "%s"\n'
            # noenhanced: implementation names contain underscores, which
            # gnuplot would otherwise typeset as subscripts.
            'set title %s noenhanced\n'
            'set xlabel "k"\n'
            'set ylabel %s\n'
            '%s'
            'set grid\n'
            'set key outside right top noenhanced\n'
            'set xrange [0:%d]\n'
            'plot %s\n'
            % (term, out, gp_quote(title),
               gp_quote(RATIO_LABEL[schema] if a.ratio else SCHEMA[schema][0]),
               'set logscale y\nset format y "10^{%T}"\n' if logscale else '',
               kmax, ", \\\n     ".join(plots)))

    subprocess.run(["gnuplot", script], check=True)
    print("wrote %s" % out)
    if a.keep:
        print("gnuplot script and data kept in %s" % tmp)
    else:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
