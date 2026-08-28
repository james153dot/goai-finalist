"""Write and run a 2-D laminar dual-jet mixer in OpenFOAM 14.

The chamber geometry is fixed. Two inlet slots carry complementary mixture
fraction Z=0 and Z=1 (OpenFOAM field name: T). Mixing delay, unmixedness, and
the Rayleigh spatial overlap are taken from the same steady field.
q_mix(x) = Var_y[Z](x) is a mixing-availability proxy, not a heat-release
prediction: it weights stations where scalar segregation remains.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cswe.geometry import H, L, NU, T_DIFFUSIVITY, W, JetLayout, jet_layout

DEFAULT_VARIANCE_THRESHOLD = 0.045
DEFAULT_N_AXIAL_BINS = 24

FOAM_BASHRC = Path("/opt/openfoam14/etc/bashrc")
HEADER = r"""/*--------------------------------*- C++ -*----------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Version:  14
\*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       {cls};
    object      {obj};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""


@dataclass
class MixingReport:
    tau: float
    Um: float
    L_mix: float
    u_bulk: float
    Cconv: bool
    backend: str
    notes: str = ""
    R_spatial: float = float("nan")
    compactness: float = float("nan")
    x_q: float = float("nan")
    q_profile: list[float] = field(default_factory=list)
    x_profile: list[float] = field(default_factory=list)
    p_profile: list[float] = field(default_factory=list)


def openfoam_available() -> bool:
    return FOAM_BASHRC.is_file()


def mixing_delay_from_variance(
    x_mid: list[float] | np.ndarray,
    variances: list[float] | np.ndarray,
    u_bulk: float,
    thresh: float = DEFAULT_VARIANCE_THRESHOLD,
) -> tuple[float, float]:
    """Operational τ from an axial Var_y[Z] profile.

    x_m is the first bin-centre with V < thresh (absolute, not V/V(0)).
    No interpolation. If the threshold is never reached, x_m is the last
    bin-centre. τ = max(1e-4, x_m / U_b).
    """
    xs = [float(v) for v in x_mid]
    vs = [float(v) for v in variances]
    if not xs:
        return 1e-4, float("nan")
    x_m = xs[-1]
    for x, var in zip(xs, vs):
        if var < thresh:
            x_m = x
            break
    tau = max(1e-4, x_m / max(float(u_bulk), 1e-6))
    return float(tau), float(x_m)


def resample_axial_profile(
    x: np.ndarray, v: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly resample a stored axial profile onto n_bins stations.

    Used only for post-processing sensitivity of the τ definition.
    It does not re-bin raw OpenFOAM cells.
    """
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    n_bins = int(n_bins)
    if n_bins <= 1 or len(x) < 2:
        return x, v
    if n_bins == len(x):
        return x, v
    x_new = np.linspace(float(x[0]), float(x[-1]), n_bins)
    v_new = np.interp(x_new, x, v)
    return x_new, v_new


def spatial_overlap_from_profile(x: np.ndarray, q: np.ndarray) -> tuple[float, float, float]:
    """R_spatial, compactness, x_q from mixing-availability q_mix(x)."""
    xx = np.asarray(x, dtype=float)
    qq = np.asarray(q, dtype=float)
    xi = np.clip(xx / L, 0.0, 1.0)
    p_mode = np.cos(np.pi * xi)
    mass = float(np.trapezoid(np.maximum(qq, 0.0), xx)) + 1e-12
    r_spatial = float(np.trapezoid(qq * p_mode, xx) / mass)
    compactness = float(qq.max() / (qq.mean() + 1e-12)) if qq.size else 1.0
    x_q = float(np.trapezoid(qq * xx, xx) / mass)
    return r_spatial, compactness, x_q


def mixedness_at_fraction(x: np.ndarray, v: np.ndarray, frac: float = 0.45) -> float:
    xx = np.asarray(x, dtype=float)
    vv = np.asarray(v, dtype=float)
    target = float(xx[0] + frac * (xx[-1] - xx[0]))
    j = int(np.argmin(np.abs(xx - target)))
    return float(max(0.0, min(1.0, 1.0 - 4.0 * vv[j])))


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _ny(span: float) -> int:
    return max(4, int(round(28 * span / H)))


def write_case(case: Path, layout: JetLayout, n_iter: int = 280) -> None:
    y = [0.0, layout.y0_lo, layout.y0_hi, layout.y1_lo, layout.y1_hi, H]
    for i in range(1, len(y)):
        if y[i] <= y[i - 1] + 1e-6:
            y[i] = y[i - 1] + 1.5e-4
    if y[-1] < H:
        y[-1] = H

    # 6 y-levels × 2 x × 2 z
    xs = (0.0, L)
    zs = (0.0, W)
    verts = []
    for z in zs:
        for yy in y:
            for x in xs:
                verts.append((x, yy, z))

    def v(ix: int, iy: int, iz: int) -> int:
        return iz * 12 + iy * 2 + ix

    blocks = []
    nxs = 48
    for iy in range(5):
        nys = _ny(y[iy + 1] - y[iy])
        hex_ids = [
            v(0, iy, 0),
            v(1, iy, 0),
            v(1, iy + 1, 0),
            v(0, iy + 1, 0),
            v(0, iy, 1),
            v(1, iy, 1),
            v(1, iy + 1, 1),
            v(0, iy + 1, 1),
        ]
        blocks.append(
            "    hex (" + " ".join(str(i) for i in hex_ids) + f") ({nxs} {nys} 1) simpleGrading (1 1 1)"
        )

    def face(iy: int, side: str) -> str:
        # Outward normals: -x on the left, +x on the right.
        if side == "left":
            ids = [v(0, iy, 0), v(0, iy, 1), v(0, iy + 1, 1), v(0, iy + 1, 0)]
        else:
            ids = [v(1, iy, 0), v(1, iy + 1, 0), v(1, iy + 1, 1), v(1, iy, 1)]
        return "            (" + " ".join(str(i) for i in ids) + ")"

    front = "\n".join(
        "            ("
        + " ".join(str(i) for i in [v(0, iy, 0), v(0, iy + 1, 0), v(1, iy + 1, 0), v(1, iy, 0)])
        + ")"
        for iy in range(5)
    )
    back = "\n".join(
        "            ("
        + " ".join(str(i) for i in [v(0, iy, 1), v(1, iy, 1), v(1, iy + 1, 1), v(0, iy + 1, 1)])
        + ")"
        for iy in range(5)
    )

    vertex_txt = "\n".join(f"    ({x:.6g} {yy:.6g} {z:.6g})" for x, yy, z in verts)
    block_mesh = (
        HEADER.format(cls="dictionary", obj="blockMeshDict")
        + f"""
scale   1;

vertices
(
{vertex_txt}
);

blocks
(
{chr(10).join(blocks)}
);

boundary
(
    inletA
    {{
        type patch;
        faces
        (
{face(1, "left")}
        );
    }}
    inletB
    {{
        type patch;
        faces
        (
{face(3, "left")}
        );
    }}
    outlet
    {{
        type patch;
        faces
        (
{chr(10).join(face(i, "right") for i in range(5))}
        );
    }}
    walls
    {{
        type wall;
        faces
        (
{face(0, "left")}
{face(2, "left")}
{face(4, "left")}
            ({v(0, 0, 0)} {v(0, 0, 1)} {v(1, 0, 1)} {v(1, 0, 0)})
            ({v(0, 5, 0)} {v(1, 5, 0)} {v(1, 5, 1)} {v(0, 5, 1)})
        );
    }}
    frontAndBack
    {{
        type empty;
        faces
        (
{front}
{back}
        );
    }}
);
"""
    )
    _write(case / "system" / "blockMeshDict", block_mesh)

    _write(
        case / "system" / "controlDict",
        HEADER.format(cls="dictionary", obj="controlDict")
        + f"""
solver          incompressibleFluid;

startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {n_iter};
deltaT          1;

writeControl    timeStep;
writeInterval   {n_iter};
purgeWrite      1;
writeFormat     ascii;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable false;

functions
{{
    #includeFunc scalarTransport(T, diffusivity=constant, D={T_DIFFUSIVITY})
}}
""",
    )

    _write(
        case / "system" / "fvSchemes",
        HEADER.format(cls="dictionary", obj="fvSchemes")
        + """
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);
    div(phi,T)      bounded Gauss linearUpwind grad(T);
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
""",
    )

    _write(
        case / "system" / "fvSolution",
        HEADER.format(cls="dictionary", obj="fvSolution")
        + """
solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-07;
        relTol          0.05;
        smoother        GaussSeidel;
    }
    "(U|T)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-06;
        relTol          0.05;
    }
}
SIMPLE
{
    nNonOrthogonalCorrectors 0;
    consistent      yes;
    residualControl { p 1e-3; U 1e-4; T 1e-4; }
    pRefCell        0;
    pRefValue       0;
}
relaxationFactors
{
    equations { U 0.8; T 0.8; ".*" 0.8; }
}
""",
    )

    _write(
        case / "constant" / "physicalProperties",
        HEADER.format(cls="dictionary", obj="physicalProperties")
        + f"""
viscosityModel  constant;
nu              {NU};
""",
    )
    _write(
        case / "constant" / "momentumTransport",
        HEADER.format(cls="dictionary", obj="momentumTransport")
        + """
simulationType  laminar;
""",
    )

    u0x, u0y = layout.u0
    u1x, u1y = layout.u1
    _write(
        case / "0" / "U",
        HEADER.format(cls="volVectorField", obj="U")
        + f"""
dimensions      [velocity];
internalField   uniform ({0.5 * layout.u_ref} 0 0);
boundaryField
{{
    inletA {{ type fixedValue; value uniform ({u0x:.6g} {u0y:.6g} 0); }}
    inletB {{ type fixedValue; value uniform ({u1x:.6g} {u1y:.6g} 0); }}
    outlet {{ type zeroGradient; }}
    walls  {{ type noSlip; }}
    frontAndBack {{ type empty; }}
}}
""",
    )
    _write(
        case / "0" / "p",
        HEADER.format(cls="volScalarField", obj="p")
        + """
dimensions      [kinematicPressure];
internalField   uniform 0;
boundaryField
{
    inletA { type zeroGradient; }
    inletB { type zeroGradient; }
    outlet { type fixedValue; value uniform 0; }
    walls  { type zeroGradient; }
    frontAndBack { type empty; }
}
""",
    )
    _write(
        case / "0" / "T",
        HEADER.format(cls="volScalarField", obj="T")
        + """
dimensions      [temperature];
internalField   uniform 0.5;
boundaryField
{
    inletA { type fixedValue; value uniform 0; }
    inletB { type fixedValue; value uniform 1; }
    outlet { type zeroGradient; }
    walls  { type zeroGradient; }
    frontAndBack { type empty; }
}
""",
    )


def _foam_cmd(cmd: str, cwd: Path, timeout: int = 180) -> subprocess.CompletedProcess:
    full = f"source {FOAM_BASHRC} && {cmd}"
    env = os.environ.copy()
    env["FOAM_SIGFPE"] = "0"
    return subprocess.run(
        ["bash", "-lc", full],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _parse_internal_field(path: Path) -> list[float] | list[tuple[float, float, float]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"internalField\s+nonuniform\s+List<(\w+)>\s+(\d+)\s*\(", text)
    if m:
        kind = m.group(1)
        n = int(m.group(2))
        rest = text[m.end() :]
        if kind == "vector":
            tuples = re.findall(r"\(([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\)", rest)
            return [(float(a), float(b), float(c)) for a, b, c in tuples[:n]]
        nums: list[float] = []
        for tok in rest.replace(")", " ").split():
            if tok == ";":
                break
            try:
                nums.append(float(tok))
            except ValueError:
                if nums:
                    break
            if len(nums) >= n:
                break
        return nums[:n]
    for line in text.splitlines():
        if "internalField" in line and "uniform" in line:
            rest = line.split("uniform", 1)[1].strip().rstrip(";")
            if rest.startswith("("):
                a, b, c = rest.strip("()").split()
                return [(float(a), float(b), float(c))]
            return [float(rest)]
    raise ValueError(f"Could not parse {path}")


def _latest_time(case: Path) -> Path | None:
    times = []
    for p in case.iterdir():
        if p.name.replace(".", "", 1).isdigit() and p.name != "0":
            times.append(p)
    if not times:
        return None
    return max(times, key=lambda p: float(p.name))


def _metrics_from_fields(case: Path, layout: JetLayout) -> MixingReport:
    tdir = _latest_time(case)
    if tdir is None:
        return MixingReport(float("nan"), float("nan"), float("nan"), float("nan"), False, "openfoam", "no_time_dir")
    # Cell centres: write once if missing.
    cc = case / "0" / "C"
    if not cc.exists() and not (tdir / "C").exists():
        _foam_cmd("foamPostProcess -func writeCellCentres -time 0", case, timeout=60)
    cpath = tdir / "C" if (tdir / "C").exists() else case / "0" / "C"
    tpath = tdir / "T"
    upath = tdir / "U"
    if not tpath.exists() or not upath.exists() or not cpath.exists():
        return MixingReport(float("nan"), float("nan"), float("nan"), float("nan"), False, "openfoam", "missing_fields")

    C = _parse_internal_field(cpath)
    T = _parse_internal_field(tpath)
    U = _parse_internal_field(upath)
    n = min(len(C), len(T), len(U))
    if n < 20:
        return MixingReport(float("nan"), float("nan"), float("nan"), float("nan"), False, "openfoam", "short_field")

    xs = [C[i][0] for i in range(n)]
    xmin, xmax = min(xs), max(xs)
    nbins = DEFAULT_N_AXIAL_BINS
    bins = [[] for _ in range(nbins)]
    ubins = [[] for _ in range(nbins)]
    for i in range(n):
        b = min(nbins - 1, int((xs[i] - xmin) / (xmax - xmin + 1e-12) * nbins))
        tval = T[i] if not isinstance(T[i], tuple) else T[i][0]
        bins[b].append(float(tval))
        ubins[b].append(U[i][0] if isinstance(U[i], tuple) else 0.0)

    variances = []
    xmid = []
    for b in range(nbins):
        if len(bins[b]) < 3:
            variances.append(1.0)
        else:
            mu = sum(bins[b]) / len(bins[b])
            variances.append(sum((t - mu) ** 2 for t in bins[b]) / len(bins[b]))
        xmid.append(xmin + (b + 0.5) * (xmax - xmin) / nbins)

    # Mixing-availability proxy q_mix(x) = Var_y[Z](x). This is not a
    # heat-release prediction. It identifies axial stations where scalar
    # segregation remains and is the declared mixing-side weighting of the
    # Rayleigh analog. Stored in JSON as q_profile. OpenFOAM field name: T.
    q_profile = [float(v) for v in variances]
    x_profile = [float(v) for v in xmid]

    q = np.array(q_profile, dtype=float)
    xx = np.array(x_profile, dtype=float)
    R_spatial, compactness, x_q = spatial_overlap_from_profile(xx, q)
    xi = np.clip(xx / L, 0.0, 1.0)
    p_mode = np.cos(np.pi * xi)

    u_bulk = 0.5 * (abs(layout.u0[0]) + abs(layout.u1[0]))
    tau, L_mix = mixing_delay_from_variance(xmid, variances, u_bulk)
    target = xmin + 0.45 * (xmax - xmin)
    j = min(range(nbins), key=lambda k: abs(xmid[k] - target))
    Um = float(max(0.0, min(1.0, 1.0 - 4.0 * variances[j])))
    if any(v != v for v in (tau, Um, L_mix, R_spatial)):
        return MixingReport(float("nan"), float("nan"), float("nan"), u_bulk, False, "openfoam", "nan_metric")
    return MixingReport(
        float(tau), Um, float(L_mix), float(u_bulk), True, "openfoam", "",
        R_spatial=R_spatial, compactness=compactness, x_q=x_q,
        q_profile=[float(v) for v in q_profile],
        x_profile=[float(v) for v in x_profile],
        p_profile=[float(v) for v in p_mode],
    )


def run_mixer(x: dict[str, float], work: Path | None = None, n_iter: int = 280) -> MixingReport:
    if not openfoam_available():
        return MixingReport(float("nan"), float("nan"), float("nan"), float("nan"), False, "openfoam", "openfoam_missing")
    layout = jet_layout(x["g"], x["d"], x["a"], x["s"], x["o"])
    tmp = Path(work) if work else Path(tempfile.mkdtemp(prefix="cswe_of_"))
    tmp.mkdir(parents=True, exist_ok=True)
    write_case(tmp, layout, n_iter=n_iter)
    mesh = _foam_cmd("blockMesh", tmp, timeout=60)
    if mesh.returncode != 0:
        return MixingReport(float("nan"), float("nan"), float("nan"), float("nan"), False, "openfoam", "blockMesh_failed:" + mesh.stderr[-400:])
    run = _foam_cmd("foamRun", tmp, timeout=240)
    if run.returncode != 0:
        return MixingReport(float("nan"), float("nan"), float("nan"), float("nan"), False, "openfoam", "foamRun_failed:" + (run.stderr or run.stdout)[-500:])
    report = _metrics_from_fields(tmp, layout)
    if work is None:
        shutil.rmtree(tmp, ignore_errors=True)
    return report
