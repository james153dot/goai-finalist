"""OpenFOAM mixing atlas → interpolator used by the acoustic n-τ layer.

The agent never sees a planted algebraic map. τ and unmixedness come from
2-D laminar dual-jet CFD (OpenFOAM 14). The interpolator is only a cache so
exploration campaigns can reuse the CFD budget.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
import warnings

from cswe.geometry import CLASSICAL_INJECTORS
from cswe.openfoam import MixingReport, openfoam_available, run_mixer

warnings.filterwarnings("ignore", category=ConvergenceWarning)

ROOT = Path(__file__).resolve().parents[2]
ATLAS_PATH = ROOT / "artifacts" / "mixing_atlas.json"

# physics.py also defines PARAM_NAMES — keep a local copy to avoid circular import.
_PARAM_NAMES = ("g", "d", "a", "s", "o")
_BOUNDS = {
    "g": (0.0, 2.0),
    "d": (0.0, 1.0),
    "a": (0.0, 1.0),
    "s": (0.0, 1.0),
    "o": (0.0, 1.0),
}


def _kernel() -> ConstantKernel:
    return ConstantKernel(1.0, (1e-2, 50.0)) * Matern(
        length_scale=np.ones(5), length_scale_bounds=(0.08, 6.0), nu=2.5
    ) + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 0.05))


class MixingAtlas:
    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.gp_tau: GaussianProcessRegressor | None = None
        self.gp_um: GaussianProcessRegressor | None = None
        self.gp_R: GaussianProcessRegressor | None = None
        self.gp_c: GaussianProcessRegressor | None = None

    def fit(self, rows: list[dict]) -> None:
        valid = [r for r in rows if r.get("Cconv")]
        if len(valid) < 6:
            raise ValueError(f"Need at least 6 converged OpenFOAM cases, got {len(valid)}")
        self.rows = rows
        X = np.array([[r[n] for n in _PARAM_NAMES] for r in valid], dtype=float)
        def gp() -> GaussianProcessRegressor:
            return GaussianProcessRegressor(
                kernel=_kernel(), normalize_y=True, random_state=0, n_restarts_optimizer=1
            )
        self.gp_tau = gp()
        self.gp_um = gp()
        self.gp_R = gp()
        self.gp_c = gp()
        self.gp_tau.fit(X, np.array([r["tau"] for r in valid], dtype=float))
        self.gp_um.fit(X, np.array([r["Um"] for r in valid], dtype=float))
        self.gp_R.fit(X, np.array([r.get("R_spatial", 0.35) for r in valid], dtype=float))
        self.gp_c.fit(X, np.array([r.get("compactness", 1.4) for r in valid], dtype=float))

    def predict(self, x: dict[str, float]) -> MixingReport:
        if self.gp_tau is None:
            raise RuntimeError("Mixing atlas is empty. Run `cswe atlas`.")
        v = np.array([[x[n] for n in _PARAM_NAMES]], dtype=float)
        tau = float(self.gp_tau.predict(v)[0])
        um = float(np.clip(self.gp_um.predict(v)[0], 0.0, 1.0))
        R_spatial = float(self.gp_R.predict(v)[0])
        compactness = float(max(self.gp_c.predict(v)[0], 0.2))
        return MixingReport(
            tau=max(tau, 1e-4), Um=um, L_mix=float("nan"), u_bulk=float("nan"),
            Cconv=True, backend="openfoam_atlas",
            R_spatial=R_spatial, compactness=compactness,
        )

    def save(self, path: Path = ATLAS_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"rows": self.rows}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = ATLAS_PATH) -> "MixingAtlas":
        data = json.loads(path.read_text(encoding="utf-8"))
        atlas = cls()
        atlas.fit(data["rows"])
        return atlas


_ATLAS: MixingAtlas | None = None


def get_atlas() -> MixingAtlas:
    global _ATLAS
    if _ATLAS is None:
        if not ATLAS_PATH.exists():
            raise FileNotFoundError(f"Missing {ATLAS_PATH}. Run `cswe atlas` with OpenFOAM.")
        _ATLAS = MixingAtlas.load()
    return _ATLAS


def mix(x: dict[str, float]) -> MixingReport:
    return get_atlas().predict(x)


def _sample_points(n: int, seed: int) -> list[dict[str, float]]:
    sampler = qmc.LatinHypercube(d=5, seed=seed)
    unit = sampler.random(n=n)
    lo = np.array([_BOUNDS[k][0] for k in _PARAM_NAMES])
    hi = np.array([_BOUNDS[k][1] for k in _PARAM_NAMES])
    scaled = qmc.scale(unit, lo, hi)
    pts = [{k: float(row[i]) for i, k in enumerate(_PARAM_NAMES)} for row in scaled]
    for name, vec in CLASSICAL_INJECTORS.items():
        q = dict(vec)
        q["label"] = name
        pts.append(q)
    return pts


def _one(x: dict[str, float], n_iter: int) -> dict:
    label = x.pop("label", "")
    report = run_mixer(x, n_iter=n_iter)
    row = {
        **x,
        "tau": report.tau,
        "Um": report.Um,
        "L_mix": report.L_mix,
        "R_spatial": report.R_spatial,
        "compactness": report.compactness,
        "x_q": report.x_q,
        "q_profile": report.q_profile,
        "x_profile": report.x_profile,
        "p_profile": report.p_profile,
        "Cconv": report.Cconv,
        "notes": report.notes,
        "label": label,
    }
    return row


def build_atlas(n: int = 36, seed: int = 7, n_iter: int = 120, workers: int = 4) -> MixingAtlas:
    if not openfoam_available():
        raise RuntimeError("OpenFOAM 14 is not available. Source /opt/openfoam14/etc/bashrc.")
    from concurrent.futures import ProcessPoolExecutor, as_completed

    pts = _sample_points(n, seed)
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_one, dict(p), n_iter): p for p in pts}
        for i, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            rows.append(row)
            status = "ok" if row["Cconv"] else row["notes"]
            print(f"[{i}/{len(pts)}] g={row['g']:.2f} s={row['s']:.2f} Cconv={row['Cconv']} tau={row.get('tau')} {status}", flush=True)
    atlas = MixingAtlas()
    atlas.fit(rows)
    atlas.save()
    global _ATLAS
    _ATLAS = atlas
    return atlas


TEST_PATH = ROOT / "artifacts" / "of_test.json"


def build_test_set(n: int = 20, seed: int = 123, n_iter: int = 100, workers: int = 4) -> list[dict]:
    """Independent OpenFOAM hold-out set. Never used to train campaigns or the atlas."""
    if not openfoam_available():
        raise RuntimeError("OpenFOAM 14 is not available.")
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from cswe.physics import _acoustics

    sampler = qmc.LatinHypercube(d=5, seed=seed)
    unit = sampler.random(n=n)
    lo = np.array([_BOUNDS[k][0] for k in _PARAM_NAMES])
    hi = np.array([_BOUNDS[k][1] for k in _PARAM_NAMES])
    pts = [{k: float(row[i]) for i, k in enumerate(_PARAM_NAMES)} for row in qmc.scale(unit, lo, hi)]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_one, dict(p), n_iter) for p in pts]
        for i, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            n_index, omega, rayleigh, sigma, S, phase = _acoustics(
                row["tau"], row["Um"], row["o"], row.get("R_spatial"), row.get("compactness")
            )
            row["S"] = S
            row["sigma"] = sigma
            row["stable"] = int(S < 1.0)
            row["R"] = rayleigh
            row["phase"] = phase
            rows.append(row)
            print(f"[test {i}/{n}] Cconv={row['Cconv']} S={S:.3f} stable={row['stable']}", flush=True)
    TEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEST_PATH.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    return rows
