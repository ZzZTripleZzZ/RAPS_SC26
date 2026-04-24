"""Traffic template loader and modulo-tiling (paper §5 ingestion).

Captures per-proxy-app affinity matrices produced by
``traffic_gen/src/analyze_research.py`` (SST-Dumpi → N×N rank-pair byte counts)
and scales them to arbitrary target sizes ``M`` by the rule

    T_syn[u, v] = T_base[u mod N, v mod N]

which preserves block-periodic spatial locality (stencil stays stencil,
all-to-all stays all-to-all) without re-running the tracer at the new scale.
"""
from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

import numpy as np


# Location of affinity JSON / static NPY templates (repo-relative default).
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATRIX_DIR = _REPO_ROOT / "data" / "matrices"


_FILENAME_RE = re.compile(
    r"^(?P<app>[a-z0-9]+)"          # app name (lulesh, comd, hpgmg, cosp2, quicksilver)
    r"(?:_[a-z0-9]+)*?"             # optional suffix tokens (dense, sparse, mc, …)
    r"_n(?P<nodes>\d+)"             # base-rank count
    r"(?:_[a-z0-9]+)*"              # more optional suffix tokens
    r"_\d{8}_\d{6}"                 # timestamp
    r"_affinity\.json$",
    re.IGNORECASE,
)


def _list_templates(matrix_dir: Path) -> list[tuple[str, int, Path]]:
    """Return [(app, base_nodes, path), ...] for every parseable affinity JSON."""
    out: list[tuple[str, int, Path]] = []
    for p in matrix_dir.glob("*_affinity.json"):
        m = _FILENAME_RE.match(p.name)
        if not m:
            continue
        app = m.group("app").lower()
        base_nodes = int(m.group("nodes"))
        out.append((app, base_nodes, p))
    return out


def _affinity_to_matrix(affinity: dict) -> np.ndarray:
    """Build a symmetric N×N float matrix from the affinity edge list."""
    n = int(affinity["num_nodes"])
    mat = np.zeros((n, n), dtype=np.float64)
    for edge in affinity.get("edges", ()):
        u = int(edge["source"])
        v = int(edge["target"])
        w = float(edge["weight"])
        if 0 <= u < n and 0 <= v < n and u != v:
            # Edge list is undirected; split weight symmetrically so
            # row sums balance send/recv contribution.
            mat[u, v] += w * 0.5
            mat[v, u] += w * 0.5
    return mat


def load_base_template(
    app: str,
    base_nodes: int | None = None,
    *,
    matrix_dir: str | os.PathLike | None = None,
) -> np.ndarray:
    """Load a row-normalized N×N template matrix for ``app``.

    Parameters
    ----------
    app : str
        Proxy-app name (``lulesh``, ``comd``, ``hpgmg``, ``cosp2``,
        ``quicksilver``).  Case-insensitive.
    base_nodes : int, optional
        Specific rank count to request.  If omitted, the largest available
        template for the app is used.
    matrix_dir : path-like, optional
        Directory to search; defaults to ``data/matrices/`` at the repo root.

    Returns
    -------
    np.ndarray
        N×N matrix whose rows sum to 1 (or 0 for isolated ranks).  Entry
        ``[u, v]`` is the fraction of rank-u traffic addressed to rank v.
    """
    matrix_dir = Path(matrix_dir) if matrix_dir is not None else DEFAULT_MATRIX_DIR
    candidates = [
        (n, p) for (a, n, p) in _list_templates(matrix_dir) if a == app.lower()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No traffic template for app={app!r} under {matrix_dir}"
        )

    if base_nodes is None:
        # Pick the largest available.
        candidates.sort(key=lambda x: x[0], reverse=True)
        _, path = candidates[0]
    else:
        match = [c for c in candidates if c[0] == base_nodes]
        if not match:
            raise FileNotFoundError(
                f"No {app}_n{base_nodes} template under {matrix_dir}; "
                f"available: {sorted(n for n, _ in candidates)}"
            )
        path = match[0][1]

    with open(path, "r") as f:
        affinity = json.load(f)

    mat = _affinity_to_matrix(affinity)
    return _row_normalize(mat)


def _row_normalize(mat: np.ndarray) -> np.ndarray:
    """Normalize each row to sum to 1 (rows with zero sum stay zero)."""
    row_sum = mat.sum(axis=1, keepdims=True)
    nz = row_sum > 0
    out = np.zeros_like(mat)
    np.divide(mat, row_sum, out=out, where=nz)
    return out


def tile_template(base: np.ndarray, target_size: int) -> np.ndarray:
    """Modulo-tile an N×N template to an M×M matrix.

    ``T_syn[u, v] = T_base[u mod N, v mod N]``; then re-normalize rows so the
    total outbound weight per rank stays 1 after any duplication / self-loop
    folding.
    """
    if target_size <= 0:
        raise ValueError(f"target_size must be positive, got {target_size}")
    if base.ndim != 2 or base.shape[0] != base.shape[1]:
        raise ValueError(f"template must be square 2D, got shape {base.shape}")
    n = base.shape[0]
    if target_size == n:
        return _row_normalize(base.copy())

    # NumPy broadcast trick for u,v ∈ [0, M) → (u % N, v % N):
    idx = np.arange(target_size) % n
    tiled = base[np.ix_(idx, idx)].astype(np.float64, copy=True)
    # Zero the diagonal — modulo folding can create fake self-loops where the
    # original matrix did not have them (distinct ranks can collide into the
    # same mod-N pair).
    np.fill_diagonal(tiled, 0.0)
    return _row_normalize(tiled)


def get_template_for_job(
    proxy_app: str,
    num_ranks: int,
    *,
    matrix_dir: str | os.PathLike | None = None,
) -> np.ndarray:
    """Convenience: pick the best base template for ``proxy_app`` and tile it
    to ``num_ranks`` with modulo mapping."""
    base = load_base_template(proxy_app, matrix_dir=matrix_dir)
    return tile_template(base, num_ranks)


def list_available_apps(matrix_dir: str | os.PathLike | None = None) -> dict[str, list[int]]:
    """Return {app: sorted_base_node_counts} for every discoverable template."""
    matrix_dir = Path(matrix_dir) if matrix_dir is not None else DEFAULT_MATRIX_DIR
    out: dict[str, set[int]] = {}
    for app, base, _ in _list_templates(matrix_dir):
        out.setdefault(app, set()).add(base)
    return {k: sorted(v) for k, v in out.items()}
