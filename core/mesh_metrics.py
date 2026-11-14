"""
core/mesh_metrics.py
====================
メッシュの品質を数字にする純粋関数。「良くなったか」を、目で見ずに比べるために使う。

- ``chamfer_distance``  : 2 つのメッシュの表面の、平均の近さ(対角線で割った相対値)
- ``mesh_quality``      : 水密か・部品の数・浮島・退化面・体積・表面積

依存: numpy, scipy, trimesh
"""

from __future__ import annotations

import numpy as np


def _diagonal(mesh) -> float:
    return float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])) or 1.0


def chamfer_distance(mesh_a, mesh_b, n_samples: int = 20_000, seed: int = 0) -> dict:
    """表面の点を取って、双方向の最近傍距離を測る。

    Returns
    -------
    dict
        ``chamfer``  : 平均(2 方向の平均)を、基準メッシュ(b)の対角線で割った相対値。小さいほど近い。
        ``hausdorff``: 最大(2 方向のうち大きい方)を、同じく相対値にしたもの。
    """
    from scipy.spatial import cKDTree

    rng_state = np.random.get_state()
    try:
        np.random.seed(seed)
        pa = mesh_a.sample(n_samples)
        pb = mesh_b.sample(n_samples)
    finally:
        np.random.set_state(rng_state)

    d_ab = cKDTree(pb).query(pa)[0]
    d_ba = cKDTree(pa).query(pb)[0]
    diag = _diagonal(mesh_b)
    return {
        "chamfer": float((d_ab.mean() + d_ba.mean()) / 2.0 / diag),
        "hausdorff": float(max(d_ab.max(), d_ba.max()) / diag),
    }


def mesh_quality(mesh, floater_ratio: float = 0.02) -> dict:
    """メッシュの状態を数字にする。

    Returns
    -------
    dict
        ``watertight``       : 閉じているか
        ``components``       : 連結成分の数
        ``floaters``         : 最大の成分の ``floater_ratio`` 未満の面数しか持たない成分の数(浮島)
        ``degenerate_faces`` : 面積がほぼ 0 の面の数
        ``boundary_edges``   : 穴の縁の辺の数(0 なら穴がない)
        ``volume`` / ``area``: 体積(符号なし)・表面積
    """
    import trimesh

    parts = mesh.split(only_watertight=False)
    sizes = sorted((len(p.faces) for p in parts), reverse=True) or [0]
    floaters = sum(1 for s in sizes[1:] if s < sizes[0] * floater_ratio)

    tri = mesh.triangles
    area = trimesh.triangles.area(tri) if len(tri) else np.array([])
    degenerate = int((area < 1e-12).sum())

    edges = np.sort(mesh.edges, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return {
        "watertight": bool(mesh.is_watertight),
        "components": len(parts),
        "floaters": int(floaters),
        "degenerate_faces": degenerate,
        "boundary_edges": int((counts == 1).sum()),
        "volume": float(abs(mesh.volume)) if mesh.is_watertight else float("nan"),
        "area": float(mesh.area),
    }
