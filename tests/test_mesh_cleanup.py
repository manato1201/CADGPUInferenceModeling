"""推論後の整形(clean_inference_mesh)と、品質の指標(mesh_metrics)の検査。

GPU の推論はここでは動かさない。代わりに、マーチングキューブの出力に似た「ノイズ・浮島・穴」を人工的に作り、
整形で指標が良くなることを確かめる。
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.mesh_metrics import chamfer_distance, mesh_quality  # noqa: E402
from core.postprocessor import clean_inference_mesh  # noqa: E402


def _sphere():
    return trimesh.creation.icosphere(subdivisions=3, radius=1.0)


def test_metrics_of_identical_meshes_are_zero():
    s = _sphere()
    d = chamfer_distance(s, s.copy())
    assert d["chamfer"] < 5e-3  # 同じ表面から別々にサンプルするので、ゼロにはならない


def test_chamfer_grows_with_distance():
    a, b = _sphere(), _sphere()
    b.apply_translation((0.2, 0, 0))
    assert chamfer_distance(a, b)["chamfer"] > chamfer_distance(a, a.copy())["chamfer"] * 5


def test_quality_counts_floaters_and_holes():
    s = _sphere()
    speck = trimesh.creation.icosphere(subdivisions=0, radius=0.05)
    speck.apply_translation((3, 0, 0))
    q = mesh_quality(trimesh.util.concatenate([s, speck]))
    assert q["components"] == 2 and q["floaters"] == 1

    holed = s.copy()
    holed.update_faces(~(holed.faces == 0).any(axis=1))
    qh = mesh_quality(holed)
    assert not qh["watertight"] and qh["boundary_edges"] > 0


def test_cleanup_removes_floaters_and_fills_holes():
    s = _sphere()
    s.update_faces(~(s.faces == 0).any(axis=1))  # 頂点 0 のまわりの面を抜く(きれいな多角形の穴)
    speck = trimesh.creation.icosphere(subdivisions=0, radius=0.05)
    speck.apply_translation((3, 0, 0))
    dirty = trimesh.util.concatenate([s, speck])
    before = mesh_quality(dirty)
    clean = clean_inference_mesh(dirty.copy())
    after = mesh_quality(clean)
    assert before["floaters"] == 1 and after["floaters"] == 0
    assert after["watertight"] and after["boundary_edges"] == 0
    assert abs(after["volume"] - 4 / 3 * np.pi) / (4 / 3 * np.pi) < 0.05


def test_cleanup_keeps_a_real_second_part():
    # 浮島ではない大きさの部品は、残す
    a = trimesh.creation.box(extents=(1, 1, 1))
    b = trimesh.creation.box(extents=(0.8, 0.8, 0.8))
    b.apply_translation((3, 0, 0))
    out = clean_inference_mesh(trimesh.util.concatenate([a, b]))
    assert mesh_quality(out)["components"] == 2


def test_taubin_smoothing_reduces_noise_without_shrinking_the_volume():
    rng = np.random.default_rng(0)
    truth = _sphere()
    noisy = truth.copy()
    noisy.vertices = truth.vertices * (1.0 + rng.normal(0, 0.02, (len(truth.vertices), 1)))
    base = chamfer_distance(noisy, truth)["chamfer"]
    smooth = clean_inference_mesh(noisy.copy(), smooth_iterations=10)
    better = chamfer_distance(smooth, truth)["chamfer"]
    assert better < base * 0.9, (base, better)
    assert abs(abs(smooth.volume) - abs(truth.volume)) / abs(truth.volume) < 0.02, "体積が縮んでいる"


def test_a_large_opening_is_left_open():
    # 底が開いた半球の底を、でたらめな扇形で塞がない
    s = _sphere()
    s.update_faces(s.triangles_center[:, 2] > 0)
    out = clean_inference_mesh(s.copy())
    assert not out.is_watertight
