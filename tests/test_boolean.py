"""ブーリアン演算(開口の切り抜き・Union)の検査。

trimesh 4.0.x の "manifold" エンジンは manifold3d 3.x と合わず、毎回失敗して Blender を起動し、
しかも Blender の戻り値(Scene)を扱えず、開口が切り抜かれないまま約 110 秒かかっていた。
manifold3d を直接呼ぶ ``_boolean`` が、速く、正しく切ることを守る。
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import floor_plan_extruder as fe  # noqa: E402


def _no_fallback(monkeypatch):
    """trimesh のエンジン(Blender)に戻ったら、失敗にする。"""
    def boom(*a, **k):
        raise AssertionError("trimesh のエンジンに戻った(manifold3d の直接呼び出しが失敗している)")
    monkeypatch.setattr(trimesh.boolean, "difference", boom)
    monkeypatch.setattr(trimesh.boolean, "union", boom)


def test_difference_removes_exactly_the_cutter_volume(monkeypatch):
    _no_fallback(monkeypatch)
    wall = trimesh.creation.box(extents=(2.0, 0.15, 2.5))
    cutter = trimesh.creation.box(extents=(0.5, 0.4, 1.0))  # 壁の厚みを貫く
    out = fe._boolean("difference", [wall, cutter])
    assert out is not None
    assert abs(out.volume) == pytest.approx(2.0 * 0.15 * 2.5 - 0.5 * 0.15 * 1.0, rel=1e-6)


def test_difference_accepts_inside_out_meshes(monkeypatch):
    # 押し出しとカッターは面の向きが裏返しで作られる(volume < 0)ことがある
    _no_fallback(monkeypatch)
    wall = trimesh.creation.box(extents=(2.0, 0.15, 2.5))
    cutter = trimesh.creation.box(extents=(0.5, 0.4, 1.0))
    wall.faces = wall.faces[:, ::-1]
    cutter.faces = cutter.faces[:, ::-1]
    assert wall.volume < 0 and cutter.volume < 0
    out = fe._boolean("difference", [wall, cutter])
    assert abs(out.volume) == pytest.approx(0.75 - 0.075, rel=1e-6)


def test_union_does_not_double_count_the_overlap(monkeypatch):
    _no_fallback(monkeypatch)
    a = trimesh.creation.box(extents=(1, 1, 1))
    b = trimesh.creation.box(extents=(1, 1, 1))
    b.apply_translation((0.5, 0, 0))
    out = fe._boolean("union", [a, b])
    assert out.volume == pytest.approx(1.5, rel=1e-6)


def test_sample_plan_gets_its_openings_cut_and_is_fast():
    sample = ROOT / "2DDXF_Sample.dxf"
    if not sample.exists():
        pytest.skip("サンプルの DXF が無い")
    common = dict(ceiling_height=2500.0, default_height=2500.0, floor_thickness=200.0,
                  wall_thickness=150.0, wall_snap_tol=50.0, use_union=False)
    t0 = time.perf_counter()
    cut = fe.extrude_floor_plan(str(sample), config=fe.ExtrusionConfig(cut_openings=True, **common))
    elapsed = time.perf_counter() - t0
    uncut = fe.extrude_floor_plan(str(sample), config=fe.ExtrusionConfig(cut_openings=False, **common))
    assert abs(cut.volume) < abs(uncut.volume) - 1.0, "開口が切り抜かれていない"
    assert elapsed < 30, f"遅い({elapsed:.1f} 秒)。Blender に戻っていないか"
