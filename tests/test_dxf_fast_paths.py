"""mesh_to_dxf の高速化(ベクトル化した可視判定・LINE の一括書き出し)が、遅い実装と同じ結果を返すことを守る。"""

import sys
from pathlib import Path

import ezdxf
import numpy as np
import pytest
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import mesh_to_dxf as m2d  # noqa: E402


def _edge_dict(faces):
    """従来の実装(dict)。順序は最初に現れた順、向きは最後に現れたときの向き。"""
    d = {}
    for face in faces:
        for i in range(3):
            a, b = face[i], face[(i + 1) % 3]
            d[(min(a, b), max(a, b))] = (a, b)
    return list(d.values())


def test_unique_edges_matches_the_dict_version():
    rng = np.random.default_rng(1)
    faces = rng.integers(0, 40, size=(300, 3))
    expected = _edge_dict(faces)
    got = [tuple(e) for e in m2d._unique_edges(faces)]
    assert got == expected


def test_batch_visibility_matches_the_per_edge_function():
    rng = np.random.default_rng(2)
    size = 64
    z_buf = rng.random((size, size)).astype(np.float32)
    z_buf[rng.random((size, size)) < 0.2] = np.inf
    params = (0.0, 1.0, 0.0, 1.0, 50.0, 50.0, 3.0, 3.0, size)
    n = 500
    x0, y0, x1, y1 = (rng.random(n) * 1.2 - 0.1 for _ in range(4))  # 範囲の外も含める(clip の確認)
    d0, d1 = rng.random(n), rng.random(n)
    batch = m2d._edge_visibility_zbuffer_batch(x0, y0, x1, y1, d0, d1, z_buf, params)
    for i in range(n):
        one = m2d._edge_visibility_zbuffer(x0[i], y0[i], x1[i], y1[i], d0[i], d1[i], z_buf, params) == "visible"
        assert bool(batch[i]) == one, f"edge {i}"


def _lines(path):
    doc = ezdxf.readfile(path)  # 厳密に読む(壊れた DXF なら、ここで例外)
    msp = doc.modelspace()
    lines = [e for e in msp if e.dxftype() == "LINE"]
    arr = np.array([[e.dxf.start.x, e.dxf.start.y, e.dxf.end.x, e.dxf.end.y] for e in lines])
    layers = sorted(e.dxf.layer for e in lines)
    handles = [e.dxf.handle for e in msp]
    return arr, layers, handles


def test_bulk_lines_save_the_same_dxf_as_add_line(tmp_path, monkeypatch):
    mesh = trimesh.creation.icosphere(subdivisions=2)
    cfg = m2d.MeshToDxfConfig(mode="triview")
    src = tmp_path / "m.obj"
    mesh.export(src)

    fast = tmp_path / "fast.dxf"
    m2d.mesh_to_dxf(src, fast, config=cfg)

    # 比較用: 一括書き出しを使わず、1 本ずつ add_line する
    monkeypatch.setattr(m2d, "_BulkLines", type("NoBulk", (), {
        "__init__": lambda self: None, "add": lambda *a: None, "__len__": lambda self: 0}))
    orig_add_edges = m2d._add_edges
    monkeypatch.setattr(m2d, "_add_edges", lambda msp, bulk, *a: orig_add_edges(msp, None, *a))
    slow = tmp_path / "slow.dxf"
    m2d.mesh_to_dxf(src, slow, config=cfg)

    a, la, ha = _lines(fast)
    b, lb, hb = _lines(slow)
    assert la == lb and len(a) == len(b) > 100
    assert np.allclose(a[np.lexsort(a.T)], b[np.lexsort(b.T)])
    assert len(set(ha)) == len(ha), "ハンドルが重なっている"


def test_the_written_dxf_passes_the_ezdxf_audit(tmp_path):
    mesh = trimesh.creation.icosphere(subdivisions=2)
    src = tmp_path / "m.obj"
    mesh.export(src)
    out = tmp_path / "o.dxf"
    m2d.mesh_to_dxf(src, out, config=m2d.MeshToDxfConfig(mode="triview"))
    doc = ezdxf.readfile(out)
    auditor = doc.audit()
    assert not auditor.has_errors, [e.message for e in auditor.errors]
