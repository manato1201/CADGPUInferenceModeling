"""
core/roof_generator.py
======================
押し出し3Dメッシュの天井面から屋根メッシュを自動生成する。

対応屋根タイプ:
  flat      陸屋根（パラペット付き平屋根）
  shed      片流れ屋根（一方向に傾斜）
  gable     切妻屋根（両側に傾斜・最も一般的な住宅屋根）

処理フロー:
  1. 天井面（Z=ceiling）の輪郭ポリゴンを抽出
  2. 指定タイプで屋根ジオメトリを生成
  3. trimesh.Trimesh として返す

依存: trimesh, numpy, shapely
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)

RoofType = Literal["flat", "shed", "gable"]


# ─────────────────────────────────────────────
# 設定
# ─────────────────────────────────────────────

@dataclass
class RoofConfig:
    """屋根生成設定。"""
    roof_type: RoofType = "gable"       # 屋根タイプ
    ridge_height: float = 1.5           # 棟の高さ [m]（flat以外）
    overhang: float = 0.3               # 軒の出 [m]（外壁からの張り出し）
    parapet_height: float = 0.2         # パラペット高さ [m]（flat のみ）
    thickness: float = 0.15             # 屋根スラブ厚 [m]


# ─────────────────────────────────────────────
# 天井輪郭の抽出
# ─────────────────────────────────────────────

def _extract_ceiling_footprint(mesh, z_tol: float = 0.1):
    """
    天井面（Z最大付近）の輪郭ポリゴンを shapely で返す。
    """
    from shapely.geometry import MultiPolygon, Polygon
    from shapely.ops import unary_union

    verts = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    z_max = verts[:, 2].max()

    # 天井面の三角形を抽出
    face_z = verts[faces].mean(axis=1)[:, 2]
    top_mask = face_z > (z_max - z_tol)
    top_faces = faces[top_mask]

    if len(top_faces) == 0:
        logger.warning("天井面が検出できませんでした。バウンディングボックスを使用します。")
        xmin, xmax = verts[:, 0].min(), verts[:, 0].max()
        ymin, ymax = verts[:, 1].min(), verts[:, 1].max()
        return Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])

    # 三角形を XY 平面ポリゴンに変換して Union
    polys = []
    for tri in top_faces:
        pts = verts[tri][:, :2]
        try:
            p = Polygon(pts)
            if p.is_valid and p.area > 1e-6:
                polys.append(p)
        except Exception:
            continue

    if not polys:
        xmin, xmax = verts[:, 0].min(), verts[:, 0].max()
        ymin, ymax = verts[:, 1].min(), verts[:, 1].max()
        return Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])

    union = unary_union(polys)
    if union.geom_type == "MultiPolygon":
        union = max(union.geoms, key=lambda p: p.area)
    return union


# ─────────────────────────────────────────────
# ポリゴン → 押し出しスラブ
# ─────────────────────────────────────────────

def _poly_to_slab(
    poly,
    z_bottom: float,
    z_top: float,
    include_sides: bool = True,
) -> object:
    """shapely Polygon → 薄いスラブ trimesh。"""
    import trimesh
    from shapely.geometry import Polygon

    coords = list(poly.exterior.coords)
    if coords[0] == coords[-1]:
        coords = coords[:-1]
    n = len(coords)
    if n < 3:
        return None

    verts_b = [[x, y, z_bottom] for x, y in coords]
    verts_t = [[x, y, z_top]    for x, y in coords]
    verts = np.array(verts_b + verts_t, dtype=np.float64)

    faces = []
    # 上面（fan）
    for i in range(1, n - 1):
        faces.append([n, n + i, n + i + 1])
    # 下面（反転）
    for i in range(1, n - 1):
        faces.append([0, i + 1, i])
    # 側面
    if include_sides:
        for i in range(n):
            j = (i + 1) % n
            faces.append([i, j, i + n])
            faces.append([j, j + n, i + n])

    return trimesh.Trimesh(
        vertices=verts,
        faces=np.array(faces, dtype=np.int64),
        process=False,
    )


# ─────────────────────────────────────────────
# 屋根タイプ別生成
# ─────────────────────────────────────────────

def _gen_flat_roof(footprint, z_ceiling: float, cfg: RoofConfig):
    """陸屋根（パラペット付き）を生成する。"""
    parts = []
    # 屋根スラブ本体
    expanded = footprint.buffer(cfg.overhang)
    slab = _poly_to_slab(expanded, z_ceiling, z_ceiling + cfg.thickness)
    if slab is not None:
        parts.append(slab)

    # パラペット（外周の立ち上がり）
    inner = footprint.buffer(cfg.overhang - 0.1)
    outer = footprint.buffer(cfg.overhang + 0.1)
    parapet_ring = outer.difference(inner)
    if not parapet_ring.is_empty:
        if parapet_ring.geom_type == "MultiPolygon":
            geoms = list(parapet_ring.geoms)
        else:
            geoms = [parapet_ring]
        for g in geoms:
            p = _poly_to_slab(g, z_ceiling, z_ceiling + cfg.parapet_height)
            if p is not None:
                parts.append(p)

    return parts


def _gen_shed_roof(footprint, z_ceiling: float, cfg: RoofConfig):
    """片流れ屋根を生成する。"""
    import trimesh

    expanded = footprint.buffer(cfg.overhang)
    coords = list(expanded.exterior.coords)
    if coords[0] == coords[-1]:
        coords = coords[:-1]
    n = len(coords)

    # Y座標で高低を決定（大きいY側が高い）
    ys = [c[1] for c in coords]
    y_min, y_max = min(ys), max(ys)
    y_range = max(y_max - y_min, 1e-3)

    verts = []
    for x, y in coords:
        t = (y - y_min) / y_range
        z = z_ceiling + t * cfg.ridge_height
        verts.append([x, y, z])
    verts = np.array(verts, dtype=np.float64)

    faces = []
    for i in range(1, n - 1):
        faces.append([0, i, i + 1])
    # 裏面
    for i in range(1, n - 1):
        faces.append([0, i + 1, i])

    mesh = trimesh.Trimesh(
        vertices=verts,
        faces=np.array(faces, dtype=np.int64),
        process=False,
    )
    mesh.fix_normals()
    return [mesh]


def _gen_gable_roof(footprint, z_ceiling: float, cfg: RoofConfig):
    """
    切妻屋根を生成する。

    バウンディングボックスの長軸方向に棟を通し、
    両側に傾斜面を生成する。
    """
    from shapely.geometry import LineString, Polygon
    import trimesh

    expanded = footprint.buffer(cfg.overhang)
    bounds = expanded.bounds   # (xmin, ymin, xmax, ymax)
    xmin, ymin, xmax, ymax = bounds
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    width  = xmax - xmin   # X方向
    depth  = ymax - ymin   # Y方向

    z_ridge = z_ceiling + cfg.ridge_height

    if width >= depth:
        # X方向が長い → 棟はX方向（Y中心に棟）
        ridge_pts = [(xmin - cfg.overhang, cy, z_ridge),
                     (xmax + cfg.overhang, cy, z_ridge)]
        # 正面・背面の妻壁
        front_pts = [(xmin - cfg.overhang, ymin, z_ceiling),
                     (xmax + cfg.overhang, ymin, z_ceiling)]
        back_pts  = [(xmin - cfg.overhang, ymax, z_ceiling),
                     (xmax + cfg.overhang, ymax, z_ceiling)]
    else:
        # Y方向が長い → 棟はY方向（X中心に棟）
        ridge_pts = [(cx, ymin - cfg.overhang, z_ridge),
                     (cx, ymax + cfg.overhang, z_ridge)]
        front_pts = [(xmin, ymin - cfg.overhang, z_ceiling),
                     (xmin, ymax + cfg.overhang, z_ceiling)]
        back_pts  = [(xmax, ymin - cfg.overhang, z_ceiling),
                     (xmax, ymax + cfg.overhang, z_ceiling)]

    # 屋根面：前側スロープ・後ろ側スロープ
    parts = []
    r0, r1 = np.array(ridge_pts[0]), np.array(ridge_pts[1])
    f0, f1 = np.array(front_pts[0]), np.array(front_pts[1])
    b0, b1 = np.array(back_pts[0]),  np.array(back_pts[1])

    for face_verts in [
        [f0, f1, r1, r0],   # 前側（F→R）
        [r0, r1, b1, b0],   # 後側（R→B）
    ]:
        verts = np.array(face_verts, dtype=np.float64)
        faces_idx = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        m = trimesh.Trimesh(vertices=verts, faces=faces_idx, process=False)
        m.fix_normals()
        parts.append(m)

    # 屋根スラブ厚（薄い層）
    for face_verts in [
        [f0, f1, r1, r0],
        [r0, r1, b1, b0],
    ]:
        pts = np.array(face_verts, dtype=np.float64)
        # 法線方向に少し下げる
        normal = np.array([0, 0, -cfg.thickness])
        pts_b = pts + normal
        verts_all = np.vstack([pts, pts_b])
        faces_idx = np.array([
            [0,1,2],[0,2,3],   # 上面
            [4,6,5],[4,7,6],   # 下面
            [0,4,5],[0,5,1],   # 側面
            [2,6,7],[2,7,3],
            [0,3,7],[0,7,4],
            [1,5,6],[1,6,2],
        ], dtype=np.int64)
        # 重複するので上面のみ使う
        m = trimesh.Trimesh(vertices=pts, faces=np.array([[0,1,2],[0,2,3]]), process=False)
        m.fix_normals()

    return parts


# ─────────────────────────────────────────────
# メイン API
# ─────────────────────────────────────────────

def generate_roof(
    wall_mesh,
    config: Optional[RoofConfig] = None,
) -> object:  # trimesh.Trimesh
    """
    押し出し壁メッシュから屋根メッシュを生成して返す。

    Parameters
    ----------
    wall_mesh : trimesh.Trimesh（押し出し済み）
    config    : RoofConfig（省略時はデフォルト）

    Returns
    -------
    trimesh.Trimesh  屋根メッシュ（壁メッシュとは別オブジェクト）
    """
    import trimesh

    cfg = config or RoofConfig()
    verts = np.array(wall_mesh.vertices)
    z_ceiling = verts[:, 2].max()
    logger.info(f"屋根生成: type={cfg.roof_type}, z_ceiling={z_ceiling:.2f}m, "
                f"ridge_height={cfg.ridge_height:.2f}m")

    footprint = _extract_ceiling_footprint(wall_mesh)
    logger.info(f"天井輪郭: area={footprint.area:.2f}m²")

    if cfg.roof_type == "flat":
        parts = _gen_flat_roof(footprint, z_ceiling, cfg)
    elif cfg.roof_type == "shed":
        parts = _gen_shed_roof(footprint, z_ceiling, cfg)
    elif cfg.roof_type == "gable":
        parts = _gen_gable_roof(footprint, z_ceiling, cfg)
    else:
        raise ValueError(f"未知の屋根タイプ: {cfg.roof_type}")

    parts = [p for p in parts if p is not None and len(p.faces) > 0]
    if not parts:
        logger.warning("屋根メッシュが生成できませんでした。")
        return None

    roof = trimesh.util.concatenate(parts)
    roof.merge_vertices()
    roof.fix_normals()
    logger.info(f"屋根完成: verts={len(roof.vertices)}, faces={len(roof.faces)}")
    return roof


def attach_roof(wall_mesh, config: Optional[RoofConfig] = None) -> object:
    """
    壁メッシュと屋根メッシュを結合して返す。
    """
    import trimesh

    roof = generate_roof(wall_mesh, config)
    if roof is None:
        return wall_mesh
    combined = trimesh.util.concatenate([wall_mesh, roof])
    combined.merge_vertices()
    combined.fix_normals()
    return combined
