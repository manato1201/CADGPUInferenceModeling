"""
core/parser.py
==============
DXF/DWG ファイルを読み込み、三面図画像・メタデータ・メッシュを返す純粋関数群。

DXFの種類を自動判定して2ルートに分岐する:

  Route A  2D図面 (LINE/ARC/CIRCLE 等、Z=0)
             → 三面図画像を生成して Zero123++ 推論へ

  Route B  3D POLYFACEメッシュ (Z値あり、メッシュが既に定義済み)
             → メッシュを直接抽出して推論をスキップ

依存: ezdxf, matplotlib, numpy, Pillow, trimesh
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import ezdxf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 定数
# ─────────────────────────────────────────────

_INSUNITS_TO_MM: dict[int, float] = {
    0: 1.0,     # 未定義
    1: 25.4,    # inch
    2: 304.8,   # foot
    4: 1.0,     # mm
    5: 10.0,    # cm
    6: 1000.0,  # m
}

_INSUNITS_NAME: dict[int, str] = {
    1: "inch", 2: "foot", 4: "mm", 5: "cm", 6: "m",
}

# POLYFACE MESH フラグ (DXF仕様: bit6)
_POLYFACE_FLAG = 0x40


# ─────────────────────────────────────────────
# データ型定義
# ─────────────────────────────────────────────

@dataclass
class CADMeta:
    """CAD図面から抽出したメタデータ。"""
    source_path: str
    layers: list[str] = field(default_factory=list)
    bbox_min: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bbox_max: np.ndarray = field(default_factory=lambda: np.zeros(3))
    unit: str = "mm"
    entity_count: int = 0
    # 判定結果: "2d" / "3d_polyface" / "floor_plan"
    dxf_type: str = "2d"

    @property
    def dimensions(self) -> np.ndarray:
        """X/Y/Z の寸法を返す（unit に応じた座標系）。"""
        return self.bbox_max - self.bbox_min

    @property
    def aspect_ratio(self) -> tuple[float, float, float]:
        d = self.dimensions
        m = d.max() if d.max() > 0 else 1.0
        return tuple((d / m).tolist())


@dataclass
class CADParseResult:
    """パース結果をまとめて保持する。"""
    meta: CADMeta
    # Route A: 三面図画像
    views: dict[str, Image.Image] = field(default_factory=dict)
    # Route B: 直接抽出したメッシュ (trimesh.Trimesh)
    mesh: Optional[object] = None
    # 点群（共通）
    point_cloud: Optional[np.ndarray] = None

    @property
    def is_3d(self) -> bool:
        """POLYFACEメッシュが直接抽出できた場合 True。"""
        return self.mesh is not None

    @property
    def is_floor_plan(self) -> bool:
        """建築平面図として判定された場合 True。"""
        return self.meta.dxf_type == "floor_plan"


# ─────────────────────────────────────────────
# ユーティリティ
# ─────────────────────────────────────────────

def _get_unit(doc: ezdxf.document.Drawing) -> tuple[float, str]:
    code = doc.header.get("$INSUNITS", 4)
    scale = _INSUNITS_TO_MM.get(code, 1.0)
    name  = _INSUNITS_NAME.get(code, "mm")
    return scale, name


def _has_polyface(msp) -> bool:
    """モデルスペースに POLYFACE MESH が存在するか判定。"""
    for e in msp:
        if e.dxftype() == "POLYLINE" and (e.dxf.get("flags", 0) & _POLYFACE_FLAG):
            vlist = list(e.vertices)
            if vlist:
                # 頂点がZ値を持っていれば3Dメッシュと確定
                try:
                    z_vals = [v.dxf.location.z for v in vlist[:20]]
                    if max(abs(z) for z in z_vals) > 1e-6:
                        return True
                except Exception:
                    pass
    return False


def _is_floor_plan(msp, layer_cats: dict) -> bool:
    """
    建築平面図かどうかを判定する。
    "wall" カテゴリのレイヤーが存在し、かつ Z=0 の2D図面であれば True。
    """
    has_wall = any(cat == "wall" for cat in layer_cats.values())
    if not has_wall:
        return False
    # Z値がすべて0かチェック（サンプリング）
    z_vals = []
    count = 0
    for e in msp:
        if count > 50:
            break
        try:
            if e.dxftype() == "LINE":
                z_vals.append(abs(e.dxf.start.z) + abs(e.dxf.end.z))
                count += 1
        except Exception:
            continue
    if not z_vals:
        return has_wall
    return has_wall and max(z_vals) < 1.0  # Z=0の2D図面


# ─────────────────────────────────────────────
# Route B: POLYFACE直接抽出
# ─────────────────────────────────────────────

def _extract_polyface_mesh(msp, scale_to_m: float):
    """
    POLYFACE MESH を頂点・面リストに変換して trimesh.Trimesh を返す。

    DXF POLYFACE の頂点フラグ:
      bit7 (0x80) が立っている → face record (vtx0〜3 がインデックス定義)
      立っていない           → 座標頂点
      両方立っている (0xC0)  → car.dxf のように全頂点が面定義を兼ねる場合は
                               グリッド面 (m×n) で面を生成する
    """
    import trimesh

    all_verts: list[list[float]] = []
    all_faces: list[list[int]]   = []

    for poly in msp:
        if poly.dxftype() != "POLYLINE":
            continue
        if not (poly.dxf.get("flags", 0) & _POLYFACE_FLAG):
            continue

        vlist = list(poly.vertices)
        if not vlist:
            continue

        offset = len(all_verts)

        # ── ① 通常 POLYFACE: geo頂点 と face record を分離 ──
        geo_verts: list[list[float]] = []
        face_recs:  list[list[int]]  = []
        for v in vlist:
            vf = v.dxf.get("flags", 0)
            if vf & 0x80:
                fi = []
                for attr in ("vtx0", "vtx1", "vtx2", "vtx3"):
                    idx = v.dxf.get(attr, 0)
                    if idx != 0:
                        fi.append(abs(idx) - 1)
                face_recs.append(fi)
            else:
                loc = v.dxf.location
                geo_verts.append([loc.x, loc.y, loc.z])

        # ── ② 全頂点が 0xC0 (geo+face 兼用) → m×n グリッドで面生成 ──
        if not geo_verts:
            geo_verts = [
                [v.dxf.location.x, v.dxf.location.y, v.dxf.location.z]
                for v in vlist
            ]
            m   = poly.dxf.get("m_count", 0)
            n_c = poly.dxf.get("n_count", 0)
            total = len(geo_verts)
            if m > 0 and n_c > 0 and m * n_c <= total:
                for i in range(m - 1):
                    for j in range(n_c - 1):
                        a = offset + i * n_c + j
                        b = offset + i * n_c + (j + 1)
                        c = offset + (i + 1) * n_c + (j + 1)
                        d = offset + (i + 1) * n_c + j
                        if max(a, b, c, d) < offset + total:
                            all_faces.append([a, b, c])
                            all_faces.append([a, c, d])

        all_verts.extend(geo_verts)

        # ── ③ face records から三角形・四角形を追加 ──
        for fi in face_recs:
            fi_off = [i + offset for i in fi]
            if len(fi_off) == 3:
                all_faces.append(fi_off)
            elif len(fi_off) >= 4:
                all_faces.append([fi_off[0], fi_off[1], fi_off[2]])
                all_faces.append([fi_off[0], fi_off[2], fi_off[3]])

    if not all_verts or not all_faces:
        return None

    verts = np.array(all_verts, dtype=np.float64) * scale_to_m
    faces = np.array(all_faces, dtype=np.int64)

    # インデックス範囲チェック
    valid = (faces.max(axis=1) < len(verts)) & (faces.min(axis=1) >= 0)
    faces = faces[valid]
    if len(faces) == 0:
        return None

    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.merge_vertices()
    mesh.fix_normals()

    logger.info(
        f"POLYFACE抽出完了: verts={len(mesh.vertices)}, faces={len(mesh.faces)}  "
        f"bbox X={verts[:,0].min():.3f}~{verts[:,0].max():.3f}m  "
        f"Y={verts[:,1].min():.3f}~{verts[:,1].max():.3f}m  "
        f"Z={verts[:,2].min():.3f}~{verts[:,2].max():.3f}m"
    )
    return mesh


# ─────────────────────────────────────────────
# Route A: 2D図面用 点群・セグメント収集
# ─────────────────────────────────────────────

def _collect_points(msp, scale: float) -> np.ndarray:
    pts: list[np.ndarray] = []
    for e in msp:
        dxftype = e.dxftype()
        try:
            if dxftype == "LINE":
                pts += [np.array([e.dxf.start.x, e.dxf.start.y, e.dxf.start.z]),
                        np.array([e.dxf.end.x,   e.dxf.end.y,   e.dxf.end.z])]
            elif dxftype == "LWPOLYLINE":
                for x, y, *_ in e.get_points():
                    pts.append(np.array([x, y, 0.0]))
            elif dxftype in ("CIRCLE", "ARC"):
                cx, cy, cz = e.dxf.center.x, e.dxf.center.y, e.dxf.center.z
                r = e.dxf.radius
                for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                    pts.append(np.array([cx + r * np.cos(a), cy + r * np.sin(a), cz]))
            elif dxftype == "SPLINE":
                for p in e.control_points:
                    pts.append(np.array([p.x, p.y, p.z]))
            elif dxftype == "POLYLINE":
                for v in e.vertices:
                    loc = v.dxf.location
                    pts.append(np.array([loc.x, loc.y, loc.z]))
        except Exception:
            continue
    if not pts:
        return np.zeros((1, 3))
    return np.array(pts, dtype=np.float64) * scale


def _collect_segments(msp, scale: float) -> list[tuple[np.ndarray, np.ndarray]]:
    segs: list[tuple[np.ndarray, np.ndarray]] = []
    for e in msp:
        dxftype = e.dxftype()
        try:
            if dxftype == "LINE":
                s = np.array([e.dxf.start.x, e.dxf.start.y, e.dxf.start.z]) * scale
                t = np.array([e.dxf.end.x,   e.dxf.end.y,   e.dxf.end.z])   * scale
                segs.append((s, t))

            elif dxftype == "LWPOLYLINE":
                verts = [(x * scale, y * scale) for x, y, *_ in e.get_points()]
                if getattr(e, "is_closed", False) and len(verts) > 1:
                    verts = verts + [verts[0]]
                for i in range(len(verts) - 1):
                    segs.append((np.array([verts[i][0],   verts[i][1],   0.0]),
                                 np.array([verts[i+1][0], verts[i+1][1], 0.0])))

            elif dxftype == "POLYLINE":
                # POLYFACEは三面図では輪郭エッジのみ抽出
                flags = e.dxf.get("flags", 0)
                if flags & _POLYFACE_FLAG:
                    # シルエット用: XY投影した全エッジを追加
                    vlist = list(e.vertices)
                    pts_p = [np.array([v.dxf.location.x, v.dxf.location.y, v.dxf.location.z]) * scale
                             for v in vlist]
                    m   = e.dxf.get("m_count", 0)
                    n_c = e.dxf.get("n_count", 0)
                    if m > 0 and n_c > 0 and m * n_c <= len(pts_p):
                        # グリッドエッジのみ（全面を描くと塗りつぶしになる）
                        for i in range(0, m, max(1, m // 20)):
                            for j in range(n_c - 1):
                                segs.append((pts_p[i*n_c+j], pts_p[i*n_c+j+1]))
                        for j in range(0, n_c, max(1, n_c // 20)):
                            for i in range(m - 1):
                                segs.append((pts_p[i*n_c+j], pts_p[(i+1)*n_c+j]))
                else:
                    pts_p = [np.array([v.dxf.location.x, v.dxf.location.y, v.dxf.location.z]) * scale
                             for v in e.vertices]
                    for i in range(len(pts_p) - 1):
                        segs.append((pts_p[i], pts_p[i+1]))

            elif dxftype == "CIRCLE":
                cx, cy, cz = e.dxf.center.x*scale, e.dxf.center.y*scale, e.dxf.center.z*scale
                r = e.dxf.radius * scale
                n = min(max(32, int(2*np.pi*r / max(scale*10, 1e-9))), 128)
                pts_c = [np.array([cx + r*np.cos(a), cy + r*np.sin(a), cz])
                         for a in np.linspace(0, 2*np.pi, n, endpoint=False)]
                for i in range(len(pts_c)):
                    segs.append((pts_c[i], pts_c[(i+1) % len(pts_c)]))

            elif dxftype == "ARC":
                cx, cy, cz = e.dxf.center.x*scale, e.dxf.center.y*scale, e.dxf.center.z*scale
                r = e.dxf.radius * scale
                a0, a1 = np.radians(e.dxf.start_angle), np.radians(e.dxf.end_angle)
                if a1 <= a0:
                    a1 += 2*np.pi
                pts_a = [np.array([cx + r*np.cos(a), cy + r*np.sin(a), cz])
                         for a in np.linspace(a0, a1, 32)]
                for i in range(len(pts_a) - 1):
                    segs.append((pts_a[i], pts_a[i+1]))

            elif dxftype == "SPLINE":
                try:
                    pts_s = [np.array([p.x, p.y, p.z]) * scale for p in e.flattening(0.01)]
                except Exception:
                    pts_s = [np.array([p.x, p.y, p.z]) * scale for p in e.control_points]
                for i in range(len(pts_s) - 1):
                    segs.append((pts_s[i], pts_s[i+1]))

            elif dxftype == "ELLIPSE":
                pts_e = [np.array([p.x, p.y, p.z]) * scale for p in e.flattening(0.01)]
                for i in range(len(pts_e) - 1):
                    segs.append((pts_e[i], pts_e[i+1]))

        except Exception:
            continue
    return segs


# ─────────────────────────────────────────────
# レンダリング
# ─────────────────────────────────────────────

def _render_view(
    segs: list[tuple[np.ndarray, np.ndarray]],
    pts: np.ndarray,
    axis_h: int,
    axis_v: int,
    image_size: int,
    margin: float = 0.08,
) -> Image.Image:
    """セグメントを指定軸に投影して PNG 画像を返す（Zero123++入力用）。"""
    dpi = 128
    fig, ax = plt.subplots(figsize=(image_size/dpi, image_size/dpi), dpi=dpi)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    if segs:
        lc = LineCollection(
            [[(s[axis_h], s[axis_v]), (t[axis_h], t[axis_v])] for s, t in segs],
            linewidths=3.0, colors="black", capstyle="round", joinstyle="round",
        )
        ax.add_collection(lc)

    if pts.shape[0] > 1:
        xmin, xmax = pts[:, axis_h].min(), pts[:, axis_h].max()
        ymin, ymax = pts[:, axis_v].min(), pts[:, axis_v].max()
        xpad = max(xmax - xmin, 1e-3) * margin
        ypad = max(ymax - ymin, 1e-3) * margin
        ax.set_xlim(xmin - xpad, xmax + xpad)
        ax.set_ylim(ymin - ypad, ymax + ypad)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    img = Image.open(buf).convert("L").resize((image_size, image_size), Image.LANCZOS)
    img = img.point(lambda x: 0 if x < 200 else 255, "L")
    return img.convert("RGBA")


# ─────────────────────────────────────────────
# 公開 API
# ─────────────────────────────────────────────

def parse_dxf(
    path: str | Path,
    image_size: int = 512,
) -> CADParseResult:
    """
    DXF を読み込んで CADParseResult を返す。

    POLYFACE MESH が検出された場合 (result.is_3d == True):
        result.mesh に trimesh.Trimesh が入る（単位: m）
        result.views には三面図のシルエット画像も入る（確認用）

    2D 図面の場合 (result.is_3d == False):
        result.views に三面図が入る → Zero123++ 推論へ渡す
        result.mesh は None
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"DXF file not found: {path}")

    doc = ezdxf.readfile(str(path))
    msp = doc.modelspace()
    scale, unit = _get_unit(doc)
    layers = [layer.dxf.name for layer in doc.layers]
    entity_count = len(list(msp))

    # ── 3D POLYFACE 検出 ──────────────────────
    from core.floor_plan_extruder import _classify_layers
    layer_cats = _classify_layers(doc)

    mesh = None
    dxf_type = "2d"
    if _has_polyface(msp):
        # mm → m 変換して抽出
        _to_m = {"mm": 1e-3, "cm": 1e-2, "m": 1.0, "inch": 25.4e-3, "foot": 0.3048}
        scale_to_m = _to_m.get(unit, 1e-3)
        mesh = _extract_polyface_mesh(msp, scale_to_m)
        if mesh is not None:
            dxf_type = "3d_polyface"
            logger.info("Route B: 3D POLYFACEメッシュとして処理します（推論スキップ可）")
    elif _is_floor_plan(msp, layer_cats):
        dxf_type = "floor_plan"
        logger.info("Route C: 建築平面図として処理します（押し出し変換）")

    # ── 点群・segs 収集（三面図生成用・共通） ──
    pts  = _collect_points(msp, scale)
    segs = _collect_segments(msp, scale)

    # bbox は POLYFACE の場合はメッシュから取る
    if mesh is not None:
        verts = np.array(mesh.vertices)
        # m → unit に戻して meta に持つ
        _to_mm = {"mm": 1.0, "cm": 0.1, "m": 1e-3, "inch": 1/25.4, "foot": 1/304.8}
        inv = 1.0 / _to_mm.get(unit, 1.0)
        bbox_min = verts.min(axis=0) * inv
        bbox_max = verts.max(axis=0) * inv
    else:
        bbox_min = pts.min(axis=0)
        bbox_max = pts.max(axis=0)

    meta = CADMeta(
        source_path=str(path),
        layers=layers,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        unit=unit,
        entity_count=entity_count,
        dxf_type=dxf_type,
    )

    views = {
        "front": _render_view(segs, pts, axis_h=0, axis_v=2, image_size=image_size),
        "side":  _render_view(segs, pts, axis_h=1, axis_v=2, image_size=image_size),
        "top":   _render_view(segs, pts, axis_h=0, axis_v=1, image_size=image_size),
    }

    return CADParseResult(meta=meta, views=views, mesh=mesh, point_cloud=pts)


def save_views(result: CADParseResult, output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = {}
    for name, img in result.views.items():
        p = output_dir / f"{name}.png"
        img.save(p)
        saved[name] = p
    return saved
