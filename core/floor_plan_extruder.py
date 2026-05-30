"""
core/floor_plan_extruder.py  v2
================================
建築平面図（2D DXF）→ 3D メッシュ変換モジュール（品質改善版）。

改善点:
  ① 端点スナップ       近接端点を統合して壁の隙間・浮きを解消
  ② 壁厚自動推定       平行線対を検出して実際の壁厚に合わせる
  ③ 開口部高さ分類     ドア（床〜2m）/ 窓（床から0.9m〜2m）を別高さで処理
  ④ 複数階の統合       1階/2階レイヤーを別高さに積み上げ
  ⑤ 天井スラブ追加     各階の天井を薄い板として生成

依存: ezdxf, trimesh, numpy, (scipy optional)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import ezdxf
import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 設定
# ─────────────────────────────────────────────

@dataclass
class ExtrusionConfig:
    ceiling_height: float  = 2500.0   # 階高 [mm]
    floor_thickness: float = 200.0    # 床スラブ厚 [mm]
    ceiling_thickness: float = 150.0  # 天井スラブ厚 [mm] ★新規
    wall_thickness: float  = 150.0    # 壁厚デフォルト [mm]
    min_wall_length: float = 300.0    # 壁最小長 [mm]
    max_wall_length: float = 15000.0  # 壁最大長 [mm]
    wall_snap_tol: float   = 80.0     # 端点スナップ許容距離 [mm] ★拡大
    default_height: Optional[float] = None
    use_union: bool = True            # デフォルトTrue ★変更
    cut_openings: bool = True
    opening_height: float = 2000.0    # ドア開口高さ [mm]
    window_sill: float  = 900.0       # 窓台高さ [mm] ★新規
    window_height: float = 1100.0     # 窓高さ [mm] ★新規
    # 複数階統合
    multi_floor: bool = True          # ★新規
    floor_height_step: float = 3000.0 # 階高ステップ（1階→2階）[mm] ★新規
    # 壁厚自動推定
    auto_wall_thickness: bool = True  # ★新規
    parallel_detect_tol: float = 30.0 # 平行線検出許容角度差 [deg] ★新規


# ─────────────────────────────────────────────
# レイヤー分類
# ─────────────────────────────────────────────

def _decode_layer(name: str) -> str:
    for method in [
        lambda n: n.encode("utf-8", "surrogateescape").decode("cp932"),
        lambda n: n.encode("latin1").decode("cp932"),
    ]:
        try:
            return method(name)
        except Exception:
            pass
    return name


def _cp932_moj(text: str) -> str:
    """CP932テキストをezdxf内部のcp1252 Mojibake文字列に変換。"""
    b = text.encode("cp932")
    result = []
    for byte in b:
        if 0x80 <= byte <= 0xFF:
            try:
                result.append(bytes([byte]).decode("cp1252"))
            except ValueError:
                result.append(chr(0xDC00 + byte))
        else:
            result.append(chr(byte))
    return "".join(result)


_WALL_MOJ      = [_cp932_moj(k) for k in ["壁"]]
_OPENING_MOJ   = [_cp932_moj(k) for k in ["建具", "ドア", "窓", "開口"]]
_FURNITURE_MOJ = [_cp932_moj(k) for k in ["家具", "設備"]]
_STAIR_MOJ     = [_cp932_moj(k) for k in ["階段"]]
_IGNORE_MOJ    = [_cp932_moj(k) for k in [
    "寸法", "文字", "部屋名", "注記", "法線", "三斜", "石材",
    "敷地", "道路", "隣地", "高低差", "間取",
]]
# 複数階判定用
_FLOOR1_MOJ    = [_cp932_moj(k) for k in ["1階", "一階"]]
_FLOOR2_MOJ    = [_cp932_moj(k) for k in ["2階", "二階"]]


def _raw_match(raw: str, patterns: list[str]) -> bool:
    return any(p and p in raw for p in patterns)


def _classify_layers(doc) -> dict[str, str]:
    """レイヤー名 → カテゴリ。floor番号も付与（例: "wall_1", "wall_2"）。"""
    cats: dict[str, str] = {}
    for layer in doc.layers:
        raw = layer.dxf.name
        d = _decode_layer(raw).lower()

        # 階数判定
        floor_num = ""
        if _raw_match(raw, _FLOOR2_MOJ) or "2階" in d or "2f" in d:
            floor_num = "_2"
        elif _raw_match(raw, _FLOOR1_MOJ) or "1階" in d or "1f" in d:
            floor_num = "_1"

        if "wall" in d or _raw_match(raw, _WALL_MOJ):
            cats[raw] = f"wall{floor_num}"
        elif any(k in d for k in ["door","window","opening"]) or _raw_match(raw, _OPENING_MOJ):
            cats[raw] = f"opening{floor_num}"
        elif any(k in d for k in ["furniture","equip"]) or _raw_match(raw, _FURNITURE_MOJ):
            cats[raw] = "furniture"
        elif "stair" in d or _raw_match(raw, _STAIR_MOJ):
            cats[raw] = f"stair{floor_num}"
        elif any(k in d for k in ["dim","text","anno"]) or _raw_match(raw, _IGNORE_MOJ):
            cats[raw] = "ignore"
        else:
            cats[raw] = f"other{floor_num}"
    return cats


# ─────────────────────────────────────────────
# 線分収集
# ─────────────────────────────────────────────

def _collect_segments(
    msp,
    layer_cats: dict[str, str],
    cfg: ExtrusionConfig,
    target_cats: set[str],
) -> list[tuple[float,float,float,float]]:
    segs = []
    for e in msp:
        cat = layer_cats.get(e.dxf.layer, "other")
        if cat not in target_cats:
            continue
        try:
            t = e.dxftype()
            cands = []
            if t == "LINE":
                cands.append((e.dxf.start.x, e.dxf.start.y,
                               e.dxf.end.x,   e.dxf.end.y))
            elif t in ("LWPOLYLINE", "POLYLINE"):
                if t == "LWPOLYLINE":
                    pts = [(x,y) for x,y,*_ in e.get_points()]
                    closed = getattr(e, "is_closed", False)
                else:
                    pts = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
                    closed = bool(e.dxf.get("flags",0) & 0x1)
                if closed and pts:
                    pts = pts + [pts[0]]
                for i in range(len(pts)-1):
                    cands.append((pts[i][0],pts[i][1],pts[i+1][0],pts[i+1][1]))
            elif t == "ARC":
                cx,cy = e.dxf.center.x, e.dxf.center.y
                r = e.dxf.radius
                a0 = np.radians(e.dxf.start_angle)
                a1 = np.radians(e.dxf.end_angle)
                if a1<=a0: a1+=2*np.pi
                aps = [(cx+r*np.cos(a),cy+r*np.sin(a)) for a in np.linspace(a0,a1,8)]
                for i in range(len(aps)-1):
                    cands.append((aps[i][0],aps[i][1],aps[i+1][0],aps[i+1][1]))
            for s in cands:
                l = np.hypot(s[2]-s[0], s[3]-s[1])
                if cfg.min_wall_length <= l <= cfg.max_wall_length:
                    segs.append(s)
        except Exception:
            continue
    return segs


def _collect_by_floor(
    msp,
    layer_cats: dict[str, str],
    cfg: ExtrusionConfig,
    base_cat: str,
) -> dict[str, list]:
    """カテゴリの階数別線分を {floor_key: [segs]} で返す。"""
    result = {}
    for floor_suffix in ["_1", "_2", ""]:
        target = f"{base_cat}{floor_suffix}"
        segs = _collect_segments(msp, layer_cats, cfg, {target, f"{base_cat}"})
        if floor_suffix == "":
            # 階数なしは other 扱いで全部に含める
            segs += _collect_segments(msp, layer_cats, cfg, {f"other{floor_suffix}"})
        if segs:
            result[floor_suffix or "0"] = segs
    return result


# ─────────────────────────────────────────────
# ① 端点スナップ
# ─────────────────────────────────────────────

def _snap_endpoints(
    segs: list[tuple],
    tol: float,
) -> list[tuple]:
    """
    tol [mm] 以内の端点を同一座標に統合して壁の隙間を解消する。
    """
    if not segs:
        return segs

    # 全端点を収集
    pts = np.array([[s[0],s[1]] for s in segs] + [[s[2],s[3]] for s in segs],
                   dtype=np.float64)

    # Union-Find で近接端点をクラスタリング
    n = len(pts)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(n):
        for j in range(i+1, n):
            if np.hypot(pts[i,0]-pts[j,0], pts[i,1]-pts[j,1]) < tol:
                union(i, j)

    # クラスタの重心を代表点とする
    clusters: dict[int, list] = {}
    for i in range(n):
        root = find(i)
        clusters.setdefault(root, []).append(i)

    snapped = np.zeros_like(pts)
    for root, members in clusters.items():
        centroid = pts[members].mean(axis=0)
        for m in members:
            snapped[m] = centroid

    # 線分を再構築
    n_segs = len(segs)
    result = []
    for i in range(n_segs):
        sx, sy = snapped[i]
        ex, ey = snapped[i + n_segs]
        length = np.hypot(ex-sx, ey-sy)
        if length >= cfg_min_snap:
            result.append((sx, sy, ex, ey))
    return result

# モジュールレベル定数（スナップ後の最小長）
cfg_min_snap = 50.0


# ─────────────────────────────────────────────
# ② 壁厚自動推定
# ─────────────────────────────────────────────

def _estimate_wall_thickness(
    segs: list[tuple],
    default_thickness: float,
    detect_tol_deg: float = 30.0,
) -> float:
    """
    平行な線分ペアを検出して壁厚を自動推定する。

    同方向で近接した線分ペアの距離の中央値を壁厚として返す。
    信頼できる推定ができない場合は default_thickness を返す。
    """
    if len(segs) < 4:
        return default_thickness

    # 各線分の角度・中点・長さ
    data = []
    for s in segs:
        dx, dy = s[2]-s[0], s[3]-s[1]
        angle = np.degrees(np.arctan2(dy, dx)) % 180
        cx, cy = (s[0]+s[2])/2, (s[1]+s[3])/2
        length = np.hypot(dx, dy)
        data.append((angle, cx, cy, length, s))

    # 平行ペアの距離を収集
    distances = []
    for i in range(len(data)):
        a1, cx1, cy1, l1, s1 = data[i]
        for j in range(i+1, len(data)):
            a2, cx2, cy2, l2, s2 = data[j]
            # 角度差が小さい（平行）
            angle_diff = abs(a1 - a2)
            angle_diff = min(angle_diff, 180 - angle_diff)
            if angle_diff > detect_tol_deg:
                continue
            # 端点間距離が小さい（隣接）
            dist = np.hypot(cx2-cx1, cy2-cy1)
            if dist < 100 or dist > 600:  # 100〜600mm の範囲が壁厚候補
                continue
            # 線分の向きに垂直な距離を計算
            angle_rad = np.radians(a1)
            perp_dist = abs(
                (cy2-cy1)*np.cos(angle_rad) - (cx2-cx1)*np.sin(angle_rad)
            )
            if 80 <= perp_dist <= 500:
                distances.append(perp_dist)

    if len(distances) < 3:
        return default_thickness

    estimated = float(np.median(distances))
    logger.info(f"壁厚自動推定: {estimated:.0f}mm (サンプル数={len(distances)})")
    return estimated


# ─────────────────────────────────────────────
# 重複除去
# ─────────────────────────────────────────────

def _deduplicate(segs: list[tuple], tol: float = 50.0) -> list[tuple]:
    if not segs:
        return []
    unique = []
    for s in segs:
        is_dup = False
        for u in unique:
            d1 = max(abs(s[0]-u[0]),abs(s[1]-u[1]),abs(s[2]-u[2]),abs(s[3]-u[3]))
            d2 = max(abs(s[0]-u[2]),abs(s[1]-u[3]),abs(s[2]-u[0]),abs(s[3]-u[1]))
            if min(d1,d2) < tol:
                is_dup = True
                break
        if not is_dup:
            unique.append(s)
    return unique


# ─────────────────────────────────────────────
# 壁ボックス
# ─────────────────────────────────────────────

def _wall_box(
    x0,y0,x1,y1,
    thickness, z_bottom, z_top,
    to_m=1e-3,
):
    import trimesh
    dx,dy = x1-x0, y1-y0
    length = np.hypot(dx,dy)
    if length < 1e-6:
        return None
    ux,uy = dx/length, dy/length
    nx,ny = -uy, ux
    half_t = thickness/2.0
    c = np.array([
        [x0+nx*half_t, y0+ny*half_t],
        [x1+nx*half_t, y1+ny*half_t],
        [x1-nx*half_t, y1-ny*half_t],
        [x0-nx*half_t, y0-ny*half_t],
    ])*to_m
    zb,zt = z_bottom*to_m, z_top*to_m
    verts = [[cx,cy,zb] for cx,cy in c] + [[cx,cy,zt] for cx,cy in c]
    n=4
    faces=[]
    for i in range(n):
        j=(i+1)%n
        faces+=[[i,j,i+n],[j,j+n,i+n]]
    for i in range(1,n-1):
        faces.append([n,n+i,n+i+1])
    for i in range(1,n-1):
        faces.append([0,i+1,i])
    return trimesh.Trimesh(
        vertices=np.array(verts,dtype=np.float64),
        faces=np.array(faces,dtype=np.int64),
        process=False,
    )


# ─────────────────────────────────────────────
# ③ 開口部の高さ分類
# ─────────────────────────────────────────────

def _cut_opening_typed(
    wall_mesh,
    x0,y0,x1,y1,
    opening_type: str,  # "door" or "window"
    cfg: ExtrusionConfig,
    z_floor_offset: float = 0.0,
    to_m: float = 1e-3,
):
    """
    開口部タイプに応じた高さで壁を切り抜く。
    door:   z=z_floor_offset 〜 z_floor_offset + opening_height
    window: z=z_floor_offset + window_sill 〜 z_floor_offset + window_sill + window_height
    """
    import trimesh

    length = np.hypot(x1-x0, y1-y0)
    if length < 1e-6:
        return wall_mesh

    dx,dy = (x1-x0)/length, (y1-y0)/length
    nx,ny = -dy, dx
    half_t = cfg.wall_thickness * 1.6 / 2.0

    c = np.array([
        [x0+nx*half_t, y0+ny*half_t],
        [x1+nx*half_t, y1+ny*half_t],
        [x1-nx*half_t, y1-ny*half_t],
        [x0-nx*half_t, y0-ny*half_t],
    ])*to_m

    if opening_type == "window":
        zb = (z_floor_offset + cfg.window_sill) * to_m
        zt = (z_floor_offset + cfg.window_sill + cfg.window_height) * to_m
    else:  # door
        zb = z_floor_offset * to_m
        zt = (z_floor_offset + cfg.opening_height) * to_m

    n=4
    verts = [[cx,cy,zb] for cx,cy in c] + [[cx,cy,zt] for cx,cy in c]
    faces=[]
    for i in range(n):
        j=(i+1)%n
        faces+=[[i,j,i+n],[j,j+n,i+n]]
    for i in range(1,n-1):
        faces.append([n,n+i,n+i+1])
    for i in range(1,n-1):
        faces.append([0,i+1,i])

    cutter = trimesh.Trimesh(
        vertices=np.array(verts,dtype=np.float64),
        faces=np.array(faces,dtype=np.int64),
        process=False,
    )

    for engine in ("manifold", "blender"):
        try:
            result = trimesh.boolean.difference([wall_mesh, cutter], engine=engine)
            if result is not None and len(result.faces) > 0:
                return result
        except Exception:
            pass
    return wall_mesh


# ─────────────────────────────────────────────
# 床スラブ・天井スラブ
# ─────────────────────────────────────────────

def _floor_slab(segs, thickness, to_m=1e-3, margin=300.0):
    import trimesh
    if not segs:
        return None
    xs = [s[0] for s in segs]+[s[2] for s in segs]
    ys = [s[1] for s in segs]+[s[3] for s in segs]
    xmin,xmax = (min(xs)-margin)*to_m, (max(xs)+margin)*to_m
    ymin,ymax = (min(ys)-margin)*to_m, (max(ys)+margin)*to_m
    zb,zt = -thickness*to_m, 0.0
    corners = [[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]]
    n=4
    verts = [[cx,cy,zb] for cx,cy in corners]+[[cx,cy,zt] for cx,cy in corners]
    faces=[]
    for i in range(n):
        j=(i+1)%n
        faces+=[[i,j,i+n],[j,j+n,i+n]]
    for i in range(1,n-1):
        faces.append([n,n+i,n+i+1])
    for i in range(1,n-1):
        faces.append([0,i+1,i])
    return trimesh.Trimesh(
        vertices=np.array(verts,dtype=np.float64),
        faces=np.array(faces,dtype=np.int64),
        process=False,
    )


def _ceiling_slab(segs, z_ceiling, thickness, to_m=1e-3, margin=200.0):
    """⑤ 天井スラブを生成する。"""
    import trimesh
    if not segs:
        return None
    xs = [s[0] for s in segs]+[s[2] for s in segs]
    ys = [s[1] for s in segs]+[s[3] for s in segs]
    xmin,xmax = (min(xs)-margin)*to_m, (max(xs)+margin)*to_m
    ymin,ymax = (min(ys)-margin)*to_m, (max(ys)+margin)*to_m
    zb = z_ceiling * to_m
    zt = (z_ceiling + thickness) * to_m
    corners = [[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]]
    n=4
    verts = [[cx,cy,zb] for cx,cy in corners]+[[cx,cy,zt] for cx,cy in corners]
    faces=[]
    for i in range(n):
        j=(i+1)%n
        faces+=[[i,j,i+n],[j,j+n,i+n]]
    for i in range(1,n-1):
        faces.append([n,n+i,n+i+1])
    for i in range(1,n-1):
        faces.append([0,i+1,i])
    return trimesh.Trimesh(
        vertices=np.array(verts,dtype=np.float64),
        faces=np.array(faces,dtype=np.int64),
        process=False,
    )


# ─────────────────────────────────────────────
# Union・修復
# ─────────────────────────────────────────────

def repair_mesh(mesh):
    import trimesh
    mesh.merge_vertices()
    try:
        from scipy import ndimage
        unique_mask = trimesh.triangles.nondegenerate(mesh.triangles)
        mesh.update_faces(unique_mask)
    except Exception:
        try:
            mesh.remove_degenerate_faces()
        except Exception:
            pass
    try:
        mesh.update_faces(mesh.unique_faces())
    except Exception:
        try:
            mesh.remove_duplicate_faces()
        except Exception:
            pass
    mesh.fix_normals()
    logger.info(f"Mesh repaired: verts={len(mesh.vertices)}, faces={len(mesh.faces)}")
    return mesh


def _union_meshes(meshes: list, batch_size: int = 48):
    import trimesh
    if not meshes:
        return trimesh.Trimesh()
    if len(meshes) == 1:
        return meshes[0]

    # バッチ Union
    batches = [meshes[i:i+batch_size] for i in range(0, len(meshes), batch_size)]
    batch_results = []
    for bi, batch in enumerate(batches):
        merged = trimesh.util.concatenate(batch)
        for engine in ("manifold", "blender"):
            try:
                result = trimesh.boolean.union(batch, engine=engine)
                if result is not None and len(result.faces) > 0:
                    merged = result
                    break
            except Exception:
                pass
        batch_results.append(merged)

    if len(batch_results) == 1:
        return batch_results[0]
    final = trimesh.util.concatenate(batch_results)
    for engine in ("manifold", "blender"):
        try:
            result = trimesh.boolean.union(batch_results, engine=engine)
            if result is not None and len(result.faces) > 0:
                return result
        except Exception:
            pass
    return final


# ─────────────────────────────────────────────
# Z=0 判定
# ─────────────────────────────────────────────

def _get_effective_height(doc, cfg: ExtrusionConfig) -> float:
    msp = doc.modelspace()
    z_vals = []
    count = 0
    for e in msp:
        if count > 200:
            break
        try:
            if e.dxftype() == "LINE":
                z_vals += [abs(e.dxf.start.z), abs(e.dxf.end.z)]
                count += 1
        except Exception:
            continue
    max_z = max(z_vals) if z_vals else 0.0
    if max_z < 1.0:
        height = cfg.default_height if cfg.default_height is not None else cfg.ceiling_height
        logger.info(f"Z=0 の2D図面を検出 → 定数高さ {height:.0f}mm を適用")
        return height
    logger.info(f"Z値あり（max={max_z:.1f}mm） → Z値から高さを使用")
    return max_z


# ─────────────────────────────────────────────
# 判定ユーティリティ
# ─────────────────────────────────────────────

def _has_polyface(msp) -> bool:
    for e in msp:
        if e.dxftype() == "POLYLINE" and (e.dxf.get("flags",0) & 0x40):
            try:
                z_vals = [v.dxf.location.z for v in list(e.vertices)[:20]]
                if max(abs(z) for z in z_vals) > 1e-6:
                    return True
            except Exception:
                pass
    return False


def _is_floor_plan(msp, layer_cats: dict) -> bool:
    has_wall = any("wall" in cat for cat in layer_cats.values())
    if not has_wall:
        return False
    z_vals = []
    count = 0
    for e in msp:
        if count > 50:
            break
        try:
            if e.dxftype() == "LINE":
                z_vals.append(abs(e.dxf.start.z)+abs(e.dxf.end.z))
                count += 1
        except Exception:
            continue
    if not z_vals:
        return has_wall
    return has_wall and max(z_vals) < 1.0


# ─────────────────────────────────────────────
# メイン API
# ─────────────────────────────────────────────

def extrude_floor_plan(
    dxf_path: str | Path,
    config: Optional[ExtrusionConfig] = None,
) -> object:
    import trimesh

    cfg = config or ExtrusionConfig()
    dxf_path = Path(dxf_path)
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()

    # レイヤー分類
    layer_cats = _classify_layers(doc)
    cat_summary: dict[str,list] = {}
    for raw, cat in layer_cats.items():
        cat_summary.setdefault(cat, []).append(_decode_layer(raw))
    for cat, names in sorted(cat_summary.items()):
        logger.info(f"  [{cat}] {names}")

    # Z=0判定・有効高さ
    effective_height = _get_effective_height(doc, cfg)

    # ── 階数別に壁線分を収集 ─────────────────────
    # 1階・2階・階数なし の3グループ
    floor_groups = {}
    for suffix, z_base in [("_1", 0.0), ("_2", cfg.floor_height_step), ("0", 0.0)]:
        targets = {f"wall{suffix}", f"other{suffix}"}
        segs = _collect_segments(msp, layer_cats, cfg, targets)
        if suffix == "0":
            # 階数なし → 1階に統合
            segs += _collect_segments(msp, layer_cats, cfg, {"wall", "other"})
        segs = _deduplicate(segs, tol=cfg.wall_snap_tol)
        if segs:
            floor_groups[suffix] = {"segs": segs, "z_base": z_base}

    if not floor_groups:
        raise ValueError("壁線分が検出できませんでした。")

    # 全階の線分を統合（重複除去）
    all_wall_segs = []
    for g in floor_groups.values():
        all_wall_segs.extend(g["segs"])
    all_wall_segs = _deduplicate(all_wall_segs, tol=cfg.wall_snap_tol)
    logger.info(f"全壁線分（全階）: {len(all_wall_segs)}本")

    # ── ② 壁厚自動推定 ────────────────────────────
    if cfg.auto_wall_thickness and len(all_wall_segs) >= 4:
        estimated = _estimate_wall_thickness(
            all_wall_segs, cfg.wall_thickness, cfg.parallel_detect_tol
        )
        wall_thickness = estimated
    else:
        wall_thickness = cfg.wall_thickness

    # ── ① 端点スナップ ────────────────────────────
    global cfg_min_snap
    cfg_min_snap = cfg.min_wall_length * 0.5
    all_wall_segs = _snap_endpoints(all_wall_segs, cfg.wall_snap_tol)
    logger.info(f"端点スナップ後: {len(all_wall_segs)}本")

    # ── ④ 複数階を考慮した押し出し ───────────────
    mesh_parts = []

    # 階ごとに処理
    processed_floors = set()
    for suffix, group in floor_groups.items():
        segs = group["segs"]
        z_base = group["z_base"] if cfg.multi_floor else 0.0
        z_top  = z_base + effective_height

        if not segs or suffix in processed_floors:
            continue
        processed_floors.add(suffix)

        logger.info(f"  階{suffix}: {len(segs)}本, Z={z_base:.0f}〜{z_top:.0f}mm")

        # 壁ボックス生成
        floor_meshes = []
        for seg in segs:
            box = _wall_box(seg[0],seg[1],seg[2],seg[3],
                            thickness=wall_thickness,
                            z_bottom=z_base, z_top=z_top)
            if box is not None:
                floor_meshes.append(box)

        logger.info(f"  壁ボックス: {len(floor_meshes)}個")

        # Union（use_union=True時）
        if cfg.use_union and len(floor_meshes) > 1:
            logger.info("  Union 実行中...")
            wall_combined = _union_meshes(floor_meshes)
        else:
            wall_combined = trimesh.util.concatenate(floor_meshes)

        # ── ③ 開口部切り抜き（ドア・窓を高さで分類）────
        if cfg.cut_openings:
            for open_suffix in [suffix, "_1" if suffix=="0" else suffix, ""]:
                opening_segs = _collect_segments(
                    msp, layer_cats, cfg, {f"opening{open_suffix}", "opening"}
                )
                if not opening_segs:
                    continue
                logger.info(f"  開口部: {len(opening_segs)}本")
                for seg in opening_segs:
                    seg_len = np.hypot(seg[2]-seg[0], seg[3]-seg[1])
                    # 長さでドア/窓を分類（900mm未満=窓、以上=ドア）
                    otype = "door" if seg_len >= 900 else "window"
                    wall_combined = _cut_opening_typed(
                        wall_combined,
                        seg[0],seg[1],seg[2],seg[3],
                        opening_type=otype,
                        cfg=cfg,
                        z_floor_offset=z_base,
                    )
                break  # 最初にマッチした階のみ処理

        mesh_parts.append(wall_combined)

        # ⑤ 天井スラブ
        ceil_slab = _ceiling_slab(segs, z_top, cfg.ceiling_thickness)
        if ceil_slab is not None:
            mesh_parts.append(ceil_slab)

    # 床スラブ（1階の下）
    slab = _floor_slab(all_wall_segs, cfg.floor_thickness)
    if slab is not None:
        mesh_parts.append(slab)

    if not mesh_parts:
        raise ValueError("メッシュを生成できませんでした。")

    combined = trimesh.util.concatenate(mesh_parts)
    combined.merge_vertices()
    combined.fix_normals()

    v = np.array(combined.vertices)
    logger.info(
        f"完成: verts={len(combined.vertices)}, faces={len(combined.faces)}\n"
        f"  X={v[:,0].min():.2f}~{v[:,0].max():.2f}m  "
        f"  Y={v[:,1].min():.2f}~{v[:,1].max():.2f}m  "
        f"  Z={v[:,2].min():.2f}~{v[:,2].max():.2f}m"
    )
    return combined
