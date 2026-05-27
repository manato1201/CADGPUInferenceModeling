"""
core/floor_plan_extruder.py
===========================
建築平面図（2D DXF）→ 3D メッシュ変換モジュール。

品質改善版:
  1. 重複線分の除去
  2. 壁ボックス同士の Union（ブーリアン結合）で隙間・重なりを解消
  3. 建具（ドア・窓）レイヤーの開口を壁から切り抜き
  4. Z=0 の2D図面に対して定数高さを付与して押し出し
  5. 階高 / 壁厚 / 床厚 を設定で変更可能

依存: ezdxf, trimesh, numpy, (shapely for polygon fallback)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import ezdxf
import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 設定
# ─────────────────────────────────────────────

@dataclass
class ExtrusionConfig:
    """押し出し設定。"""
    # 建物寸法
    ceiling_height: float  = 2500.0   # 階高 [mm]
    floor_thickness: float = 200.0    # 床スラブ厚 [mm]
    wall_thickness: float  = 150.0    # 壁厚 [mm]
    # 線分フィルタ
    min_wall_length: float = 300.0    # 壁とみなす最小長 [mm]
    max_wall_length: float = 15000.0  # 壁とみなす最大長 [mm]（外周敷地線除外）
    wall_snap_tol: float   = 50.0     # 重複判定の許容距離 [mm]
    # Z=0 図面への定数高さ付与
    # None の場合は ceiling_height をそのまま使う
    # 指定した場合、Z=0のエンティティに対してこの値 [mm] を高さとして使う
    default_height: Optional[float] = None
    # Union（ブーリアン結合）の実行フラグ
    # True にすると品質が上がるが処理時間が増加（壁数 * O(n²)）
    use_union: bool = True
    # 開口部切り抜きフラグ
    cut_openings: bool = True
    # 開口部のデフォルト高さ [mm]（建具レイヤーに高さ情報がない場合）
    opening_height: float = 2000.0


# ─────────────────────────────────────────────
# レイヤー分類
# ─────────────────────────────────────────────

def _decode_layer(name: str) -> str:
    """ezdxf が latin1 で読み込んだ CP932 レイヤー名を UTF-8 に変換する。"""
    for method in [
        lambda n: n.encode("utf-8", "surrogateescape").decode("cp932"),
        lambda n: n.encode("latin1").decode("cp932"),
    ]:
        try:
            return method(name)
        except Exception:
            pass
    return name


def _to_mojibake(text: str) -> str:
    """
    CP932テキストをezdxf内部のUnicode文字列に変換する。

    ezdxfはDXFファイルをcp1252で読み込む。
    cp1252で表現できないバイト(0x81,0x8D,0x8F,0x90,0x9D等)は
    surrogate (\udc81等) に変換される。

    例:
      壁   (CP932: 95 C7)  → cp1252: '•Ç'       (どちらもcp1252で定義済み)
      設備 (CP932: 90 DD)  → cp1252: '\udc90Ý'  (0x90はcp1252未定義→surrogate)
    """
    b = text.encode("cp932")
    result = []
    for byte in b:
        if 0x80 <= byte <= 0xFF:
            try:
                result.append(bytes([byte]).decode("cp1252"))
            except ValueError:
                # cp1252未定義バイト → ezdxfはsurrogateに変換
                result.append(chr(0xDC00 + byte))
        else:
            result.append(chr(byte))
    return "".join(result)


# ezdxf内部形式（cp1252 Mojibake）のパターン
_WALL_MOJ      = [_to_mojibake(k) for k in ["壁"]]
_OPENING_MOJ   = [_to_mojibake(k) for k in ["建具", "ドア", "窓", "開口"]]
_FURNITURE_MOJ = [_to_mojibake(k) for k in ["家具", "設備"]]
_STAIR_MOJ     = [_to_mojibake(k) for k in ["階段"]]
_IGNORE_MOJ    = [_to_mojibake(k) for k in [
    "寸法", "文字", "部屋名", "注記", "法線", "三斜", "石材",
    "敷地", "道路", "隣地", "高低差", "間取",
]]


def _raw_match(raw: str, patterns: list[str]) -> bool:
    """ezdxf内部のraw文字列にMojibakeパターンが含まれるか判定する。"""
    return any(p and p in raw for p in patterns)


def _classify_layers(doc) -> dict[str, str]:
    cats: dict[str, str] = {}
    for layer in doc.layers:
        raw = layer.dxf.name
        d = _decode_layer(raw).lower()
        if "wall" in d or _raw_match(raw, _WALL_MOJ):
            cats[raw] = "wall"
        elif any(k in d for k in ["door","window","opening"]) or _raw_match(raw, _OPENING_MOJ):
            cats[raw] = "opening"
        elif any(k in d for k in ["furniture","equip"]) or _raw_match(raw, _FURNITURE_MOJ):
            cats[raw] = "furniture"
        elif "stair" in d or _raw_match(raw, _STAIR_MOJ):
            cats[raw] = "stair"
        elif any(k in d for k in ["dim","text","anno"]) or _raw_match(raw, _IGNORE_MOJ):
            cats[raw] = "ignore"
        else:
            cats[raw] = "other"
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
    """指定カテゴリの線分を収集して (x0,y0,x1,y1) リストを返す。"""
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
                    closed = getattr(e,"is_closed",False)
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
                aps=[(cx+r*np.cos(a),cy+r*np.sin(a)) for a in np.linspace(a0,a1,8)]
                for i in range(len(aps)-1):
                    cands.append((aps[i][0],aps[i][1],aps[i+1][0],aps[i+1][1]))
            for s in cands:
                l = np.hypot(s[2]-s[0], s[3]-s[1])
                if cfg.min_wall_length <= l <= cfg.max_wall_length:
                    segs.append(s)
        except Exception:
            continue
    return segs


# ─────────────────────────────────────────────
# 重複線分の除去
# ─────────────────────────────────────────────

def _deduplicate(
    segs: list[tuple],
    tol: float = 50.0,
) -> list[tuple]:
    """
    端点が tol [mm] 以内で同一方向の線分を重複とみなして除去する。
    1・2階の図面が重なっている場合でも余分な壁を生成しない。
    """
    if not segs:
        return []

    unique = []
    for s in segs:
        is_dup = False
        for u in unique:
            # 正方向 or 逆方向
            d1 = max(abs(s[0]-u[0]), abs(s[1]-u[1]),
                     abs(s[2]-u[2]), abs(s[3]-u[3]))
            d2 = max(abs(s[0]-u[2]), abs(s[1]-u[3]),
                     abs(s[2]-u[0]), abs(s[3]-u[1]))
            if min(d1, d2) < tol:
                is_dup = True
                break
        if not is_dup:
            unique.append(s)
    return unique


# ─────────────────────────────────────────────
# 壁ボックス生成
# ─────────────────────────────────────────────

def _wall_box(
    x0: float, y0: float,
    x1: float, y1: float,
    thickness: float,
    z_bottom: float,
    z_top: float,
    to_m: float = 1e-3,
) -> Optional[object]:
    """線分を中心軸とした厚み付き壁ボックス（trimesh）を返す。"""
    import trimesh

    dx, dy = x1-x0, y1-y0
    length = np.hypot(dx, dy)
    if length < 1e-6:
        return None

    ux, uy = dx/length, dy/length
    nx, ny = -uy, ux
    half_t = thickness / 2.0

    c = np.array([
        [x0+nx*half_t, y0+ny*half_t],
        [x1+nx*half_t, y1+ny*half_t],
        [x1-nx*half_t, y1-ny*half_t],
        [x0-nx*half_t, y0-ny*half_t],
    ]) * to_m

    zb, zt = z_bottom*to_m, z_top*to_m
    verts = [[cx,cy,zb] for cx,cy in c] + [[cx,cy,zt] for cx,cy in c]
    n = 4
    faces = []
    for i in range(n):
        j=(i+1)%n
        faces+=[[i,j,i+n],[j,j+n,i+n]]
    for i in range(1,n-1):
        faces.append([n,n+i,n+i+1])    # 上面
    for i in range(1,n-1):
        faces.append([0,i+1,i])         # 下面

    return trimesh.Trimesh(
        vertices=np.array(verts, dtype=np.float64),
        faces=np.array(faces, dtype=np.int64),
        process=False,
    )


# ─────────────────────────────────────────────
# ブーリアン Union（壁結合）
# ─────────────────────────────────────────────

def _union_meshes(meshes: list) -> object:
    """
    trimesh ブーリアン Union で壁メッシュを結合する。
    失敗した場合は concatenate にフォールバックする。
    """
    import trimesh

    if len(meshes) == 0:
        return trimesh.Trimesh()
    if len(meshes) == 1:
        return meshes[0]

    # まず concatenate（高速・軽量）
    combined = trimesh.util.concatenate(meshes)

    # ブーリアン Union を試みる（manifold / blender backend）
    try:
        result = trimesh.boolean.union(meshes, engine="manifold")
        if result is not None and len(result.faces) > 0:
            logger.info(f"  Union成功: {len(result.faces)} faces")
            return result
    except Exception as e:
        logger.debug(f"  manifold Union失敗: {e}")

    try:
        result = trimesh.boolean.union(meshes, engine="blender")
        if result is not None and len(result.faces) > 0:
            logger.info(f"  Union成功(blender): {len(result.faces)} faces")
            return result
    except Exception as e:
        logger.debug(f"  blender Union失敗: {e}")

    logger.info("  Union不可: concatenateにフォールバック")
    return combined


# ─────────────────────────────────────────────
# 開口部切り抜き
# ─────────────────────────────────────────────

def _cut_opening(
    wall_mesh,
    x0: float, y0: float,
    x1: float, y1: float,
    opening_width: float,
    opening_height: float,
    z_bottom: float = 0.0,
    to_m: float = 1e-3,
    wall_thickness: float = 300.0,
) -> object:
    """
    壁メッシュから開口部（ドア・窓）を切り抜く。
    開口部を「少し厚い直方体」として差分を取る。
    """
    import trimesh

    dx, dy = x1-x0, y1-y0
    length = np.hypot(dx, dy)
    if length < 1e-6:
        return wall_mesh

    ux, uy = dx/length, dy/length
    nx, ny = -uy, ux
    half_t = wall_thickness * 1.5 / 2.0  # 壁を完全に貫通させる

    c = np.array([
        [x0+nx*half_t, y0+ny*half_t],
        [x1+nx*half_t, y1+ny*half_t],
        [x1-nx*half_t, y1-ny*half_t],
        [x0-nx*half_t, y0-ny*half_t],
    ]) * to_m

    zb = z_bottom * to_m
    zt = (z_bottom + opening_height) * to_m
    n = 4
    verts = [[cx,cy,zb] for cx,cy in c] + [[cx,cy,zt] for cx,cy in c]
    faces = []
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

    try:
        result = trimesh.boolean.difference([wall_mesh, cutter], engine="manifold")
        if result is not None and len(result.faces) > 0:
            return result
    except Exception:
        pass
    try:
        result = trimesh.boolean.difference([wall_mesh, cutter], engine="blender")
        if result is not None and len(result.faces) > 0:
            return result
    except Exception:
        pass

    return wall_mesh  # フォールバック


# ─────────────────────────────────────────────
# 床スラブ
# ─────────────────────────────────────────────

def _floor_slab(
    segs: list[tuple],
    thickness: float,
    to_m: float = 1e-3,
    margin: float = 300.0,
) -> Optional[object]:
    """壁線分のバウンディングボックスから床スラブを生成する。"""
    import trimesh

    if not segs:
        return None
    xs = [s[0] for s in segs] + [s[2] for s in segs]
    ys = [s[1] for s in segs] + [s[3] for s in segs]
    xmin,xmax = (min(xs)-margin)*to_m, (max(xs)+margin)*to_m
    ymin,ymax = (min(ys)-margin)*to_m, (max(ys)+margin)*to_m
    zb,zt = -thickness*to_m, 0.0

    corners = [[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]]
    n = 4
    verts = [[cx,cy,zb] for cx,cy in corners] + [[cx,cy,zt] for cx,cy in corners]
    faces = []
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
# Z=0 判定・高さ付与
# ─────────────────────────────────────────────

def _get_effective_height(doc, cfg: ExtrusionConfig) -> float:
    """
    図面の Z 範囲を確認し、有効な階高 [mm] を返す。

    - Z 値がすべて 0 の純粋な2D図面の場合:
        cfg.default_height が指定されていればその値を使用
        指定がなければ cfg.ceiling_height を使用
    - Z 値に有意な高さ情報がある場合:
        その最大 Z 値を階高として使用
    """
    msp = doc.modelspace()
    z_vals = []
    count = 0
    for e in msp:
        if count > 200:
            break
        try:
            t = e.dxftype()
            if t == "LINE":
                z_vals += [abs(e.dxf.start.z), abs(e.dxf.end.z)]
                count += 1
            elif t == "POLYLINE":
                for v in list(e.vertices)[:5]:
                    z_vals.append(abs(v.dxf.location.z))
                count += 1
        except Exception:
            continue

    max_z = max(z_vals) if z_vals else 0.0

    if max_z < 1.0:
        # Z=0 の純粋2D図面
        height = cfg.default_height if cfg.default_height is not None else cfg.ceiling_height
        logger.info(
            f"Z=0 の2D図面を検出 → 定数高さ {height:.0f}mm を適用"
            + (f" (cfg.default_height={cfg.default_height})" if cfg.default_height else "")
        )
        return height
    else:
        logger.info(f"Z値あり（max={max_z:.1f}mm） → Z値から高さ {max_z:.0f}mm を使用")
        return max_z


# ─────────────────────────────────────────────
# メイン API
# ─────────────────────────────────────────────

def extrude_floor_plan(
    dxf_path: str | Path,
    config: Optional[ExtrusionConfig] = None,
) -> object:  # trimesh.Trimesh
    """
    建築平面図 DXF → 3D メッシュ（単位: m）。

    Parameters
    ----------
    dxf_path : DXF ファイルパス
    config   : ExtrusionConfig（省略時はデフォルト値）

    Returns
    -------
    trimesh.Trimesh  単位 m、Z=0 が床面
    """
    import trimesh

    cfg = config or ExtrusionConfig()
    dxf_path = Path(dxf_path)
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()

    # ── ① レイヤー分類 ──────────────────────────
    layer_cats = _classify_layers(doc)
    cat_summary: dict[str,list] = {}
    for raw, cat in layer_cats.items():
        cat_summary.setdefault(cat, []).append(_decode_layer(raw))
    for cat, names in sorted(cat_summary.items()):
        logger.info(f"  [{cat}] {names}")

    # ── ② Z=0 判定・有効高さ取得 ─────────────────
    effective_height = _get_effective_height(doc, cfg)

    # ── ③ 壁線分収集 ─────────────────────────────
    wall_segs = _collect_segments(msp, layer_cats, cfg,
                                  target_cats={"wall","other"})
    logger.info(f"壁線分（フィルタ前）: {len(wall_segs)}本")

    # ── ④ 重複除去 ───────────────────────────────
    wall_segs = _deduplicate(wall_segs, tol=cfg.wall_snap_tol)
    logger.info(f"壁線分（重複除去後）: {len(wall_segs)}本")

    if not wall_segs:
        raise ValueError("壁線分が検出されませんでした。")

    # ── ⑤ 壁ボックス生成 ─────────────────────────
    wall_meshes = []
    for seg in wall_segs:
        box = _wall_box(
            seg[0], seg[1], seg[2], seg[3],
            thickness=cfg.wall_thickness,
            z_bottom=0.0,
            z_top=effective_height,
        )
        if box is not None:
            wall_meshes.append(box)
    logger.info(f"壁ボックス: {len(wall_meshes)}個生成")

    # ── ⑥ ブーリアン Union ───────────────────────
    if cfg.use_union and len(wall_meshes) > 1:
        logger.info("ブーリアン Union 実行中...")
        # バッチ Union（一度に全部は重いので32個ずつ）
        batch_size = 32
        batches = [wall_meshes[i:i+batch_size]
                   for i in range(0, len(wall_meshes), batch_size)]
        batch_results = []
        for bi, batch in enumerate(batches):
            logger.info(f"  batch {bi+1}/{len(batches)} ({len(batch)}個)")
            batch_results.append(_union_meshes(batch))
        wall_combined = _union_meshes(batch_results)
    else:
        wall_combined = trimesh.util.concatenate(wall_meshes)

    # ── ⑦ 開口部切り抜き ─────────────────────────
    if cfg.cut_openings:
        opening_segs = _collect_segments(msp, layer_cats, cfg,
                                         target_cats={"opening"})
        if opening_segs:
            logger.info(f"開口部切り抜き: {len(opening_segs)}本")
            for seg in opening_segs:
                wall_combined = _cut_opening(
                    wall_combined,
                    seg[0], seg[1], seg[2], seg[3],
                    opening_width=np.hypot(seg[2]-seg[0], seg[3]-seg[1]),
                    opening_height=cfg.opening_height,
                    wall_thickness=cfg.wall_thickness,
                )
        else:
            logger.info("開口部レイヤーなし（切り抜きスキップ）")

    # ── ⑧ 床スラブ ───────────────────────────────
    slab = _floor_slab(wall_segs, thickness=cfg.floor_thickness)
    if slab is not None:
        combined = trimesh.util.concatenate([wall_combined, slab])
    else:
        combined = wall_combined

    # ── ⑨ 仕上げ ─────────────────────────────────
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
