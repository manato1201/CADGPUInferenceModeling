"""
core/mesh_to_dxf.py
===================
OBJ / FBX / GLB などの 3D メッシュを DXF 図面に変換するモジュール。

出力モード:
  triview    三面図（正面・側面・上面）+ 寸法線
  floorplan  平面図（フットプリント・間取り）
  both       両方を1つのDXFファイルに出力

依存: trimesh, shapely, ezdxf, numpy
FBX対応: pip install trimesh[all] または pip install pyassimp
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional
import ezdxf
import numpy as np

logger = logging.getLogger(__name__)
OutputMode = Literal["triview", "floorplan", "both"]


@dataclass
class MeshToDxfConfig:
    mode: OutputMode = "both"
    view_gap: float = 2.0
    add_dimensions: bool = True
    dim_offset_ratio: float = 0.15
    show_hidden_lines: bool = True
    simplify_tolerance: float = 0.01
    floor_slice_ratio: float = 0.3
    output_unit: Literal["mm", "m"] = "mm"
    layer_visible: str = "VISIBLE"
    layer_hidden:  str = "HIDDEN"
    layer_dim:     str = "DIMENSION"
    layer_wall:    str = "WALL"
    layer_floor:   str = "FLOOR"
    layer_center:  str = "CENTERLINE"


def load_mesh(path: str | Path):
    import trimesh
    path = Path(path)

    if path.suffix.lower() == ".fbx":
        return _load_fbx(path)

    # DXF は trimesh で直接読むと Path2D になることがある
    # → cad2asset の parser.py 経由で POLYFACE メッシュを抽出する
    if path.suffix.lower() in (".dxf", ".dwg"):
        return _load_dxf_as_mesh(path)

    obj = trimesh.load(str(path))
    if isinstance(obj, trimesh.Scene):
        meshes = [g for g in obj.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"メッシュが見つかりません: {path}")
        obj = trimesh.util.concatenate(meshes)
    import trimesh as _tm
    # Path2D や Point など非Trimeshオブジェクトの処理
    if isinstance(obj, _tm.Trimesh):
        obj.merge_vertices()
    elif isinstance(obj, _tm.Scene):
        meshes = [g for g in obj.geometry.values() if isinstance(g, _tm.Trimesh)]
        if not meshes:
            raise ValueError("読み込んだファイルにTriangleMeshが含まれていません")
        obj = _tm.util.concatenate(meshes)
        obj.merge_vertices()
    else:
        raise ValueError(f"非対応のオブジェクト型: {type(obj)}")
    logger.info(f"読み込み完了: verts={len(obj.vertices)}, faces={len(obj.faces)}")
    return obj


def _load_dxf_as_mesh(path: Path):
    """
    DXF ファイルを3Dメッシュとして読み込む。
    parser.py の POLYFACE 抽出を使い、2Dパス問題を回避する。
    """
    import sys as _sys
    # cad2asset の core を参照
    core_dir = Path(__file__).parent.parent
    if str(core_dir) not in _sys.path:
        _sys.path.insert(0, str(core_dir))

    try:
        from core.parser import parse_dxf
        result = parse_dxf(str(path))
        if result.mesh is not None:
            logger.info(f"DXF: parser.py (Route B) で読み込み成功")
            return result.mesh
    except Exception as e:
        logger.debug(f"parser.py 読み込み失敗: {e}")

    # フォールバック: floor_plan_extruder で押し出し
    try:
        from core.floor_plan_extruder import extrude_floor_plan, ExtrusionConfig
        mesh = extrude_floor_plan(str(path), config=ExtrusionConfig())
        logger.info(f"DXF: floor_plan_extruder (押し出し) で読み込み成功")
        return mesh
    except Exception as e:
        logger.debug(f"floor_plan_extruder 失敗: {e}")


    raise RuntimeError(
        f"DXF の読み込みに失敗しました: {path}. 3D POLYFACE メッシュまたは建築平面図（壁レイヤーあり）が必要です。"
    )


def _load_fbx(path: Path):
    """
    FBX ファイルを読み込む。複数の方法を順に試みる。

    方法1: trimesh (trimesh[all] または assimp バインディングが必要)
    方法2: pyassimp + Assimp DLL
    方法3: Blender 経由 (blender がPATHにある場合)
    方法4: FBX→OBJ 変換ツール経由
    """
    import trimesh

    # 方法1: trimesh 直接
    try:
        obj = trimesh.load(str(path))
        if isinstance(obj, trimesh.Scene):
            meshes = [g for g in obj.geometry.values()
                      if isinstance(g, trimesh.Trimesh)]
            if not meshes:
                raise ValueError("Scene にメッシュが含まれていません")
            obj = trimesh.util.concatenate(meshes)
        if len(obj.vertices) > 0:
            logger.info(f"FBX: trimesh で読み込み成功 (verts={len(obj.vertices)})")
            return obj
    except Exception as e:
        logger.debug(f"trimesh FBX 失敗: {e}")

    # 方法2: pyassimp (Assimp DLL が必要)
    try:
        import pyassimp
        scene = pyassimp.load(str(path))
        verts_all, faces_all, offset = [], [], 0
        for mesh in scene.meshes:
            v = np.array([[vv.x, vv.y, vv.z] for vv in mesh.vertices])
            f = np.array([[face.indices[0], face.indices[1], face.indices[2]]
                          for face in mesh.faces if len(face.indices) == 3])
            if len(v) > 0 and len(f) > 0:
                verts_all.append(v); faces_all.append(f + offset); offset += len(v)
        pyassimp.release(scene)
        if verts_all:
            result = trimesh.Trimesh(
                vertices=np.vstack(verts_all),
                faces=np.vstack(faces_all)
            )
            logger.info(f"FBX: pyassimp で読み込み成功 (verts={len(result.vertices)})")
            return result
    except ImportError:
        logger.debug("pyassimp 未インストール")
    except Exception as e:
        logger.debug(f"pyassimp FBX 失敗: {e}")

    # 方法3: Blender CLI 経由で OBJ に変換してから読み込む
    try:
        import subprocess, tempfile, os
        tmp_obj = Path(tempfile.mktemp(suffix=".obj"))
        blender_script = f"""
import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=r"{path}")
bpy.ops.export_scene.obj(filepath=r"{tmp_obj}", use_selection=False)
"""
        script_path = Path(tempfile.mktemp(suffix=".py"))
        script_path.write_text(blender_script)

        # Blender のパスを探す
        blender_candidates = [
            "blender",
            r"C:\Program Files\Blender Foundation\Blender 4.5lender.exe",
            r"C:\Program Files\Blender Foundation\Blender 4.2lender.exe",
            r"C:\Program Files\Blender Foundation\Blender 4.1lender.exe",
            r"C:\Program Files\Blender Foundation\Blender 4.0lender.exe",
            r"C:\Program Files\Blender Foundation\Blender 3.6lender.exe",
        ]
        blender_exe = None
        for candidate in blender_candidates:
            try:
                r = subprocess.run([candidate, "--version"],
                                   capture_output=True, timeout=5)
                if r.returncode == 0:
                    blender_exe = candidate
                    break
            except Exception:
                continue

        if blender_exe:
            result = subprocess.run(
                [blender_exe, "--background", "--python", str(script_path)],
                capture_output=True, timeout=60
            )
            script_path.unlink(missing_ok=True)
            if tmp_obj.exists() and tmp_obj.stat().st_size > 0:
                obj = trimesh.load(str(tmp_obj), force="mesh")
                tmp_obj.unlink(missing_ok=True)
                logger.info(f"FBX: Blender経由で読み込み成功 (verts={len(obj.vertices)})")
                return obj
    except Exception as e:
        logger.debug(f"Blender FBX 変換失敗: {e}")

    # すべて失敗
    raise RuntimeError(
        f"FBX の読み込みに失敗しました: {path}\n\n"
        "以下のいずれかの方法で対応してください:\n"
        "\n"
        "【方法A: Assimp DLLをインストール（推奨）】\n"
        "  1. https://github.com/assimp/assimp/releases から\n"
        "     assimp-x64.dll (Windows) をダウンロード\n"
        "  2. プロジェクトフォルダまたは System32 に配置\n"
        "  3. pip install pyassimp\n"
        "\n"
        "【方法B: Blenderで事前変換】\n"
        "  Blender → File → Import → FBX\n"
        "         → File → Export → Wavefront OBJ\n"
        "  python cli.py export model.obj\n"
        "\n"
        "【方法C: FBX Converter（Autodesk無料ツール）】\n"
        "  FBX → OBJ に変換してから export コマンドを使用"
    )


def _extract_silhouette(mesh, ax_h, ax_v, simplify_tol=0.01):
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    verts = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    polys = []
    for face in faces:
        pts = verts[face][:, [ax_h, ax_v]]
        try:
            p = Polygon(pts)
            if p.is_valid and p.area > 1e-9:
                polys.append(p)
        except Exception:
            continue
    if not polys:
        return None
    union = unary_union(polys)
    return union.simplify(simplify_tol) if simplify_tol > 0 else union


# ─────────────────────────────────────────────
# Zバッファラスタライザ（深度画像ベース隠線判定）
# ─────────────────────────────────────────────

def _build_zbuffer(
    verts: np.ndarray,
    faces: np.ndarray,
    ax_h: int,
    ax_v: int,
    ax_d: int,
    size: int = 512,
    margin: float = 0.02,
) -> tuple[np.ndarray, tuple]:
    """
    Zバッファ（深度マップ）をラスタライズする。

    各三角形をピクセル単位で描画し、最も手前の深度を記録する。
    バリセントリック座標法で正確な深度補間を行う。

    Returns
    -------
    z_buf  : (size, size) float32 の深度マップ（値が小さいほど手前）
    params : 座標変換パラメータ（エッジ判定で使用）
    """
    pts_h = verts[:, ax_h].astype(np.float32)
    pts_v = verts[:, ax_v].astype(np.float32)
    pts_d = verts[:, ax_d].astype(np.float32)

    h_min, h_max = pts_h.min(), pts_h.max()
    v_min, v_max = pts_v.min(), pts_v.max()

    # モデル座標 → ピクセル座標の変換係数
    scale_h = (1.0 - 2.0*margin) * size / max(h_max - h_min, 1e-9)
    scale_v = (1.0 - 2.0*margin) * size / max(v_max - v_min, 1e-9)
    off_h = margin * size - h_min * scale_h
    off_v = margin * size - v_min * scale_v

    # 全頂点のピクセル座標を事前計算
    px_h = (pts_h * scale_h + off_h).astype(np.float32)
    px_v = ((1.0 - (pts_v * scale_v + off_v) / size) * size).astype(np.float32)

    z_buf = np.full((size, size), np.inf, dtype=np.float32)

    # 三角形ごとにラスタライズ
    for face in faces:
        a, b, c = face
        h0, h1, h2 = px_h[a], px_h[b], px_h[c]
        v0, v1, v2 = px_v[a], px_v[b], px_v[c]
        d0, d1, d2 = pts_d[a], pts_d[b], pts_d[c]

        # バウンディングボックス
        xmin = max(0,      int(min(h0, h1, h2)))
        xmax = min(size-1, int(max(h0, h1, h2)) + 1)
        ymin = max(0,      int(min(v0, v1, v2)))
        ymax = min(size-1, int(max(v0, v1, v2)) + 1)
        if xmin >= xmax or ymin >= ymax:
            continue

        # バリセントリック座標でピクセルをカバー
        area = (h1-h0)*(v2-v0) - (h2-h0)*(v1-v0)
        if abs(area) < 0.5:
            continue

        xs = np.arange(xmin, xmax, dtype=np.float32)
        ys = np.arange(ymin, ymax, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)

        w0 = ((h1-h2)*(gx-h2) + (v2-v1)*(gy-v2)) / area
        w1 = ((h2-h0)*(gx-h0) + (v0-v2)*(gy-v0)) / area
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -0.01) & (w1 >= -0.01) & (w2 >= -0.01)
        if not inside.any():
            continue

        depth  = w0*d0 + w1*d1 + w2*d2
        gyi    = gy[inside].astype(np.int32)
        gxi    = gx[inside].astype(np.int32)
        di     = depth[inside]
        update = di < z_buf[gyi, gxi]
        z_buf[gyi[update], gxi[update]] = di[update]

    params = (h_min, h_max, v_min, v_max,
              scale_h, scale_v, off_h, off_v, size)
    return z_buf, params


def _edge_visibility_zbuffer(
    x0: float, y0: float,
    x1: float, y1: float,
    d0: float, d1: float,
    z_buf: np.ndarray,
    params: tuple,
    tol_ratio: float = 0.015,
    n_samples: int = 5,
) -> str:
    """
    Zバッファを使ってエッジの可視性を判定する。

    エッジを n_samples 点でサンプリングし、各点の深度とZバッファを比較。
    過半数が「手前」にあれば visible、そうでなければ hidden。

    Parameters
    ----------
    tol_ratio : 深度許容誤差（モデルの奥行き範囲に対する割合）
    n_samples : エッジのサンプリング点数

    Returns
    -------
    "visible" or "hidden"
    """
    h_min, h_max, v_min, v_max, scale_h, scale_v, off_h, off_v, size = params
    depth_range = max(h_max - h_min, 1e-9)
    tol = depth_range * tol_ratio

    visible_votes = 0

    for t in np.linspace(0.1, 0.9, n_samples):
        mx = x0 + t * (x1 - x0)
        my = y0 + t * (y1 - y0)
        md = d0 + t * (d1 - d0)

        # ピクセル座標
        px = int(mx * scale_h + off_h)
        py = int((1.0 - (my * scale_v + off_v) / size) * size)
        px = np.clip(px, 0, size - 1)
        py = np.clip(py, 0, size - 1)

        zbuf_d = z_buf[py, px]
        if zbuf_d == np.inf or md <= zbuf_d + tol:
            visible_votes += 1

    # 過半数が可視なら可視
    return "visible" if visible_votes > n_samples // 2 else "hidden"


def _get_silhouette_and_hidden(
    mesh,
    ax_h: int,
    ax_v: int,
    ax_d: int,
    zbuf_size: int = 512,
    tol_ratio: float = 0.015,
) -> tuple[list, list]:
    """
    Zバッファ深度画像を使ってシルエットエッジと隠線を判定する。

    処理フロー:
      1. メッシュを Zバッファにラスタライズ（深度マップ生成）
      2. 全エッジの中間点の深度を Zバッファと比較
      3. 手前 → visible / 奥 → hidden に分類

    従来の法線ベースより大幅に精度が高い。
    曲面・複雑形状・自己遮蔽を正確に判定できる。

    Parameters
    ----------
    zbuf_size : Zバッファの解像度（高いほど精度が高いが遅い）
    tol_ratio : 深度判定の許容誤差（モデルサイズに対する割合）

    Returns
    -------
    silhouette_edges : list of (x0,y0,x1,y1)  可視エッジ
    hidden_edges     : list of (x0,y0,x1,y1)  隠線
    """
    verts = np.array(mesh.vertices, dtype=np.float64)
    faces = np.array(mesh.faces)

    if len(faces) == 0:
        return [], []

    logger.info(f"  Zバッファ生成中 (size={zbuf_size})...")
    z_buf, params = _build_zbuffer(
        verts.astype(np.float32), faces,
        ax_h, ax_v, ax_d, size=zbuf_size,
    )

    # 全エッジを収集
    edge_set: dict = {}
    for face in faces:
        for i in range(3):
            a, b = face[i], face[(i+1) % 3]
            key = (min(a, b), max(a, b))
            edge_set[key] = (a, b)

    logger.info(f"  エッジ可視性判定中 ({len(edge_set)}本)...")
    silhouette = []
    hidden     = []

    for a, b in edge_set.values():
        x0, y0 = verts[a, ax_h], verts[a, ax_v]
        x1, y1 = verts[b, ax_h], verts[b, ax_v]
        length = np.hypot(x1-x0, y1-y0)
        if length < 1e-9:
            continue

        d0, d1 = verts[a, ax_d], verts[b, ax_d]
        vis = _edge_visibility_zbuffer(
            x0, y0, x1, y1, d0, d1,
            z_buf, params, tol_ratio=tol_ratio,
        )
        if vis == "visible":
            silhouette.append((x0, y0, x1, y1))
        else:
            hidden.append((x0, y0, x1, y1))

    logger.info(f"  可視={len(silhouette)}, 隠線={len(hidden)}")
    return silhouette, hidden


def _get_hidden_edges(mesh, ax_h, ax_v, ax_d):
    """後方互換用ラッパー。"""
    _, hidden = _get_silhouette_and_hidden(mesh, ax_h, ax_v, ax_d)
    return hidden


def _shapely_to_dxf(msp, geom, ox, oy, scale, layer):
    geoms = list(geom.geoms) if geom.geom_type in ("MultiPolygon","GeometryCollection") else [geom]
    for poly in geoms:
        if poly.is_empty or not hasattr(poly, 'exterior'):
            continue
        pts = [(ox + x*scale, oy + y*scale) for x, y in poly.exterior.coords]
        if len(pts) >= 2:
            msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": layer})
        for interior in poly.interiors:
            pts_i = [(ox + x*scale, oy + y*scale) for x, y in interior.coords]
            if len(pts_i) >= 2:
                msp.add_lwpolyline(pts_i, close=True, dxfattribs={"layer": layer})


def _add_linear_dim(msp, p1, p2, direction, offset, layer):
    try:
        if direction == "h":
            base = ((p1[0]+p2[0])/2, p1[1]+offset)
            dim = msp.add_linear_dim(base=base, p1=p1, p2=p2, angle=0,
                                     dxfattribs={"layer": layer})
        else:
            base = (p1[0]+offset, (p1[1]+p2[1])/2)
            dim = msp.add_linear_dim(base=base, p1=p1, p2=p2, angle=90,
                                     dxfattribs={"layer": layer})
        dim.render()
    except Exception as e:
        logger.debug(f"寸法線追加失敗: {e}")


def _init_dxf(cfg: MeshToDxfConfig):
    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = 4 if cfg.output_unit == "mm" else 6
    for name, color, lt in [
        (cfg.layer_visible, 7, "CONTINUOUS"),
        (cfg.layer_hidden,  8, "DASHED"),
        (cfg.layer_dim,     2, "CONTINUOUS"),
        (cfg.layer_wall,    7, "CONTINUOUS"),
        (cfg.layer_floor,   3, "CONTINUOUS"),
        (cfg.layer_center,  1, "CENTER"),
    ]:
        try:
            layer = doc.layers.new(name)
            layer.dxf.color = color
            try:
                doc.linetypes.get(lt)
                layer.dxf.linetype = lt
            except Exception:
                pass
        except Exception:
            pass
    return doc


def _write_triview(doc, mesh, cfg, us, offset_y=0.0):
    msp = doc.modelspace()
    verts = np.array(mesh.vertices)
    ext = verts.max(axis=0) - verts.min(axis=0)  # [Xwidth, Ydepth, Zheight]
    gap = cfg.view_gap * us

    views = [
        (0, 2, 1, 0.0,              offset_y,              "FRONT (XZ)",  ext[0], ext[2]),
        (1, 2, 0, ext[0]*us + gap,  offset_y,              "SIDE (YZ)",   ext[1], ext[2]),
        (0, 1, 2, 0.0,              offset_y+ext[2]*us+gap,"TOP (XY)",    ext[0], ext[1]),
    ]

    for ax_h, ax_v, ax_d, ox, oy, label, w, h in views:
        # シルエット（可視輪郭）と隠線を分離して取得
        silhouette_edges, hidden_edges = _get_silhouette_and_hidden(
            mesh, ax_h, ax_v, ax_d
        )

        # シルエットエッジを VISIBLE レイヤーに描画
        for x0, y0, x1, y1 in silhouette_edges:
            msp.add_line(
                (ox+x0*us, oy+y0*us), (ox+x1*us, oy+y1*us),
                dxfattribs={"layer": cfg.layer_visible},
            )

        # shapely シルエットも重ねて描画（外形輪郭を補完）
        sil_poly = _extract_silhouette(mesh, ax_h, ax_v, cfg.simplify_tolerance)
        if sil_poly is not None:
            _shapely_to_dxf(msp, sil_poly, ox, oy, us, cfg.layer_visible)

        # 隠線を HIDDEN レイヤーに描画
        if cfg.show_hidden_lines:
            for x0, y0, x1, y1 in hidden_edges:
                msp.add_line(
                    (ox+x0*us, oy+y0*us), (ox+x1*us, oy+y1*us),
                    dxfattribs={"layer": cfg.layer_hidden},
                )

        # ラベル
        try:
            msp.add_text(label, dxfattribs={"layer": cfg.layer_dim,
                         "height": max(w*us*0.04, 50.0 if us>1 else 0.05)}
            ).set_placement((ox+w*us/2, oy-gap*0.35),
                            align=ezdxf.enums.TextEntityAlignment.MIDDLE_CENTER)
        except Exception:
            pass

        # 寸法線
        if cfg.add_dimensions:
            doff = h * us * cfg.dim_offset_ratio + gap * 0.25
            _add_linear_dim(msp, (ox,oy), (ox+w*us,oy), "h", -doff, cfg.layer_dim)
            _add_linear_dim(msp, (ox,oy), (ox,oy+h*us), "v", -doff, cfg.layer_dim)

    # 中心線（正面図）
    cx = ext[0]*us/2
    msp.add_line((cx, offset_y-gap*0.4), (cx, offset_y+ext[2]*us+gap*0.4),
                 dxfattribs={"layer": cfg.layer_center})
    logger.info("三面図書き込み完了")


def _write_floorplan(doc, mesh, cfg, us, offset_y=0.0):
    msp = doc.modelspace()
    verts = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    ext = verts.max(axis=0) - verts.min(axis=0)
    gap = cfg.view_gap * us

    z_min, z_max = verts[:,2].min(), verts[:,2].max()
    z_slice = z_min + (z_max - z_min) * cfg.floor_slice_ratio

    # スライス断面のエッジを抽出
    edge_set = set()
    for face in faces:
        fz = verts[face, 2]
        if not (fz.min() <= z_slice <= fz.max()):
            continue
        for i in range(3):
            a, b = face[i], face[(i+1)%3]
            key = (min(a,b), max(a,b))
            if key in edge_set:
                continue
            edge_set.add(key)
            length = np.hypot(verts[b,0]-verts[a,0], verts[b,1]-verts[a,1])
            if length >= cfg.simplify_tolerance:
                msp.add_line(
                    (verts[a,0]*us, verts[a,1]*us+offset_y),
                    (verts[b,0]*us, verts[b,1]*us+offset_y),
                    dxfattribs={"layer": cfg.layer_wall},
                )

    # 外形シルエット
    sil = _extract_silhouette(mesh, 0, 1, cfg.simplify_tolerance)
    if sil is not None:
        _shapely_to_dxf(msp, sil, 0, offset_y, us, cfg.layer_floor)

    # ラベル
    try:
        msp.add_text("FLOOR PLAN", dxfattribs={"layer": cfg.layer_dim,
                     "height": max(ext.max()*us*0.04, 50.0 if us>1 else 0.05)}
        ).set_placement((ext[0]*us/2, offset_y-gap*0.35),
                        align=ezdxf.enums.TextEntityAlignment.MIDDLE_CENTER)
    except Exception:
        pass

    # 寸法線
    if cfg.add_dimensions:
        doff = ext[1]*us*cfg.dim_offset_ratio + gap*0.25
        ox, oy = verts[:,0].min()*us, verts[:,1].min()*us+offset_y
        _add_linear_dim(msp, (ox,oy), (ox+ext[0]*us,oy), "h", -doff, cfg.layer_dim)
        _add_linear_dim(msp, (ox,oy), (ox,oy+ext[1]*us), "v", -doff, cfg.layer_dim)

    logger.info(f"平面図書き込み完了 (スライスZ={z_slice:.3f}m, エッジ={len(edge_set)}本)")


def mesh_to_dxf(
    input_path: str | Path,
    output_path: str | Path,
    config: Optional[MeshToDxfConfig] = None,
) -> Path:
    """
    3D メッシュファイルを DXF 図面に変換する。

    Parameters
    ----------
    input_path  : 入力（OBJ / FBX / GLB / STL 等）
    output_path : 出力 DXF ファイルパス
    config      : MeshToDxfConfig（省略時はデフォルト）
    """
    cfg = config or MeshToDxfConfig()
    input_path  = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mesh = load_mesh(input_path)
    us = 1000.0 if cfg.output_unit == "mm" else 1.0

    verts = np.array(mesh.vertices)
    ext = verts.max(axis=0) - verts.min(axis=0)
    logger.info(f"モデルサイズ: X={ext[0]:.3f}m Y={ext[1]:.3f}m Z={ext[2]:.3f}m")

    doc = _init_dxf(cfg)
    gap = cfg.view_gap * us

    if cfg.mode == "triview":
        _write_triview(doc, mesh, cfg, us)
    elif cfg.mode == "floorplan":
        _write_floorplan(doc, mesh, cfg, us)
    elif cfg.mode == "both":
        _write_triview(doc, mesh, cfg, us, offset_y=0.0)
        # 三面図の下に平面図を配置
        fp_offset = (ext[2] + ext[1] + gap/us*2) * us
        _write_floorplan(doc, mesh, cfg, us, offset_y=fp_offset)

    doc.saveas(str(output_path))
    logger.info(f"DXF 保存: {output_path}")
    return output_path
