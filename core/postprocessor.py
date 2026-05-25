"""
core/postprocessor.py
=====================
推論後のメッシュに対して、ゲームエンジン向けの後処理を行う純粋関数群。

処理内容:
  1. スケール補正   - CADMeta.dimensions を使って寸法を合わせる
  2. メッシュ修復   - 重複頂点・退化面の除去
  3. LOD 生成      - Quadric Decimation で段階的ポリゴン削減
  4. UV 展開       - Blender Python API（オプション）または trimesh
  5. glTF エクスポート

依存: trimesh, open3d, numpy
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 設定データクラス
# ─────────────────────────────────────────────

@dataclass
class PostprocessConfig:
    """後処理パラメータ。"""
    # LOD ポリゴン数の上限リスト（LOD0 が最高解像度）
    lod_face_counts: list[int] = field(default_factory=lambda: [10_000, 4_000, 1_000])
    # 最終スケールの単位 ("m" → Unity デフォルト, "cm" → UE デフォルト)
    target_unit: str = "m"
    # UV展開をするかどうか（trimesh ベースの簡易版）
    unwrap_uv: bool = False
    # 頂点マージ許容距離（メッシュ修復用）
    merge_tol: float = 1e-6


# ─────────────────────────────────────────────
# スケール補正
# ─────────────────────────────────────────────

def apply_scale(
    mesh,           # trimesh.Trimesh
    cad_dimensions_mm: np.ndarray,
    target_unit: str = "m",
):
    """
    CAD の実寸 (mm) に合わせてメッシュをリスケールする。

    Zero123++ が生成するメッシュは正規化空間 [-1, 1] に収まっているため、
    CADMeta.dimensions を使って実寸に戻す。

    Parameters
    ----------
    mesh               : trimesh.Trimesh  入力メッシュ
    cad_dimensions_mm  : (3,) ndarray  X/Y/Z の実寸 [mm]
    target_unit        : "m" / "cm" / "mm"
    """
    unit_scale = {"m": 1e-3, "cm": 1e-1, "mm": 1.0}[target_unit]

    # 現在のメッシュの bounding box サイズ
    extents = mesh.bounding_box.extents  # (3,)
    max_extent = extents.max()
    if max_extent < 1e-9:
        logger.warning("Mesh has near-zero extent, skipping scale.")
        return mesh

    # CAD の最大寸法（mm → target_unit）
    target_size = cad_dimensions_mm.max() * unit_scale

    scale_factor = target_size / max_extent
    mesh.apply_scale(scale_factor)
    logger.info(f"Scale applied: ×{scale_factor:.4f}  "
                f"(target max dim = {target_size:.4f} {target_unit})")
    return mesh


# ─────────────────────────────────────────────
# メッシュ修復
# ─────────────────────────────────────────────

def repair_mesh(mesh):
    """
    重複頂点・退化三角形・法線の修復。
    trimesh のバージョン差異を吸収して処理する。
    """
    import trimesh
    import numpy as np

    # 重複頂点をマージ
    mesh.merge_vertices()

    # 退化面の除去（trimesh 4.x でAPIが変わったため両対応）
    try:
        # trimesh 4.x 以降
        unique_mask = trimesh.triangles.nondegenerate(mesh.triangles)
        mesh.update_faces(unique_mask)
    except Exception:
        try:
            # trimesh 3.x 系
            mesh.remove_degenerate_faces()
            mesh.remove_duplicate_faces()
        except Exception:
            pass  # どちらも失敗した場合はスキップ

    # 重複面の除去（4.x対応）
    try:
        mesh.update_faces(mesh.unique_faces())
    except Exception:
        try:
            mesh.remove_duplicate_faces()
        except Exception:
            pass

    # 法線の再計算
    mesh.fix_normals()

    logger.info(f"Mesh repaired: verts={len(mesh.vertices)}, faces={len(mesh.faces)}")
    return mesh


# ─────────────────────────────────────────────
# LOD 生成
# ─────────────────────────────────────────────

def generate_lods(
    mesh,
    lod_face_counts: list[int],
) -> dict[str, object]:
    """
    Quadric Decimation で LOD メッシュを生成する。

    Returns
    -------
    dict: {"LOD0": trimesh, "LOD1": trimesh, ...}
    """
    lods: dict[str, object] = {"LOD0": mesh}

    for i, target in enumerate(lod_face_counts[1:], start=1):
        if len(mesh.faces) <= target:
            logger.info(f"LOD{i}: Already <= {target} faces, skipping decimation.")
            lods[f"LOD{i}"] = mesh
            continue

        try:
            import open3d as o3d

            mesh_o3d = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(mesh.vertices),
                triangles=o3d.utility.Vector3iVector(mesh.faces),
            )
            mesh_o3d.compute_vertex_normals()

            ratio = target / len(mesh.faces)
            mesh_simplified = mesh_o3d.simplify_quadric_decimation(
                target_number_of_triangles=target
            )

            import trimesh as tm
            lod_mesh = tm.Trimesh(
                vertices=np.asarray(mesh_simplified.vertices),
                faces=np.asarray(mesh_simplified.triangles),
            )
            lod_mesh.fix_normals()
            lods[f"LOD{i}"] = lod_mesh
            logger.info(f"LOD{i}: {len(mesh.faces)} → {len(lod_mesh.faces)} faces")

        except Exception as e:
            logger.warning(f"LOD{i} generation failed: {e}")
            lods[f"LOD{i}"] = mesh  # フォールバック

    return lods


# ─────────────────────────────────────────────
# エクスポート
# ─────────────────────────────────────────────

def export_meshes(
    lods: dict[str, object],
    output_dir: str | Path,
    base_name: str = "asset",
    export_format: str = "obj",
) -> dict[str, Path]:
    """
    LOD ごとにメッシュを書き出す。

    Parameters
    ----------
    export_format : "obj" (Unity標準・追加パッケージ不要) or "glb" (glTFast必要)

    Returns
    -------
    dict: {"LOD0": Path, "LOD1": Path, ...}
    """
    fmt = export_format.lower().lstrip(".")
    if fmt not in ("obj", "glb", "gltf"):
        logger.warning(f"Unknown format '{fmt}', falling back to obj")
        fmt = "obj"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = {}

    for lod_name, mesh in lods.items():
        out_path = output_dir / f"{base_name}_{lod_name}.{fmt}"
        mesh.export(str(out_path))
        saved[lod_name] = out_path
        logger.info(f"Exported: {out_path}")

    return saved


# 後方互換エイリアス
def export_gltf(lods, output_dir, base_name="asset"):
    return export_meshes(lods, output_dir, base_name, export_format="glb")


# ─────────────────────────────────────────────
# ワンショット API
# ─────────────────────────────────────────────

def postprocess(
    mesh_path: str | Path,
    cad_dimensions_mm: np.ndarray,
    output_dir: str | Path,
    base_name: str = "asset",
    config: Optional[PostprocessConfig] = None,
    cad_unit: str = "mm",
    export_format: str = "obj",
) -> dict[str, Path]:
    """
    メッシュファイルを受け取り、スケール補正・修復・LOD生成・エクスポートを一括実行。

    Parameters
    ----------
    mesh_path          : 推論が出力した .glb/.obj
    cad_dimensions_mm  : CADMeta.dimensions（CAD座標系のまま）
    output_dir         : 出力ディレクトリ
    base_name          : 出力ファイルの基名
    config             : PostprocessConfig（省略時はデフォルト）
    cad_unit           : CAD図面の単位 ("mm","cm","m","inch","foot")
    export_format      : 出力フォーマット "obj"(デフォルト) / "glb"
    """
    import trimesh

    # CAD座標値 → mm に統一してから apply_scale に渡す
    _to_mm = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "inch": 25.4, "foot": 304.8}
    scale_to_mm = _to_mm.get(cad_unit, 1.0)
    dimensions_in_mm = cad_dimensions_mm * scale_to_mm

    logger.info(f"CAD unit: {cad_unit}  ->  dimensions in mm: "
                f"{dimensions_in_mm[0]:.1f} / {dimensions_in_mm[1]:.1f} / {dimensions_in_mm[2]:.1f}")

    cfg = config or PostprocessConfig()
    mesh = trimesh.load(str(mesh_path), force="mesh")

    mesh = repair_mesh(mesh)
    mesh = apply_scale(mesh, dimensions_in_mm, target_unit=cfg.target_unit)
    lods = generate_lods(mesh, cfg.lod_face_counts)
    return export_meshes(lods, output_dir, base_name, export_format=export_format)
