"""
core/mesh_renderer.py
=====================
押し出し3Dメッシュ → Zero123++入力用シルエット画像 の変換モジュール。

処理フロー:
  1. trimeshメッシュをカメラ視点でラスタライズ
  2. 白背景・黒シルエットの画像に変換（Zero123++が期待する形式）
  3. 複数視点（正面・側面・斜め俯瞰）で出力

Open3Dのヘッドレスレンダリング or matplotlib + 射影変換 の2方式を
環境に応じて自動選択する。

依存: trimesh, numpy, Pillow, matplotlib, (open3d optional)
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter, ImageOps

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 設定
# ─────────────────────────────────────────────

@dataclass
class RenderConfig:
    """レンダリング設定。"""
    image_size: int = 512          # 出力画像の一辺 [px]
    bg_color: tuple = (255, 255, 255)  # 背景色 (RGB)
    mesh_color: tuple = (30, 30, 30)   # メッシュ色 (RGB)
    margin: float = 0.10           # 余白比率（画像サイズに対して）
    line_width: float = 1.5        # エッジ線幅 [px]
    # 生成する視点のリスト: (elevation_deg, azimuth_deg)
    # elevation: 0=水平, 30=斜め上, 90=真上
    # azimuth: 0=正面, 90=左側面, 180=背面, 270=右側面
    viewpoints: list[tuple[float, float]] = field(default_factory=lambda: [
        (25.0,   0.0),   # 正面・中程度（TripoSR推奨: 建物壁面が見える）
        (25.0,  90.0),   # 左側面
        (25.0, 180.0),   # 背面
        (25.0, 270.0),   # 右側面
    ])
    # Zero123++/TripoSR 入力として使うメインビュー（インデックス）
    primary_view_idx: int = 0


# ─────────────────────────────────────────────
# カメラ行列ユーティリティ
# ─────────────────────────────────────────────

def _look_at(
    eye: np.ndarray,
    center: np.ndarray,
    up: np.ndarray = np.array([0, 0, 1], dtype=float),
) -> np.ndarray:
    """Look-at 変換行列 (4×4) を返す。"""
    f = center - eye
    f_norm = np.linalg.norm(f)
    if f_norm < 1e-9:
        return np.eye(4)
    f = f / f_norm

    r = np.cross(f, up)
    r_norm = np.linalg.norm(r)
    if r_norm < 1e-9:
        up = np.array([0, 1, 0], dtype=float)
        r = np.cross(f, up)
        r_norm = np.linalg.norm(r)
    r = r / r_norm
    u = np.cross(r, f)

    mat = np.eye(4)
    mat[0, :3] = r
    mat[1, :3] = u
    mat[2, :3] = -f
    mat[:3, 3] = -mat[:3, :3] @ eye
    return mat


def _orbit_camera(
    elevation_deg: float,
    azimuth_deg: float,
    radius: float,
    center: np.ndarray,
) -> np.ndarray:
    """
    中心 center の周りを radius 距離で周回するカメラ位置を返す。
    elevation: 0=水平  90=真上 [deg]
    azimuth  : 0=+X方向  90=+Y方向 [deg]（右手系）
    """
    el = np.radians(elevation_deg)
    az = np.radians(azimuth_deg)
    x = center[0] + radius * np.cos(el) * np.cos(az)
    y = center[1] + radius * np.cos(el) * np.sin(az)
    z = center[2] + radius * np.sin(el)
    return np.array([x, y, z], dtype=float)


# ─────────────────────────────────────────────
# matplotlib ベースのソフトウェアレンダラ
# ─────────────────────────────────────────────

def _render_view_matplotlib(
    mesh,  # trimesh.Trimesh
    elevation_deg: float,
    azimuth_deg: float,
    cfg: RenderConfig,
) -> Image.Image:
    """
    matplotlib でワイヤーフレーム + 塗りつぶしシルエットをレンダリングする。

    Open3Dのヘッドレスレンダリングが使えない環境向けのフォールバック。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.collections import LineCollection

    verts = np.array(mesh.vertices)
    faces = np.array(mesh.faces)

    # ── カメラ設定 ──────────────────────────────
    center = verts.mean(axis=0)
    extents = verts.max(axis=0) - verts.min(axis=0)
    radius = np.linalg.norm(extents) * 0.9

    # ── 面の頂点を取得 ──────────────────────────
    tri_verts = verts[faces]  # (F, 3, 3)

    # ── matplotlib 3D プロット ──────────────────
    dpi = 100
    fig = plt.figure(figsize=(cfg.image_size/dpi, cfg.image_size/dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor([c/255 for c in cfg.bg_color])
    fig.patch.set_facecolor([c/255 for c in cfg.bg_color])

    # 面コレクション（塗りつぶし）
    poly = Poly3DCollection(
        tri_verts,
        alpha=1.0,
        facecolor=[c/255 for c in cfg.mesh_color],
        edgecolor=[c/255 for c in cfg.bg_color],
        linewidth=0.1,
    )
    ax.add_collection3d(poly)

    # 軸範囲
    margin_m = max(extents) * cfg.margin
    ax.set_xlim(verts[:,0].min()-margin_m, verts[:,0].max()+margin_m)
    ax.set_ylim(verts[:,1].min()-margin_m, verts[:,1].max()+margin_m)
    ax.set_zlim(verts[:,2].min()-margin_m, verts[:,2].max()+margin_m)
    ax.set_axis_off()

    # カメラ視点
    ax.view_init(elev=elevation_deg, azim=azimuth_deg)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)

    img = Image.open(buf).convert("RGB")
    img = img.resize((cfg.image_size, cfg.image_size), Image.LANCZOS)
    return img


# ─────────────────────────────────────────────
# シルエット変換
# ─────────────────────────────────────────────

def _to_silhouette(img: Image.Image, cfg: RenderConfig) -> Image.Image:
    """
    レンダリング画像をZero123++入力用シルエットに変換する。

    - 白背景・暗い前景 → 二値化してシャープなシルエットに
    - メッシュ色(暗)と背景色(白)のコントラストを強調
    - ガウシアンブラーで輪郭を少し滑らかに
    """
    # グレースケール化
    gray = img.convert("L")

    # 二値化（閾値180: 白背景の暗い部分をシルエットとして抽出）
    binary = gray.point(lambda x: 0 if x < 180 else 255, "L")

    # 少し膨張させてシルエットを安定させる
    binary = binary.filter(ImageFilter.MaxFilter(3))

    # ガウシアンブラーで輪郭を滑らかに
    soft = binary.filter(ImageFilter.GaussianBlur(radius=1))

    # 最終二値化
    final = soft.point(lambda x: 0 if x < 220 else 255, "L")

    # RGBA に変換（Zero123++ は RGBA を期待）
    rgba = Image.new("RGBA", final.size, (255, 255, 255, 255))
    rgba_arr = np.array(rgba)
    mask = np.array(final) < 128
    rgba_arr[mask] = [int(c) for c in cfg.mesh_color] + [255]
    return Image.fromarray(rgba_arr)


# ─────────────────────────────────────────────
# メイン API
# ─────────────────────────────────────────────

def render_mesh_views(
    mesh,
    config: Optional[RenderConfig] = None,
    output_dir: Optional[str | Path] = None,
) -> dict[str, Image.Image]:
    """
    trimesh.Trimesh を複数視点でレンダリングしてシルエット画像を返す。

    Parameters
    ----------
    mesh       : trimesh.Trimesh（単位 m）
    config     : RenderConfig（省略時はデフォルト）
    output_dir : 指定した場合、PNG として保存する

    Returns
    -------
    dict: {
        "primary": PIL.Image,    # Zero123++へ渡すメインビュー
        "view_0": PIL.Image,
        "view_1": PIL.Image,
        ...
    }
    """
    cfg = config or RenderConfig()
    results: dict[str, Image.Image] = {}

    for i, (el, az) in enumerate(cfg.viewpoints):
        logger.info(f"  view_{i}: elevation={el:.0f}° azimuth={az:.0f}°")

        # matplotlib ベースレンダリング
        rendered = _render_view_matplotlib(mesh, el, az, cfg)

        # シルエット変換
        silhouette = _to_silhouette(rendered, cfg)
        results[f"view_{i}"] = silhouette

        if i == cfg.primary_view_idx:
            results["primary"] = silhouette

    # 保存
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, img in results.items():
            p = out / f"{name}.png"
            img.save(p)
            logger.info(f"  saved: {p}")

    logger.info(f"レンダリング完了: {len(cfg.viewpoints)}視点")
    return results


# 視点プリセット: --view-angle オプション用
VIEW_ANGLE_PRESETS: dict[str, tuple[float, float]] = {
    "front":       (25.0,   0.0),   # 正面（デフォルト・TripoSR推奨）
    "front_low":   (15.0,   0.0),   # 正面・低め
    "front_high":  (40.0,   0.0),   # 正面・高め
    "corner":      (25.0,  45.0),   # 斜め正面
    "corner_low":  (15.0,  45.0),   # 斜め・低め
    "corner_high": (40.0,  45.0),   # 斜め・高め
    "side":        (25.0,  90.0),   # 側面
    "top":         (75.0,  45.0),   # 俯瞰
    "iso":         (35.0,  45.0),   # アイソメトリック
}


def render_mesh_for_zero123(
    mesh,
    image_size: int = 512,
    output_dir: Optional[str | Path] = None,
    view_angle: str = "front",
) -> Image.Image:
    """
    Zero123++/TripoSR 入力用のメインビュー画像を1枚返す簡易 API。

    Parameters
    ----------
    view_angle : プリセット名（"front"/"corner"/"side"/"top"等）
                 または "EL,AZ" 形式で直接指定（例: "25,45"）

    Returns
    -------
    PIL.Image  RGB 512×512 のシルエット画像
    """
    # view_angle の解釈
    if view_angle in VIEW_ANGLE_PRESETS:
        el, az = VIEW_ANGLE_PRESETS[view_angle]
    elif "," in view_angle:
        try:
            el, az = [float(v.strip()) for v in view_angle.split(",")]
        except ValueError:
            logger.warning(f"view_angle '{view_angle}' を解析できません → front を使用")
            el, az = VIEW_ANGLE_PRESETS["front"]
    else:
        logger.warning(f"未知の view_angle '{view_angle}' → front を使用")
        el, az = VIEW_ANGLE_PRESETS["front"]

    logger.info(f"レンダリング視点: {view_angle} (elevation={el}°, azimuth={az}°)")

    cfg = RenderConfig(
        image_size=image_size,
        viewpoints=[(el, az)],
        primary_view_idx=0,
    )
    views = render_mesh_views(mesh, config=cfg, output_dir=output_dir)
    return views["primary"]
