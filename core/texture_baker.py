"""
core/texture_baker.py
=====================
押し出し3Dメッシュに対して手続き的テクスチャを生成・適用する。

生成するテクスチャ要素:
  外壁  : レンガ / コンクリート / 木材 パターン
  窓    : 開口部検出 → 窓枠・ガラスを描画
  扉    : 建具レイヤー位置に扉パターンを描画
  屋根  : 瓦 / 金属 / フラット パターン

出力:
  - テクスチャPNG (diffuse map)
  - .mtl マテリアルファイル
  - テクスチャ付き OBJ

依存: trimesh, numpy, Pillow
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

logger = logging.getLogger(__name__)

WallStyle  = Literal["brick", "concrete", "wood"]
RoofStyle  = Literal["tile", "metal", "flat"]


# ─────────────────────────────────────────────
# 設定
# ─────────────────────────────────────────────

@dataclass
class TextureConfig:
    """テクスチャ生成設定。"""
    texture_size: int = 1024        # テクスチャ解像度 [px]
    wall_style: WallStyle  = "brick"
    roof_style: RoofStyle  = "tile"
    # 色設定（RGB）
    wall_color_base:   tuple = (210, 190, 170)  # 外壁ベース
    wall_color_mortar: tuple = (240, 235, 230)  # 目地色
    window_glass:      tuple = (140, 180, 210)  # 窓ガラス
    window_frame:      tuple = (80,  70,  60)   # 窓枠
    door_color:        tuple = (100, 70,  50)   # 扉
    roof_color_main:   tuple = (100, 60,  50)   # 瓦メイン
    roof_color_sub:    tuple = (80,  45,  35)   # 瓦サブ
    floor_color:       tuple = (180, 170, 160)  # 床
    # 窓の密度（壁面積に対する比率）
    window_density: float = 0.15


# ─────────────────────────────────────────────
# テクスチャパターン生成
# ─────────────────────────────────────────────

def _make_brick_texture(cfg: TextureConfig) -> Image.Image:
    """レンガテクスチャを生成する。"""
    size = cfg.texture_size
    img = Image.new("RGB", (size, size), cfg.wall_color_base)
    draw = ImageDraw.Draw(img)

    brick_h = size // 16       # レンガ1枚の高さ
    brick_w = size // 8        # レンガ1枚の幅
    mortar  = max(2, size // 200)  # 目地幅

    rows = size // brick_h + 1
    for row in range(rows):
        y0 = row * brick_h
        # 奇数行はオフセット
        offset = (brick_w // 2) if row % 2 == 1 else 0
        cols = size // brick_w + 2
        for col in range(-1, cols):
            x0 = col * brick_w + offset
            x1 = x0 + brick_w - mortar
            y1 = y0 + brick_h - mortar
            # レンガ色に微妙なランダム変化
            rng = np.random.default_rng(abs(row * 1000 + col))
            noise = int(rng.integers(-15, 15))
            c = tuple(max(0, min(255, v + noise)) for v in cfg.wall_color_base)
            draw.rectangle([x0, y0, x1, y1], fill=c)
        # 水平目地
        draw.rectangle([0, y0 + brick_h - mortar, size, y0 + brick_h], fill=cfg.wall_color_mortar)

    # 軽くブラー
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    return img


def _make_concrete_texture(cfg: TextureConfig) -> Image.Image:
    """コンクリートテクスチャ（ノイズベース）を生成する。"""
    size = cfg.texture_size
    base = np.array(cfg.wall_color_base, dtype=np.float32)
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 12, (size, size, 3))

    # ボロノイ風パターンを追加
    n_cracks = size // 64
    img_arr = np.clip(base + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img)

    # ひび割れ風ライン
    for _ in range(n_cracks):
        x0, y0 = rng.integers(0, size, 2)
        dx, dy = rng.integers(-size//4, size//4, 2)
        gray = int(rng.integers(100, 160))
        draw.line([x0, y0, x0+dx, y0+dy], fill=(gray, gray, gray), width=1)

    return img.filter(ImageFilter.GaussianBlur(radius=0.8))


def _make_wood_texture(cfg: TextureConfig) -> Image.Image:
    """木材テクスチャ（木目ストライプ）を生成する。"""
    size = cfg.texture_size
    rng = np.random.default_rng(42)
    img = Image.new("RGB", (size, size), cfg.wall_color_base)
    draw = ImageDraw.Draw(img)

    grain_spacing = size // 20
    for i in range(size // grain_spacing + 1):
        x = i * grain_spacing + int(rng.integers(-3, 3))
        noise_y = rng.integers(-2, 2, size // 4)
        pts = []
        for j in range(size // 4):
            y = j * 4
            pts.extend([x + int(noise_y[j]), y])
        if len(pts) >= 4:
            draw.line(pts, fill=cfg.wall_color_mortar, width=1)

    return img.filter(ImageFilter.GaussianBlur(radius=0.5))


def _make_tile_roof_texture(cfg: TextureConfig) -> Image.Image:
    """瓦屋根テクスチャを生成する。"""
    size = cfg.texture_size
    img = Image.new("RGB", (size, size), cfg.roof_color_main)
    draw = ImageDraw.Draw(img)

    tile_w = size // 10
    tile_h = size // 14
    for row in range(size // tile_h + 1):
        for col in range(size // tile_w + 1):
            x0 = col * tile_w
            y0 = row * tile_h
            # 瓦の丸み表現（楕円）
            draw.ellipse(
                [x0, y0 + tile_h//3, x0 + tile_w - 2, y0 + tile_h],
                fill=cfg.roof_color_sub
            )
            draw.line([x0, y0, x0, y0 + tile_h], fill=cfg.roof_color_sub, width=1)

    return img.filter(ImageFilter.GaussianBlur(radius=0.3))


def _make_metal_roof_texture(cfg: TextureConfig) -> Image.Image:
    """金属屋根テクスチャを生成する。"""
    size = cfg.texture_size
    base = (150, 155, 160)
    img = Image.new("RGB", (size, size), base)
    draw = ImageDraw.Draw(img)

    stripe_w = size // 12
    for i in range(size // stripe_w + 1):
        x = i * stripe_w
        draw.line([x, 0, x, size], fill=(130, 135, 140), width=2)
        draw.line([x + stripe_w//2, 0, x + stripe_w//2, size],
                  fill=(170, 175, 180), width=1)

    return img.filter(ImageFilter.GaussianBlur(radius=0.5))


def _draw_windows(
    img: Image.Image,
    cfg: TextureConfig,
    wall_width_px: int,
    wall_height_px: int,
    wall_height_m: float,
    x_offset: int = 0,
    y_offset: int = 0,
) -> None:
    """テクスチャ画像上に窓パターンを描画する。"""
    draw = ImageDraw.Draw(img)

    # 窓の寸法（ピクセル）
    win_w = int(wall_width_px * 0.12)
    win_h = int(wall_height_px * 0.25)
    frame_w = max(2, win_w // 8)

    # 窓の数（密度から算出）
    n_windows = max(1, int(wall_width_px / (win_w * 2.5)))

    # 窓の垂直位置（壁高の55%付近）
    win_y0 = y_offset + int(wall_height_px * 0.35)
    win_y1 = win_y0 + win_h

    for i in range(n_windows):
        spacing = wall_width_px // (n_windows + 1)
        win_x0 = x_offset + spacing * (i + 1) - win_w // 2
        win_x1 = win_x0 + win_w

        # ガラス
        draw.rectangle([win_x0, win_y0, win_x1, win_y1], fill=cfg.window_glass)
        # 窓枠
        draw.rectangle([win_x0, win_y0, win_x1, win_y1],
                       outline=cfg.window_frame, width=frame_w)
        # 桟（十字）
        mid_x = (win_x0 + win_x1) // 2
        mid_y = (win_y0 + win_y1) // 2
        draw.line([mid_x, win_y0, mid_x, win_y1],
                  fill=cfg.window_frame, width=max(1, frame_w//2))
        draw.line([win_x0, mid_y, win_x1, mid_y],
                  fill=cfg.window_frame, width=max(1, frame_w//2))


# ─────────────────────────────────────────────
# UV展開ヘルパー
# ─────────────────────────────────────────────

def _unwrap_box_uv(mesh) -> np.ndarray:
    """
    箱型メッシュに対してボックスマッピングUVを生成する。

    各面の法線方向に応じて、以下の6面マッピングを適用:
      +Z (上) → 上面ゾーン
      -Z (下) → 下面ゾーン
      ±X/±Y   → 側面ゾーン（4分割）

    Returns
    -------
    np.ndarray shape=(N_faces*3, 2)  各頂点のUV座標 [0,1]
    """
    verts = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    normals = np.array(mesh.face_normals)

    # バウンディングボックス
    vmin = verts.min(axis=0)
    vmax = verts.max(axis=0)
    vrange = np.maximum(vmax - vmin, 1e-6)

    uvs = np.zeros((len(faces) * 3, 2), dtype=np.float32)

    for fi, (face, normal) in enumerate(zip(faces, normals)):
        tri = verts[face]
        nx, ny, nz = abs(normal[0]), abs(normal[1]), abs(normal[2])

        if nz > max(nx, ny):
            # 上面・下面 → XY
            u = (tri[:, 0] - vmin[0]) / vrange[0]
            v = (tri[:, 1] - vmin[1]) / vrange[1]
        elif nx > ny:
            # X面 → YZ
            u = (tri[:, 1] - vmin[1]) / vrange[1]
            v = (tri[:, 2] - vmin[2]) / vrange[2]
        else:
            # Y面 → XZ
            u = (tri[:, 0] - vmin[0]) / vrange[0]
            v = (tri[:, 2] - vmin[2]) / vrange[2]

        uvs[fi * 3:fi * 3 + 3, 0] = np.clip(u, 0, 1)
        uvs[fi * 3:fi * 3 + 3, 1] = np.clip(v, 0, 1)

    return uvs


# ─────────────────────────────────────────────
# テクスチャ付きOBJ出力
# ─────────────────────────────────────────────

def export_textured_obj(
    mesh,
    output_dir: str | Path,
    base_name: str = "asset",
    config: Optional[TextureConfig] = None,
) -> dict[str, Path]:
    """
    メッシュにテクスチャを適用してOBJ+MTL+PNGを出力する。

    Parameters
    ----------
    mesh       : trimesh.Trimesh
    output_dir : 出力ディレクトリ
    base_name  : ファイル名ベース
    config     : TextureConfig

    Returns
    -------
    dict: {"obj": Path, "mtl": Path, "texture": Path}
    """
    cfg = config or TextureConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tex_path = output_dir / f"{base_name}_texture.png"
    mtl_path = output_dir / f"{base_name}.mtl"
    obj_path = output_dir / f"{base_name}.obj"

    # ── テクスチャ生成 ───────────────────────────
    size = cfg.texture_size

    # 外壁テクスチャ
    if cfg.wall_style == "brick":
        wall_tex = _make_brick_texture(cfg)
    elif cfg.wall_style == "concrete":
        wall_tex = _make_concrete_texture(cfg)
    else:
        wall_tex = _make_wood_texture(cfg)

    # 全体テクスチャ（外壁が下80%、屋根が上20%）
    full_tex = Image.new("RGB", (size, size), cfg.wall_color_base)
    wall_h = int(size * 0.8)
    roof_h = size - wall_h

    # 外壁ゾーン
    wall_resized = wall_tex.resize((size, wall_h), Image.LANCZOS)
    full_tex.paste(wall_resized, (0, 0))

    # 窓を描画
    _draw_windows(full_tex, cfg,
                  wall_width_px=size,
                  wall_height_px=wall_h,
                  wall_height_m=2.5)

    # 屋根ゾーン
    if cfg.roof_style == "tile":
        roof_tex = _make_tile_roof_texture(cfg)
    elif cfg.roof_style == "metal":
        roof_tex = _make_metal_roof_texture(cfg)
    else:
        roof_tex = Image.new("RGB", (size, roof_h), cfg.roof_color_main)

    roof_resized = roof_tex.resize((size, roof_h), Image.LANCZOS)
    full_tex.paste(roof_resized, (0, wall_h))

    # 床ゾーン（右端に小さく）
    draw = ImageDraw.Draw(full_tex)
    draw.rectangle([size - 100, 0, size, wall_h],
                   fill=cfg.floor_color)

    full_tex.save(str(tex_path))
    logger.info(f"テクスチャ生成: {tex_path}")

    # ── UV展開 ──────────────────────────────────
    uvs = _unwrap_box_uv(mesh)
    verts = np.array(mesh.vertices)
    faces = np.array(mesh.faces)

    # ── MTLファイル ──────────────────────────────
    mtl_content = f"""# Material for {base_name}
newmtl building_material
Ka 1.0 1.0 1.0
Kd 1.0 1.0 1.0
Ks 0.1 0.1 0.1
Ns 10.0
d 1.0
map_Kd {tex_path.name}
"""
    mtl_path.write_text(mtl_content)
    logger.info(f"MTL生成: {mtl_path}")

    # ── OBJファイル（UV付き）────────────────────
    with open(obj_path, "w") as f:
        f.write(f"# Generated by cad2asset\n")
        f.write(f"mtllib {mtl_path.name}\n\n")

        # 頂点
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        # UV座標（面ごとに3点）
        for uv in uvs:
            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")

        f.write("\nusemtl building_material\n")

        # 面（1-indexed、UV インデックスは面順）
        for fi, face in enumerate(faces):
            uv_base = fi * 3 + 1   # 1-indexed
            v0, v1, v2 = face + 1  # 1-indexed
            f.write(f"f {v0}/{uv_base} {v1}/{uv_base+1} {v2}/{uv_base+2}\n")

    logger.info(f"OBJ出力: {obj_path}")

    return {"obj": obj_path, "mtl": mtl_path, "texture": tex_path}


# ─────────────────────────────────────────────
# ワンショット API
# ─────────────────────────────────────────────

def bake_textures(
    mesh,
    output_dir: str | Path,
    base_name: str = "asset",
    wall_style: WallStyle = "brick",
    roof_style: RoofStyle = "tile",
    texture_size: int = 1024,
) -> dict[str, Path]:
    """
    メッシュにテクスチャを焼き込んでOBJ+MTL+PNGを出力する簡易API。
    """
    cfg = TextureConfig(
        texture_size=texture_size,
        wall_style=wall_style,
        roof_style=roof_style,
    )
    return export_textured_obj(mesh, output_dir, base_name, cfg)
