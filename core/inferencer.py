"""
core/inferencer.py
==================
Zero123++ を使った多視点画像生成 → メッシュ変換の推論クラス。

設計方針:
  - Zero123PlusPlusInferencer をインスタンス化しておき、
    generate_views() で多視点画像を、
    generate_mesh() で .glb/.obj を返す。
  - モデルロードはコンストラクタで1回だけ行う（CLI/API 共用）。
  - GPU/CPU 自動フォールバック付き。

依存: torch, diffusers>=0.27, Pillow, numpy
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 設定データクラス
# ─────────────────────────────────────────────

@dataclass
class InferenceConfig:
    """推論パラメータをまとめた設定。"""
    # Zero123++ の HuggingFace モデル ID
    model_id: str = "sudo-ai/zero123plus-v1.2"
    # 生成する多視点数（6 が標準：±30° × 3 方位）
    num_views: int = 6
    # diffusion ステップ数（少ないほど速い、多いほど品質高）
    num_inference_steps: int = 36
    # 出力画像サイズ（Zero123++ は 320×320 が標準）
    image_size: int = 320
    # VRAM 節約モード（RTX 3060 8GB 以下で推奨）
    enable_attention_slicing: bool = True
    # torch dtype
    dtype: str = "float16"  # "float32" で安定性優先
    # デバイス（None で自動選択）
    device: Optional[str] = None


# ─────────────────────────────────────────────
# 推論クラス
# ─────────────────────────────────────────────

class Zero123PlusPlusInferencer:
    """
    Zero123++ 推論ラッパー。

    使い方:
        infer = Zero123PlusPlusInferencer()
        views = infer.generate_views(front_image)
        mesh_path = infer.generate_mesh(views, output_path="out.glb")
    """

    def __init__(self, config: Optional[InferenceConfig] = None):
        self.config = config or InferenceConfig()
        self.device = self._resolve_device()
        self.pipeline = None  # 遅延ロード

    # ── デバイス選択 ────────────────────────────────

    def _resolve_device(self) -> torch.device:
        if self.config.device:
            return torch.device(self.config.device)
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)} ({vram_gb:.1f} GB VRAM)")
            return dev
        logger.warning("CUDA not available, falling back to CPU (推論が非常に遅くなります)")
        return torch.device("cpu")

    # ── モデルロード ────────────────────────────────

    def load(self) -> None:
        """Zero123++ パイプラインをロードする（初回のみ）。"""
        if self.pipeline is not None:
            return

        # diffusers の Zero123++ パイプライン
        # NOTE: diffusers 0.27+ で StableDiffusionImg2ImgPipeline ベースで動作
        try:
            from diffusers import DiffusionPipeline

            dtype = torch.float16 if self.config.dtype == "float16" else torch.float32
            logger.info(f"Loading Zero123++ from {self.config.model_id} ...")

            self.pipeline = DiffusionPipeline.from_pretrained(
                self.config.model_id,
                custom_pipeline="sudo-ai/zero123plus-pipeline",
                torch_dtype=dtype,
                trust_remote_code=True,  # Zero123++ はカスタムpipeline.pyが必要
            ).to(self.device)

            if self.config.enable_attention_slicing:
                self.pipeline.enable_attention_slicing()
                logger.info("Attention slicing enabled (VRAM 節約モード)")

            logger.info("Zero123++ loaded successfully.")

        except ImportError as e:
            raise RuntimeError(
                "diffusers がインストールされていません。\n"
                "  pip install diffusers>=0.27.0 accelerate"
            ) from e

    # ── 多視点生成 ───────────────────────────────────

    def generate_views(
        self,
        input_image: Image.Image,
        seed: int = 42,
    ) -> list[Image.Image]:
        """
        入力画像（正面図）から 6 視点の画像を生成して返す。

        Parameters
        ----------
        input_image : PIL.Image  正面図（512×512 RGBA 推奨）
        seed        : 乱数シード（再現性のため固定推奨）

        Returns
        -------
        list[PIL.Image]  6 視点分の画像
        """
        self.load()

        # Zero123++ は白背景 RGB を期待する
        bg = Image.new("RGB", input_image.size, (255, 255, 255))
        if input_image.mode == "RGBA":
            bg.paste(input_image, mask=input_image.split()[3])
            input_rgb = bg
        else:
            input_rgb = input_image.convert("RGB")

        generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipeline(
            input_rgb,
            num_inference_steps=self.config.num_inference_steps,
            generator=generator,
        )

        # Zero123++ は 2×3 グリッド画像を1枚返す → 分割
        grid: Image.Image = result.images[0]
        views = self._split_grid(grid, rows=2, cols=3)
        logger.info(f"Generated {len(views)} views.")
        return views

    @staticmethod
    def _split_grid(grid: Image.Image, rows: int, cols: int) -> list[Image.Image]:
        """2D グリッド画像を個別ビューに分割する。"""
        w, h = grid.size
        cell_w, cell_h = w // cols, h // rows
        views = []
        for r in range(rows):
            for c in range(cols):
                box = (c * cell_w, r * cell_h, (c + 1) * cell_w, (r + 1) * cell_h)
                views.append(grid.crop(box))
        return views

    # ── メッシュ生成 ─────────────────────────────────

    def generate_mesh(
        self,
        views: list[Image.Image],
        output_path: str | Path = "output.glb",
        method: str = "tsdf",
    ) -> Path:
        """
        多視点画像からメッシュを生成して保存する。

        Parameters
        ----------
        views       : generate_views() の戻り値
        output_path : 保存先（.glb / .obj）
        method      : "tsdf"（Open3D TSDF）または "instant-ngp"（将来拡張用）

        Returns
        -------
        Path  保存したファイルのパス
        """
        output_path = Path(output_path)

        if method == "tsdf":
            return self._mesh_via_tsdf(views, output_path)
        else:
            raise NotImplementedError(f"Mesh method '{method}' is not implemented yet.")

    def _mesh_via_tsdf(
        self,
        views: list[Image.Image],
        output_path: Path,
    ) -> Path:
        """
        Open3D の TSDF Fusion で多視点画像 → メッシュ化。

        NOTE: 本実装は簡易版です。
              精度を高めるには DepthEstimation（DPT/Depth-Anything）で
              深度マップを推定してから TSDF に入力することを推奨します。
        """
        try:
            import open3d as o3d
        except ImportError:
            raise RuntimeError("open3d がインストールされていません: pip install open3d")

        try:
            import trimesh
        except ImportError:
            raise RuntimeError("trimesh がインストールされていません: pip install trimesh")

        logger.info("Running TSDF Fusion ...")

        # ── ① 実際の画像サイズから Intrinsic を動的に生成 ──
        # Zero123++ のグリッド分割結果は環境によってサイズが変わるため
        # 最初の view から実サイズを取得して合わせる
        first_rgb = np.array(views[0].convert("RGB"), dtype=np.uint8)
        img_h, img_w = first_rgb.shape[:2]
        logger.info(f"View image size: {img_w}×{img_h}")

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=4.0 / 512.0,
            sdf_trunc=0.04,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

        # 画像サイズから焦点距離・主点を算出（標準的な 60° FoV 相当）
        fx = fy = max(img_w, img_h) * 0.8
        cx, cy = img_w / 2.0, img_h / 2.0
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=img_w, height=img_h,
            fx=fx, fy=fy,
            cx=cx, cy=cy,
        )
        logger.info(f"Intrinsic: w={img_w}, h={img_h}, fx={fx:.1f}, cx={cx:.1f}, cy={cy:.1f}")

        for i, view in enumerate(views):
            rgb = np.array(view.convert("RGB"), dtype=np.uint8)
            # 仮深度（一定値）: 実際は推定深度マップに差し替える
            depth_val = 1.0
            depth = np.full((rgb.shape[0], rgb.shape[1]), depth_val, dtype=np.float32)

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(rgb),
                o3d.geometry.Image(depth),
                depth_scale=1.0,
                depth_trunc=3.0,
                convert_rgb_to_intensity=False,
            )

            # 仮カメラ姿勢（方位角 60° 刻みの 6 視点を想定）
            angle = i * (2 * np.pi / len(views))
            extrinsic = _make_orbit_extrinsic(angle=angle, elevation=np.radians(30), radius=1.5)

            volume.integrate(rgbd, intrinsic, extrinsic)

        mesh_o3d = volume.extract_triangle_mesh()
        mesh_o3d.compute_vertex_normals()

        # ── ② trimesh 経由で glTF/OBJ エクスポート ──
        verts = np.asarray(mesh_o3d.vertices)
        faces = np.asarray(mesh_o3d.triangles)
        colors = (np.asarray(mesh_o3d.vertex_colors) * 255).astype(np.uint8)

        mesh_tm = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=colors)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mesh_tm.export(str(output_path))

        logger.info(f"Mesh saved → {output_path}  "
                    f"(verts={len(verts)}, faces={len(faces)})")
        return output_path


# ─────────────────────────────────────────────
# 内部ユーティリティ
# ─────────────────────────────────────────────

def _make_orbit_extrinsic(angle: float, elevation: float, radius: float) -> np.ndarray:
    """
    球面上の軌道カメラ姿勢 (4×4 extrinsic matrix) を返す。

    Parameters
    ----------
    angle     : 方位角 [rad]
    elevation : 仰角 [rad]
    radius    : カメラ距離
    """
    x = radius * np.cos(elevation) * np.cos(angle)
    y = radius * np.cos(elevation) * np.sin(angle)
    z = radius * np.sin(elevation)
    eye = np.array([x, y, z])
    center = np.zeros(3)
    up = np.array([0.0, 0.0, 1.0])

    f = center - eye
    f /= np.linalg.norm(f)
    r = np.cross(up, f)
    r_norm = np.linalg.norm(r)
    if r_norm < 1e-6:
        r = np.array([1.0, 0.0, 0.0])
    else:
        r /= r_norm
    u = np.cross(f, r)

    mat = np.eye(4)
    mat[:3, 0] = r
    mat[:3, 1] = u
    mat[:3, 2] = f
    mat[:3, 3] = eye
    return np.linalg.inv(mat)
