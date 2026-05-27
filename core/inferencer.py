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

    # ── 押し出しメッシュからのエンドツーエンドパイプライン ────────

    def generate_from_floor_plan_mesh(
        self,
        mesh,
        output_path: str | Path = "output.glb",
        image_size: int = 512,
        seed: int = 42,
        render_output_dir=None,
    ) -> Path:
        """
        押し出し3DメッシュからZero123++推論→メッシュ化を一括実行する。

        Route C（建築平面図）専用のエンドツーエンドパイプライン:
          trimesh.Trimesh
            → mesh_renderer でシルエット画像生成
            → Zero123++ で多視点画像生成
            → Depth-Anything-V2 で深度推定
            → TSDF Fusion でメッシュ化

        Parameters
        ----------
        mesh            : trimesh.Trimesh（押し出し済み、単位 m）
        output_path     : 出力ファイルパス（.glb / .obj）
        image_size      : レンダリング・推論の画像サイズ（デフォルト 512）
        seed            : Zero123++ の乱数シード
        render_output_dir: レンダリング画像の保存先（デバッグ用、省略可）

        Returns
        -------
        Path  生成したメッシュファイルのパス
        """
        from core.mesh_renderer import render_mesh_for_zero123

        logger.info("=== Route C: 押し出しメッシュ → Zero123++ パイプライン ===")

        # ① 押し出しメッシュをレンダリング
        logger.info("Step 1: メッシュをシルエット画像にレンダリング中...")
        input_image = render_mesh_for_zero123(
            mesh,
            image_size=image_size,
            output_dir=render_output_dir,
        )
        logger.info(f"  シルエット画像生成完了: {input_image.size}")

        # ② Zero123++ で多視点生成
        logger.info("Step 2: Zero123++ で多視点生成中...")
        views = self.generate_views(input_image, seed=seed)
        logger.info(f"  {len(views)}視点生成完了")

        # 生成ビューを保存（デバッグ用）
        if render_output_dir is not None:
            gen_dir = Path(render_output_dir) / "generated"
            gen_dir.mkdir(parents=True, exist_ok=True)
            for i, v in enumerate(views):
                v.save(gen_dir / f"view_{i:02d}.png")

        # ③ メッシュ生成
        logger.info("Step 3: TSDF Fusion でメッシュ化中...")
        return self.generate_mesh(views, output_path=output_path)

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

    # ── Depth-Anything-V2 深度推定 ───────────────────

    def _estimate_depth(self, image: Image.Image) -> np.ndarray:
        """
        Depth-Anything-V2 で単眼深度推定を行い、正規化された深度マップを返す。

        Returns
        -------
        np.ndarray shape=(H,W) dtype=float32  値域 [0.2, 2.0] (メートル相当)
        """
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError:
            logger.warning("transformers 未インストール → 仮深度にフォールバック")
            arr = np.array(image.convert("RGB"))
            return np.full((arr.shape[0], arr.shape[1]), 1.0, dtype=np.float32)

        # モデルの遅延ロード（初回のみDL ~400MB）
        if not hasattr(self, "_depth_pipe") or self._depth_pipe is None:
            logger.info("Depth-Anything-V2 をロード中 (初回のみDL ~400MB)...")
            self._depth_pipe = hf_pipeline(
                "depth-estimation",
                model="depth-anything/Depth-Anything-V2-Small-hf",
                device=0 if str(self.device) == "cuda" else -1,
            )
            logger.info("Depth-Anything-V2 ロード完了")

        # 推論
        result = self._depth_pipe(image.convert("RGB"))
        depth_pil = result["depth"]                          # PIL.Image (grayscale)
        depth_arr = np.array(depth_pil, dtype=np.float32)   # shape (H, W)

        # 正規化: [0,255] → [0.2, 2.0] メートル（線形スケール）
        d_min, d_max = depth_arr.min(), depth_arr.max()
        if d_max - d_min < 1e-6:
            return np.full_like(depth_arr, 1.0)
        depth_norm = (depth_arr - d_min) / (d_max - d_min)  # [0, 1]
        depth_m = 0.2 + depth_norm * 1.8                    # [0.2, 2.0] m

        # Depth-Anything は近いほど大きい値 → TSDF 用に反転
        depth_m = 2.2 - depth_m                             # 反転して遠いほど大きく

        return depth_m.astype(np.float32)

    def _mesh_via_tsdf(
        self,
        views: list[Image.Image],
        output_path: Path,
    ) -> Path:
        """
        Depth-Anything-V2 で深度推定 → Open3D TSDF Fusion でメッシュ化。
        """
        try:
            import open3d as o3d
        except ImportError:
            raise RuntimeError("open3d がインストールされていません: pip install open3d")

        try:
            import trimesh
        except ImportError:
            raise RuntimeError("trimesh がインストールされていません: pip install trimesh")

        logger.info("Running TSDF Fusion with Depth-Anything-V2 ...")

        # ── ① 画像サイズから Intrinsic を動的に生成 ──
        first_rgb = np.array(views[0].convert("RGB"), dtype=np.uint8)
        img_h, img_w = first_rgb.shape[:2]
        logger.info(f"View image size: {img_w}×{img_h}")

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=4.0 / 512.0,
            sdf_trunc=0.04,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

        fx = fy = max(img_w, img_h) * 0.8
        cx, cy = img_w / 2.0, img_h / 2.0
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=img_w, height=img_h,
            fx=fx, fy=fy, cx=cx, cy=cy,
        )
        logger.info(f"Intrinsic: w={img_w}, h={img_h}, fx={fx:.1f}")

        for i, view in enumerate(views):
            rgb = np.array(view.convert("RGB"), dtype=np.uint8)

            # ── ② Depth-Anything-V2 で深度推定 ──────────
            view_resized = view.resize((img_w, img_h))
            depth = self._estimate_depth(view_resized)      # shape (H, W) float32

            # 画像サイズと深度マップのサイズを合わせる
            if depth.shape != (img_h, img_w):
                from PIL import Image as PILImage
                depth_pil = PILImage.fromarray(depth).resize(
                    (img_w, img_h), PILImage.BILINEAR
                )
                depth = np.array(depth_pil, dtype=np.float32)

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(rgb),
                o3d.geometry.Image(depth),
                depth_scale=1.0,
                depth_trunc=3.0,
                convert_rgb_to_intensity=False,
            )

            angle = i * (2 * np.pi / len(views))
            extrinsic = _make_orbit_extrinsic(
                angle=angle, elevation=np.radians(30), radius=1.5
            )
            volume.integrate(rgbd, intrinsic, extrinsic)

        mesh_o3d = volume.extract_triangle_mesh()
        mesh_o3d.compute_vertex_normals()

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
