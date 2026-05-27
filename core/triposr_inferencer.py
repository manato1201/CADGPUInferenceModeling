"""
core/triposr_inferencer.py
==========================
TripoSR による単視点画像 → 3D メッシュ変換モジュール。

インストール方法:
    # ① TripoSR リポジトリを clone
    git clone https://github.com/VAST-AI-Research/TripoSR.git
    cd TripoSR

    # ② 依存パッケージをインストール
    pip install --upgrade setuptools
    pip install -r requirements.txt

    # ③ sys.path に追加（または環境変数 TRIPOSR_PATH を設定）
    set TRIPOSR_PATH=C:\\path\\to\\TripoSR   # Windows
    export TRIPOSR_PATH=/path/to/TripoSR     # Linux/Mac

依存:
    - torch >= 2.0 (CUDA対応版推奨)
    - TripoSR リポジトリのコード
    - trimesh, Pillow, numpy
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 設定
# ─────────────────────────────────────────────

@dataclass
class TripoSRConfig:
    """TripoSR 推論設定。"""
    model_id: str = "stabilityai/TripoSR"
    # TripoSR リポジトリのパス（None で TRIPOSR_PATH 環境変数 or 自動検索）
    triposr_repo_path: Optional[str] = None
    # 前景が画像に占める割合（0.85 推奨）
    foreground_ratio: float = 0.85
    # メッシュ抽出解像度（256 が標準、下げると速い）
    mc_resolution: int = 256
    # 背景除去（白背景シルエットの場合は False でOK）
    remove_background: bool = False
    # デバイス（None で自動選択）
    device: Optional[str] = None
    # chunk size（VRAM節約。0=無効）
    chunk_size: int = 8192


# ─────────────────────────────────────────────
# TripoSR リポジトリのパス解決
# ─────────────────────────────────────────────

def _find_triposr_path(hint: Optional[str] = None) -> Optional[Path]:
    """
    TripoSR リポジトリのパスを探す。

    探索順:
      1. hint 引数
      2. 環境変数 TRIPOSR_PATH
      3. カレントディレクトリ付近の TripoSR フォルダ
    """
    candidates = []
    if hint:
        candidates.append(Path(hint))
    env_path = os.environ.get("TRIPOSR_PATH")
    if env_path:
        candidates.append(Path(env_path))

    # よくある配置場所を探索
    cwd = Path.cwd()
    for parent in [cwd, cwd.parent, Path.home() / "Desktop" / "GameDevelopment"]:
        candidates.append(parent / "TripoSR")

    for p in candidates:
        if p.exists() and (p / "tsr").exists():
            return p

    return None


def _setup_triposr_path(repo_path: Optional[str] = None) -> Path:
    """TripoSR リポジトリを sys.path に追加する。"""
    found = _find_triposr_path(repo_path)
    if found is None:
        raise RuntimeError(
            "TripoSR リポジトリが見つかりません。\n\n"
            "以下の手順でセットアップしてください:\n"
            "  1. git clone https://github.com/VAST-AI-Research/TripoSR.git\n"
            "  2. cd TripoSR && pip install --upgrade setuptools && pip install -r requirements.txt\n"
            "  3. 環境変数を設定: set TRIPOSR_PATH=C:\\path\\to\\TripoSR\n"
            "  または --triposr-path オプションで直接指定"
        )
    if str(found) not in sys.path:
        sys.path.insert(0, str(found))
    logger.info(f"TripoSR リポジトリ: {found}")
    return found


# ─────────────────────────────────────────────
# 画像前処理
# ─────────────────────────────────────────────

def preprocess_image(
    image: Image.Image,
    remove_bg: bool = False,
    foreground_ratio: float = 0.85,
    output_size: int = 512,
) -> Image.Image:
    """
    TripoSR への入力画像を前処理する。

    remove_bg=False の場合:
        白背景・黒前景のシルエット画像として処理
        （mesh_renderer の出力に適している）

    remove_bg=True の場合:
        rembg で背景除去してから処理
    """
    if remove_bg:
        try:
            from rembg import remove
            image = remove(image)
        except ImportError:
            logger.warning("rembg 未インストール → 閾値ベース背景除去")
            img_arr = np.array(image.convert("RGBA"))
            white = (img_arr[:,:,0]>240) & (img_arr[:,:,1]>240) & (img_arr[:,:,2]>240)
            img_arr[white, 3] = 0
            image = Image.fromarray(img_arr)

    # RGBA に変換
    image = image.convert("RGBA")
    arr = np.array(image)
    alpha = arr[:,:,3]

    rows = np.any(alpha > 10, axis=1)
    cols = np.any(alpha > 10, axis=0)

    if not rows.any() or not cols.any():
        # 前景なしの場合はそのまま
        return image.resize((output_size, output_size), Image.LANCZOS)

    rmin, rmax = np.where(rows)[0][[0,-1]]
    cmin, cmax = np.where(cols)[0][[0,-1]]
    fg = arr[rmin:rmax+1, cmin:cmax+1]
    h, w = fg.shape[:2]
    size = max(h, w)

    # 正方形キャンバスに中央配置
    canvas = np.zeros((size, size, 4), dtype=np.uint8)
    canvas[:,:,3] = 255  # 白背景のalphaを不透明に
    canvas[:,:,:3] = 255  # 白
    pad_y = (size - h) // 2
    pad_x = (size - w) // 2
    canvas[pad_y:pad_y+h, pad_x:pad_x+w] = fg

    # foreground_ratio で余白調整
    final_size = int(size / foreground_ratio)
    final = np.full((final_size, final_size, 4), 255, dtype=np.uint8)
    offset = (final_size - size) // 2
    final[offset:offset+size, offset:offset+size] = canvas

    img = Image.fromarray(final)
    return img.resize((output_size, output_size), Image.LANCZOS)


# ─────────────────────────────────────────────
# TripoSR 推論クラス
# ─────────────────────────────────────────────

class TripoSRInferencer:
    """
    TripoSR 推論ラッパー。

    使い方:
        infer = TripoSRInferencer()
        mesh_path = infer.generate_mesh(image, "output.obj")
    """

    def __init__(self, config: Optional[TripoSRConfig] = None):
        self.config = config or TripoSRConfig()
        self.model = None

    def _resolve_device(self):
        import torch
        if self.config.device:
            return self.config.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def load(self) -> None:
        if self.model is not None:
            return

        _setup_triposr_path(self.config.triposr_repo_path)

        import torch
        from tsr.system import TSR

        device = self._resolve_device()
        logger.info(f"TripoSR ロード中: {self.config.model_id} → {device}")
        logger.info("初回は約 2.5GB のダウンロードが発生します...")

        self.model = TSR.from_pretrained(
            self.config.model_id,
            weight_name="model.ckpt",
            config_name="config.yaml",
        )
        self.model = self.model.to(device)
        self.model.eval()
        self._device = device
        logger.info("TripoSR ロード完了")

    def generate_mesh(
        self,
        image: Image.Image,
        output_path: str | Path = "output.obj",
    ) -> Path:
        """
        単視点画像から 3D メッシュを生成して保存する。

        Parameters
        ----------
        image       : 入力画像（PIL.Image）
        output_path : 出力パス（.obj / .glb）

        Returns
        -------
        Path
        """
        import torch

        self.load()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 前処理
        logger.info("TripoSR: 画像前処理中...")
        processed = preprocess_image(
            image,
            remove_bg=self.config.remove_background,
            foreground_ratio=self.config.foreground_ratio,
        )
        # TripoSR は RGB（3ch）を期待するので RGBA → RGB に変換（白背景合成）
        if processed.mode == "RGBA":
            bg = Image.new("RGB", processed.size, (255, 255, 255))
            bg.paste(processed, mask=processed.split()[3])
            processed = bg
        elif processed.mode != "RGB":
            processed = processed.convert("RGB")
        logger.info(f"  入力画像: {processed.mode} {processed.size}")

        import trimesh as tm
        # 推論
        logger.info(f"TripoSR: 推論中 (resolution={self.config.mc_resolution})...")
        with torch.no_grad():
            scene_codes = self.model(
                [processed],
                device=self._device,
            )
            try:
                meshes = self.model.extract_mesh(
                    scene_codes,
                    resolution=self.config.mc_resolution,
                    has_vertex_color=False,
                )
            except Exception as e:
                err_str = str(e).lower()
                if any(k in err_str for k in ["torchmcubes", "marching_cubes", "no module"]):
                    logger.warning(f"torchmcubes 未インストール → skimage フォールバック: {e}")
                    try:
                        meshes = self.model.extract_mesh(
                            scene_codes,
                            resolution=min(self.config.mc_resolution, 128),
                            has_vertex_color=False,
                            use_torch_mc=False,
                        )
                    except TypeError:
                        meshes = self.model.extract_mesh(
                            scene_codes,
                            resolution=min(self.config.mc_resolution, 128),
                            has_vertex_color=False,
                        )
                    logger.info("skimage marching_cubes で成功")
                else:
                    raise

        # trimesh 経由でエクスポート
        mesh = meshes[0]

        # TripoSR の出力形式に応じて変換
        if hasattr(mesh, "verts_packed"):
            v = mesh.verts_packed().cpu().numpy()
            f = mesh.faces_packed().cpu().numpy()
            tm_mesh = tm.Trimesh(vertices=v, faces=f)
        elif hasattr(mesh, "vertices"):
            tm_mesh = mesh if isinstance(mesh, tm.Trimesh) else \
                      tm.Trimesh(vertices=np.array(mesh.vertices), faces=np.array(mesh.faces))
        else:
            raise RuntimeError(f"未対応のメッシュ形式: {type(mesh)}")

        tm_mesh.fix_normals()
        tm_mesh.export(str(output_path))
        logger.info(
            f"TripoSR: 保存 → {output_path} "
            f"(verts={len(tm_mesh.vertices)}, faces={len(tm_mesh.faces)})"
        )
        return output_path


# ─────────────────────────────────────────────
# 押し出しメッシュ → TripoSR パイプライン
# ─────────────────────────────────────────────

def generate_from_mesh_triposr(
    mesh,
    output_path: str | Path = "output.obj",
    config: Optional[TripoSRConfig] = None,
    render_output_dir: Optional[str | Path] = None,
) -> Path:
    """
    押し出し3Dメッシュ → レンダリング → TripoSR → メッシュ のパイプライン。
    """
    from core.mesh_renderer import render_mesh_for_zero123

    logger.info("=== TripoSR パイプライン ===")
    logger.info("Step 1: 押し出しメッシュをレンダリング中...")
    input_image = render_mesh_for_zero123(
        mesh,
        image_size=512,
        output_dir=render_output_dir,
    )
    logger.info(f"  完了: {input_image.size}")

    logger.info("Step 2: TripoSR で3Dメッシュ生成中...")
    infer = TripoSRInferencer(config)
    return infer.generate_mesh(input_image, output_path=output_path)
