"""
cli.py
======
フェーズ1 エントリポイント。
core/ の各モジュールを繋いで DXF → glTF を実行する。

使い方:
    python cli.py run input.dxf --output-dir ./out
    python cli.py run input.dxf --output-dir ./out --no-infer   # 三面図だけ確認
    python cli.py info input.dxf                                  # メタデータ表示
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

# ── ロギング設定 ──────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    format="%(message)s",
)
logger = logging.getLogger("cad2asset")

app = typer.Typer(
    name="cad2asset",
    help="CAD図面（DXF）→ ゲーム用3Dアセット（glTF）変換パイプライン",
    add_completion=False,
)
console = Console()


# ─────────────────────────────────────────────
# コマンド: run
# ─────────────────────────────────────────────

@app.command()
def run(
    dxf_path: Path = typer.Argument(..., help="入力 DXF ファイルパス"),
    output_dir: Path = typer.Option(Path("./output"), help="出力ディレクトリ"),
    image_size: int  = typer.Option(512,  help="三面図の解像度 (px)"),
    infer_steps: int = typer.Option(36,   help="Zero123++ 推論ステップ数"),
    no_infer: bool   = typer.Option(False, "--no-infer", help="推論をスキップして三面図だけ確認"),
    seed: int        = typer.Option(42,   help="乱数シード"),
    lod0: int        = typer.Option(10_000, help="LOD0 最大ポリゴン数"),
    lod1: int        = typer.Option(4_000,  help="LOD1 最大ポリゴン数"),
    lod2: int        = typer.Option(1_000,  help="LOD2 最大ポリゴン数"),
):
    """DXF → 三面図レンダリング → Zero123++ 推論 → glTF 出力 を一括実行する。"""

    # ── Step 1: DXF パース ───────────────────────
    console.rule("[bold cyan]Step 1: DXFパース・三面図レンダリング")
    from core.parser import parse_dxf, save_views

    result = parse_dxf(dxf_path, image_size=image_size)
    meta = result.meta

    # メタデータ表示
    table = Table(title="CAD Metadata", show_header=True)
    table.add_column("項目", style="cyan")
    table.add_column("値", style="white")
    table.add_row("ファイル",      meta.source_path)
    table.add_row("単位",          meta.unit)
    table.add_row("エンティティ数", str(meta.entity_count))
    table.add_row("レイヤー",      ", ".join(meta.layers[:10]))
    dims = meta.dimensions
    table.add_row("寸法 (X/Y/Z)", f"{dims[0]:.1f} / {dims[1]:.1f} / {dims[2]:.1f} {meta.unit}")
    console.print(table)

    # 三面図保存
    view_dir = output_dir / "views"
    saved_views = save_views(result, view_dir)
    console.print(f"[green]✓ 三面図を保存: {view_dir}")
    for name, path in saved_views.items():
        console.print(f"  {name}: {path}")

    if no_infer:
        console.print("[yellow]--no-infer が指定されているため推論をスキップします。")
        raise typer.Exit()

    # ── Step 2: GPU 推論 ─────────────────────────
    console.rule("[bold cyan]Step 2: Zero123++ 多視点生成")
    from core.inferencer import Zero123PlusPlusInferencer, InferenceConfig

    config = InferenceConfig(num_inference_steps=infer_steps)
    infer = Zero123PlusPlusInferencer(config)

    # 正面図を入力として多視点生成
    front_img = result.views["front"]
    console.print("推論中... (初回はモデルダウンロードが入ります)")
    views = infer.generate_views(front_img, seed=seed)
    console.print(f"[green]✓ {len(views)} 視点の画像を生成しました")

    # 生成ビューを保存（デバッグ用）
    gen_dir = output_dir / "generated_views"
    gen_dir.mkdir(parents=True, exist_ok=True)
    for i, v in enumerate(views):
        v.save(gen_dir / f"view_{i:02d}.png")
    console.print(f"[green]✓ 生成ビューを保存: {gen_dir}")

    # メッシュ生成
    raw_mesh_path = output_dir / "raw_mesh.glb"
    infer.generate_mesh(views, output_path=raw_mesh_path)
    console.print(f"[green]✓ 粗メッシュを保存: {raw_mesh_path}")

    # ── Step 3: 後処理 ───────────────────────────
    console.rule("[bold cyan]Step 3: メッシュ後処理・LOD生成")
    from core.postprocessor import postprocess, PostprocessConfig

    post_cfg = PostprocessConfig(
        lod_face_counts=[lod0, lod1, lod2],
        target_unit="m",  # Unity 向け
    )
    asset_dir = output_dir / "assets"
    base_name = Path(dxf_path).stem

    exported = postprocess(
        mesh_path=raw_mesh_path,
        cad_dimensions_mm=meta.dimensions,
        output_dir=asset_dir,
        base_name=base_name,
        config=post_cfg,
    )

    # 結果サマリ
    console.rule("[bold green]完了")
    summary = Table(title="Output Assets", show_header=True)
    summary.add_column("LOD",  style="cyan")
    summary.add_column("パス", style="white")
    for lod_name, path in exported.items():
        summary.add_row(lod_name, str(path))
    console.print(summary)
    console.print(f"\n[bold green]✅ 完了！ Unity / Unreal Engine に {asset_dir} をインポートしてください。")


# ─────────────────────────────────────────────
# コマンド: info  （メタデータだけ確認したいとき）
# ─────────────────────────────────────────────

@app.command()
def info(
    dxf_path: Path = typer.Argument(..., help="調べたい DXF ファイルパス"),
):
    """DXF のメタデータ（寸法・レイヤー・エンティティ数）を表示する。"""
    from core.parser import parse_dxf

    console.rule("[bold cyan]DXF メタデータ")
    result = parse_dxf(dxf_path, image_size=256)  # 軽量モードで
    meta = result.meta

    table = Table(show_header=True)
    table.add_column("項目", style="cyan")
    table.add_column("値")
    dims = meta.dimensions
    table.add_row("ファイル",      meta.source_path)
    table.add_row("単位",          meta.unit)
    table.add_row("エンティティ数", str(meta.entity_count))
    table.add_row("寸法 X",        f"{dims[0]:.3f} {meta.unit}")
    table.add_row("寸法 Y",        f"{dims[1]:.3f} {meta.unit}")
    table.add_row("寸法 Z",        f"{dims[2]:.3f} {meta.unit}")
    table.add_row("縦横比",        " : ".join(f"{v:.2f}" for v in meta.aspect_ratio))
    table.add_row("レイヤー",      "\n".join(meta.layers))
    console.print(table)


# ─────────────────────────────────────────────
# エントリ
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app()
