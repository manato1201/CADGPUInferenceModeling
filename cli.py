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
    no_infer: bool   = typer.Option(True,  "--no-infer/--infer", help="推論スキップ（デフォルト）/ --infer で Zero123++ 推論を有効化"),
    seed: int        = typer.Option(42,   help="乱数シード"),
    lod0: int        = typer.Option(10_000, help="LOD0 最大ポリゴン数"),
    lod1: int        = typer.Option(4_000,  help="LOD1 最大ポリゴン数"),
    lod2: int        = typer.Option(1_000,  help="LOD2 最大ポリゴン数"),
    fmt: str         = typer.Option("obj", "--format", help="出力フォーマット: obj / glb (デフォルト: obj)"),
    force_unit: str  = typer.Option("", "--force-unit", help="単位を強制指定: mm / cm / m / inch / foot (省略時はDXFヘッダから自動判定)"),
    default_height: float = typer.Option(2500.0, "--default-height", help="Z=0の2D図面に適用する定数高さ [mm]（デフォルト: 2500mm = 2.5m）"),
    wall_thickness: float = typer.Option(150.0,  "--wall-thickness",  help="壁厚 [mm]（デフォルト: 150mm）"),
    use_union: bool       = typer.Option(False,  "--use-union",       help="壁ブーリアンUnionを実行（品質向上・低速）"),
    roof_type: str        = typer.Option("gable", "--roof-type",      help="屋根タイプ: flat/shed/gable (デフォルト: gable)"),
    roof_height: float    = typer.Option(1.5,    "--roof-height",    help="棟の高さ [m]（デフォルト: 1.5m）"),
    wall_style: str       = typer.Option("brick", "--wall-style",    help="外壁テクスチャ: brick/concrete/wood (デフォルト: brick)"),
    roof_style: str       = typer.Option("tile",  "--roof-style",    help="屋根テクスチャ: tile/metal/flat (デフォルト: tile)"),
    no_texture: bool      = typer.Option(False,  "--no-texture",     help="テクスチャ生成をスキップ"),
    texture_size: int     = typer.Option(1024,   "--texture-size",   help="テクスチャ解像度 [px]（デフォルト: 1024）"),
    no_roof: bool         = typer.Option(False,  "--no-roof",        help="屋根生成をスキップ"),
    model: str            = typer.Option("triposr", "--model",         help="推論モデル: triposr / zero123 (--infer 時のみ有効)"),
    triposr_resolution: int = typer.Option(256,   "--triposr-res",    help="TripoSR メッシュ解像度 32〜256（デフォルト: 256）"),
):
    """DXF → 押し出し3Dメッシュ + 屋根 + テクスチャ → ゲーム用アセット出力。\n\n--infer オプションを付けると Zero123++ 推論を試みます（実験的機能）。"""

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

    from core.postprocessor import postprocess, PostprocessConfig, export_meshes, generate_lods, repair_mesh
    import trimesh as _trimesh

    post_cfg = PostprocessConfig(
        lod_face_counts=[lod0, lod1, lod2],
        target_unit="m",
    )
    asset_dir = output_dir / "assets"
    base_name = Path(dxf_path).stem
    effective_unit = force_unit if force_unit else meta.unit
    if force_unit:
        console.print(f"[yellow]単位を強制指定: {force_unit}  (DXFヘッダ: {meta.unit})")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Route C: 建築平面図 → 押し出し変換
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if result.is_floor_plan:
        console.rule("[bold green]Route C: 建築平面図 → 押し出し + Zero123++ 外観生成")
        from core.floor_plan_extruder import extrude_floor_plan, ExtrusionConfig
        from core.inferencer import Zero123PlusPlusInferencer, InferenceConfig

        # ── C-1: 押し出し3Dメッシュ生成 ─────────────
        ext_cfg = ExtrusionConfig(
            ceiling_height=2500.0,
            default_height=default_height,
            floor_thickness=200.0,
            wall_thickness=wall_thickness,
            wall_snap_tol=50.0,
            use_union=use_union,
            cut_openings=True,
        )
        console.print(f"  定数高さ: {default_height:.0f}mm  壁厚: {wall_thickness:.0f}mm  Union: {use_union}")
        console.print("押し出し変換中...")
        floor_mesh = extrude_floor_plan(dxf_path, config=ext_cfg)
        console.print(f"[green]✓ 押し出しメッシュ: verts={len(floor_mesh.vertices)}, faces={len(floor_mesh.faces)}")

        # 押し出しメッシュ単体も出力（確認・フォールバック用）
        extrude_dir = output_dir / "extrude"
        extrude_dir.mkdir(parents=True, exist_ok=True)
        extrude_path = extrude_dir / f"{base_name}_extrude.{fmt}"
        floor_mesh.export(str(extrude_path))
        console.print(f"[green]✓ 押し出しメッシュ保存: {extrude_path}")

        # ── C-1.5: 屋根生成（--no-roof でスキップ）──────
        if no_roof:
            console.print("[yellow]--no-roof: 屋根生成をスキップします")
        else:
            console.rule("[bold cyan]屋根生成")
            from core.roof_generator import attach_roof, RoofConfig
            roof_cfg = RoofConfig(
                roof_type=roof_type,
                ridge_height=roof_height,
                overhang=0.3,
            )
            try:
                floor_mesh = attach_roof(floor_mesh, config=roof_cfg)
                console.print(f"[green]✓ 屋根追加: {roof_type} 棟高{roof_height:.1f}m  "
                              f"verts={len(floor_mesh.vertices)}, faces={len(floor_mesh.faces)}")
            except Exception as e:
                console.print(f"[yellow]⚠ 屋根生成失敗（スキップ）: {e}")

        if no_infer:
            console.print("[yellow]--no-infer: 押し出し+屋根メッシュを出力します")
            floor_mesh = repair_mesh(floor_mesh)
            lods = generate_lods(floor_mesh, post_cfg.lod_face_counts)

            if no_texture:
                exported = export_meshes(lods, asset_dir, base_name, export_format=fmt)
            else:
                # テクスチャ付きOBJ出力
                console.rule("[bold cyan]テクスチャ生成")
                from core.texture_baker import bake_textures
                tex_result = bake_textures(
                    floor_mesh,
                    output_dir=asset_dir,
                    base_name=base_name,
                    wall_style=wall_style,
                    roof_style=roof_style,
                    texture_size=texture_size,
                )
                console.print(f"[green]✓ テクスチャ生成: {tex_result['texture'].name}")
                exported = {"LOD0_textured": tex_result["obj"]}
                # LOD1/2はテクスチャなしで追加
                lod_paths = export_meshes(
                    {"LOD1": lods.get("LOD1", floor_mesh),
                     "LOD2": lods.get("LOD2", floor_mesh)},
                    asset_dir, base_name, export_format=fmt
                )
                exported.update(lod_paths)
        else:
            # ── C-2: 推論（TripoSR or Zero123++）────────
            render_dir = output_dir / "render"
            render_dir.mkdir(parents=True, exist_ok=True)

            if model.lower() == "triposr":
                console.rule("[bold cyan]Route C-2: TripoSR 外観生成")
                from core.triposr_inferencer import TripoSRConfig, generate_from_mesh_triposr
                raw_mesh_path = output_dir / "raw_mesh.obj"
                triposr_cfg = TripoSRConfig(mc_resolution=triposr_resolution)
                console.print("押し出しメッシュ → TripoSR で外観生成中...")
                generate_from_mesh_triposr(
                    floor_mesh,
                    output_path=raw_mesh_path,
                    config=triposr_cfg,
                    render_output_dir=render_dir,
                )
                console.print(f"[green]✓ TripoSR 完了: {raw_mesh_path}")
            else:
                console.rule("[bold cyan]Route C-2: Zero123++ 外観生成")
                from core.inferencer import Zero123PlusPlusInferencer, InferenceConfig
                infer_cfg = InferenceConfig(num_inference_steps=infer_steps)
                infer = Zero123PlusPlusInferencer(infer_cfg)
                raw_mesh_path = output_dir / "raw_mesh.glb"
                console.print("レンダリング + Zero123++ 推論中...")
                infer.generate_from_floor_plan_mesh(
                    floor_mesh,
                    output_path=raw_mesh_path,
                    image_size=image_size,
                    seed=seed,
                    render_output_dir=render_dir,
                )
                console.print(f"[green]✓ 推論完了: {raw_mesh_path}")

            # ── C-3: 後処理 ────────────────────────────
            # Route C: スケール基準は押し出しメッシュの実寸（Z=0の図面は寸法が0になるため）
            # 押し出しメッシュの最大辺長 [m] → mm に変換して使う
            import numpy as _np
            _floor_extents = _np.array(floor_mesh.bounding_box.extents)  # m単位
            _max_dim_mm = float(_floor_extents.max() * 1000.0)           # m → mm
            _floor_dims_mm = _floor_extents * 1000.0                     # m → mm
            console.print(f"[cyan]スケール基準: 押し出しメッシュ実寸 "
                          f"{_floor_dims_mm[0]:.0f}×{_floor_dims_mm[1]:.0f}×{_floor_dims_mm[2]:.0f}mm")

            console.rule("[bold cyan]Step 3: メッシュ後処理・LOD生成")
            exported = postprocess(
                mesh_path=raw_mesh_path,
                cad_dimensions_mm=_floor_dims_mm,
                output_dir=asset_dir,
                base_name=base_name,
                config=post_cfg,
                cad_unit="mm",              # 押し出しメッシュはmm換算済み
                export_format=fmt,
            )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Route B: 3D POLYFACE → 推論スキップして直接後処理
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    elif result.is_3d:
        console.rule("[bold green]Route B: 3D POLYFACEメッシュを直接抽出（推論スキップ）")
        console.print(f"[green]✓ メッシュ検出: verts={len(result.mesh.vertices)}, faces={len(result.mesh.faces)}")

        mesh = repair_mesh(result.mesh)

        # 押し出しメッシュ（抽出元）を extrude/ に保存
        extrude_dir = output_dir / "extrude"
        extrude_dir.mkdir(parents=True, exist_ok=True)
        extrude_path = extrude_dir / f"{base_name}_extrude.{fmt}"
        mesh.export(str(extrude_path))
        console.print(f"[green]✓ 抽出メッシュ保存: {extrude_path}")

        # 屋根生成（--no-roof でスキップ可）
        if not no_roof:
            try:
                from core.roof_generator import attach_roof, RoofConfig
                console.rule("[bold cyan]屋根生成")
                roof_cfg = RoofConfig(roof_type=roof_type, ridge_height=roof_height, overhang=0.3)
                mesh = attach_roof(mesh, config=roof_cfg)
                console.print(f"[green]✓ 屋根追加: {roof_type}")
            except Exception as e:
                console.print(f"[yellow]⚠ 屋根生成スキップ: {e}")

        console.rule("[bold cyan]Step 3: メッシュ後処理・LOD生成")
        lods = generate_lods(mesh, post_cfg.lod_face_counts)

        if no_texture:
            exported = export_meshes(lods, asset_dir, base_name, export_format=fmt)
        else:
            try:
                from core.texture_baker import bake_textures
                console.rule("[bold cyan]テクスチャ生成")
                tex_result = bake_textures(
                    mesh, output_dir=asset_dir, base_name=base_name,
                    wall_style=wall_style, roof_style=roof_style, texture_size=texture_size,
                )
                console.print(f"[green]✓ テクスチャ生成: {tex_result['texture'].name}")
                exported = {"LOD0_textured": tex_result["obj"]}
                lod_paths = export_meshes(
                    {"LOD1": lods.get("LOD1", mesh), "LOD2": lods.get("LOD2", mesh)},
                    asset_dir, base_name, export_format=fmt)
                exported.update(lod_paths)
            except Exception as e:
                console.print(f"[yellow]⚠ テクスチャ生成スキップ: {e}")
                exported = export_meshes(lods, asset_dir, base_name, export_format=fmt)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Route A: 2D図面 → 押し出しフォールバック + Zero123++ 推論
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    else:
        console.rule("[bold cyan]Route A: 2D図面")

        # ── extrude/ フォルダを作成 ────────────────
        extrude_dir = output_dir / "extrude"
        extrude_dir.mkdir(parents=True, exist_ok=True)

        # ── A-0: 押し出しOBJを常に出力（フォールバック）──
        # Route C と判定されなかった2D図面でも押し出しを試みる
        _extrude_mesh = None
        try:
            from core.floor_plan_extruder import extrude_floor_plan, ExtrusionConfig
            _ext_cfg = ExtrusionConfig(
                ceiling_height=2500.0,
                default_height=default_height,
                floor_thickness=200.0,
                wall_thickness=wall_thickness,
                wall_snap_tol=50.0,
                use_union=False,
                cut_openings=True,
            )
            _extrude_mesh = extrude_floor_plan(dxf_path, config=_ext_cfg)
            _extrude_path = extrude_dir / f"{base_name}_extrude.{fmt}"
            _extrude_mesh.export(str(_extrude_path))
            console.print(f"[green]✓ 押し出しメッシュ保存: {_extrude_path}")
        except Exception as e:
            console.print(f"[yellow]⚠ 押し出しフォールバック失敗: {e}")
            # 三面図は常に保存
            result.views["top"].save(str(extrude_dir / f"{base_name}_top_view.png"))

        # ── --no-infer: 押し出しを最終出力にして終了 ──
        if no_infer:
            if _extrude_mesh is not None:
                console.print("[yellow]--no-infer: 押し出しメッシュを最終出力にします")
                _extrude_mesh = repair_mesh(_extrude_mesh)
                if not no_roof:
                    try:
                        from core.roof_generator import attach_roof, RoofConfig
                        _extrude_mesh = attach_roof(
                            _extrude_mesh,
                            RoofConfig(roof_type=roof_type, ridge_height=roof_height, overhang=0.3)
                        )
                        console.print(f"[green]✓ 屋根追加: {roof_type}")
                    except Exception as e:
                        console.print(f"[yellow]⚠ 屋根スキップ: {e}")
                lods = generate_lods(_extrude_mesh, post_cfg.lod_face_counts)
                if no_texture:
                    exported = export_meshes(lods, asset_dir, base_name, export_format=fmt)
                else:
                    try:
                        from core.texture_baker import bake_textures
                        tex_result = bake_textures(
                            _extrude_mesh, output_dir=asset_dir, base_name=base_name,
                            wall_style=wall_style, roof_style=roof_style, texture_size=texture_size,
                        )
                        console.print(f"[green]✓ テクスチャ生成: {tex_result['texture'].name}")
                        exported = {"LOD0_textured": tex_result["obj"]}
                        exported.update(export_meshes(
                            {"LOD1": lods.get("LOD1", _extrude_mesh),
                             "LOD2": lods.get("LOD2", _extrude_mesh)},
                            asset_dir, base_name, export_format=fmt))
                    except Exception as e:
                        console.print(f"[yellow]⚠ テクスチャスキップ: {e}")
                        exported = export_meshes(lods, asset_dir, base_name, export_format=fmt)
            else:
                console.print("[yellow]--no-infer: 押し出し失敗。三面図のみ保存しました。")
                raise typer.Exit()

        # ── A-1: Zero123++ 推論 ────────────────────
        else:
            render_dir = output_dir / "render"
            render_dir.mkdir(parents=True, exist_ok=True)
            raw_mesh_path = output_dir / "raw_mesh.obj"

            # ── モデル選択: TripoSR or Zero123++ ──────────
            if model.lower() == "triposr":
                console.rule("[bold cyan]Route A-1: TripoSR 推論")
                from core.triposr_inferencer import TripoSRInferencer, TripoSRConfig, generate_from_mesh_triposr

                triposr_cfg = TripoSRConfig(mc_resolution=triposr_resolution)

                # 押し出しメッシュがあればそれを入力に使う（より正確な形状）
                if _extrude_mesh is not None:
                    console.print("押し出しメッシュ → TripoSR パイプラインで生成中...")
                    generate_from_mesh_triposr(
                        _extrude_mesh,
                        output_path=raw_mesh_path,
                        config=triposr_cfg,
                        render_output_dir=render_dir,
                    )
                else:
                    console.print("top.png → TripoSR で生成中...")
                    infer = TripoSRInferencer(triposr_cfg)
                    infer.generate_mesh(result.views["top"], output_path=raw_mesh_path)

                console.print(f"[green]✓ TripoSR 完了: {raw_mesh_path}")

            else:
                console.rule("[bold cyan]Route A-1: Zero123++ 推論")
                from core.inferencer import Zero123PlusPlusInferencer, InferenceConfig

                z123_config = InferenceConfig(num_inference_steps=infer_steps)
                infer = Zero123PlusPlusInferencer(z123_config)

                input_img = result.views["top"]
                console.print("Zero123++ 推論中...")
                views = infer.generate_views(input_img, seed=seed)
                console.print(f"[green]✓ {len(views)} 視点生成")
                for i, v in enumerate(views):
                    v.save(render_dir / f"view_{i:02d}.png")
                raw_mesh_path_glb = output_dir / "raw_mesh.glb"
                infer.generate_mesh(views, output_path=raw_mesh_path_glb)
                raw_mesh_path = raw_mesh_path_glb

            # ── 後処理（共通）────────────────────────────
            console.rule("[bold cyan]Step 3: メッシュ後処理・LOD生成")
            # Route A の場合は押し出しメッシュの実寸をスケール基準にする
            if _extrude_mesh is not None:
                import numpy as _np2
                _ea = _np2.array(_extrude_mesh.bounding_box.extents) * 1000.0
                _scale_dims = _ea
                _scale_unit = "mm"
            else:
                _scale_dims = meta.dimensions
                _scale_unit = effective_unit

            exported = postprocess(
                mesh_path=raw_mesh_path,
                cad_dimensions_mm=_scale_dims,
                output_dir=asset_dir,
                base_name=base_name,
                config=post_cfg,
                cad_unit=_scale_unit,
                export_format=fmt,
            )

            # 屋根 + テクスチャ
            if not no_texture:
                try:
                    import trimesh as _tm
                    _mesh_a = _tm.load(str(list(exported.values())[0]), force="mesh")
                    if not no_roof:
                        from core.roof_generator import attach_roof, RoofConfig
                        _mesh_a = attach_roof(
                            _mesh_a,
                            RoofConfig(roof_type=roof_type, ridge_height=roof_height)
                        )
                    from core.texture_baker import bake_textures
                    console.rule("[bold cyan]屋根 + テクスチャ生成")
                    tex_result = bake_textures(
                        _mesh_a, output_dir=asset_dir, base_name=f"{base_name}_textured",
                        wall_style=wall_style, roof_style=roof_style, texture_size=texture_size,
                    )
                    exported["LOD0_textured"] = tex_result["obj"]
                    console.print(f"[green]✓ テクスチャ付き: {tex_result['obj'].name}")
                except Exception as e:
                    console.print(f"[yellow]⚠ テクスチャ生成スキップ: {e}")

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
