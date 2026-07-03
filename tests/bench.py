"""
tests/bench.py
===============
Phase 1 (テスト・評価ハーネス構築) の計測ベースラインスクリプト。

これから行う複数フェーズのリファクタリング（モデルシングルトン化、
O(N^2) スナップの KDTree 化、Z バッファ隠線判定の高速化など）の効果測定に
使うベースライン計測値を `tests/baseline.json` に記録する。

対象サンプル: 2DDXF_Sample.dxf（建築平面図, Route C）

計測項目:
  - 押し出し処理時間   : core.floor_plan_extruder.extrude_floor_plan()
  - 隠線判定時間       : core.mesh_to_dxf.mesh_to_dxf() (triview モード、
                          内部で Z バッファ計算を含む)
  - 推論時間           : 今回は計測しない（GPU依存・モデルダウンロードが
                          必要なため）。JSON には "inference_sec": null を出力。

単体スクリプトとして実行可能:
    python tests/bench.py
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.floor_plan_extruder import extrude_floor_plan, ExtrusionConfig
from core.mesh_to_dxf import mesh_to_dxf, MeshToDxfConfig

SAMPLE_FILE = "2DDXF_Sample.dxf"
BASELINE_JSON = Path(__file__).resolve().parent / "baseline.json"


def main() -> None:
    sample_path = REPO_ROOT / SAMPLE_FILE
    if not sample_path.exists():
        raise FileNotFoundError(f"サンプルファイルが見つかりません: {sample_path}")

    print(f"=== ベースライン計測開始: {SAMPLE_FILE} ===")

    # ── 1. 押し出し処理時間 ─────────────────────
    # cli.py の Route C 実行時デフォルト値 (cli.py:125-133) に合わせる。
    # ExtrusionConfig データクラス自体の既定値 (use_union=True) とは異なるため注意。
    ext_cfg = ExtrusionConfig(
        ceiling_height=2500.0,
        default_height=2500.0,
        floor_thickness=200.0,
        wall_thickness=150.0,
        wall_snap_tol=50.0,
        use_union=False,
        cut_openings=True,
    )
    print("[1/2] 押し出し処理 (extrude_floor_plan) 計測中...")
    t0 = time.perf_counter()
    mesh = extrude_floor_plan(str(sample_path), config=ext_cfg)
    extrusion_sec = time.perf_counter() - t0
    print(f"      完了: {extrusion_sec:.3f} 秒  "
          f"(verts={len(mesh.vertices)}, faces={len(mesh.faces)})")

    # ── 2. 隠線判定時間 (mesh_to_dxf triview) ───
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        obj_path = tmp_dir_path / "bench_extruded.obj"
        mesh.export(str(obj_path))

        out_dxf_path = tmp_dir_path / "bench_triview.dxf"
        print("[2/2] 隠線判定 (mesh_to_dxf triview) 計測中...")
        t1 = time.perf_counter()
        mesh_to_dxf(obj_path, out_dxf_path, config=MeshToDxfConfig(mode="triview"))
        hidden_line_sec = time.perf_counter() - t1
        print(f"      完了: {hidden_line_sec:.3f} 秒")

    # ── 3. 推論時間 ── 今回はスキップ ───────────
    # GPU依存・モデルダウンロードが必要なため、このフェーズでは計測しない。
    inference_sec = None

    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "sample_file": SAMPLE_FILE,
        "extrusion_sec": extrusion_sec,
        "hidden_line_sec": hidden_line_sec,
        "inference_sec": inference_sec,  # GPU/モデルダウンロード依存のため今回は未計測
        "mesh_stats": {
            "verts": len(mesh.vertices),
            "faces": len(mesh.faces),
        },
    }

    BASELINE_JSON.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== 計測結果 ===")
    print(f"  押し出し処理時間  : {extrusion_sec:.3f} 秒")
    print(f"  隠線判定時間      : {hidden_line_sec:.3f} 秒")
    print(f"  推論時間          : 未計測 (GPU依存のためスキップ)")
    print(f"  メッシュ統計      : verts={result['mesh_stats']['verts']}, "
          f"faces={result['mesh_stats']['faces']}")
    print(f"\nベースラインを書き出しました: {BASELINE_JSON}")


if __name__ == "__main__":
    main()
