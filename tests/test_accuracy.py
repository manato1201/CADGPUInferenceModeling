"""
tests/test_accuracy.py
=======================
Phase 1 (テスト・評価ハーネス構築) の精度回帰テスト。

これから行う複数フェーズのリファクタリング（モデルシングルトン化、
O(N^2) スナップの KDTree 化、Z バッファ隠線判定の高速化など）に対する
安全網として、現時点での「正しい」出力を固定する。

期待値の出典: README.md の「精度・一致度」セクション実測値。
  - car.dxf (Route B, POLYFACE メッシュ):
      幅 2160.0mm × 奥行 4853.0mm × 高さ 1702.0mm、誤差 0mm（完全一致）
  - 可視輪郭（VISIBLE）検出率: 78%（Zバッファ 512×512、tol_ratio=0.015 既定）
      README 記載の許容下限は 75%

GPU/推論系（TripoSR, Zero123++, torch）には一切依存しない。
core.parser / core.postprocessor / core.mesh_to_dxf は CPU のみで完結する。
"""

from __future__ import annotations

from pathlib import Path

import ezdxf
import numpy as np
import pytest

from core.parser import parse_dxf
from core.postprocessor import repair_mesh
from core.mesh_to_dxf import mesh_to_dxf, MeshToDxfConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
CAR_DXF = REPO_ROOT / "car.dxf"

# README.md 実測値（mm単位、DXFネイティブ座標系: X=幅, Y=奥行, Z=高さ、小数第1位で丸めた表記）
EXPECTED_DIMENSIONS_MM = np.array([2160.0, 4853.0, 1702.0])

# car.dxf の生座標自体が float32 精度で保存されている
# （例: X座標が -1079.999877929687 のように格納されており、
#  厳密な範囲は 2159.999755859374mm であって 2160.0mm ちょうどではない）。
# README の値は小数第1位に丸めた表記のため、比較には float32 精度
# （実測で最大 0.00025mm 程度のずれ）を許容する 1e-2mm を用いる。
DIM_ATOL_MM = 1e-2

# README.md: 可視輪郭（VISIBLE）検出率 78%（許容下限 75%）
MIN_VISIBLE_RATIO = 0.75


@pytest.fixture(scope="module")
def car_parse_result():
    assert CAR_DXF.exists(), f"サンプルファイルが見つかりません: {CAR_DXF}"
    return parse_dxf(str(CAR_DXF))


class TestRouteBDimensionAccuracy:
    """Route B (POLYFACE メッシュ) の寸法精度を検証する。"""

    def test_is_3d_route_b(self, car_parse_result):
        assert car_parse_result.is_3d is True
        assert car_parse_result.mesh is not None

    def test_parsed_dimensions_match_readme(self, car_parse_result):
        """
        result.meta.dimensions は DXF ネイティブ単位（mm）のはずだが、
        現状の実装では car.dxf ($INSUNITS=4=mm) の場合、POLYFACE抽出時に
        m 単位でメッシュを保持しつつ bbox を mm へ戻すスケール変換が
        実質的に恒等変換になっており、返る数値は「メートル数値」になっている
        （例: 2160.0mm のはずが 2.16 という値になる）。
        これは実測されたバグであり、Phase 1 では修正せず、
        ここでは「mm 換算した値が README の mm 値と一致する」ことを固定する。
        """
        dims = car_parse_result.meta.dimensions
        assert dims.shape == (3,)

        # 実際に返ってくる数値スケールを吸収するため、
        # 大きい方の値（mm 相当）に正規化してから比較する。
        # dims の最大成分が 1000 未満ならメートル相当とみなし ×1000 する。
        dims_mm = dims * 1000.0 if dims.max() < 1000.0 else dims

        np.testing.assert_allclose(
            dims_mm, EXPECTED_DIMENSIONS_MM, atol=DIM_ATOL_MM,
            err_msg=f"car.dxf の寸法が README 実測値と一致しません: {dims_mm}",
        )

    def test_repaired_mesh_extents_match_readme(self, car_parse_result):
        """repair_mesh() 後のメッシュの bounding box extents が
        README「生成OBJ」の値（幅2160.0mm/奥行4853.0mm/高さ1702.0mm）と一致する。

        core.parser._extract_polyface_mesh は mm を m に変換して mesh.vertices
        を保持するため、extents はメートル単位で得られる。README の mm 値と
        比較するため ×1000 して照合する。
        """
        mesh = repair_mesh(car_parse_result.mesh)
        extents_m = mesh.bounding_box.extents
        extents_mm = extents_m * 1000.0

        np.testing.assert_allclose(
            extents_mm, EXPECTED_DIMENSIONS_MM, atol=DIM_ATOL_MM,
            err_msg=f"repair_mesh 後の extents が README 実測値と一致しません: {extents_mm}",
        )


class TestZBufferVisibilityRatio:
    """Z バッファ隠線判定の可視輪郭検出率を検証する。"""

    def test_visible_ratio_at_least_75_percent(self, car_parse_result, tmp_path):
        """
        car.dxf を Route B でメッシュ抽出 → repair_mesh → 一時OBJへexport
        → mesh_to_dxf() で triview モードのDXFを生成 → 出力DXFを読み込み、
        VISIBLE / (VISIBLE + HIDDEN) が README 実測値 78% に対する
        許容下限 75% 以上であることを検証する。
        """
        mesh = repair_mesh(car_parse_result.mesh)

        obj_path = tmp_path / "car.obj"
        mesh.export(str(obj_path))

        out_dxf_path = tmp_path / "car_triview.dxf"
        cfg = MeshToDxfConfig(mode="triview")
        mesh_to_dxf(obj_path, out_dxf_path, config=cfg)

        assert out_dxf_path.exists()

        doc = ezdxf.readfile(str(out_dxf_path))
        msp = doc.modelspace()

        visible_count = sum(1 for e in msp if e.dxf.layer == cfg.layer_visible)
        hidden_count = sum(1 for e in msp if e.dxf.layer == cfg.layer_hidden)

        assert visible_count + hidden_count > 0, "VISIBLE/HIDDEN エンティティが1件もありません"

        ratio = visible_count / (visible_count + hidden_count)
        assert ratio >= MIN_VISIBLE_RATIO, (
            f"可視輪郭検出率が許容下限を下回りました: "
            f"visible={visible_count}, hidden={hidden_count}, ratio={ratio:.4f} "
            f"(要求 >= {MIN_VISIBLE_RATIO})"
        )
