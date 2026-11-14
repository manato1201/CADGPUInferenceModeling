"""docs/diagrams/ の構成図(SVG)を作る。直すときは、ここを直して `python tools/make_diagrams.py` を実行する。

  architecture.svg   DXF ↔ 3D アセットの双方向パイプライン

部品は tools/svgkit.py。座標は手で決める。矢印はカードの横から出入りする(カードの下には文字がある)。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from svgkit import Diagram  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "diagrams"


def architecture() -> Diagram:
    d = Diagram(1500, 860, "CADGPUInferenceModeling の構成",
                "DXF の図面は、端点のスナップと壁厚の推定を経て押し出され、屋根とテクスチャが付いて、Unity や UE 向けの 3D アセットになる。"
                "逆向きでは、メッシュを Z バッファで隠線判定し、三面図と平面図の DXF にする。GPU 推論(TripoSR・Zero123++)は、外観の推論に使う。")
    d.text(40, 44, "CADGPUInferenceModeling — 構成", cls="hd")
    d.text(40, 66, "上の流れ = 図面 → 3D / 下の流れ = 3D → 図面(青=主な流れ / 灰の破線=任意の GPU 推論)", cls="note")

    d.group(40, 100, 250, 640, "INPUT / OUTPUT", color="gray")
    dxf = d.card(150, 220, "doc", "DXF の図面", "平面図・三面図\n(CP932 の文字化けに注意)")
    mesh = d.card(150, 540, "film", "3D メッシュ", "OBJ・FBX・GLB")

    d.group(480, 100, 1010, 640, "CORE — core/", color="teal")
    parser = d.card(640, 220, "terminal", "parser", "ルート自動判定\n三面図")
    ext = d.card(900, 220, "gear", "floor_plan_extruder", "端点スナップ・壁厚の推定\n開口を切って押し出し")
    roof = d.card(1160, 220, "tag", "roof / texture", "屋根(陸・片流れ・切妻)\n手続き的テクスチャ")
    z = d.card(900, 540, "check", "mesh_to_dxf", "Z バッファで隠線判定\nnumba で高速化")
    gpu = d.card(640, 380, "chart", "GPU 推論", "TripoSR・Zero123++\n(--infer)")
    post = d.card(1160, 540, "scissors", "postprocessor", "メッシュ修復・LOD\nエクスポート")
    d.text(500, 722, "入口: cli.py(run / export / info)と api.py(FastAPI・ジョブキュー)", cls="note")

    d.arrow([dxf.right, parser.left], "読む", at=0)
    d.arrow([parser.right, ext.left], "", at=0)
    d.arrow([ext.right, roof.left], "壁のメッシュ", at=0)
    d.arrow([roof.right, (1330, 220), (1330, 540), post.right], "3D アセット", at=1, dx=-34, dy=0)
    d.arrow([mesh.right, z.left], "三面図 + 平面図を DXF に", at=0)
    d.arrow([gpu.right, (780, 380), (780, 220)], "外観の推論", kind="secondary", at=0, dx=0, dy=-10)
    return d


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "architecture.svg").write_text(architecture().svg(), encoding="utf-8")
    print("wrote", OUT / "architecture.svg")


if __name__ == "__main__":
    main()
