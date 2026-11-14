# 2. コードの読み方(どこに何があるか)

## ファイルの地図

| ファイル | 行数の目安 | 役割 | 読む順 |
|---|---|---|---|
| `core/parser.py` | 510 | DXF の読み込み・単位・**ルート判定**・三面図(`parse_dxf`・`CADParseResult`) | ① |
| `core/floor_plan_extruder.py` | 800 | レイヤー分類・**端点スナップ**・壁厚・開口・押し出し(`extrude_floor_plan`) | ② |
| `core/roof_generator.py` | 350 | 屋根(陸・片流れ・切妻)・天井の輪郭の抽出(`attach_roof`) | ③ |
| `core/texture_baker.py` | 410 | 手続き的テクスチャ(レンガ・コンクリート・木・屋根)・UV・OBJ の出力 | |
| `core/postprocessor.py` | 270 | スケール合わせ・修復・**LOD**・エクスポート(`postprocess`) | ④ |
| `core/mesh_to_dxf.py` | 700 | メッシュ → DXF(**Z バッファ**・シルエット・三面図・平面図・寸法) | ⑤ |
| `core/mesh_renderer.py` | 320 | 押し出しメッシュ → シルエット画像(視点プリセット) | |
| `core/inferencer.py`・`triposr_inferencer.py` | 450・430 | GPU 推論(Zero123++ の TSDF Fusion・TripoSR の直接メッシュ)。**シングルトン** | |
| `cli.py` | 560 | `run`(ルートで分岐)・`export`・`info` | ⑥ |
| `api.py` | 330 | FastAPI(ジョブ・TTL・エラーの分類) | ⑦ |
| `fbx2obj.py` | 200 | FBX → OBJ(Blender 経由) | |
| `TripoSR/` | – | **外部リポジトリの配置**(`git clone`)。このリポジトリのコードではない | |

**読み方**: ①②(入力の理解と押し出し)→ ③④(屋根と後処理)→ ⑤(逆変換)→ ⑥⑦(入口)。

## テスト・計測

| 場所 | 内容 |
|---|---|
| `tests/test_accuracy.py` | 寸法の精度・Z バッファの可視率(75% 以上) |
| `tests/test_singleton.py` | 推論モデルのシングルトン化(同じ設定なら同じインスタンス) |
| `tests/test_api_robustness.py` | エラーの分類・ジョブの TTL と上限・作業フォルダの掃除 |
| `tests/bench.py`・`baseline.json` | 処理時間の計測 |

## 文書

| 場所 | 内容 |
|---|---|
| `overview.md` | 技術資料(アーキテクチャ・アルゴリズム) |
| `docs/PERF_RESULTS.md` | 性能改善の実測 |
| `IMPROVEMENT_PLAN.md` | 改善計画(フェーズ別) |
| `lecture.html`・`index.html` | 講義資料・ポートフォリオ(HTML) |
