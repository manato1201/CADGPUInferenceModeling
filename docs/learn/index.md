# CADGPUInferenceModeling 学習資料(簡易版)

保守・拡張の詳しい資料は**未作成**です。基礎は [講義](../lecture/index.md)(第 0〜3 回)、アーキテクチャと各アルゴリズムは [overview.md](../../overview.md)(Mermaid の図つき)、性能は [PERF_RESULTS.md](../PERF_RESULTS.md)、旧 README は [setup-and-operations.md](../reference/setup-and-operations.md)。

## コードの地図

| ファイル | 役割 |
|---|---|
| `core/parser.py`(約 510 行) | DXF のパース・単位・三面図・**ルート自動判定**(平面図か、3D か) |
| `core/floor_plan_extruder.py`(約 800 行) | レイヤー分類・**端点スナップ**・壁厚の推定・開口を切る・押し出し |
| `core/roof_generator.py`・`texture_baker.py` | 屋根(陸・片流れ・切妻)・手続き的テクスチャ(UV 付き OBJ) |
| `core/mesh_to_dxf.py`(約 700 行) | メッシュ → DXF。**Z バッファ**・シルエット・三面図・平面図・寸法 |
| `core/inferencer.py`・`triposr_inferencer.py` | GPU 推論(Zero123++ の TSDF Fusion・TripoSR の直接メッシュ) |
| `core/postprocessor.py` | メッシュ修復・LOD・エクスポート |
| `cli.py`・`api.py` | CLI(run / export / info)・FastAPI(ジョブキュー) |
| `tests/` | 寸法精度・Z バッファの可視率・推論のシングルトン・API の堅牢性・`bench.py` |

## 既知の問題・注意(講義の作成中に読んで見つけたもの。コードは変えていない)

1. **「壁寸法」が wall に分類される**: `_classify_layers` は上から順に最初の一致で決めるため、「壁」に先に当たり、寸法線を壁として押し出す可能性がある(講義第 3 回で確認)。
2. **`_snap_endpoints` の最小の長さ(`cfg_min_snap` = 50 mm)がモジュールの定数**で、`ExtrusionConfig` から変えられない。許容距離を大きくすると、端点が推移的につながって、関係のない壁が崩れる。
3. **`_decode_layer` は、化けた名前を元に戻せないことがある**(そのまま返す)。そのため、分類はパターンを化けさせて照合している。
4. **GPU 推論(TripoSR・Zero123++)は、実機(RTX 3070)での動作確認**で、このリポジトリの自動テストには含まれない。`TripoSR/` は外部リポジトリの配置。
5. 依存が重い(`open3d`・`torch`・`numba` など)。Z バッファの初回呼び出しは、`numba` のコンパイルで数秒かかる。

## 用語

**DXF**(CAD の交換形式)・**押し出し**(2D の線を高さ方向に立ち上げて 3D にする)・**KD 木**(空間の近傍探索)・**Union-Find**(素集合データ構造)・**Z バッファ**(各ピクセルの最小の深さ)・**重心座標**(三角形の内部の点を 3 つの重みで表す)・**隠線**(手前の面に隠れた辺)・**LOD**(距離に応じた簡略モデル)・**CP932**(Shift_JIS の拡張)・**TSDF**(符号つき距離場による形状の融合)。
