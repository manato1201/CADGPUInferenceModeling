# 3. 内側のしくみ

## 1. ルート判定(`parse_dxf`)

1. `ezdxf.readfile` で読み、**単位**(`mm`・`cm`・`m`・`inch`・`foot`)から、m への換算係数を決める。
2. **POLYFACE MESH があれば → ルート B**(メッシュを m 単位で抽出)。
3. なければ、**平面図らしさ**(`_is_floor_plan`: 壁・建具のレイヤーがあるか)→ **ルート C**。
4. どちらでもなければ → **ルート A**(2D)。
5. 点群・線分を集め、三面図(front・top・side)を作る(共通)。

`CADParseResult.is_3d` / `is_floor_plan` が、`cli.py` の分岐に使われる。

## 2. 押し出し(`extrude_floor_plan`)

講義[第 1・3 回](../lecture/01-snap-endpoints.md)のとおり。順序:

1. **レイヤー分類**(壁・建具・家具・階段・無視・その他。**上から順の最初の一致**)。
2. 階の判定(「1 階」「2 階」・`1f`・`2f`)。複数階なら、`floor_height_step`(3000mm)で積む。
3. **線分の収集**(円弧・ポリラインも線分に)→ **端点のスナップ**(80mm)→ **重複の除去**。
4. **壁厚の推定**(平行な線の組の間隔。`auto_wall_thickness`)。
5. **壁の箱**を作る(`_wall_box`)。**開口**(ドア 高さ 2000mm・窓 台 900mm・高さ 1100mm)を、壁から**切り抜く**。
6. **床・天井のスラブ**を作る。メッシュを**結合**(`use_union`。48 個ずつのバッチ)→ **修復**(`repair_mesh`)。

## 3. 屋根(`attach_roof`)

天井の**輪郭**(`_extract_ceiling_footprint`: 天井の高さの面を取り出す)から、屋根を作って、壁のメッシュに付ける。種類: **陸屋根**(パラペットつき)・**片流れ**・**切妻**(棟の高さ・軒の出)。輪郭の演算は **shapely**。

## 4. 後処理(`postprocess`)

CAD の実寸(`cad_dimensions_mm`)に**スケールを合わせ**(`apply_scale`)→ **修復**(退化した三角形の除去など)→ **LOD**(`generate_lods`: LOD0・LOD1・LOD2)→ **エクスポート**(OBJ・glTF)。

## 5. GPU 推論

- **Zero123++**: 1 枚の図(front)から、複数の視点の画像を作り、**TSDF Fusion** でメッシュにする。
- **TripoSR**: 1 枚から、メッシュを**直接**出す(速い)。
- モデルは**シングルトン**(同じ設定なら、同じインスタンスを使い回す。モデルの読み込みを 1 回にする)。

## 6. API のジョブ(`api.py`)

1. `POST /convert`: 拡張子(`.dxf`・`.dwg`)を確かめ、`_cleanup_jobs()`(TTL を超えた完了済みを削除 → それでも `MAX_JOBS` なら古い完了済みから削除)→ ジョブ ID を作り、作業フォルダにアップロードを保存 → **`202 Accepted`** を即返し、バックグラウンドで変換。
2. `_run_pipeline`: パース → 三面図 → **推論**(`generate_views` → `generate_mesh`)→ `postprocess`。
3. **時間切れ**(`JOB_TIMEOUT_SEC`): `ThreadPoolExecutor` で包んで、`future.result(timeout=...)`。**スレッドは強制終了できない**ので、API の応答上の「時間切れ」にすぎず、処理は裏で続きうる(GPU のメモリも解放されない)。コードのコメントも、この限界を明記している。
4. **エラーの分類**(`_classify_error`): `input_error`(壊れた DXF・ファイル無し)・`inference_error`(CUDA OOM・実行時エラー)・`timeout`・`system_error`。**トレースバック全文は応答に含めない**(情報漏洩を防ぐ)。
5. `GET /jobs/{id}`(状態)・`GET /jobs/{id}/download/{LOD}`(完了後だけ。未完了は 409)。
