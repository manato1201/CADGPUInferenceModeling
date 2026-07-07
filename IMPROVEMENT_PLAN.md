# CADGPUInferenceModeling 改善・リファクタリング計画書

**改善指標: 性能・精度の強化**
作成日: 2026-07-03 / 調査範囲: 全4,879 LOC

各フェーズは独立して新しいセッションで実行できるよう、参照ファイル・検証手順・アンチパターンを明記している。

---

## Phase 0: 現状分析(調査済み)

### アーキテクチャ
DXF ↔ ゲーム用3Dアセット(OBJ/GLB)の双方向変換パイプライン。

- **Route C**: 建築平面図 → 押し出しメッシュ → TripoSR/Zero123++ 外観生成
- **Route B**: 3D POLYFACE → 直接抽出(寸法精度100%、推論スキップ)
- **Route A**: 一般2D図面 → フォールバック
- **逆方向**: 3Dメッシュ → Zバッファ隠線判定 → DXF(可視率78% @v3)

### 主要モジュールと利用可能API
| ファイル | 役割 |
|---|---|
| `core/floor_plan_extruder.py` (786行) | 平面図押し出し。`wall_snap_tol=80.0`(L41)、`parallel_detect_tol=30.0`(L52-53) |
| `core/mesh_to_dxf.py` (675行) | Zバッファ隠線判定(512×512深度マップ、CPU numpy実装) |
| `core/triposr_inferencer.py` (402行) | TripoSR推論。`mc_resolution=256`(L53)、`preprocess_image()`(L116-149、`foreground_ratio=0.85`) |
| `core/inferencer.py` (427行) | Zero123++推論(L87-111でロード) |
| `core/postprocessor.py` (268行) | trimeshによるメッシュ修復・LOD生成 |
| `api.py` (159行) | FastAPI。`_run_pipeline()`(L57-95)は素朴なtry-except のみ |

依存: ezdxf 1.3+, trimesh 4.3+, open3d 0.18+, torch 2.2+(CUDA 12.1), diffusers 0.27+。VRAM: RTX 3070 8GBで動作確認済み。TripoSRはvendored(未改変)。

### 特定済みボトルネック
1. **端点スナップが O(N²)** — `floor_plan_extruder.py:206-250`。全端点ペアを二重ループで距離判定。1000線分超で支配的
2. **Zバッファ隠線判定がCPU逐次** — 3ビューで約18秒(10k頂点)
3. **モデルのシングルトン欠落** — APIリクエストごとに`from_pretrained()`が走り得る(TripoSR 1.7GB + Zero123++ 5.6GB)
4. **自動テスト・精度評価スクリプトが皆無** — 手動確認(car.dxf)のみ

### アンチパターン(全フェーズ共通)
- trimeshのAPIはバージョン差異が大きい。`requirements.txt`のバージョン指定範囲内のAPIのみ使用し、存在確認せず新メソッドを呼ばない
- TripoSR本体(`TripoSR/`)は改変しない。ラッパー側(`core/triposr_inferencer.py`)で吸収する
- 精度に影響するパラメータ変更(mc_resolution等)は、Phase 1の評価ハーネスで前後比較してから採用する

---

## Phase 1: テスト・評価ハーネス構築(最優先・後続フェーズの安全網)

**実装内容:**
1. `tests/test_accuracy.py` を新規作成
   - `car.dxf`のRoute B抽出で既知寸法(README記載の実測値)と一致することを検証
   - Zバッファ隠線判定の可視率が75%以上であることを検証
2. `tests/bench.py` を新規作成
   - `2DDXF_Sample.dxf`の押し出し処理時間、隠線判定時間、推論時間(モデルロード除く)を`time.perf_counter`で計測しJSON出力
   - **このベースライン値を`tests/baseline.json`として保存**(Phase 2以降の改善効果測定に使う)

**参照:** リポジトリ同梱のサンプルDXF(`car.dxf`, `boat.dxf`, `2DDXF_Sample.dxf`, `06PC183.dxf`)をフィクスチャとして使用。README.mdの実測値表を期待値の出典とする。

**検証チェックリスト:**
- [ ] `pytest tests/ -v` が全件パス
- [ ] `python tests/bench.py` が baseline.json を生成
- [ ] GPU無し環境でも推論以外のテストがスキップではなくパスする

**アンチパターン:** 期待値をテスト実行結果からコピーして自己成就させない(READMEの実測値を出典にする)。

---

## Phase 2: モデルロードのシングルトン化(工数小・効果大)

**実装内容:**
1. `core/triposr_inferencer.py` / `core/inferencer.py` にモジュールレベルのファクトリを追加:
   ```python
   _INSTANCE = None
   def get_inferencer(config) -> TripoSRInferencer:
       global _INSTANCE
       if _INSTANCE is None:
           _INSTANCE = TripoSRInferencer(config)
       return _INSTANCE
   ```
2. `api.py` の `_run_pipeline()` と `cli.py` を直接コンストラクタ呼び出しからファクトリ経由に変更
3. FastAPI起動時ウォームアップ(`lifespan`でモデルを事前ロード)をオプション化(環境変数 `PRELOAD_MODELS=1`)

**検証チェックリスト:**
- [ ] APIに同一ジョブを2回投げ、2回目にモデルロードログが出ないこと
- [ ] `nvidia-smi`でVRAMがリクエスト間で増加しないこと
- [ ] Phase 1のテストが引き続きパス

**アンチパターン:** configが異なる2回目の呼び出しで古いインスタンスを黙って返さない(config不一致時は再生成 or 明示エラー)。

---

## Phase 3: 端点スナップの空間インデックス化(性能: 4〜10倍)

**実装内容:**
`core/floor_plan_extruder.py:206-250` の O(N²) ループを `scipy.spatial.cKDTree` に置換:
```python
from scipy.spatial import cKDTree
tree = cKDTree(pts)
pairs = tree.query_pairs(r=tol)   # tol以内の全ペアを O(N log N) で取得
for i, j in pairs:
    union(i, j)
```
scipyは既存依存(requirements.txt記載)のため追加依存なし。

**検証チェックリスト:**
- [ ] Phase 1の精度テストがパス(スナップ結果が同一 = 生成メッシュの頂点数・bounds一致)
- [ ] `tests/bench.py`でbaseline比の高速化を確認・記録
- [ ] `2DDXF_Sample.dxf`と`06PC183.dxf`の両方で出力メッシュがbaselineと一致

**アンチパターン:** `query_pairs`は「r以内」の判定。既存コードが `< tol`(strict)なら境界値の扱いを合わせる。

---

## Phase 4: Zバッファ隠線判定の高速化(性能: 5〜20倍)

**実装内容(2段階、まず4a、足りなければ4b):**
1. **4a. Numba JIT化**: `core/mesh_to_dxf.py` のラスタライズ内ループを`@numba.njit`でコンパイル。依存追加は`numba`のみ、コード構造は不変。目標: 18秒 → 4秒以下
2. **4b. マルチビュー並列化**: 3ビューのラスタライズを`concurrent.futures.ProcessPoolExecutor`で並列実行(各ビューは独立)

**検証チェックリスト:**
- [ ] 可視率(VISIBLE率)がbaselineの78%から低下しない
- [ ] 出力DXFのエンティティ数がbaselineと一致
- [ ] bench.pyで処理時間を記録

**アンチパターン:** OpenGL/GPUラスタライズへの全面書き換えは初手でやらない(ヘッドレス環境のGLコンテキスト問題で工数爆発しがち)。Numba+並列化で目標未達の場合のみ検討。

---

## Phase 5: 精度チューニング(評価ハーネス駆動)

**実装内容:** 以下のノブをPhase 1のハーネスで前後比較し、優位なら既定値を変更:
| ノブ | 現在値 | 試行値 | 期待効果 |
|---|---|---|---|
| Zバッファ深度マップ解像度 | 512² | 1024² | 可視率 78%→82%(+約2秒) |
| `mc_resolution` | 256 | 96〜128 | 速度3倍・品質はハーネスで判定 |
| `foreground_ratio` | 0.85 | 0.80/0.90 | シルエット依存の推論品質 |
| `wall_snap_tol` | 80mm固定 | 推定壁厚の±5%に動的化 | 誤結合の削減 |
| 壁厚推定 | 平行線ペア | 厚さ分布の中央値+外れ値除外 | 推定の安定化 |

**検証チェックリスト:**
- [ ] 各ノブ変更を個別コミットにし、bench.py/精度テストの前後値をコミットメッセージに記録
- [ ] 既定値の変更はREADME.mdの性能表にも反映

---

## Phase 6: APIロバスト化

**実装内容(`api.py:57-95`):**
1. エラー分類(入力エラー/推論エラー/システムエラー)とHTTPステータスの対応付け
2. `traceback.format_exc()`をジョブレコードに保存(レスポンスにはメッセージのみ)
3. ジョブごとのタイムアウト(GPUハング検出)
4. ジョブ辞書`_JOBS`のサイズ上限とTTL(メモリリーク防止)

**検証チェックリスト:**
- [ ] 壊れたDXFを投げて`failed`+分類済みエラーメッセージが返る
- [ ] 正常系のテストが引き続きパス

---

## Final Phase: 統合検証

- [ ] `pytest tests/ -v` 全パス
- [ ] `python tests/bench.py` の結果をbaseline.jsonと比較し、改善サマリー表を`docs/PERF_RESULTS.md`に記録
- [ ] 全サンプルDXF(6ファイル)でCLIパイプラインがエラーなく完走
- [ ] `grep -rn "from_pretrained" api.py cli.py` — ファクトリ経由以外の直接ロードが残っていないこと
- [ ] README.mdの性能・精度表を実測値で更新

**期待効果合計: 処理時間 約60%短縮、可視率 78%→82%、API多重リクエストのVRAM問題解消**
