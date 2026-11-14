# 5. テストの書き方と実行

## 実行

```bash
.venv/Scripts/python -m pytest -q                     # 全部(文書の検査を含む)
.venv/Scripts/python -m pytest tests/test_api_robustness.py -q
.venv/Scripts/python tests/bench.py                   # 処理時間の計測(baseline.json を出す)
```

## テストの置き場所と、何を確かめるか

| ファイル | 確かめるもの |
|---|---|
| `test_accuracy.py` | サンプル DXF(`car.dxf`)の寸法・修復後のメッシュの寸法・**Z バッファの可視率 75% 以上**(README の数字との一致) |
| `test_singleton.py` | 推論クラスのシングルトン(**モデルを読み込まずに**確かめる) |
| `test_api_robustness.py` | 壊れた DXF → `input_error`・マジックバイトの無いファイル・ファイル未存在・**TTL・上限・作業フォルダの掃除** |
| `test_docs.py` | 文書の検査(下) |

## よく使う型

1. **実データの小さなサンプル**(`car.dxf` など)で、**寸法を数字で**確かめる(README に書いた数字と一致するか)。
2. **重いものを避ける**: シングルトンのテストは、モデルにも GPU にも触れない(インスタンスの同一性だけ)。
3. **API は `TestClient`**・ジョブのストアは直接いじり、時刻を差し替えて TTL を確かめる(待たない)。
4. **エラーの分類**: 例外の種類ごとに、`error_category` が正しいことを確かめる。
5. **計測**は `bench.py` で、ベースラインと比べる(回帰の検出)。

## 新しい機能のテストを書く手順

1. **失敗するテストを先に**。2. 実装して通す。3. **壊れた入力**(壊れた DXF・空の図面・単位が不明)。4. 既定の挙動が**変わらない**こと。

## 文書の検査

```bash
python tools/check_docs.py             # 検査(標準ライブラリだけ)
python tools/check_docs.py --update    # 引用の更新・実行例の出力の作り直し
python tools/check_mermaid.py          # Mermaid が描画できるか(mmdc が要る)
python tools/make_diagrams.py          # 構成図(SVG)を作り直す
```

`tests/test_docs.py` が、同じ検査を pytest で行います(引用がソースと一致・実行例が 2 回動いて出力が一致・講義の標準の節・リンク・Mermaid の標準行)。

## 確かめていないこと(正直に)

- **GPU 推論(Zero123++・TripoSR)の結果**は、自動テストに含まれません(実機 RTX 3070 での確認)。
- **Unity・UE への取り込み**は未確認。
- 図面の種類(他社の CAD・他の単位・レイヤー名)への**汎用性**は、サンプルの範囲でしか確かめていません。
