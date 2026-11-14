# CADGPUInferenceModeling

**CAD 図面(DXF)↔ ゲーム用 3D アセット の双方向変換パイプライン**

```
DXF → 自動ルート判定 → 押し出し 3D メッシュ → 屋根 → テクスチャ → Unity / UE
                              ↓(--infer のとき)
                      TripoSR / Zero123++ で外観を推論

OBJ / FBX / GLB → Z バッファで隠線判定 → 三面図 + 平面図 → DXF
```

動作確認済みの環境: Windows 11 / NVIDIA GeForce RTX 3070(8.6GB VRAM)/ Python 3.12

![構成図](docs/diagrams/architecture.svg)

## 状態

| 項目 | 内容 |
|---|---|
| テスト | `.venv/Scripts/python -m pytest -q`(寸法精度・Z バッファ・API の堅牢性 + 文書の検査) |
| GPU 推論 | 実機で確認。自動テストには**含まれない** |
| 学習資料 | 講義 4 回あり。**保守・拡張の詳しい資料は未作成**(簡易版のみ) |
| 既知の問題 | [docs/learn/index.md](docs/learn/index.md)(「壁寸法」が壁になる 等) |

## クイックスタート

セットアップ・CLI・API・出力の詳細は [旧 README の詳細](docs/reference/setup-and-operations.md)。

```bash
.venv/Scripts/python -m pytest -q       # テスト
.venv/Scripts/python cli.py info <file.dxf>
```

## ドキュメント

| 読みたいもの | 場所 |
|---|---|
| 全体の地図 | [docs/index.md](docs/index.md) |
| **基礎から学ぶ**(講義 4 回・実行できる例つき) | [docs/lecture/](docs/lecture/index.md) |
| コードの地図・既知の問題 | [docs/learn/index.md](docs/learn/index.md) |
| セットアップ・使い方の**詳細**(旧 README) | [docs/reference/setup-and-operations.md](docs/reference/setup-and-operations.md) |
| 技術資料(Mermaid) | [overview.md](overview.md)・[PERF_RESULTS.md](docs/PERF_RESULTS.md) |
| 改善計画 | [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) |
| 講義資料・ポートフォリオ(HTML) | `lecture.html`・`index.html` |

## 限界(正直に)

- 図面の品質(端点の隙間・レイヤー名の付け方)に依存します。レイヤーの分類は**名前の照合**で、順序の影響を受けます。
- GPU 推論の結果は、入力と VRAM に依存し、再現性は保証しません。
