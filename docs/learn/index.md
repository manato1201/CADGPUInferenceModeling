# CADGPUInferenceModeling 学習資料(保守・拡張の実務編)

このフォルダは、**このツールを直す・足す人**のための資料です。基礎から順に学ぶ講義は [../lecture/index.md](../lecture/index.md)(第 0〜3 回)。既存の技術資料は [overview.md](../../overview.md)(Mermaid の図つき)・[PERF_RESULTS.md](../PERF_RESULTS.md)(性能の実測)。

| # | 資料 | いつ読むか |
|---|---|---|
| 1 | [全体像](01-overview.md) | ルート(A・B・C)・パイプライン・データの置き場所・API |
| 2 | [コードの読み方](02-code-tour.md) | どのファイルから読むか |
| 3 | [内側のしくみ](03-internals.md) | ルート判定・押し出しの手順・屋根・後処理・API のジョブ |
| 4 | [拡張のレシピ](04-extend.md) | 屋根・レイヤー・テクスチャ・書き出し・API を**足したい** |
| 5 | [テストの書き方と実行](05-testing.md) | テストを回す・足す・文書の検査 |
| 6 | [トラブルと保守](06-troubleshooting-maintenance.md) | 動かない・**既知の問題**・定期の保守 |
| – | [用語集](glossary.md) | 分からない言葉が出た |

## 最初の 15 分

1. [README](../../README.md) の手順で `.venv` を確認し、`pytest -q` を通す。
2. `cli.py info <file.dxf>` で、サンプルの DXF(`car.dxf` など)の種類(どのルートか)を見る。
3. [構成図](../diagrams/architecture.svg)を見ながら、[全体像](01-overview.md)を読む。
