# 第 1 回 端点のスナップ(KD 木 + Union-Find)

> 所要時間の目安: 2.5 時間 / 前提: 第 0 回

## この回のゴール

- 近くにある端点を**ひとまとめ**にする方法(**KD 木で近い組を探し、Union-Find でグループ化**)を説明できる。
- 計算量が **O(N²) から O(N log N)** になる理由を説明できる。
- スナップの**許容距離**と、**短すぎる線の除去**の意味を説明できる。

## 1. 問題

壁の線分 A の終点と、壁 B の始点が、**30 mm だけ離れている**とします。人の目には「つながっている」が、プログラムには**別の点**です。そのまま押し出すと、**隙間**の空いた壁になります。

そこで、**許容距離(tol)以内の端点を、同じ座標にそろえます**。

## 2. 実装

```python
# 引用: core/floor_plan_extruder.py::_snap_endpoints
def _snap_endpoints(
    segs: list[tuple],
    tol: float,
) -> list[tuple]:
    """
    tol [mm] 以内の端点を同一座標に統合して壁の隙間を解消する。
    """
    if not segs:
        return segs

    # 全端点を収集
    pts = np.array([[s[0],s[1]] for s in segs] + [[s[2],s[3]] for s in segs],
                   dtype=np.float64)

    # Union-Find で近接端点をクラスタリング
    n = len(pts)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # 空間インデックス(KD-Tree)で近接端点ペアを O(N log N) で取得。
    # 注意: query_pairs(r=tol) は distance <= tol (以下) で判定するため、
    # 旧実装の distance < tol (未満) とは境界値 (distance == tol) の扱いのみ異なる。
    # 浮動小数点座標では実質的に無視できる差であり、Union-Findのロジックは変更しない。
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=tol)
    for i, j in pairs:
        union(i, j)

    # クラスタの重心を代表点とする
    clusters: dict[int, list] = {}
    for i in range(n):
        root = find(i)
        clusters.setdefault(root, []).append(i)

    snapped = np.zeros_like(pts)
    for root, members in clusters.items():
        centroid = pts[members].mean(axis=0)
        for m in members:
            snapped[m] = centroid

    # 線分を再構築
    n_segs = len(segs)
    result = []
    for i in range(n_segs):
        sx, sy = snapped[i]
        ex, ey = snapped[i + n_segs]
        length = np.hypot(ex-sx, ey-sy)
        if length >= cfg_min_snap:
            result.append((sx, sy, ex, ey))
    return result
```

手順:

1. すべての線分の**両端**を集め、`(2 × 線分の数)` 個の点にする。
2. **KD 木**(`cKDTree`)で、距離が `tol` 以内の**点の組**をすべて求める(`query_pairs`)。
3. その組を、**Union-Find**(素集合データ構造)で**グループ化**する。
4. 各グループの**重心**を、代表点にして、全員をそこへ動かす。
5. 線分を作り直す。**長さが `cfg_min_snap`(50 mm)未満**になったものは、**捨てる**。

| 部品 | 役割 |
|---|---|
| **KD 木** | 空間を分割して持ち、「この点の近く」を**全部の点と比べずに**探せる。全組を比べると **O(N²)**、KD 木なら **O(N log N)** 程度 |
| **Union-Find** | 「A と B は同じグループ」を**効率よく**まとめる。`find`(代表を探す。**経路圧縮**で速くする)と `union`(つなぐ) |
| **重心** | グループの点の平均。どれか 1 つに寄せるより、**偏りが少ない** |

**推移的につながる**: A–B が近く、B–C が近ければ、A–C が遠くても、**A・B・C は 1 グループ**になる(Union-Find の性質)。許容距離を大きくしすぎると、**関係のない壁まで連鎖**して崩れる。

```python
# 実行できる
from core.floor_plan_extruder import _snap_endpoints

# 壁 1(0→1000)の終点と、壁 2(1020→)の始点は 20 mm 離れている
segs = [(0, 0, 1000, 0), (1020, 0, 1020, 1000)]
print("20 mm の隙間(tol=50):", _snap_endpoints(segs, 50.0))

# 40 mm と 80 mm の隙間: 許容距離 50 mm なら、40 mm だけ閉じる
segs = [(0, 0, 1000, 0), (1040, 0, 2000, 0), (2080, 0, 3000, 0)]
print("40 mm と 80 mm      :", _snap_endpoints(segs, 50.0))

# 短すぎる線は捨てられる
print("30 mm の線          :", _snap_endpoints([(0, 0, 30, 0)], 50.0))
```

```text
# 出力
20 mm の隙間(tol=50): [(0.0, 0.0, 1010.0, 0.0), (1010.0, 0.0, 1020.0, 1000.0)]
40 mm と 80 mm      : [(0.0, 0.0, 1020.0, 0.0), (1020.0, 0.0, 2000.0, 0.0), (2080.0, 0.0, 3000.0, 0.0)]
30 mm の線          : []
```

注目: 20 mm 離れた端点は、**重心 1010** にそろい、壁 1 の終点が**動いています**(壁の長さが、1000 → 1010 に変わる)。**スナップは、図面を少し変える**操作です。

## 確認問題

1. KD 木を使わずに、全部の点の組を比べると、なぜ遅いですか。
2. 重心にそろえる理由は。
3. 許容距離を大きくしすぎると、何が起きますか。
4. 50 mm 未満の線を捨てるのは、なぜですか(スナップの後で)。

## 解答

1. N 個の点で、N²/2 組を比べるので、N が増えると**急激に遅くなる**。KD 木は、近くの点だけを探す。
2. どれか 1 点に寄せると、**その点の側に偏る**。平均なら、元の位置から**均等に**動く。
3. 推移的につながり、**関係のない壁の端点まで 1 つ**になって、形が崩れる。
4. スナップで、**ほぼ点になった**線(重なって縮んだ線)は、壁として意味を持たず、**押し出しの失敗**(退化した形)の原因になる。

## まとめ

- 端点のスナップ = **KD 木で近い組を探し → Union-Find でグループ化 → 重心にそろえる**。
- 計算量は O(N log N)。許容距離は、**図面の精度に合わせる**(大きすぎると連鎖)。
- 次の回は、3D → 図面の**隠線判定**(Z バッファ)。
