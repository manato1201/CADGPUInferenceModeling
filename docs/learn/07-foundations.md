# 7. 基礎の補強(言語・設計思想・アルゴリズム・用語)

講義と学習資料の**前提になっている知識**を、ここで補います。「Python・NumPy・文字コードの基本」を、**このツールで実際に使われている形**で、動かして確かめます。

## 1. 言語の基礎: CAD ツールで使う Python

### 1.1 NumPy: 座標を「配列」で扱う

図面の点は、**(x, y) の組の並び**です。NumPy の配列は、**全部の点を一度に**計算する(ベクトル化)。`shape` は(点の数, 2)。

```python
# 実行できる
import numpy as np

points = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]])    # 長方形の 4 隅
print("形:", points.shape)
print("x だけ:", points[:, 0], " y の最大:", points[:, 1].max())

moved = points + np.array([100.0, 50.0])                 # ブロードキャスト: 全部の点に同じ値を足す
print("平行移動:", moved[0].tolist())

width, height = np.ptp(points, axis=0)                   # 最大 − 最小
print("幅と高さ:", width, height)
```

```text
# 出力
形: (4, 2)
x だけ: [ 0. 10. 10.  0.]  y の最大: 5.0
平行移動: [100.0, 50.0]
幅と高さ: 10.0 5.0
```

### 1.2 `@dataclass` と既定値で、設定をまとめる

押し出しの設定(壁の高さ・厚み)のような**数の組**は、`@dataclass` にして、**名前つきで持つ**。既定値を決めておけば、**必要なものだけ指定**できる。

```python
# 実行できる
from dataclasses import dataclass


@dataclass
class ExtrusionConfig:
    wall_height: float = 2.7        # メートル
    wall_thickness: float = 0.2


default = ExtrusionConfig()
tall = ExtrusionConfig(wall_height=3.5)             # 一部だけ変える
print(default)
print(tall)
print("体積の目安(壁 1 m):", round(tall.wall_height * tall.wall_thickness * 1.0, 2), "m³")
```

```text
# 出力
ExtrusionConfig(wall_height=2.7, wall_thickness=0.2)
ExtrusionConfig(wall_height=3.5, wall_thickness=0.2)
体積の目安(壁 1 m): 0.7 m³
```

### 1.3 例外: 想定内の失敗を分ける

「壊れた図面」は、**よく起きる想定内の失敗**。**入力のミス(利用者に直してもらう)**と、**内部の不具合**を、例外の種類で分ける。API は、前者を `400` 系で返す。

```python
# 実行できる
class InputError(ValueError):
    """図面が壊れている・読めない(利用者の入力の問題)"""


def read_drawing(text: str) -> list[str]:
    if not text.startswith("0"):
        raise InputError("DXF の先頭(マジックバイト)がありません")
    return text.splitlines()


for sample in ["0\nSECTION\n", "<html>これは図面ではない"]:
    try:
        print("読めた:", read_drawing(sample))
    except InputError as error:
        print("入力エラー(400 を返す):", error)
```

```text
# 出力
読めた: ['0', 'SECTION']
入力エラー(400 を返す): DXF の先頭(マジックバイト)がありません
```

### 1.4 文字コード: バイト列と文字

コンピューターは**バイト列**を保存し、**文字コード**で文字に直す。日本の古い CAD の図面は **CP932(Shift_JIS 系)**、新しいものは **UTF-8**。**違う文字コードで読むと、文字化け**する。

```python
# 実行できる
name = "壁寸法"
as_cp932 = name.encode("cp932")
as_utf8 = name.encode("utf-8")
print("CP932 のバイト数:", len(as_cp932), " UTF-8 のバイト数:", len(as_utf8))
print("正しく読む:", as_cp932.decode("cp932"))
print("取り違えると:", as_utf8.decode("cp932", errors="replace"))          # 文字化け
try:
    as_cp932.decode("utf-8")
except UnicodeDecodeError:
    print("CP932 を UTF-8 として読むと、エラーになることがある")
```

```text
# 出力
CP932 のバイト数: 6  UTF-8 のバイト数: 9
正しく読む: 壁寸法
取り違えると: 螢∝ｯｸ豕�
CP932 を UTF-8 として読むと、エラーになることがある
```

## 2. 設計思想

| 原則 | 意味 | このツールのどこに出るか | 破るとどうなるか |
|---|---|---|---|
| **ルートを分けて、段階的に使う** | 図面の種類(A・B・C)ごとに、処理を選ぶ | CAD → 押し出し / 画像 → 3D | 1 つの手順で、全部を無理に処理して失敗する |
| **GPU の重い処理を、入口と出口で切る** | 推論は重いので、前処理・後処理と分けて測る | `PERF_RESULTS.md` | どこが遅いか分からない |
| **入力の失敗を、分類して返す** | 壊れた図面は 400 系、内部の不具合は 500 系 | `api.py` の例外の分類 | 利用者が、自分で直せるか分からない |
| **作業フォルダを分け、掃除する** | ジョブごとの作業場所を、期限つきで消す | API の TTL・上限 | ディスクが埋まる |
| **推論は、シングルトンで 1 回だけ読む** | 重いモデルを、何度も読み込まない | 推論クラス(`test_singleton.py`) | 毎回、数十秒かかる |
| **実測で語る** | 数字は、測った環境つきで書く | `PERF_RESULTS.md`・README | 古い数字を、今の性能と誤解する |

## 3. 関数・アルゴリズム

| 名前 | 場所 | 何をする | 計算量・注意 |
|---|---|---|---|
| **KD 木による近傍探索** | `core/floor_plan_extruder.py`(`cKDTree`) | 空間を木で分けて、**最も近い点**を探す | 構築 O(n log n)・探索 平均 O(log n)。全部と比べる O(n) より速い |
| **端点のスナップ** | 同上 | 近い端点(許容内)を、同じ点にそろえる | 許容を大きくすると、別の点まで混ざる |
| **Z バッファ** | 講義 第 2 回 | 画素ごとに、**最も手前の深さ**だけを残す | O(三角形 × 画素) |
| **重心座標** | 同上 | 三角形の中の点を、3 頂点の**重みつきの和**で表す(深さの補間・内外判定) | O(1) / 点 |
| **ポリゴンの押し出し** | `floor_plan_extruder.py` | 2D の輪郭を、高さ方向へ伸ばして 3D の壁にする | O(頂点数) |

```python
# 実行できる
import numpy as np
from scipy.spatial import cKDTree

endpoints = np.array([[0.0, 0.0], [10.0, 0.0], [10.01, 0.02], [20.0, 0.0]])    # 10.0 付近に、ほぼ同じ点が 2 つ
tree = cKDTree(endpoints)
tolerance = 0.05

pairs = sorted(tree.query_pairs(r=tolerance))
print("許容 0.05 以内のペア:", pairs)
distance, index = tree.query([10.0, 0.0], k=2)
print("(10, 0) に近い 2 点の距離:", [round(float(d), 3) for d in distance], " 位置:", index.tolist())

# Z バッファの核: 画素ごとに、最小の深さ(最も手前)を残す
depth = np.full(4, np.inf)
pixel_of = np.array([0, 1, 1, 2, 1])                         # 5 つの面が、どの画素に載るか
z = np.array([5.0, 3.0, 2.0, 9.0, 4.0])                      # それぞれの深さ
np.minimum.at(depth, pixel_of, z)
print("画素ごとの最も手前の深さ:", depth.tolist())
```

```text
# 出力
許容 0.05 以内のペア: [(1, 2)]
(10, 0) に近い 2 点の距離: [0.0, 0.022]  位置: [1, 2]
画素ごとの最も手前の深さ: [5.0, 2.0, 9.0, inf]
```

## 4. IT・エンタメの用語の補強

| 用語 | 意味 | このツールでの出番 |
|---|---|---|
| **CAD / DXF** | 設計用の図面ソフト / その図面の交換形式 | 入力 |
| **レイヤー(図層)** | 図面の要素を、分類して重ねる仕組み(壁・窓・寸法など) | 壁の抽出 |
| **ポリライン / ブロック** | 連なった線 / 部品としてまとめた図形 | `ezdxf` で読む |
| **メッシュ(頂点・面)** | 3D の形を、点と三角形で表したもの | 出力(OBJ・FBX) |
| **OBJ / FBX / glTF** | 3D モデルのファイル形式 | `fbx2obj.py` |
| **UV / テクスチャ** | 模様を貼る座標 / 貼る画像 | 3D アセット |
| **単一画像からの 3D 生成** | 1 枚の絵から、AI が 3D を推定する | Zero123++・TripoSR |
| **GPU / CUDA / VRAM** | 並列計算の装置 / その計算の仕組み / その専用メモリ | 推論(RTX 3070) |
| **推論(inference)** | 学習済みモデルに、入力を与えて答えを得ること | 画像 → 3D |
| **Z バッファ / 深度** | 手前の物だけを描く仕組み / 奥行きの値 | 可視率の確認 |
| **FastAPI / ジョブ / TTL** | Python の Web フレームワーク / 長い処理を受け付けて後で結果を返す仕組み / 期限 | `api.py` |
| **文字コード(CP932・UTF-8)** | 文字をバイトに直す規則 | 日本の図面の名前 |
| **Unity / Unreal への取り込み** | ゲームエンジンへ、モデルを入れる | 想定する出力先(未確認) |

## 確認問題

1. NumPy のブロードキャストで、`points + np.array([100.0, 50.0])` は何をしますか。
2. 壊れた図面を、内部の不具合と区別して例外にするのは、なぜですか。
3. KD 木が、全部の点と比べるより速いのは、なぜですか。
4. CP932 のバイト列を UTF-8 として読むと、何が起きますか。

## 解答

1. 全部の点に、(100, 50) を足す(平行移動)。ループは要らない。
2. 利用者が直せる入力のミス(400 系)と、開発者が直す不具合(500 系)を、API の応答で分けるため。
3. 空間を木で分け、近くにない枝を、まとめて捨てられるから(平均 O(log n))。
4. 文字化けする、または読めずにエラー(`UnicodeDecodeError`)になることがある。
