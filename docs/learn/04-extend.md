# 4. 拡張のレシピ

どれも「**設定(データ)→ 関数 → テスト**」の順で、小さく進めます。**既定の挙動を変えない**(新しい引数・設定は、既定で今と同じ)のが原則です。

## 設定を読む

押し出しと屋根は、**設定のデータクラス**で決まります。まず既定値を確かめます。

```python
# 実行できる
from core.floor_plan_extruder import ExtrusionConfig
from core.roof_generator import RoofConfig

e = ExtrusionConfig()
print("階高", e.ceiling_height, "mm / 壁厚", e.wall_thickness, "mm / スナップ", e.wall_snap_tol, "mm / 2 階の上げ幅", e.floor_height_step, "mm")
print("ドア", e.opening_height, "mm / 窓 台", e.window_sill, "mm・高さ", e.window_height, "mm")
r = RoofConfig()
print("屋根:", r.roof_type, "/ 棟", r.ridge_height, "m / 軒の出", r.overhang, "m")
```

```text
# 出力
階高 2500.0 mm / 壁厚 150.0 mm / スナップ 80.0 mm / 2 階の上げ幅 3000.0 mm
ドア 2000.0 mm / 窓 台 900.0 mm・高さ 1100.0 mm
屋根: gable / 棟 1.5 m / 軒の出 0.3 m
```

## レシピ 1: 屋根の種類を足す(例: 寄棟 hip)

1. `RoofType`(`Literal`)に `"hip"` を足し、`_gen_hip_roof(footprint, z_ceiling, cfg)` を書く(`_gen_gable_roof` を参考に)。
2. `generate_roof` の**振り分け**に分岐を足す(未知の種類は例外のまま)。
3. **テスト**: 単純な長方形の輪郭で、**閉じたメッシュ**(`is_watertight`)・高さ(棟の高さ)を確かめる。
4. CLI の引数(`--roof`)の選択肢にも足す。

## レシピ 2: レイヤー分類のルールを足す(例: 「ゾーン」)

講義[第 3 回](../lecture/03-encoding-and-capstone.md)の総合演習が完成形です。

1. 日本語の語は `_cp932_moj("語")` で**化けさせた形**にして、`_..._MOJ` のリストに足す(英語の語は、`d`(小文字)の部分一致)。
2. `_classify_layers` の**どこに**置くか決める(**順序が結果を決める**。「壁寸法」が壁になる、という落とし穴)。
3. `_collect_segments` が、新しいカテゴリを**対象にするか・外すか**を決める。
4. **テスト**: 化けた名前・英語名・順序の取り合い(偽のレイヤーのオブジェクトで足りる)。

## レシピ 3: 押し出しの設定を足す(例: 手すり)

1. `ExtrusionConfig` に項目を足す(**既定は、今と同じ挙動**)。
2. 押し出しの関数で、設定を読んで反映する。**メッシュの結合・修復**(`repair_mesh`)の前に足す(結合の後だと、修復が崩す)。
3. `cli.py` の引数に足し、`ExtrusionConfig` へ渡す。**API からも使うなら**、`api.py` にも渡す(CLI と API で、設定が食い違わないように)。
4. **テスト**: 既定で結果が変わらないこと・設定したときの寸法(`test_accuracy.py` の形)。

## レシピ 4: テクスチャを足す

`texture_baker.py` に `_make_<name>_texture(cfg) -> Image` を足し、`bake_textures` の振り分けに足す。**手続き的**(乱数のシードを固定して、毎回同じ画像)にする。UV は `_unwrap_box_uv`。

## レシピ 5: 書き出し形式を足す

`postprocess.export_meshes` に形式を足す。**LOD ごと**に出す。単位(m)・向き(Z 上か Y 上か)を、受け側(Unity・UE)に合わせる。

## レシピ 6: API を足す

`api.py` に `@app.get / post` を足す。**パスにクライアントの値を使うときは、検証する**(ファイル名・LOD 名。[6 トラブル](06-troubleshooting-maintenance.md)の既知の問題 1)。**認証・上限は、今は無い**。

## 変更の前に

- **ルート判定(A・B・C)の順序**を変えない(B → C → A)。
- 変更したら `pytest -q` と `python tools/check_docs.py`([5 テスト](05-testing.md))を通す。
