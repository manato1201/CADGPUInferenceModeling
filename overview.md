# CADGPUInferenceModeling — 技術概要

> CAD図面（DXF）↔ ゲーム用3Dアセット（OBJ/GLB）双方向変換パイプライン  
> 建築平面図・3D POLYFACEメッシュの自動変換 + GPU推論 + Zバッファ隠線判定

---

## 目次

1. [背景・目的](#背景目的)
2. [システム構成](#システム構成)
3. [処理ルートの自動判定](#処理ルートの自動判定)
4. [コアアルゴリズム](#コアアルゴリズム)
5. [精度・性能](#精度性能)
6. [技術スタック](#技術スタック)
7. [使い方](#使い方)

---

## 背景・目的

ゲーム開発において建築・乗り物・設備を扱うタイトルでは、  
**「CAD図面 → ゲームアセット」の変換を手動でモデリングする工程**が存在し、コストが高い。

本プロジェクトはこの工程を **アルゴリズム + GPU推論** で自動化することを目的とする。  
逆方向（3Dモデル → DXF図面）も実装し、両方向で **寸法精度 100%** を達成した。

---

## システム構成

```
CADGPUInferenceModeling/
├── core/
│   ├── parser.py               DXFパース・ルート自動判定
│   ├── floor_plan_extruder.py  建築平面図 → 3Dメッシュ（v2: 5項目改善）
│   ├── mesh_renderer.py        メッシュ → シルエット画像（視点プリセット付き）
│   ├── roof_generator.py       屋根自動生成（陸屋根/片流れ/切妻）
│   ├── texture_baker.py        手続き的テクスチャ生成
│   ├── mesh_to_dxf.py          3Dメッシュ → DXF（Zバッファ隠線判定）
│   ├── inferencer.py           Zero123++ GPU推論
│   ├── triposr_inferencer.py   TripoSR GPU推論
│   └── postprocessor.py        LOD生成・エクスポート
├── cli.py                      CLIエントリ（run / export / info）
├── fbx2obj.py                  FBX → OBJ 変換（Blender経由）
└── api.py                      FastAPI サーバー
```

---

## 処理ルートの自動判定

入力 DXF の種類を解析し、最適なルートで処理する。

```
DXF 読み込み
 │
 ├─ Route C: 建築平面図
 │   判定条件: 壁レイヤーあり + Z=0 の2D図面
 │   処理: 壁線分抽出 → 端点スナップ → 壁厚推定 → 押し出し
 │         → ドア/窓高さ分類 → 天井スラブ → 屋根 → テクスチャ
 │
 ├─ Route B: 3D POLYFACEメッシュ
 │   判定条件: Z値あり・POLYLINE flags=0x40
 │   処理: 頂点/面を直接抽出 → 屋根 → テクスチャ
 │
 └─ Route A: その他2D図面
     処理: 押し出しフォールバック → 屋根 → テクスチャ
           ※ --infer 時は TripoSR / Zero123++ で外観推論
```

---

## コアアルゴリズム

### 1. Zバッファ深度画像による隠線判定

3DモデルをDXF図面に変換する際の可視エッジ判定。  
GPU不使用・numpy のみでバリセントリック座標法を実装。

**処理フロー:**

```
全三角形をバリセントリック座標法でラスタライズ
  → 512×512 float32 の深度マップを生成

各エッジを5点サンプリング
  → 各点の深度値 vs 深度マップの値 を比較
  → 過半数が「手前」 → VISIBLE（可視）
  → 過半数が「奥」   → HIDDEN（隠線）
```

**精度の改善履歴:**

| バージョン | 手法 | VISIBLE比率 |
|---|---|---|
| v1 | 深度中央値による単純分類 | 0.1% |
| v2 | 面法線による前面/背面判定 | 27.7% |
| v3（現行） | Zバッファ深度画像（512×512） | **78.0%** |

**核心コード:**

```python
def _build_zbuffer(verts, faces, ax_h, ax_v, ax_d, size=512):
    z_buf = np.full((size, size), np.inf, dtype=np.float32)
    for face in faces:
        a, b, c = face
        # バリセントリック座標でピクセルをカバー
        area = (h1-h0)*(v2-v0) - (h2-h0)*(v1-v0)
        w0 = ((h1-h2)*(gx-h2) + (v2-v1)*(gy-v2)) / area
        w1 = ((h2-h0)*(gx-h0) + (v0-v2)*(gy-v0)) / area
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -0.01) & (w1 >= -0.01) & (w2 >= -0.01)
        depth  = w0*d0 + w1*d1 + w2*d2
        z_buf[gy[inside], gx[inside]] = np.minimum(
            z_buf[gy[inside], gx[inside]], depth[inside]
        )
    return z_buf
```

---

### 2. CP932 Mojibakeパターンマッチング

日本語CADソフトのDXFはレイヤー名がCP932エンコードされているが、  
ezdxfはcp1252 Mojibakeとして読み込む。  
キーワードを事前にCP932→cp1252変換してパターンマッチングを行う。

```python
def _cp932_moj(text: str) -> str:
    """CP932テキスト → ezdxf内部のcp1252 Mojibake文字列に変換"""
    b = text.encode("cp932")
    result = []
    for byte in b:
        if 0x80 <= byte <= 0xFF:
            try:
                result.append(bytes([byte]).decode("cp1252"))
            except ValueError:
                result.append(chr(0xDC00 + byte))  # surrogate
        else:
            result.append(chr(byte))
    return "".join(result)

# 壁1階/壁2階を自動検出
_WALL_MOJ   = [_cp932_moj("壁")]   # ezdxf内部: "•Ç" にマッチ
_FLOOR1_MOJ = [_cp932_moj("1階")]  # → Z = 0〜2500mm
_FLOOR2_MOJ = [_cp932_moj("2階")]  # → Z = 3000〜5500mm
```

---

### 3. Union-Find による端点スナップ

壁線分の接合点が微妙にずれている問題を Union-Find で解決。  
80mm以内の近接端点をクラスタリングして重心に統合する。

```
全端点を収集 (N点)
  ↓
Union-Find で80mm以内のペアを結合
  ↓
クラスタの重心を代表点として線分を再構築
  ↓
隙間・浮きのない壁メッシュを生成
```

---

### 4. 平行線ペア検出による壁厚自動推定

建築CADの壁は2本の平行線で表現される慣習を利用。

```
各線分の角度・中点を計算
  ↓
角度差 < 30° かつ 垂直距離 80〜500mm のペアを検出
  ↓
垂直距離の中央値 = 実際の壁厚
  ↓
実測: デフォルト150mm → 自動推定 100mm
```

---

### 5. 建築平面図 v2 の5項目改善

| 項目 | 改善内容 | 効果 |
|---|---|---|
| ① 端点スナップ | Union-Find (tol=80mm) | 壁の隙間・浮きを解消 |
| ② 壁厚自動推定 | 平行線ペア検出 | 150mm → 実測100mmに修正 |
| ③ 開口部高さ分類 | 線分長さでドア/窓を判別 | ドア(床〜2m)・窓(90cm〜2m) |
| ④ 複数階統合 | レイヤー名から階数を検出 | 1階+2階を正確に積み上げ |
| ⑤ 天井スラブ | 各階天井に150mm板を追加 | 建物としてのリアリティ向上 |

---

## 精度・性能

### 寸法精度（DXF → OBJ, car.dxf 実測）

| 軸 | 元DXF | 生成OBJ | 誤差 |
|---|---|---|---|
| 幅（X） | 2160.0mm | 2160.0mm | **0.0mm** |
| 奥行（Y） | 4853.0mm | 4853.0mm | **0.0mm** |
| 高さ（Z） | 1702.0mm | 1702.0mm | **0.0mm** |

### 建築平面図 押し出し実測（2DDXF_Sample.dxf）

```
1階壁線分:  61本 (Z=0〜2500mm)
2階壁線分:  97本 (Z=3000〜5500mm)
壁厚推定:   100mm（平行線対から自動算出）
完成メッシュ: verts=1098, faces=1932
高さ:       -0.20〜5.65m（2階建て正確に再現）
```

### 処理時間（RTX 3070 / 8.6GB VRAM）

| 処理 | 時間 | GPU |
|---|---|---|
| 押し出し+屋根+テクスチャ | ~10秒 | 不使用 |
| TripoSR 推論（256res） | ~30秒 | ~6GB |
| 3Dメッシュ→DXF変換 | ~18秒 | 不使用 |

---

## 技術スタック

| カテゴリ | ライブラリ | 用途 |
|---|---|---|
| CAD処理 | ezdxf | DXF読み書き |
| 3Dメッシュ | trimesh | メッシュ処理・エクスポート |
| 幾何演算 | shapely, scipy | 2Dポリゴン・Union-Find |
| 数値計算 | numpy | Zバッファ・行列演算 |
| GPU推論 | PyTorch + CUDA | TripoSR・Zero123++ |
| 深度推定 | Depth-Anything-V2 | 単眼深度推定 |
| API | FastAPI + typer | サーバー・CLI |
| 画像処理 | Pillow, matplotlib | レンダリング・前処理 |

---

## 使い方

### DXF → 3Dアセット

```bash
# 推論なし（押し出し+屋根+テクスチャ）
python cli.py run building.dxf --output-dir ./out

# TripoSR推論あり
python cli.py run building.dxf --output-dir ./out --infer --model triposr

# 陸屋根 + コンクリートテクスチャ
python cli.py run building.dxf --output-dir ./out --roof-type flat --wall-style concrete
```

### 3Dモデル → DXF図面

```bash
# OBJ → 三面図+平面図（Zバッファ隠線判定付き）
python cli.py export model.obj

# 隠線なし・三面図のみ
python cli.py export model.obj --mode triview --no-hidden

# FBX は事前変換
python fbx2obj.py model.fbx
python cli.py export model.obj
```

---

## 入手できるテストデータ

| サイト | URL | 形式 | 内容 |
|---|---|---|---|
| Free3D | free3d.com/3d-models/dxf | 3D POLYFACE DXF | 車・建物・動物など174種 |
| 3D ContentCentral | 3dcontentcentral.com | DXF/STEP | 高品質な機械部品 |
| CADblocksfree.com | cadblocksfree.com | 2D DXF | 建築平面図 |
| GrabCAD | grabcad.com | DXF/FBX/OBJ | 何でも |
