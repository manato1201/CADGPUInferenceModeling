# CADGPUInferenceModeling

**CAD図面（DXF）↔ ゲーム用3Dアセット 双方向変換パイプライン**

```
DXF → 自動ルート判定 → 押し出し3Dメッシュ → 屋根 → テクスチャ → Unity/UE
                              ↓ (--infer 時)
                      TripoSR / Zero123++ で外観推論

OBJ / FBX / GLB → Zバッファ隠線判定 → 三面図 + 平面図 → DXF
```

動作確認済み環境: Windows 11 / NVIDIA GeForce RTX 3070 (8.6GB VRAM) / Python 3.12

---


<img width="1196" height="786" alt="スクリーンショット 2026-05-26 215409" src="https://github.com/user-attachments/assets/1c64b99a-406a-47c7-988e-08d765f592f8" />
<img width="829" height="709" alt="スクリーンショット 2026-05-26 215317" src="https://github.com/user-attachments/assets/33af048f-a110-4e39-bed9-4643654c59cb" />
<img width="1311" height="605" alt="スクリーンショット 2026-05-30 225934" src="https://github.com/user-attachments/assets/e9dde388-c198-4c40-8c86-889934a63867" />
<img width="349" height="700" alt="スクリーンショット 2026-05-30 225943" src="https://github.com/user-attachments/assets/8a7fcfb8-06a1-4e6c-97c1-8bfce0ba3222" />
<img width="921" height="689" alt="スクリーンショット 2026-05-26 204210" src="https://github.com/user-attachments/assets/ee4e55e2-7345-4deb-80b8-08f604455602" />


## ディレクトリ構成

```
cad2asset/
├── core/
│   ├── parser.py              # DXFパース・三面図生成・ルート自動判定
│   ├── floor_plan_extruder.py # 建築平面図 → 押し出し3Dメッシュ（v2: 5項目改善）
│   ├── mesh_renderer.py       # 押し出しメッシュ → シルエット画像（視点プリセット付き）
│   ├── roof_generator.py      # 屋根自動生成（陸屋根/片流れ/切妻）
│   ├── texture_baker.py       # 手続き的テクスチャ生成（UV付きOBJ出力）
│   ├── mesh_to_dxf.py         # 3Dメッシュ → DXF変換（Zバッファ隠線判定）
│   ├── inferencer.py          # Zero123++ GPU推論（TSDF Fusion）
│   ├── triposr_inferencer.py  # TripoSR GPU推論（直接メッシュ出力）
│   └── postprocessor.py       # メッシュ修復・LOD生成・エクスポート
├── cli.py                     # CLIエントリポイント（run / export / info コマンド）
├── fbx2obj.py                 # FBX → OBJ 変換ツール（Blender経由）
├── api.py                     # FastAPI エントリポイント（フェーズ2）
├── TripoSR/                   # TripoSR リポジトリ（git clone で配置）
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 精度・一致度

### 図面 → 3Dモデル（DXF → OBJ）

| ルート | 対象 | 寸法精度 | 形状精度 | 備考 |
|---|---|---|---|---|
| **Route B** | POLYFACEメッシュ（car.dxf等） | **100%（誤差0mm）** | ★★★★★ | 座標をそのまま抽出 |
| **Route C** | 建築平面図（壁レイヤーあり） | **寸法保存100%** | ★★★☆☆ | 押し出しアルゴリズムで近似 |
| **Route A** | その他2D図面 | 押し出し依存 | ★★☆☆☆ | フォールバック処理 |

**Route B の実測値（car.dxf）:**
```
元DXF:       幅 2160.0mm × 奥行 4853.0mm × 高さ 1702.0mm
生成OBJ:     幅 2160.0mm × 奥行 4853.0mm × 高さ 1702.0mm
誤差:        X=0.0mm  Y=0.0mm  Z=0.0mm  （完全一致）
```

**Route C の実測値（2DDXF_Sample.dxf・2階建て住宅）:**
```
平面寸法:    10.38m × 10.21m（DXF座標範囲と一致）
高さ:        5.65m（1階 2.5m + 空隙 0.5m + 2階 2.5m + 天井スラブ）
壁厚自動推定: 100mm（DXFの平行線対から実測）
```

### 3Dモデル → 図面（OBJ → DXF）

| 項目 | 精度 | 備考 |
|---|---|---|
| **寸法保存率** | **100%** | 座標変換のみ・丸め誤差なし |
| **可視輪郭（VISIBLE）検出率** | **78%** | Zバッファ隠線判定（512×512深度マップ） |
| **隠線（HIDDEN）比率** | 21% | 前版は99.9%隠線→大幅改善 |
| **三面図レイアウト** | 正確 | 正面(XZ)・側面(YZ)・上面(XY)を自動配置 |
| **寸法線精度** | バウンディングボックス基準 | 外形寸法は正確・内部詳細寸法は手動追加要 |

**往復精度（DXF → OBJ → DXF）:**
```
元DXF寸法:   2160mm × 4853mm × 1702mm
OBJ経由後:   2160mm × 4853mm × 1702mm
寸法誤差:    0mm（完全一致）
隠線判定:    Zバッファ法（深度画像512×512）で78%が可視に正しく分類
```

**隠線判定の改善履歴:**

| バージョン | 手法 | VISIBLE比率 |
|---|---|---|
| v1（初期） | 深度中央値による単純分類 | 0.1% |
| v2（法線ベース） | 面法線による前面/背面判定 | 27.7% |
| **v3（現行）** | **Zバッファ深度画像** | **78.0%** |

---

## セットアップ

### 1. Python 仮想環境

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 2. PyTorch CUDA版

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 3. 依存パッケージ

```bash
pip install -r requirements.txt
pip install shapely scipy scikit-image onnxruntime
```

### 4. TripoSR セットアップ（--infer --model triposr を使う場合）

```bash
git clone https://github.com/VAST-AI-Research/TripoSR.git
cd TripoSR

# torchmcubesを除いてインストール
(Get-Content requirements.txt) | Where-Object { $_ -notmatch "torchmcubes" } | Set-Content requirements_no_mcubes.txt
pip install -r requirements_no_mcubes.txt
pip install onnxruntime scikit-image
```

`TripoSR\tsr\models\isosurface.py` の先頭を修正:

```python
try:
    from torchmcubes import marching_cubes
except ImportError:
    from skimage.measure import marching_cubes as _ski_mc
    def marching_cubes(volume, level):
        import torch
        v, f, *_ = _ski_mc(volume.cpu().numpy(),
                           level=level.item() if hasattr(level, 'item') else float(level))
        return torch.from_numpy(v.copy()), torch.from_numpy(f.copy().astype('int64'))
```

```powershell
$env:TRIPOSR_PATH = "C:\path\to\CADGPUInferenceModeling\TripoSR"
```

---

## コマンド: run（DXF → 3Dアセット）

### 基本実行

```bash
# 押し出し + 屋根 + テクスチャ（デフォルト・推論なし）
python cli.py run your_drawing.dxf --output-dir ./out --format obj

# TripoSR 推論あり
python cli.py run your_drawing.dxf --output-dir ./out --infer --model triposr
```

### 出力ファイル構成

```
out/
├── views/                        # 三面図（確認用）
│   ├── front.png / side.png / top.png
├── extrude/
│   └── {name}_extrude.obj        # 押し出しメッシュ（屋根・テクスチャ前）
├── render/                       # レンダリング画像（--infer 時）
│   ├── primary.png               # TripoSR/Zero123++への入力画像
│   └── view_0.png ...
└── assets/
    ├── {name}_LOD0.obj           # 最高解像度メッシュ
    ├── {name}_LOD1.obj           # 中解像度
    ├── {name}_LOD2.obj           # 低解像度
    ├── {name}.obj                # テクスチャ付き
    ├── {name}.mtl                # マテリアルファイル
    └── {name}_texture.png        # テクスチャ画像（デフォルト 1024×1024）
```

### run オプション一覧

| オプション | デフォルト | 説明 |
|---|---|---|
| `--output-dir PATH` | `./output` | 出力先ディレクトリ |
| `--format obj\|glb` | `obj` | 出力フォーマット |
| `--no-infer / --infer` | `--no-infer` | 推論スキップ（デフォルト）/ 推論あり |
| `--model triposr\|zero123` | `triposr` | 推論モデルの選択 |
| `--triposr-res INT` | `256` | TripoSR メッシュ解像度（32〜256） |
| `--view-angle PRESET` | `front` | レンダリング視点（下記参照） |
| `--infer-steps INT` | `36` | Zero123++ ステップ数 |
| `--roof-type flat\|shed\|gable` | `gable` | 屋根タイプ |
| `--roof-height FLOAT` | `1.5` | 棟の高さ [m] |
| `--no-roof` | `False` | 屋根生成をスキップ |
| `--wall-style brick\|concrete\|wood` | `brick` | 外壁テクスチャ |
| `--roof-style tile\|metal\|flat` | `tile` | 屋根テクスチャ |
| `--no-texture` | `False` | テクスチャ生成をスキップ |
| `--texture-size INT` | `1024` | テクスチャ解像度 [px] |
| `--default-height FLOAT` | `2500.0` | Z=0図面への定数高さ [mm] |
| `--wall-thickness FLOAT` | `150.0` | 壁厚 [mm]（自動推定が優先） |
| `--use-union` | `False` | 壁ボックスをブーリアンUnionで結合 |
| `--force-unit mm\|cm\|m` | (自動) | 単位を強制指定 |

### --view-angle プリセット

| プリセット | elevation | azimuth | 特徴 |
|---|---|---|---|
| `front` | 25° | 0° | **デフォルト・TripoSR推奨** |
| `front_low` | 15° | 0° | 正面・低め |
| `front_high` | 40° | 0° | 正面・高め |
| `corner` | 25° | 45° | 斜め正面 |
| `corner_low` | 15° | 45° | 斜め・低め |
| `corner_high` | 40° | 45° | 斜め・高め |
| `side` | 25° | 90° | 側面 |
| `top` | 75° | 45° | 俯瞰 |
| `iso` | 35° | 45° | アイソメトリック |
| `"EL,AZ"` | 任意 | 任意 | 直接指定（例: `"20,30"`） |

---

## コマンド: export（3DメッシュをDXFに変換）

### 基本実行

```bash
# OBJ → 三面図+平面図 DXF（デフォルト）
python cli.py export model.obj

# 三面図のみ
python cli.py export model.obj --mode triview

# 平面図のみ・m単位
python cli.py export model.obj --mode floorplan --unit m

# 出力先を指定
python cli.py export model.glb -o ./drawings/output.dxf

# 隠線なし（輪郭線のみ・すっきりした図面）
python cli.py export model.obj --no-hidden

# DXFをビューアで確認
python -m ezdxf view output.dxf
```

### export オプション一覧

| オプション | デフォルト | 説明 |
|---|---|---|
| `--output / -o PATH` | (入力と同フォルダ) | 出力 DXF パス |
| `--mode triview\|floorplan\|both` | `both` | 出力モード |
| `--unit mm\|m` | `mm` | 出力単位 |
| `--no-dim` | `False` | 寸法線をスキップ |
| `--no-hidden` | `False` | 隠線をスキップ（図面がすっきりする） |
| `--slice FLOAT` | `0.3` | 平面図スライス高さ（建物高さに対する割合） |
| `--gap FLOAT` | `2.0` | 三面図ビュー間の余白 [m] |

### 出力 DXF のレイヤー構成

| レイヤー名 | 色 | 線種 | 内容 |
|---|---|---|---|
| `VISIBLE` | 白 | 実線 | 可視輪郭線（Zバッファで判定） |
| `HIDDEN` | グレー | 破線 | 隠線（遮蔽されたエッジ） |
| `DIMENSION` | 黄 | 実線 | 寸法線・ビューラベル |
| `WALL` | 白 | 実線 | 平面図の壁断面エッジ |
| `FLOOR` | 緑 | 実線 | 平面図の外形輪郭 |
| `CENTERLINE` | 赤 | 一点鎖線 | 中心線 |

### Zバッファ隠線判定の仕組み

```
メッシュ全三角形をラスタライズ → 512×512の深度マップ生成
  ↓
各エッジを5点サンプリング
  ↓ 各点の深度 vs 深度マップを比較
  ↓ 過半数が「手前」→ VISIBLE / 「奥」→ HIDDEN
```

GPU不使用・numpy のみで実装。精度向上の過程:

| バージョン | 手法 | VISIBLE比率 |
|---|---|---|
| v1 | 深度中央値による単純分類 | 0.1% |
| v2 | 面法線による前面/背面判定 | 27.7% |
| **v3（現行）** | **Zバッファ深度画像（512×512）** | **78.0%** |

---

## FBX の変換（fbx2obj.py）

pyassimpはWindowsでDLL本体が必要なため、Blender CLI経由で変換する。

```bash
# 1ファイル変換
python fbx2obj.py model.fbx

# 複数ファイルを一括変換
python fbx2obj.py *.fbx

# Blenderのパスを直接指定
python fbx2obj.py model.fbx --blender "C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"

# 変換後にDXFへ
python cli.py export model.obj
```

Blenderのインストールパスを自動検索（4.5 → 4.2 → 4.1 → 4.0 → 3.6 の順）。

---

## floor_plan_extruder v2 の5つの改善

### ① 端点スナップ
Union-Find で80mm以内の端点を統合し、壁の隙間・浮きを解消。

### ② 壁厚自動推定
平行線ペアを検出して実際の壁厚を自動推定（例: 100mm）。`--wall-thickness` で上書き可能。

### ③ 開口部高さ分類
線分長さでドア（900mm以上・床〜2m）と窓（900mm未満・床90cm〜2m）を分類して別高さで切り抜き。

### ④ 複数階の統合
`壁1階`/`壁2階` レイヤーを自動検出し、1階（Z=0〜2.5m）・2階（Z=3.0〜5.5m）として積み上げ。

### ⑤ 天井スラブ
各階の天井に 150mm 厚のスラブを追加。

**実測結果（2DDXF_Sample.dxf・2階建て）:**

| 項目 | 値 |
|---|---|
| 1階壁線分 | 61本（Z=0〜2500mm） |
| 2階壁線分 | 97本（Z=3000〜5500mm） |
| 壁厚自動推定 | 100mm（実測値） |
| 完成メッシュ | verts=1098, faces=1932 |
| 高さ | -0.20〜5.65m（2階建て） |

---

## Unity / Unreal Engine へのインポート

### OBJ形式（推奨）

`assets/` フォルダごと Unity の Assets にコピー。`.obj` / `.mtl` / `_texture.png` が同じフォルダにある必要がある。

### GLB形式

```
Window → Package Manager → Add package by name
→ com.unity.cloud.gltfast → Install
```

---

## VRAM・性能目安（RTX 3070 / 8.6GB）

| 処理 | 所要時間 | VRAM |
|---|---|---|
| 押し出し + 屋根 + テクスチャ | 約5〜10秒 | GPU不使用 |
| TripoSR 推論（256解像度） | 約10〜30秒 | 約6GB |
| Zero123++ 推論（36ステップ） | 約11秒 | 約8GB |
| 3Dメッシュ → DXF（Zバッファ付き） | 約18〜30秒 | GPU不使用 |
| FBX → OBJ（Blender経由） | 約10〜30秒 | GPU不使用 |

初回ダウンロード:

| モデル | サイズ |
|---|---|
| TripoSR | 約1.7GB |
| Zero123++ | 約5.6GB |
| Depth-Anything-V2-Small | 約400MB |

---

## トラブルシューティング

### `ModuleNotFoundError: No module named 'shapely'`
```bash
pip install shapely
```

### `ModuleNotFoundError: No module named 'typer'`
仮想環境が無効。`.venv\Scripts\Activate.ps1` を実行してから再試行。

### `ModuleNotFoundError: No module named 'torchmcubes'`
`TripoSR\tsr\models\isosurface.py` を skimage フォールバックに修正（セットアップ参照）。

### `ModuleNotFoundError: No module named 'onnxruntime'`
```bash
pip install onnxruntime
```

### `pyassimp.errors.AssimpError: assimp library not found`
WindowsでpyassimpはDLL本体が別途必要。`fbx2obj.py` でBlender経由変換を使う（FBXの変換参照）。

### TripoSR関連エラー一覧

| エラー | 原因 | 対処 |
|---|---|---|
| `got an unexpected keyword argument 'chunk_size'` | このバージョンは非対応 | 現バージョンで修正済み |
| `missing 1 required positional argument: 'has_vertex_color'` | 引数不足 | 現バージョンで `has_vertex_color=False` 追加済み |
| `The size of tensor a (4) must match tensor b (3)` | RGBA→RGB未変換 | 現バージョンで前処理時にRGB変換済み |

### 建築平面図なのに Route A に入る
ログの `[wall_1]`/`[wall_2]` を確認。`other` に分類されている場合は `floor_plan_extruder.py` の `_WALL_MOJ` にキーワードを追加。

### GLBがUnityで表示されない
`--format obj` を使うか `glTFast` パッケージをインストール。

### DWGが読み込めない
[ODA File Converter](https://www.opendesign.com/guestfiles/oda_file_converter)（無料）でDXFに変換してから使用。

---

## 既知の制限

| 項目 | 現状 | 備考 |
|---|---|---|
| 屋根形状 | 陸屋根/片流れ/切妻の3種類 | 寄棟・入母屋は未対応 |
| テクスチャ | 手続き的生成（写真品質ではない） | AI生成テクスチャとの統合で改善可能 |
| 推論品質 | 建物外観の正確な再現は困難 | 押し出しメッシュ単体が最も正確 |
| FBX読み込み | Blender経由が必要 | `fbx2obj.py` で変換してから使用 |
| 隠線判定速度 | 約18秒/モデル（3ビュー） | zbuf_size=256 で約4倍高速化可能 |
| DWG | 直接読み込み不可 | ODA File Converter で変換 |
| 寸法線 | 外形寸法のみ自動生成 | 内部詳細寸法は手動追加要 |
