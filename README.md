# cad2asset

**CAD図面（DXF）→ 3Dアセット自動生成パイプライン**

```
DXF → 自動ルート判定 → 押し出し / 直接抽出 → 屋根生成 → テクスチャ → Unity/UE
                  ↓ (オプション)
              TripoSR / Zero123++ で外観推論
```

動作確認済み環境: Windows 11 / NVIDIA GeForce RTX 3070 (8.6GB VRAM) / Python 3.12

---

## ディレクトリ構成

```
cad2asset/
├── core/
│   ├── parser.py              # DXFパース・三面図生成・ルート判定
│   ├── inferencer.py          # Zero123++ GPU推論（TSDF Fusion）
│   ├── triposr_inferencer.py  # TripoSR GPU推論（直接メッシュ出力）
│   ├── postprocessor.py       # メッシュ修復・LOD・エクスポート
│   ├── floor_plan_extruder.py # 建築平面図 → 押し出し3Dメッシュ
│   ├── mesh_renderer.py       # 押し出しメッシュ → シルエット画像
│   ├── roof_generator.py      # 屋根自動生成（陸屋根/片流れ/切妻）
│   └── texture_baker.py       # 手続き的テクスチャ生成（UV付きOBJ出力）
├── cli.py                     # CLIエントリポイント
├── api.py                     # FastAPI エントリポイント（フェーズ2）
├── TripoSR/                   # TripoSR リポジトリ（git clone で配置）
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 処理ルートの自動判定

DXFの種類を自動判定して最適なルートで処理する。

```
DXF 読み込み
 ├─ Route C: 建築平面図（壁レイヤーあり・Z=0）
 │     壁線分抽出 → 押し出し → 屋根生成 → テクスチャ
 │     ※ --infer でTripoSR/Zero123++による外観生成も可能
 │
 ├─ Route B: 3D POLYFACEメッシュ（Z値あり・car.dxf等）
 │     メッシュ直接抽出 → 屋根生成 → テクスチャ
 │
 └─ Route A: その他2D図面
       押し出しフォールバック → 屋根生成 → テクスチャ
       ※ --infer でTripoSR/Zero123++による外観生成も可能
```

---

## セットアップ

### 1. Python 仮想環境

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Linux / Mac
source .venv/bin/activate
```

### 2. PyTorch CUDA版（RTX 30/40系・CUDA 12.x）

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

GPU認識を確認:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# True  NVIDIA GeForce RTX 3070
```

### 3. 依存パッケージ

```bash
pip install -r requirements.txt
pip install shapely scipy scikit-image onnxruntime
```

### 4. TripoSR のセットアップ（推論機能を使う場合）

```bash
# プロジェクトフォルダ内にclone
git clone https://github.com/VAST-AI-Research/TripoSR.git
cd TripoSR

# torchmcubesを除いてインストール（Windows環境はビルド不可のため）
(Get-Content requirements.txt) | Where-Object { $_ -notmatch "torchmcubes" } | Set-Content requirements_no_mcubes.txt
pip install -r requirements_no_mcubes.txt
pip install onnxruntime

# torchmcubesのかわりにskimageを使うようにisosurface.pyを修正
# TripoSR\tsr\models\isosurface.py の6行目を以下に変更:
```

```python
# 変更前
from torchmcubes import marching_cubes

# 変更後
try:
    from torchmcubes import marching_cubes
except ImportError:
    from skimage.measure import marching_cubes as _ski_mc
    def marching_cubes(volume, level):
        import torch
        v, f, *_ = _ski_mc(volume.cpu().numpy(), level=level.item() if hasattr(level, 'item') else float(level))
        return torch.from_numpy(v.copy()), torch.from_numpy(f.copy().astype('int64'))
```

---

## 基本的な使い方

### デフォルト実行（推論なし・高速）

```bash
# 切妻屋根 + レンガテクスチャ付きアセットを生成（デフォルト）
python cli.py run your_drawing.dxf --output-dir ./out --format obj
```

出力ファイル構成:

```
out/
├── views/                        # 三面図（確認用）
│   ├── front.png / side.png / top.png
├── extrude/
│   └── {name}_extrude.obj        # 押し出しメッシュ（屋根・テクスチャ前）
├── render/                       # レンダリング画像（推論時）
│   └── primary.png / view_0.png
└── assets/
    ├── {name}_LOD0.obj           # 最高解像度
    ├── {name}_LOD1.obj           # 中解像度
    ├── {name}_LOD2.obj           # 低解像度
    ├── {name}.obj                # テクスチャ付き（LOD0相当）
    ├── {name}.mtl                # マテリアルファイル
    └── {name}_texture.png        # テクスチャ画像（1024×1024）
```

### TripoSR 推論モード（高品質・初回2.5GBダウンロード）

```bash
# 環境変数でTripoSRリポジトリのパスを設定（PowerShell）
$env:TRIPOSR_PATH = "C:\path\to\CADGPUInferenceModeling\TripoSR"

# TripoSRで推論
python cli.py run your_drawing.dxf --output-dir ./out --infer --model triposr
```

### オプション一覧

| オプション | デフォルト | 説明 |
|---|---|---|
| `--output-dir PATH` | `./output` | 出力先ディレクトリ |
| `--format obj\|glb` | `obj` | 出力フォーマット（Unity標準はobj） |
| `--no-infer / --infer` | `--no-infer` | 推論スキップ（デフォルト）/ 推論あり |
| `--model triposr\|zero123` | `triposr` | 推論モデルの選択 |
| `--triposr-res INT` | `256` | TripoSR メッシュ解像度（32〜256） |
| `--infer-steps INT` | `36` | Zero123++ ステップ数 |
| `--roof-type flat\|shed\|gable` | `gable` | 屋根タイプ |
| `--roof-height FLOAT` | `1.5` | 棟の高さ [m] |
| `--no-roof` | `False` | 屋根生成をスキップ |
| `--wall-style brick\|concrete\|wood` | `brick` | 外壁テクスチャ |
| `--roof-style tile\|metal\|flat` | `tile` | 屋根テクスチャ |
| `--no-texture` | `False` | テクスチャ生成をスキップ |
| `--texture-size INT` | `1024` | テクスチャ解像度 [px] |
| `--default-height FLOAT` | `2500.0` | Z=0図面への定数高さ [mm] |
| `--wall-thickness FLOAT` | `150.0` | 壁厚 [mm] |
| `--force-unit mm\|cm\|m\|inch` | `` | 単位を強制指定（自動判定上書き） |
| `--lod0 INT` | `10000` | LOD0 最大ポリゴン数 |
| `--lod1 INT` | `4000` | LOD1 最大ポリゴン数 |
| `--lod2 INT` | `1000` | LOD2 最大ポリゴン数 |

### 使用例

```bash
# 陸屋根 + コンクリートテクスチャ
python cli.py run building.dxf --output-dir ./out --roof-type flat --wall-style concrete

# テクスチャなし・屋根なし（壁のみ）
python cli.py run building.dxf --output-dir ./out --no-texture --no-roof

# 階高3m・壁厚200mmで生成
python cli.py run building.dxf --output-dir ./out --default-height 3000 --wall-thickness 200

# DXFの単位が誤認識される場合に強制指定
python cli.py run drawing.dxf --output-dir ./out --force-unit mm

# メタデータだけ確認
python cli.py info your_drawing.dxf
```

---

## Unity / Unreal Engine へのインポート

### OBJ形式（推奨・追加パッケージ不要）

`assets/` フォルダごと Unity の Assets にコピーする。
`.obj` / `.mtl` / `_texture.png` の3ファイルが同じフォルダにある必要がある。

### GLB形式

Unity で GLB を読むには `glTFast` パッケージが必要:

```
Window → Package Manager → Add package by name
→ com.unity.cloud.gltfast → Install
```

---

## VRAM・性能目安（RTX 3070 / 8.6GB）

| 処理 | 所要時間 | VRAM使用量 |
|---|---|---|
| 押し出し + 屋根 + テクスチャ | 約5〜10秒 | GPU不使用 |
| TripoSR 推論（256解像度） | 約10〜30秒 | 約6GB |
| Zero123++ 推論（36ステップ） | 約11秒 | 約8GB |
| Depth-Anything-V2（Small） | 約1秒/枚 | 約2GB |

初回実行時のモデルダウンロード:

| モデル | サイズ | 保存先 |
|---|---|---|
| TripoSR | 約1.7GB | `~/.cache/huggingface/` |
| Zero123++ | 約5.6GB | `~/.cache/huggingface/` |
| Depth-Anything-V2-Small | 約400MB | `~/.cache/huggingface/` |

---

## フェーズ2: APIサーバーとして動かす

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Swagger UI: http://localhost:8000/docs

| Method | Path | 説明 |
|---|---|---|
| `POST` | `/convert` | DXFをアップロードしてジョブ登録 |
| `GET` | `/jobs/{job_id}` | ジョブ状態確認 |
| `GET` | `/jobs/{job_id}/download/{lod}` | アセットをダウンロード |
| `GET` | `/health` | ヘルスチェック |

---

## トラブルシューティング

### `ModuleNotFoundError: No module named 'shapely'`
```bash
pip install shapely
```

### `ModuleNotFoundError: No module named 'torchmcubes'`
TripoSR の `isosurface.py` を skimage フォールバックに修正する（セットアップ参照）。

### `ModuleNotFoundError: No module named 'onnxruntime'`
```bash
pip install onnxruntime
```

### `TypeError: TSR.extract_mesh() missing 1 required positional argument: 'has_vertex_color'`
`triposr_inferencer.py` に `has_vertex_color=False` が設定されているか確認。現バージョンでは修正済み。

### `RuntimeError: The size of tensor a (4) must match...`
画像がRGBAのままTripoSRに渡されている。`triposr_inferencer.py` でRGB変換されているか確認。現バージョンでは修正済み。

### GLBをUnityにインポートしても表示されない
Unity 標準はGLBに未対応。`--format obj` を使うか、`glTFast` パッケージをインストールする。

### DWGファイルを渡すとエラーになる
ezdxf は DWG を直接読めない。[ODA File Converter](https://www.opendesign.com/guestfiles/oda_file_converter)（無料）で DXF に変換してから渡す。

### 建築平面図なのに Route A に入る
レイヤー名が「壁」以外のキーワードになっている可能性がある。ログの `[wall]` 行を確認し、壁レイヤーが空の場合は `floor_plan_extruder.py` の `_WALL_MOJ` にキーワードを追加する。

---

## 既知の制限

| 項目 | 現状 | 備考 |
|---|---|---|
| 屋根形状 | 陸屋根 / 片流れ / 切妻の3種類 | 寄棟・入母屋は未対応 |
| テクスチャ | 手続き的生成（写真品質ではない） | TEXTure3D統合で改善可能 |
| 推論品質 | 建物の外観を正確に再現するのは困難 | 押し出しメッシュ単体が最も正確 |
| 入力形式 | DXF のみ | STEP/IGES（3D CAD）は未対応 |
| ジョブキュー | FastAPI BackgroundTasks | Celery + Redis で並列化可能 |
| DWG | 直接読み込み不可 | ODA File Converter で変換 |
