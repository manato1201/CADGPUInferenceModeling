# cad2asset

**CAD図面（DXF）→ GPU推論 → ゲーム用3Dアセット（glTF）変換パイプライン**

```
DXF/DWG → 三面図レンダリング → Zero123++ 推論 → メッシュ最適化・LOD生成 → Unity/UE
```

動作確認済み環境: Windows 11 / NVIDIA GeForce RTX 3070 (8.6GB VRAM) / Python 3.11

---

## ディレクトリ構成

```
cad2asset/
├── core/
│   ├── parser.py          # DXFパース・三面図生成（純粋関数）
│   ├── inferencer.py      # Zero123++ GPU推論（クラス）
│   └── postprocessor.py   # メッシュ修復・LOD・glTFエクスポート（純粋関数）
├── cli.py                 # フェーズ1: CLIエントリポイント
├── api.py                 # フェーズ2: FastAPI エントリポイント（スケルトン済み）
├── requirements.txt
├── .gitignore
└── README.md
```

`cli.py` と `api.py` はどちらも `core/` を呼ぶだけの設計。
フェーズ2への移行は `api.py` を起動するだけで完了する。

---

## セットアップ

### 1. Python 仮想環境の作成

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# WSL2 / Linux
source .venv/bin/activate
```

### 2. PyTorch CUDA版を先にインストール

通常の `pip install` とは別コマンドが必要。RTX 30/40系（CUDA 12.x）の場合:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

インストール後、GPU認識を確認する:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# True
# NVIDIA GeForce RTX 3070   ← こう出ればOK
```

`False` が出た場合は `nvidia-smi` で CUDA バージョンを確認し、対応する PyTorch を選ぶ。

### 3. 残りの依存関係

```bash
pip install -r requirements.txt
```

> **Note**: `scipy` が不足している場合は別途 `pip install scipy` を実行すること。
> trimesh の法線修復処理が内部で scipy を使用している。

---

## フェーズ1: CLIで動かす

### Step 1 — メタデータ確認（推論なし・数秒）

```bash
python cli.py info your_drawing.dxf
```

出力例:
```
┌────────────────┬──────────────────────────────────────────┐
│ 項目           │ 値                                       │
├────────────────┼──────────────────────────────────────────┤
│ ファイル       │ Sample_Drawing.DXF                       │
│ 単位           │ inch                                     │
│ エンティティ数 │ 242                                      │
│ レイヤー       │ 0, 作業レイヤ, 図形, 2, 3, 4, 5, 6, 7, 8 │
│ 寸法 (X/Y/Z)   │ 8387.2 / 7340.6 / 0.0 inch               │
└────────────────┴──────────────────────────────────────────┘
```

### Step 2 — 三面図だけ確認（推論スキップ）

```bash
python cli.py run your_drawing.dxf --output-dir ./out --no-infer
```

`./out/views/` に `front.png` / `side.png` / `top.png` が生成される。
**2D図面（Z=0）の場合は `top.png` に図形が集中するのが正常。**

### Step 3 — フルパイプライン実行

```bash
python cli.py run your_drawing.dxf --output-dir ./out
```

初回は Zero123++ モデル（約5.6GB）が HuggingFace から自動ダウンロードされる（約15〜30分）。
2回目以降はキャッシュが使われるため、**推論 + 後処理のみ（約30秒）** で完了する。

出力されるファイル:

```
out/
├── views/                          # 三面図（入力）
│   ├── front.png
│   ├── side.png
│   └── top.png
├── generated_views/                # Zero123++ が生成した6視点（品質確認用）
│   ├── view_00.png 〜 view_05.png
├── raw_mesh.glb                    # 後処理前の粗メッシュ
└── assets/
    ├── {name}_LOD0.glb             # 最高解像度（〜10,000ポリゴン）
    ├── {name}_LOD1.glb             # 中解像度（〜4,000ポリゴン）
    └── {name}_LOD2.glb             # 低解像度（〜1,000ポリゴン）
```

### CLIオプション一覧

| オプション | デフォルト | 説明 |
|---|---|---|
| `--output-dir PATH` | `./output` | 出力先ディレクトリ |
| `--image-size INT` | `512` | 三面図解像度 (px) |
| `--infer-steps INT` | `36` | Zero123++ ステップ数（速度重視なら `20`） |
| `--seed INT` | `42` | 乱数シード（再現性のため固定推奨） |
| `--lod0 INT` | `10000` | LOD0 最大ポリゴン数 |
| `--lod1 INT` | `4000` | LOD1 最大ポリゴン数 |
| `--lod2 INT` | `1000` | LOD2 最大ポリゴン数 |
| `--no-infer` | `False` | 三面図生成のみ（推論スキップ） |

---

## Unity / Unreal Engine へのインポート

`out/assets/` に生成された `.glb` ファイルをそのままドラッグ&ドロップでインポートできる。

**スケールについて**: 出力メッシュは CAD 図面の実寸を `m` 単位に変換済み。
`inch` / `mm` / `cm` / `m` / `foot` の単位は DXF ヘッダ（`$INSUNITS`）から自動判定される。

---

## VRAM 目安（RTX 30/40系）

| 処理 | 最低VRAM | 推奨VRAM | 備考 |
|------|---------|---------|------|
| Zero123++ 推論 | 6 GB | 8 GB | `--infer-steps 20` でさらに軽量化可 |
| TSDF Fusion | 1 GB | 2 GB | CPU でも動作可 |

RTX 3070 (8.6GB) での実測: **推論 約11秒 / 36ステップ**

---

## フェーズ2: APIサーバーとして動かす

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Swagger UI: http://localhost:8000/docs

### エンドポイント

| Method | Path | 説明 |
|--------|------|------|
| `POST` | `/convert` | DXFをアップロードしてジョブ登録（202 即返し） |
| `GET` | `/jobs/{job_id}` | ジョブ状態確認（queued / running / done / failed） |
| `GET` | `/jobs/{job_id}/download/{lod}` | 完成アセット（glTF）をダウンロード |
| `GET` | `/health` | ヘルスチェック |

### Unity からのリクエスト例

```csharp
// DXF をアップロードしてジョブを登録
var form = new WWWForm();
form.AddBinaryData("file", dxfBytes, "model.dxf", "application/octet-stream");
var req = UnityWebRequest.Post("http://localhost:8000/convert", form);
yield return req.SendWebRequest();
var jobId = JSON.Parse(req.downloadHandler.text)["job_id"];

// ジョブ完了を確認してダウンロード
var dlReq = UnityWebRequest.Get($"http://localhost:8000/jobs/{jobId}/download/LOD0");
yield return dlReq.SendWebRequest();
// dlReq.downloadHandler.data → glTF バイナリ
```

---

## トラブルシューティング

### Zero123++ ロード時に `trust_remote_code` エラー

`inferencer.py` の `DiffusionPipeline.from_pretrained()` に `trust_remote_code=True` が必要。
現バージョンでは修正済み。

### `AttributeError: 'Trimesh' object has no attribute 'remove_degenerate_faces'`

trimesh 4.x 系でAPIが変更された。現バージョンでは両対応済み。

### `ModuleNotFoundError: No module named 'scipy'`

```bash
pip install scipy
```

trimesh の法線修復が内部で使用している。`requirements.txt` に追加済み。

### glbをUnityにインポートしても何も表示されない

出力スケールが大きすぎる場合がある。ログの `Scale applied` の値を確認すること。
`cad_unit` の自動判定が正しく動いていれば Unity の標準カメラ範囲内に収まる（〜数m）。

### DWGファイルを渡すとエラーになる

ezdxf は DWG を直接読めない。[ODA File Converter](https://www.opendesign.com/guestfiles/oda_file_converter)（無料）で DXF に変換してから渡す。

---

## 既知の制限と今後の拡張

| 項目 | 現状 | 拡張案 |
|------|------|--------|
| 深度推定 | TSDF の深度は仮値（一定値） | `Depth-Anything-V2` 統合で精度向上 |
| テクスチャ | PBRテクスチャ未生成（頂点カラーのみ） | `TEXTure3D` 統合 |
| ジョブキュー | FastAPI BackgroundTasks（シングルプロセス） | Celery + Redis で並列化 |
| 入力形式 | DXF のみ（2D図面想定） | STEP/IGES（3D CAD）対応 |
| メッシュ品質 | TSDF Fusion ベース | InstantMesh / One-2-3-45 への差し替え |
