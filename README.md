# cad2asset

**CAD図面（DXF）→ GPU推論 → ゲーム用3Dアセット（glTF）変換パイプライン**

```
DXF/DWG → 三面図レンダリング → Zero123++ 推論 → メッシュ最適化 → Unity/UE
```

---

## ディレクトリ構成

```
cad2asset/
├── core/
│   ├── parser.py          # DXFパース・三面図生成（純粋関数）
│   ├── inferencer.py      # Zero123++ GPU推論（クラス）
│   └── postprocessor.py   # メッシュ修復・LOD・glTFエクスポート（純粋関数）
├── cli.py                 # フェーズ1: CLIエントリポイント
├── api.py                 # フェーズ2: FastAPI エントリポイント
└── requirements.txt
```

---

## セットアップ

### 1. Python 環境

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2. PyTorch（CUDA版）を先にインストール

RTX 30/40系（CUDA 12.x）の場合:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. 残りの依存関係

```bash
pip install -r requirements.txt
```

---

## フェーズ1: CLIで動かす

### メタデータ確認（推論なし・高速）

```bash
python cli.py info your_drawing.dxf
```

### 三面図だけ確認（推論スキップ）

```bash
python cli.py run your_drawing.dxf --output-dir ./out --no-infer
```
→ `./out/views/` に `front.png`, `side.png`, `top.png` が生成される

### フルパイプライン実行

```bash
python cli.py run your_drawing.dxf --output-dir ./out
```
→ `./out/assets/` に `{name}_LOD0.glb`, `{name}_LOD1.glb`, `{name}_LOD2.glb` が生成される

### オプション一覧

```
--output-dir PATH     出力先ディレクトリ（デフォルト: ./output）
--image-size INT      三面図解像度 px（デフォルト: 512）
--infer-steps INT     Zero123++ ステップ数（デフォルト: 36, 速度重視: 20）
--seed INT            乱数シード（デフォルト: 42）
--lod0 INT            LOD0 最大ポリゴン数（デフォルト: 10000）
--lod1 INT            LOD1 最大ポリゴン数（デフォルト: 4000）
--lod2 INT            LOD2 最大ポリゴン数（デフォルト: 1000）
--no-infer            三面図生成のみ（推論スキップ）
```

---

## フェーズ2: APIサーバーとして動かす

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### エンドポイント

| Method | Path | 説明 |
|--------|------|------|
| POST | `/convert` | DXFをアップロードしてジョブ登録 |
| GET | `/jobs/{job_id}` | ジョブ状態確認 |
| GET | `/jobs/{job_id}/download/{lod}` | 完成アセットをダウンロード |
| GET | `/health` | ヘルスチェック |

### Unity からのリクエスト例

```csharp
// UnityWebRequest で DXF を送って glTF を受け取る
var form = new WWWForm();
form.AddBinaryData("file", dxfBytes, "model.dxf", "application/octet-stream");
var req = UnityWebRequest.Post("http://localhost:8000/convert", form);
yield return req.SendWebRequest();
var jobId = JSON.Parse(req.downloadHandler.text)["job_id"];
```

---

## VRAM 目安（RTX 30/40系）

| モデル | 最低VRAM | 推奨VRAM | 備考 |
|--------|---------|---------|------|
| Zero123++ | 6 GB | 8 GB | `--infer-steps 20` でさらに軽量化 |
| TSDF Fusion | 1 GB | 2 GB | CPU でも動作可 |

---

## 既知の制限と今後の拡張

- **深度推定**: 現状 TSDF の深度は仮値。`Depth-Anything-V2` を組み込むと精度が上がる
- **テクスチャ**: PBRテクスチャ生成は未実装（TEXTure3D 統合が次のステップ）
- **DWG**: ezdxf は DWG を直接読めない。`ODA File Converter`（無料）で DXF 変換が必要
- **Celery化**: フェーズ2で複数ジョブを並列処理するには `api.py` の BackgroundTasks を Celery に差し替える
