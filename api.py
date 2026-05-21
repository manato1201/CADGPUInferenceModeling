"""
api.py
======
フェーズ2 エントリポイント。
core/ を再利用して REST API として公開する。
フェーズ1 の cli.py と同じ core/ を呼ぶだけで動く設計。

起動:
    uvicorn api:app --reload --host 0.0.0.0 --port 8000

エンドポイント:
    POST /convert          - DXF をアップロードして変換ジョブを登録
    GET  /jobs/{job_id}    - ジョブの状態を確認
    GET  /jobs/{job_id}/download/{lod} - 変換済みアセットをダウンロード
"""

from __future__ import annotations

import uuid
import logging
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel

logger = logging.getLogger("cad2asset.api")

app = FastAPI(
    title="cad2asset API",
    description="CAD図面（DXF）→ ゲーム用3Dアセット（glTF）変換 REST API",
    version="0.1.0",
)

# ── ジョブストア（フェーズ2では Redis + Celery に差し替える） ──
_JOBS: dict[str, dict] = {}
WORK_DIR = Path("./api_workspace")
WORK_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# スキーマ定義
# ─────────────────────────────────────────────

class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "done", "failed"]
    message: str = ""
    lods: list[str] = []  # 完了時に利用可能な LOD 名リスト


# ─────────────────────────────────────────────
# 変換ワーカー（フェーズ2では Celery タスクに差し替え）
# ─────────────────────────────────────────────

def _run_pipeline(job_id: str, dxf_path: Path) -> None:
    """
    バックグラウンドで cad2asset パイプラインを実行する。
    フェーズ1 の cli.py と同じ core/ 関数群を呼ぶだけ。
    """
    _JOBS[job_id]["status"] = "running"
    output_dir = WORK_DIR / job_id

    try:
        from core.parser import parse_dxf, save_views
        from core.inferencer import Zero123PlusPlusInferencer, InferenceConfig
        from core.postprocessor import postprocess, PostprocessConfig

        # Step 1: パース
        result = parse_dxf(dxf_path)
        save_views(result, output_dir / "views")

        # Step 2: 推論
        infer = Zero123PlusPlusInferencer(InferenceConfig())
        views = infer.generate_views(result.views["front"])
        raw_mesh = output_dir / "raw_mesh.glb"
        infer.generate_mesh(views, output_path=raw_mesh)

        # Step 3: 後処理
        exported = postprocess(
            mesh_path=raw_mesh,
            cad_dimensions_mm=result.meta.dimensions,
            output_dir=output_dir / "assets",
            base_name=dxf_path.stem,
        )

        _JOBS[job_id]["status"] = "done"
        _JOBS[job_id]["lods"] = list(exported.keys())
        _JOBS[job_id]["exported"] = {k: str(v) for k, v in exported.items()}

    except Exception as e:
        logger.exception(f"Job {job_id} failed")
        _JOBS[job_id]["status"] = "failed"
        _JOBS[job_id]["message"] = str(e)


# ─────────────────────────────────────────────
# エンドポイント
# ─────────────────────────────────────────────

@app.post("/convert", response_model=JobStatus, status_code=202)
async def convert(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="DXF ファイル"),
):
    """
    DXF をアップロードして変換ジョブを登録する。
    202 Accepted を即返し、変換はバックグラウンドで実行される。
    """
    if not file.filename.lower().endswith((".dxf", ".dwg")):
        raise HTTPException(400, "DXF または DWG ファイルをアップロードしてください。")

    job_id = str(uuid.uuid4())
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True)

    dxf_path = job_dir / file.filename
    with open(dxf_path, "wb") as f:
        f.write(await file.read())

    _JOBS[job_id] = {"status": "queued", "message": "", "lods": [], "exported": {}}
    background_tasks.add_task(_run_pipeline, job_id, dxf_path)

    logger.info(f"Job queued: {job_id}")
    return JobStatus(job_id=job_id, status="queued")


@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str):
    """ジョブの状態を返す。"""
    if job_id not in _JOBS:
        raise HTTPException(404, f"Job '{job_id}' not found.")
    j = _JOBS[job_id]
    return JobStatus(job_id=job_id, **j)


@app.get("/jobs/{job_id}/download/{lod_name}")
async def download_asset(job_id: str, lod_name: str):
    """
    変換済みアセット（glTF）をダウンロードする。
    lod_name: "LOD0" / "LOD1" / "LOD2"
    """
    if job_id not in _JOBS:
        raise HTTPException(404, f"Job '{job_id}' not found.")
    job = _JOBS[job_id]
    if job["status"] != "done":
        raise HTTPException(409, f"Job is not done yet (status: {job['status']}).")
    if lod_name not in job["exported"]:
        raise HTTPException(404, f"LOD '{lod_name}' not found. Available: {list(job['exported'].keys())}")

    file_path = Path(job["exported"][lod_name])
    return FileResponse(str(file_path), media_type="model/gltf-binary", filename=file_path.name)


@app.get("/health")
async def health():
    """ヘルスチェック。"""
    return {"status": "ok"}
