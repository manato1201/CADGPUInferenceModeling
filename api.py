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

import os
import shutil
import time
import traceback
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import ezdxf
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel

logger = logging.getLogger("cad2asset.api")

# ─────────────────────────────────────────────
# 設定（環境変数、Phase 6: APIロバスト化）
# ─────────────────────────────────────────────

# ジョブごとのタイムアウト（秒）。GPUハング等でパイプラインが応答しなくなった場合の
# 「APIレスポンス上の」検出用（下記 _run_pipeline のコメント参照。真のプロセスkillではない）。
JOB_TIMEOUT_SEC = float(os.environ.get("JOB_TIMEOUT_SEC", "600"))

# _JOBS の最大保持件数、および完了済みジョブの生存時間（秒）。
MAX_JOBS = int(os.environ.get("MAX_JOBS", "500"))
JOB_TTL_SEC = float(os.environ.get("JOB_TTL_SEC", "3600"))

# _run_pipeline の実処理をここに投げて timeout 付きで待つ。
# BackgroundTasks はもともとワーカースレッドプールで実行されるため、
# ここでさらに ThreadPoolExecutor に投げても二重にスレッドを使うだけで
# 非同期実行モデルとは矛盾しない（GPU推論はいずれにせよ別スレッドで動く）。
_PIPELINE_EXECUTOR = ThreadPoolExecutor(max_workers=2)


# ─────────────────────────────────────────────
# ライフサイクル（起動時ウォームアップ、オプション）
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    アプリ起動時のフック。

    環境変数 PRELOAD_MODELS=1 の場合のみ、起動時に Zero123++ を
    事前ロードしてシングルトンをウォームアップする（初回リクエストの
    レイテンシを削減）。未設定・それ以外の値の場合は何もせず、
    従来通り初回リクエスト時の遅延ロードのままにする。
    """
    if os.environ.get("PRELOAD_MODELS") == "1":
        logger.info("PRELOAD_MODELS=1: Zero123++ を起動時にウォームアップロード中...")
        from core.inferencer import get_inferencer

        get_inferencer().load()
        logger.info("Zero123++ ウォームアップロード完了。")
    yield


app = FastAPI(
    title="cad2asset API",
    description="CAD図面（DXF）→ ゲーム用3Dアセット（glTF）変換 REST API",
    version="0.1.0",
    lifespan=lifespan,
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
    # "" (未失敗) / "input_error" / "inference_error" / "timeout" / "system_error"
    # NOTE: traceback 全文はここに含めない（情報漏洩防止）。詳細は _JOBS[job_id]["traceback"] にのみ保持しログ出力する。
    error_category: str = ""


# ─────────────────────────────────────────────
# 変換ワーカー（フェーズ2では Celery タスクに差し替え）
# ─────────────────────────────────────────────

def _classify_error(e: BaseException) -> str:
    """
    例外を input_error / inference_error / system_error に分類する。

    - input_error: 壊れたDXF・ファイル未存在・パースのバリデーション失敗
    - inference_error: 推論（core.inferencer / core.triposr_inferencer）の実行時エラー
      （CUDA OOM を含む）
    - system_error: 上記以外の予期しない例外
    """
    import torch  # 遅延import（重いのでエラー分類時のみ）

    # CUDA OOM は torch のバージョンにより存在有無が異なるため hasattr で確認する。
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_cls is not None and isinstance(e, oom_cls):
        return "inference_error"

    if isinstance(e, (ezdxf.DXFError, FileNotFoundError, ValueError)):
        return "input_error"

    # ezdxf.readfile() はマジックバイト（先頭の "0/SECTION"）すら無いファイルに対しては
    # DXFError ではなく素の OSError("... is not a DXF file.") を送出する
    # (ezdxf.filemanagement.readfile 参照)。これも実質的には入力エラーなので拾う。
    # ただし FileNotFoundError は OSError のサブクラスだが上で既に判定済み。
    if isinstance(e, OSError) and "is not a DXF file" in str(e):
        return "input_error"

    if isinstance(e, RuntimeError):
        # torch/CUDA まわりの実行時エラーは大半が RuntimeError として送出される
        return "inference_error"

    return "system_error"


def _run_pipeline(job_id: str, dxf_path: Path) -> None:
    """
    cad2asset パイプラインの実処理。フェーズ1 の cli.py と同じ core/ 関数群を呼ぶだけ。
    正常系ロジック（parse_dxf→推論→postprocess）は変更しない。
    """
    _JOBS[job_id]["status"] = "running"
    output_dir = WORK_DIR / job_id

    try:
        from core.parser import parse_dxf, save_views
        from core.inferencer import get_inferencer
        from core.postprocessor import postprocess, PostprocessConfig

        # Step 1: パース
        result = parse_dxf(dxf_path)
        save_views(result, output_dir / "views")

        # Step 2: 推論
        infer = get_inferencer()
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
        _JOBS[job_id]["error_category"] = _classify_error(e)
        _JOBS[job_id]["traceback"] = traceback.format_exc()


def _run_pipeline_with_timeout(job_id: str, dxf_path: Path) -> None:
    """
    _run_pipeline を ThreadPoolExecutor 経由で実行し、JOB_TIMEOUT_SEC でタイムアウト監視する
    ラッパー。background_tasks.add_task にはこちらを渡す。

    重要な制約:
    Python のスレッドは強制終了できないため、ここでタイムアウトを検出しても
    _PIPELINE_EXECUTOR 上のバックグラウンドスレッドでは処理が継続し続ける可能性がある。
    その結果、タイムアウト扱いにした後でも GPU メモリが解放されない、あるいは
    処理完了後に _JOBS[job_id] へ遅れて書き込みが発生する、といった事態が起こり得る。
    これはあくまで「APIレスポンス上のタイムアウト検出」であり、真のプロセスkillによる
    GPUハング対策（プロセス分離・強制終了）は本フェーズのスコープ外。
    FastAPI の BackgroundTasks はもともとワーカースレッドプールで実行されるため、
    ここで別途 ThreadPoolExecutor に投げても実行モデル上の矛盾は生じない
    （どのみち別スレッドで動く処理を、タイムアウト監視できる形に置き換えているだけ）。
    """
    future = _PIPELINE_EXECUTOR.submit(_run_pipeline, job_id, dxf_path)
    try:
        future.result(timeout=JOB_TIMEOUT_SEC)
    except FutureTimeoutError:
        logger.error(f"Job {job_id} timed out after {JOB_TIMEOUT_SEC}s")
        _JOBS[job_id]["status"] = "failed"
        _JOBS[job_id]["error_category"] = "timeout"
        _JOBS[job_id]["message"] = f"処理が{JOB_TIMEOUT_SEC}秒でタイムアウトしました"
        # NOTE: future 自体はキャンセルされず、バックグラウンドスレッドで動き続ける
        # （上記docstring参照）。完了後に _JOBS[job_id] を上書きする可能性がある。


# ─────────────────────────────────────────────
# ジョブストアのクリーンアップ（サイズ上限・TTL、Phase 6）
# ─────────────────────────────────────────────

def _cleanup_jobs() -> None:
    """
    完了済み（done/failed）ジョブのうち TTL を超えたものを削除し、
    それでも MAX_JOBS を超える場合は古い完了済みジョブから追加削除する。
    queued/running のジョブは経過時間・件数に関わらず削除しない。
    ジョブに紐づく一時ファイル（WORK_DIR / job_id）も併せて削除する。
    """
    now = time.time()

    def _is_done(j: dict) -> bool:
        return j.get("status") in ("done", "failed")

    # 1) TTL超過の完了済みジョブを削除
    expired = [
        jid for jid, j in _JOBS.items()
        if _is_done(j) and (now - j.get("created_at", now)) > JOB_TTL_SEC
    ]
    for jid in expired:
        _remove_job(jid)

    # 2) それでも MAX_JOBS 以上なら、完了済みジョブを created_at が古い順に削除
    if len(_JOBS) >= MAX_JOBS:
        done_jobs = sorted(
            (jid for jid, j in _JOBS.items() if _is_done(j)),
            key=lambda jid: _JOBS[jid].get("created_at", 0.0),
        )
        for jid in done_jobs:
            if len(_JOBS) < MAX_JOBS:
                break
            _remove_job(jid)


def _remove_job(job_id: str) -> None:
    """_JOBS からジョブを削除し、紐づく一時ファイルも削除する。"""
    _JOBS.pop(job_id, None)
    job_dir = WORK_DIR / job_id
    try:
        if job_dir.exists():
            shutil.rmtree(job_dir)
    except Exception:
        logger.warning(f"Failed to remove work dir for job {job_id}", exc_info=True)


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

    # 新規ジョブ登録前に、TTL超過・上限超過分の完了済みジョブを掃除する
    _cleanup_jobs()

    job_id = str(uuid.uuid4())
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True)

    dxf_path = job_dir / file.filename
    with open(dxf_path, "wb") as f:
        f.write(await file.read())

    _JOBS[job_id] = {
        "status": "queued",
        "message": "",
        "lods": [],
        "exported": {},
        "error_category": "",
        "traceback": "",
        "created_at": time.time(),
    }
    background_tasks.add_task(_run_pipeline_with_timeout, job_id, dxf_path)

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
