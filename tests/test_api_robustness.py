"""
tests/test_api_robustness.py
=============================
Phase 6 (APIロバスト化) の検証テスト。

検証対象:
  1. 壊れたDXFをアップロードした際、ジョブが status="failed",
     error_category="input_error" になること（GPU推論やモデルダウンロードには
     到達しない。parse_dxf の段階でエラーになるケースのみを使う）。
  2. JobStatus のレスポンスに traceback 全文が含まれないこと。
  3. _cleanup_jobs() が TTL 超過・MAX_JOBS 超過の完了済みジョブを正しく削除し、
     queued/running のジョブは削除しないこと（created_at を手動で過去に
     書き換えたジョブを使い、実際に1時間待つことなく検証する）。

GPU推論・モデルダウンロードは一切行わない。
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

import api as api_module
from api import app, _JOBS, _cleanup_jobs, _remove_job, WORK_DIR


@pytest.fixture(autouse=True)
def _clear_jobs():
    """各テスト前後で _JOBS をクリアし、他テストへの汚染を防ぐ。"""
    _JOBS.clear()
    yield
    _JOBS.clear()


class TestBrokenDxfErrorClassification:
    def test_broken_dxf_marks_job_failed_with_input_error(self):
        """
        SECTIONタグはあるが構造が壊れているDXF風テキストをアップロードすると、
        ezdxf.readfile() が DXFStructureError (DXFError のサブクラス) を送出し、
        parse_dxf の段階で失敗する。GPU/推論には到達しない。

        NOTE: マジックバイト（先頭の "0\\nSECTION"）すら無い完全なプレーン
        テキストの場合、ezdxf.readfile() は DXFError ではなく素の
        OSError("... is not a DXF file.") を送出する（is_dxf_file() の
        事前チェックによる分岐）。そのケースは現在の分類ロジックでは
        system_error 扱いになる。ここでは仕様書が明示する DXFError系の
        パスを再現するため、構造だけ壊れた DXF を使う。

        TestClient は BackgroundTasks をリクエスト処理と同じスレッドで
        同期的に実行する。_run_pipeline_with_timeout は ThreadPoolExecutor に
        submit して future.result() で待つ実装だが、これは呼び出し元スレッド
        (=TestClient がリクエストを処理しているスレッド) をブロックして
        完了を待つだけなので、レスポンスが返った時点で処理は完了している。
        """
        client = TestClient(app)
        broken_content = (
            b"0\r\nSECTION\r\n2\r\nHEADER\r\n9\r\n$ACADVER\r\n1\r\nAC1015\r\n"
            b"0\r\nENDSEC\r\n0\r\nSECTION\r\n2\r\nENTITIES\r\n0\r\nLINE\r\n"
            b"5\r\nnot a valid group code line at all\r\ngarbage garbage\r\n"
        )

        resp = client.post(
            "/convert",
            files={"file": ("broken.dxf", broken_content, "application/octet-stream")},
        )
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]

        # TestClient 経由では BackgroundTasks 完了後にレスポンスが返るため、
        # この時点で GET すれば結果が確定しているはず。
        get_resp = client.get(f"/jobs/{job_id}")
        assert get_resp.status_code == 200
        body = get_resp.json()

        assert body["status"] == "failed"
        assert body["error_category"] == "input_error"
        assert body["message"]  # 何らかのメッセージが入っている

        # traceback 全文はレスポンスボディに含めない（情報漏洩防止）
        assert "traceback" not in body

        # ジョブ内部（_JOBS）には traceback が保存されている
        assert _JOBS[job_id].get("traceback")

    def test_plain_text_without_dxf_magic_bytes_is_input_error(self):
        """
        マジックバイトすら無い完全なプレーンテキストの場合、ezdxf.readfile() は
        DXFError ではなく OSError("... is not a DXF file.") を送出する。
        これも _classify_error 側でメッセージ内容から input_error として拾う。
        """
        client = TestClient(app)
        broken_content = b"This is not a DXF file at all, just plain text.\n" * 5

        resp = client.post(
            "/convert",
            files={"file": ("plain.dxf", broken_content, "application/octet-stream")},
        )
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]

        get_resp = client.get(f"/jobs/{job_id}")
        body = get_resp.json()
        assert body["status"] == "failed"
        assert body["error_category"] == "input_error"

    def test_missing_file_path_is_input_error(self):
        """FileNotFoundError も input_error に分類されることを直接関数レベルで確認する。"""
        from pathlib import Path

        job_id = "test-missing-file"
        _JOBS[job_id] = {
            "status": "queued", "message": "", "lods": [], "exported": {},
            "error_category": "", "traceback": "", "created_at": time.time(),
        }
        api_module._run_pipeline(job_id, Path("this_file_does_not_exist_xyz.dxf"))

        assert _JOBS[job_id]["status"] == "failed"
        assert _JOBS[job_id]["error_category"] == "input_error"


class TestJobStoreCleanup:
    def test_ttl_removes_only_expired_done_jobs(self):
        now = time.time()
        _JOBS.clear()

        # 期限切れの完了済みジョブ（削除されるべき）
        _JOBS["old_done"] = {
            "status": "done", "message": "", "lods": [], "exported": {},
            "error_category": "", "traceback": "", "created_at": now - 10_000,
        }
        # 新しい完了済みジョブ（TTL内なので残るべき）
        _JOBS["fresh_done"] = {
            "status": "done", "message": "", "lods": [], "exported": {},
            "error_category": "", "traceback": "", "created_at": now,
        }
        # 古いが実行中のジョブ（削除されてはいけない）
        _JOBS["old_running"] = {
            "status": "running", "message": "", "lods": [], "exported": {},
            "error_category": "", "traceback": "", "created_at": now - 10_000,
        }

        old_ttl = api_module.JOB_TTL_SEC
        api_module.JOB_TTL_SEC = 60  # 60秒より古い完了済みジョブは削除対象
        try:
            _cleanup_jobs()
        finally:
            api_module.JOB_TTL_SEC = old_ttl

        assert "old_done" not in _JOBS
        assert "fresh_done" in _JOBS
        assert "old_running" in _JOBS  # running は経過時間に関わらず残る

    def test_max_jobs_evicts_oldest_done_jobs_first(self):
        now = time.time()
        _JOBS.clear()

        # 5件の完了済みジョブ（created_at が古い順に 0,1,2,3,4）
        for i in range(5):
            _JOBS[f"done_{i}"] = {
                "status": "done", "message": "", "lods": [], "exported": {},
                "error_category": "", "traceback": "", "created_at": now + i,
            }
        # running は上限に関わらず残る
        _JOBS["running_job"] = {
            "status": "running", "message": "", "lods": [], "exported": {},
            "error_category": "", "traceback": "", "created_at": now,
        }

        old_max = api_module.MAX_JOBS
        old_ttl = api_module.JOB_TTL_SEC
        api_module.MAX_JOBS = 3
        api_module.JOB_TTL_SEC = 10_000_000  # TTL には引っかからないようにする
        try:
            _cleanup_jobs()
        finally:
            api_module.MAX_JOBS = old_max
            api_module.JOB_TTL_SEC = old_ttl

        # running_job は必ず残る。それ以外は MAX_JOBS 未満になるまで
        # created_at が古い順（done_0, done_1, ...）に削除される。
        assert "running_job" in _JOBS
        assert len(_JOBS) < 6  # 少なくとも何件かは削除された
        assert "done_0" not in _JOBS  # 最も古い完了済みジョブから削除される

    def test_cleanup_removes_job_workdir(self, tmp_path, monkeypatch):
        """クリーンアップ対象ジョブの一時ファイル（WORK_DIR/job_id）も削除されること。"""
        monkeypatch.setattr(api_module, "WORK_DIR", tmp_path)

        job_id = "job_with_files"
        job_dir = tmp_path / job_id
        job_dir.mkdir()
        (job_dir / "dummy.txt").write_text("hello")
        assert job_dir.exists()

        _JOBS.clear()
        _JOBS[job_id] = {
            "status": "done", "message": "", "lods": [], "exported": {},
            "error_category": "", "traceback": "", "created_at": time.time(),
        }

        _remove_job(job_id)

        assert job_id not in _JOBS
        assert not job_dir.exists()

    def test_remove_job_does_not_raise_when_dir_missing(self):
        """紐づく一時ファイルが既に存在しない場合でも例外を送出しない。"""
        _JOBS.clear()
        job_id = "job_without_dir"
        _JOBS[job_id] = {
            "status": "done", "message": "", "lods": [], "exported": {},
            "error_category": "", "traceback": "", "created_at": time.time(),
        }
        _remove_job(job_id)  # 例外が出ないことを確認
        assert job_id not in _JOBS
