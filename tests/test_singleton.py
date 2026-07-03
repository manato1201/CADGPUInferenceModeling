"""
tests/test_singleton.py
========================
Phase 2 (モデルロードのシングルトン化) の単体テスト。

TripoSRInferencer / Zero123PlusPlusInferencer は重い依存（torch, diffusers,
TripoSR 本体）を要求するが、実際のモデルロード（`.load()`）や推論
（`.generate_mesh()` / `.generate_views()`）は一切呼び出さない。
検証するのはコンストラクタ・config 比較ロジックのみ:

  - 同一 config（省略時のデフォルト同士）で `get_inferencer()` を2回呼ぶと
    同一インスタンス（`is`）が返ること
  - config が異なる場合は新しい（別の）インスタンスが返ること
    （Phase 2 のアンチパターン: 古いインスタンスを黙って返さない）

実際の HuggingFace ダウンロードや GPU 推論は行わない。
"""

from __future__ import annotations

import core.inferencer as inferencer_module
import core.triposr_inferencer as triposr_module
from core.inferencer import InferenceConfig, get_inferencer as get_zero123_inferencer
from core.triposr_inferencer import TripoSRConfig, get_inferencer as get_triposr_inferencer


def _reset_singletons():
    """各テスト前後でモジュールレベルのシングルトンをリセットする。"""
    triposr_module._INSTANCE = None
    inferencer_module._INSTANCE = None


class TestTripoSRSingleton:
    def setup_method(self):
        _reset_singletons()

    def teardown_method(self):
        _reset_singletons()

    def test_same_default_config_returns_same_instance(self):
        infer1 = get_triposr_inferencer()
        infer2 = get_triposr_inferencer()
        assert infer1 is infer2

    def test_same_explicit_config_returns_same_instance(self):
        cfg = TripoSRConfig(mc_resolution=128)
        infer1 = get_triposr_inferencer(cfg)
        infer2 = get_triposr_inferencer(TripoSRConfig(mc_resolution=128))
        assert infer1 is infer2

    def test_different_config_returns_new_instance(self):
        infer1 = get_triposr_inferencer(TripoSRConfig(mc_resolution=256))
        infer2 = get_triposr_inferencer(TripoSRConfig(mc_resolution=128))
        assert infer1 is not infer2
        # 新しいインスタンスが返る（古いインスタンスを黙って返さない）
        assert infer2.config.mc_resolution == 128

    def test_no_model_or_pipeline_access(self):
        """コンストラクタのみが走り、.model には None のまま（.load() 未実行）。"""
        infer = get_triposr_inferencer()
        assert infer.model is None


class TestZero123PlusPlusSingleton:
    def setup_method(self):
        _reset_singletons()

    def teardown_method(self):
        _reset_singletons()

    def test_same_default_config_returns_same_instance(self):
        infer1 = get_zero123_inferencer()
        infer2 = get_zero123_inferencer()
        assert infer1 is infer2

    def test_same_explicit_config_returns_same_instance(self):
        cfg = InferenceConfig(num_inference_steps=10)
        infer1 = get_zero123_inferencer(cfg)
        infer2 = get_zero123_inferencer(InferenceConfig(num_inference_steps=10))
        assert infer1 is infer2

    def test_different_config_returns_new_instance(self):
        infer1 = get_zero123_inferencer(InferenceConfig(num_inference_steps=36))
        infer2 = get_zero123_inferencer(InferenceConfig(num_inference_steps=10))
        assert infer1 is not infer2
        assert infer2.config.num_inference_steps == 10

    def test_no_pipeline_access(self):
        """コンストラクタのみが走り、.pipeline には None のまま（.load() 未実行）。"""
        infer = get_zero123_inferencer()
        assert infer.pipeline is None
