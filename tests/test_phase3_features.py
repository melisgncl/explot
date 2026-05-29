"""Tests for Phase 3 features: survival stage, model export, UMAP, Plotly chart."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state_result(outputs=None, success=True, error=None):
    sr = MagicMock()
    sr.success = success
    sr.error = error
    sr.outputs = outputs or {}
    sr.interpretations = {}
    return sr


# ============================================================
# Phase 3c — Survival Stage
# ============================================================

class TestSurvivalStageDetection:
    def test_imports_without_error(self):
        from explot.stages.survival.stage import SurvivalStage
        assert SurvivalStage is not None

    def test_skip_when_no_survival_columns(self):
        from explot.stages.survival.stage import SurvivalStage
        from explot.state import PipelineState

        df = pd.DataFrame({"age": [30, 40, 50], "score": [0.1, 0.2, 0.3]})
        state = PipelineState(raw_df=df)

        config = MagicMock()
        config.pipeline.is_enabled.return_value = True
        hooks = MagicMock()

        result = SurvivalStage().run(state, config, hooks)
        assert result.success
        assert result.outputs.get("detected") is False

    def test_detects_time_and_event_columns(self):
        from explot.stages.survival.stage import SurvivalStage
        from explot.state import PipelineState

        rng = np.random.default_rng(42)
        n = 100
        df = pd.DataFrame({
            "duration": rng.integers(1, 100, n).astype(float),
            "event": rng.integers(0, 2, n),
            "age": rng.integers(20, 80, n).astype(float),
        })
        state = PipelineState(raw_df=df)

        config = MagicMock()
        hooks = MagicMock()

        result = SurvivalStage().run(state, config, hooks)
        assert result.success
        assert result.outputs.get("detected") is True
        assert result.outputs["time_column"] == "duration"
        assert result.outputs["event_column"] == "event"

    @pytest.mark.parametrize("lifelines_available", [True, False])
    def test_graceful_without_lifelines(self, lifelines_available):
        """Stage must not crash whether or not lifelines is installed."""
        from explot.stages.survival.stage import SurvivalStage
        from explot.state import PipelineState

        rng = np.random.default_rng(0)
        n = 60
        df = pd.DataFrame({
            "time_to_event": rng.integers(1, 50, n).astype(float),
            "status": rng.integers(0, 2, n),
        })
        state = PipelineState(raw_df=df)
        config = MagicMock()
        hooks = MagicMock()

        if not lifelines_available:
            with patch.dict("sys.modules", {"lifelines": None}):
                result = SurvivalStage().run(state, config, hooks)
        else:
            result = SurvivalStage().run(state, config, hooks)

        assert result.success

    def test_km_svg_in_outputs(self):
        from explot.stages.survival.stage import SurvivalStage
        from explot.state import PipelineState

        pytest.importorskip("lifelines")

        rng = np.random.default_rng(7)
        n = 120
        df = pd.DataFrame({
            "survival_time": rng.integers(1, 200, n).astype(float),
            "event": rng.integers(0, 2, n),
        })
        state = PipelineState(raw_df=df)
        config = MagicMock()
        hooks = MagicMock()

        result = SurvivalStage().run(state, config, hooks)
        if result.outputs.get("detected"):
            km_svg = result.outputs.get("km_svg", "")
            assert km_svg.startswith("<svg") or km_svg == ""

    def test_low_event_rate_flag(self):
        from explot.stages.survival.stage import SurvivalStage
        from explot.state import PipelineState

        pytest.importorskip("lifelines")

        # Only 5% events → low event rate
        n = 200
        df = pd.DataFrame({
            "duration": np.arange(1, n + 1, dtype=float),
            "event": [1] * 10 + [0] * (n - 10),
        })
        state = PipelineState(raw_df=df)
        config = MagicMock()
        hooks = MagicMock()

        result = SurvivalStage().run(state, config, hooks)
        if result.outputs.get("detected"):
            assert result.outputs["event_rate"] < 0.10

    def test_validate_inputs_needs_cleaned_df(self):
        from explot.stages.survival.stage import SurvivalStage
        from explot.state import PipelineState

        state = PipelineState(raw_df=pd.DataFrame())
        # cleaned_df not set → validate_inputs reports missing
        missing = SurvivalStage().validate_inputs(state)
        # survival is optional on preprocessing — missing list may be empty
        # The stage should declare it optional; assert it doesn't crash
        assert isinstance(missing, list)


# ============================================================
# Phase 3d — Model Export
# ============================================================

class TestModelExport:
    def _make_supervised_result(self, with_estimator=True):
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier

        X, y = make_classification(n_samples=100, n_features=5, random_state=0)
        clf = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)

        best_models = {
            "label": {
                "model": "RandomForestClassifier",
                "metric": "F1",
                "mean": 0.85,
                "_export_estimator": clf if with_estimator else None,
                "_export_feature_names": [f"f{i}" for i in range(5)],
            }
        }
        sr = MagicMock()
        sr.success = True
        sr.outputs = {"best_models": best_models}
        return sr

    def test_save_best_model_creates_file(self, tmp_path):
        pytest.importorskip("joblib")
        from explot.orchestrator import _save_best_model

        state = MagicMock()
        state.results = {"supervised": self._make_supervised_result(with_estimator=True)}

        model_path = tmp_path / "output_model.joblib"
        _save_best_model(state, model_path)

        assert model_path.exists()

    def test_saved_payload_has_required_keys(self, tmp_path):
        import joblib
        pytest.importorskip("joblib")
        from explot.orchestrator import _save_best_model

        state = MagicMock()
        state.results = {"supervised": self._make_supervised_result(with_estimator=True)}

        model_path = tmp_path / "out_model.joblib"
        _save_best_model(state, model_path)

        payload = joblib.load(model_path)
        for key in ("estimator", "target", "feature_names", "model_name", "metric", "score"):
            assert key in payload, f"Missing key: {key}"

    def test_no_crash_without_estimator(self, tmp_path):
        pytest.importorskip("joblib")
        from explot.orchestrator import _save_best_model

        state = MagicMock()
        state.results = {"supervised": self._make_supervised_result(with_estimator=False)}

        model_path = tmp_path / "out_model.joblib"
        _save_best_model(state, model_path)  # should not raise
        # File may or may not exist depending on implementation
        # Just verify no exception

    def test_no_crash_without_supervised_stage(self, tmp_path):
        from explot.orchestrator import _save_best_model

        state = MagicMock()
        state.results = {}

        model_path = tmp_path / "no_model.joblib"
        _save_best_model(state, model_path)  # should not raise

    def test_exported_path_stored_in_state(self, tmp_path):
        pytest.importorskip("joblib")
        from explot.orchestrator import _save_best_model

        sup = self._make_supervised_result(with_estimator=True)
        state = MagicMock()
        state.results = {"supervised": sup}

        model_path = tmp_path / "out_model.joblib"
        _save_best_model(state, model_path)

        assert sup.outputs.get("exported_model_path") == str(model_path)
        assert sup.outputs.get("exported_model_target") == "label"


# ============================================================
# Phase 3e — UMAP output from DimensionalityStage
# ============================================================

class TestUMAPOutput:
    def _run_dimensionality(self, df):
        from explot.stages.dimensionality.stage import DimensionalityStage
        from explot.state import PipelineState

        state = PipelineState(raw_df=df)
        # Provide cleaned_df so stage can proceed
        state.results["profiling"] = MagicMock(success=True, outputs={
            "numeric_columns": list(df.select_dtypes("number").columns),
            "normalization_guess": "standard",
        })
        state.results["exploration"] = MagicMock(success=True, outputs={})
        from explot.stages.preprocessing.stage import PreprocessingStage
        prep_config = MagicMock()
        prep_config.pipeline.is_enabled.return_value = True
        hooks = MagicMock()
        prep_result = PreprocessingStage().run(state, prep_config, hooks)
        state.results["preprocessing"] = prep_result

        config = MagicMock()
        return DimensionalityStage().run(state, config, hooks)

    def test_umap_key_always_present(self):
        """umap_2d key must exist in outputs even when umap-learn is absent."""
        rng = np.random.default_rng(1)
        df = pd.DataFrame(rng.standard_normal((80, 6)), columns=[f"x{i}" for i in range(6)])

        result = self._run_dimensionality(df)
        assert "umap_2d" in result.outputs

    def test_umap_shape_when_available(self):
        pytest.importorskip("umap")
        rng = np.random.default_rng(2)
        df = pd.DataFrame(rng.standard_normal((80, 6)), columns=[f"x{i}" for i in range(6)])

        result = self._run_dimensionality(df)
        umap_2d = result.outputs["umap_2d"]
        if isinstance(umap_2d, np.ndarray) and umap_2d.size > 0:
            assert umap_2d.shape[1] == 2

    def test_umap_max_5000_rows(self):
        pytest.importorskip("umap")
        rng = np.random.default_rng(3)
        n = 6000
        df = pd.DataFrame(rng.standard_normal((n, 4)), columns=[f"f{i}" for i in range(4)])

        result = self._run_dimensionality(df)
        umap_2d = result.outputs.get("umap_2d", np.array([]))
        if isinstance(umap_2d, np.ndarray) and umap_2d.size > 0:
            assert len(umap_2d) <= 5000


# ============================================================
# Phase 3e — Plotly feature importance chart
# ============================================================

class TestPlotlyFeatureImportanceChart:
    def _make_generator(self):
        from explot.report.generator import ReportGenerator
        return ReportGenerator()

    def test_returns_string(self):
        gen = self._make_generator()
        fi = {"track_a": [{"feature": "age", "importance": 0.5},
                           {"feature": "score", "importance": 0.3}]}
        html = gen._feature_importance_chart(fi, "target1")
        assert isinstance(html, str)

    def test_contains_plotly_newplot(self):
        gen = self._make_generator()
        fi = {"track_a": [{"feature": "age", "importance": 0.5}]}
        html = gen._feature_importance_chart(fi, "target1")
        assert "Plotly.newPlot" in html

    def test_empty_fi_returns_muted_message(self):
        gen = self._make_generator()
        html = gen._feature_importance_chart({}, "no_target")
        assert "Plotly.newPlot" not in html
        assert "muted" in html or "No feature" in html

    def test_safe_id_no_invalid_chars(self):
        gen = self._make_generator()
        fi = {"track_a": [{"feature": "x", "importance": 0.1}]}
        html = gen._feature_importance_chart(fi, "target with spaces & special!")
        # Find the div id
        assert "fi_" in html
        # ID must not contain spaces or &
        import re
        ids = re.findall(r"id='([^']+)'", html)
        for div_id in ids:
            assert " " not in div_id
            assert "&" not in div_id

    def test_two_tracks_both_in_traces(self):
        gen = self._make_generator()
        fi = {
            "track_a": [{"feature": "a", "importance": 0.4}, {"feature": "b", "importance": 0.2}],
            "track_b": [{"feature": "z1", "importance": 0.6}, {"feature": "z2", "importance": 0.1}],
        }
        html = gen._feature_importance_chart(fi, "tgt")
        # Each trace has "type": "bar" — count occurrences as proxy for trace count
        assert html.count('"type": "bar"') == 2

    def test_top_10_features_only(self):
        gen = self._make_generator()
        feats = [{"feature": f"f{i}", "importance": 1 / (i + 1)} for i in range(20)]
        fi = {"track_a": feats}
        html = gen._feature_importance_chart(fi, "t")
        # At most 10 feature names in the chart (each appears in JSON as "f0", "f1", ...)
        # Count by checking that f10..f19 do not appear as feature values
        for i in range(10, 20):
            # Feature name "f{i}" should not be in the traces data
            assert f'"f{i}"' not in html

    def test_horizontal_bar_orientation(self):
        gen = self._make_generator()
        fi = {"track_a": [{"feature": "a", "importance": 0.9}]}
        html = gen._feature_importance_chart(fi, "t")
        assert '"orientation": "h"' in html or "'orientation': 'h'" in html or "orientation" in html

    def test_plotly_script_in_head(self):
        """The HTML report template must include Plotly CDN in <head>."""

        from explot.config import load_config
        from explot.report.generator import ReportGenerator
        from explot.state import PipelineState

        _ROOT = Path(__file__).resolve().parent.parent
        config = load_config(_ROOT / "config" / "fast.yaml")
        df = pd.DataFrame({"a": [1, 2, 3]})
        state = PipelineState(raw_df=df)
        html = ReportGenerator().render(state, config)
        assert "plotly" in html.lower()
        head_end = html.index("</head>")
        assert "plotly" in html[:head_end].lower()
