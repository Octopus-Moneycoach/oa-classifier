from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.pipelines.train_model import TrainModelPipeline


@pytest.fixture(name="data")
def dummy_data_fixture():
    # Small, balanced sample for train/test split
    return pd.DataFrame(
        {"feature1": [1, 2, 3, 4], "feature2": [5, 6, 7, 8], "target": [0, 1, 0, 1]}
    )


@pytest.fixture(name="pipeline")
def mock_pipeline_fixture(data):
    # Patch where TrainModelPipeline looks up these names (module scope)
    with (
        patch("src.pipelines.train_model.read_config") as mock_config,
        patch("src.pipelines.train_model.read_data") as mock_read,
    ):
        mock_config.return_value = {
            "model_name": "xgboost",
            "model_params": {
                "objective": "binary:logistic",
                "n_estimators": 10,
            },  # make compatible with different models
            # Optional split settings; defaults exist in code
            "test_size": 0.5,
            "random_state": 123,
            "stratify": True,
        }
        mock_read.return_value = data

        pipeline_instance = TrainModelPipeline()

    return pipeline_instance


def test_pipeline_run(pipeline, data):
    """Test the pipeline's run method end-to-end (with heavy bits mocked)."""
    with (
        patch("mlflow.set_tracking_uri"),
        patch("mlflow.start_run"),
        patch("mlflow.log_metric"),
        patch("mlflow.log_params"),
        patch("mlflow.sklearn.log_model"),
        patch("mlflow.log_artifact"),
        patch("src.pipelines.train_model.read_data", return_value=data),
        patch("src.pipelines.train_model.write_data") as mock_write,
        patch.object(pipeline, "_log_shap_analysis"),  # SHAP needs realistic data
    ):
        pipeline.run()
        assert pipeline.model is not None

        # NEW: Assert new write_data logic
        mock_write.assert_called_once()
        _, kwargs = mock_write.call_args
        data = mock_write.call_args[0][0]
        table_name = kwargs["table_name"]
        schema_obj = kwargs["schema_obj"]
        assert "PREDICTION" in data.columns
        assert table_name == "OA_DATASET_OUTPUT"
        assert schema_obj is None


def test_train_method(pipeline, data):
    """Directly test the public train method."""
    X, y = data.drop(columns=["target"]), data["target"]
    pipeline.train(X, y)
    assert pipeline.model is not None


# ---------------------------------------------------------------------------
# SHAP analysis tests
# ---------------------------------------------------------------------------


@pytest.fixture(name="shap_setup")
def shap_setup_fixture():
    """Pipeline with mock model and test data for SHAP tests."""
    with patch("src.pipelines.train_model.read_config") as mock_config:
        mock_config.return_value = {
            "model_name": "xgboost",
            "model_params": {},
            "shap_max_samples": 1000,
        }
        pl = TrainModelPipeline()

    model = MagicMock()
    model.predict_proba.return_value = np.array(
        [[0.8, 0.2], [0.3, 0.7], [0.5, 0.5], [0.1, 0.9], [0.6, 0.4]]
    )
    pl.model = model

    X = pd.DataFrame({"feat_a": [1, 2, 3, 4, 5], "feat_b": [5, 4, 3, 2, 1.0]})
    y = pd.Series([0, 1, 0, 1, 0], index=X.index)
    return pl, X, y


def _shap_patches():
    """Common patches to isolate SHAP from real computation and I/O."""
    return [
        patch("shap.plots.scatter"),
        patch("shap.plots.bar"),
        patch("shap.plots.beeswarm"),
        patch("shap.plots.waterfall"),
        patch("shap.summary_plot"),
        patch("mlflow.log_artifact"),
        patch("matplotlib.pyplot.savefig"),
        patch("matplotlib.pyplot.close"),
        patch("matplotlib.pyplot.figure"),
    ]


def _mock_explainer(shap_vals, expected_value=0.5):
    """Create a mock explainer with given return values."""
    exp = MagicMock()
    exp.shap_values.return_value = shap_vals
    exp.expected_value = expected_value
    return exp


def test_shap_explainer_selection_tree(shap_setup):
    """_is_tree_model returns True for XGBoost, leading to TreeExplainer."""
    pl, X, _ = shap_setup
    import xgboost as xgb

    assert TrainModelPipeline._is_tree_model(xgb.XGBClassifier()) is True
    assert TrainModelPipeline._is_tree_model(LogisticRegression()) is False


def test_shap_explainer_selection_linear(shap_setup):
    """_is_linear_model returns True for LogisticRegression."""
    pl, X, y = shap_setup
    assert TrainModelPipeline._is_linear_model(LogisticRegression()) is True

    import xgboost as xgb

    assert TrainModelPipeline._is_linear_model(xgb.XGBClassifier()) is False


def test_shap_normalizes_class_indexed_output():
    """_normalize_shap_explanation selects positive class from list output."""
    rng = np.random.default_rng(42)
    class_0 = rng.standard_normal((5, 2))
    class_1 = rng.standard_normal((5, 2))

    mock_explainer = MagicMock()
    mock_explainer.expected_value = np.array([-0.5, 0.5])

    X = pd.DataFrame({"a": range(5), "b": range(5)})
    result = TrainModelPipeline._normalize_shap_explanation(
        [class_0, class_1], mock_explainer, X
    )

    np.testing.assert_array_equal(result.values, class_1)
    assert result.base_values == 0.5


def test_shap_logs_expected_artifacts(shap_setup):
    """Verifies key artifact names are logged to MLflow."""
    pl, X, y = shap_setup
    mock_exp = _mock_explainer(np.zeros((len(X), X.shape[1])))

    with ExitStack() as stack:
        for p in _shap_patches():
            stack.enter_context(p)
        stack.enter_context(
            patch.object(pl, "_make_shap_explainer", return_value=mock_exp)
        )
        mock_log = stack.enter_context(patch("mlflow.log_artifact"))
        pl._log_shap_analysis(X, y)

        paths = [call.args[0] for call in mock_log.call_args_list]
        assert any("shap_feature_importance.csv" in p for p in paths)
        assert any("shap_bar.png" in p for p in paths)
        assert any("shap_beeswarm.png" in p for p in paths)
        assert any("shap_waterfall" in p for p in paths)
