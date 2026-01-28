from unittest.mock import patch

import pandas as pd
import pytest

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
        assert table_name == "TEST_DS_TABLE_IRIS_OUTPUT"
        assert schema_obj == "output_data"

        # # Ensure predictions were written for the full dataset under the expected key/env
        # mock_write.assert_called_once()
        # args, _ = mock_write.call_args
        # df_out, table_name, schema_obj = args
        # assert "PREDICTION" in df_out.columns
        # assert table_name == "TEST_DS_TABLE_IRIS_OUTPUT"
        # assert schema_obj == "output_data"


def test_train_method(pipeline, data):
    """Directly test the public train method."""
    X, y = data.drop(columns=["target"]), data["target"]
    pipeline.train(X, y)
    assert pipeline.model is not None
