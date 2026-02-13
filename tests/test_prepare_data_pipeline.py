from unittest.mock import patch

import pandas as pd
import pytest

from src.pipelines.prepare_data import PrepareDataPipeline


@pytest.fixture(name="data")
def sample_df_fixture():
    return pd.DataFrame(
        {
            "HARBOUR_ID": [101, 102, 103],
            "GENDER": ["F", "M", "F"],
            "AGE": [29, 41, 35],
            "CHANNEL": ["Web", "Phone", "Web"],
            "INDUSTRY": ["Tech", "Finance", "Tech"],
            "EMPLOYER_NAME": ["Acme", "Beta", "Acme"],
            "COACH_NAME": ["Alice", "Bob", "Charlie"],
            "HASOA": [1, 0, 1],
        }
    )


def test_run_pipeline(data):
    with (
        patch("src.pipelines.prepare_data.read_data", return_value=data) as mock_read,
        patch("src.pipelines.prepare_data.write_data") as mock_write,
    ):
        pipeline = PrepareDataPipeline()
        pipeline.run()

        mock_read.assert_called_once()
        _, read_kwargs = mock_read.call_args
        assert (
            read_kwargs["schema_obj"] is None
        )  # "input_data" add back schema validation once implemented
        assert "select" in read_kwargs["sql_query"].lower()

        mock_write.assert_called_once()
        _, kwargs = mock_write.call_args
        print(kwargs)
        assert kwargs["table_name"] == "OA_DATASET_PREPARED"
        assert (
            kwargs["schema_obj"] is None
        )  # "prepared_data" add back schema validation once implemented
