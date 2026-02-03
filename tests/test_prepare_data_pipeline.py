from unittest.mock import patch

import pandas as pd
import pytest

from src.pipelines.prepare_data import PrepareDataPipeline


@pytest.fixture(name="data")
def sample_df_fixture():
    return pd.DataFrame({"col1": [1, 2, 3], "col2": ["A", "B", "C"]})


def test_run_pipeline(data):
    with (
        patch("src.pipelines.prepare_data.read_data", return_value=data) as mock_read,
        patch("src.pipelines.prepare_data.write_data") as mock_write,
    ):
        pipeline = PrepareDataPipeline()
        pipeline.run()

        mock_read.assert_called_once_with(schema_obj="input_data")

        mock_write.assert_called_once()
        _, kwargs = mock_write.call_args
        print(kwargs)
        assert kwargs["table_name"] == "OA_DATASET_PREPARED"
        assert kwargs["schema_obj"] is None
