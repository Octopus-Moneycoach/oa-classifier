from unittest.mock import patch

import pandas as pd
import pytest

from src.pipelines.prepare_data import PrepareDataPipeline


@pytest.fixture(name="data")
def sample_df_fixture():
    return pd.DataFrame({"col1": [1, 2, 3], "col2": ["A", "B", "C"]})


def test_run_pipeline(data):
    # NEW: Test to add applied data transformations
    expected_df = data.copy()
    expected_df["col1"] = expected_df["col1"] * 10  # as per _func1 transformation

    with (
        patch("src.pipelines.prepare_data.read_data", return_value=data) as mock_read,
        patch("src.pipelines.prepare_data.write_data") as mock_write,
    ):
        pipeline = PrepareDataPipeline()
        pipeline.run()

        # NEW: Verify table write function called once with expected args
        mock_read.assert_called_once_with(
            table_name="TEST_DS_TABLE_IRIS", schema_obj="input_data"
        )

        # NEW: Assert new write_data logic
        mock_write.assert_called_once()
        _, kwargs = mock_write.call_args
        print(kwargs)
        pd.testing.assert_frame_equal(data, expected_df)
        assert kwargs["table_name"] == "TEST_DS_TABLE_IRIS_PREPARED"
        assert kwargs["schema_obj"] == "prepared_data"

        # # Assert write_data called with df, key
        # args, _ = mock_write.call_args
        # written_df, file_key = args
        # pd.testing.assert_frame_equal(written_df, data)
        # assert file_key == "prepared_data"
