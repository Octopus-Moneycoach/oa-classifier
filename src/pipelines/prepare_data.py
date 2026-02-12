import logging

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from src.pipelines.pipeline import Pipeline
from src.utils.utils import read_data, write_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class PrepareDataPipeline(Pipeline):
    """Pipeline that loads raw data, runs placeholder transforms, and persists results."""

    def __init__(self) -> None:
        """Initialise configuration."""
        logger.info("Initializing the PrepareDataPipeline.")

    def run(self) -> None:
        """Execute the data preparation workflow end to end."""
        logger.info("Starting the data preparation pipeline.")

        # remember to exclude data leakage columns from transformations and encoding, and to not apply same transformations to target column as to features
        df = read_data(
            sql_query="""
                select * exclude (OA_PURCHASE_DATE) from TEST_DS_DATABASE.PUBLIC.OA_DATASET
            """,
            schema_obj=None,  # Ignore schema validation for now
        )
        logger.info("Input data read. Shape: %s", df.shape)

        df = (
            df.pipe(self._dedup, idx_col="HARBOUR_ID")
            .pipe(self._impute)
            # .pipe(self._feature_engineer_int, cols=["AGE"], target_col="HASOA") # omitted for now
            .pipe(
                self._feature_engineer_date,
                date_cols=[
                    "EQ_SUBMITTED_DATE",
                    "PMG_START_DATE",
                    "PLANNING_SESSION_CREATED_DATE",
                    "PLANNING_SESSION_DATE",
                    "ACTION_SESSION_CREATED_DATE",
                    "ACTION_SESSION_DATE",
                    "FIRST_FORECAST_CREATION_DATE",
                    "PLAN_ACTIVATED_DATE",
                ],
                target_col="HASOA",
            )  # no date columns for now
            # Don't encode high cardinality columns here
            .pipe(
                self._encoder,
                cat_cols=[
                    "GENDER",
                    "CHANNEL",
                ],  # Choose low cardinality categorical columns to encode
                id_col="HARBOUR_ID",
                target_col="HASOA",
            )
        )
        logger.info("Data preparation transformations applied. Shape: %s", df.shape)

        write_data(
            df,
            table_name="OA_DATASET_PREPARED",
            schema_obj=None,
            database="TEST_DS_DATABASE",
            schema="PUBLIC",
        )
        logger.info("Processed data saved.")

    @staticmethod
    def _dedup(df: pd.DataFrame, idx_col: str = "HARBOUR_ID") -> pd.DataFrame:
        """Drop duplicate rows based on index column."""
        df = df.drop_duplicates(subset=[idx_col]).reset_index(drop=True)
        return df

    @staticmethod
    def _impute(
        df: pd.DataFrame,
        num_cols: list[str] | None = None,
        cat_cols: list[str] | None = None,
        target_col: str = "HASOA",
    ) -> pd.DataFrame:
        """Impute missing values in DataFrame with median for numerical and "Unknown" for categorical columns."""
        if num_cols is None:
            num_cols = (
                df.drop(columns=[target_col]).select_dtypes(include="number").columns
            )
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

        if cat_cols is None:
            cat_cols = (
                df.drop(columns=[target_col])
                .select_dtypes(include=["object", "category"])
                .columns
            )
        df[cat_cols] = df[cat_cols].fillna("Unknown")
        return df

    # TODO: create different feature engineering functions for different types of data.
    @staticmethod
    def _feature_engineer_int(
        df: pd.DataFrame, cols: list[str], bins: int = 11, target_col: str = "HASOA"
    ) -> pd.DataFrame:
        """Feature engineer binned numerical columns.

        Args:
            df: Input DataFrame.
            cols: List of numerical columns to bin.
            bins: Number of bins to create.
            target_col: Target column to exclude from binning.
        """
        assert (
            df[cols].dtypes.isin(["int64", "float64"]).all()
        ), "All cols must be numerical."
        assert bins > 1, "Number of bins must be greater than 1."
        for col in cols:
            if col in df.columns and col != target_col:
                # Use pd.qcut to create quantile bins and generate appropriate labels
                quantiles = pd.qcut(df[col], q=bins, duplicates="raise")
                bin_edges = quantiles.cat.categories
                bin_labels = [
                    f"{int(interval.left)}-{int(interval.right)}"
                    for interval in bin_edges
                ]
                df[f"{col}"] = pd.qcut(
                    df[col], q=bins, duplicates="raise", labels=bin_labels
                )
        return df

    @staticmethod
    def _feature_engineer_date(
        df: pd.DataFrame, date_cols: list[str], target_col: str = "HASOA"
    ) -> pd.DataFrame:
        """Feature engineer date columns into year, month, day.

        Args:
            df: Input DataFrame.
            date_cols: List of date columns to transform.
            target_col: Target column to exclude from transformation.
        """
        for col in date_cols:
            if col in df.columns and col != target_col:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                df[f"{col}_YEAR"] = df[col].dt.year
                df[f"{col}_MONTH"] = df[col].dt.month
                df[f"{col}_DAY"] = df[col].dt.day
                df.drop(columns=[col], inplace=True)
        return df

    @staticmethod
    def _encoder(
        df: pd.DataFrame,
        cat_cols: list[str],
        id_col: str = "HARBOUR_ID",
        target_col: str = "HASOA",
    ) -> pd.DataFrame:
        """Encode target and categorical columns using Label Encoding and One-Hot Encoding.

        Args:
            df: Input DataFrame.
            cat_cols: List of categorical columns to encode.
            id_col: Identifier column to exclude from encoding.
            target_col: Target column to label encode.

        """
        le = LabelEncoder()
        ohc = OneHotEncoder(sparse_output=False, drop="first")

        # Label encode target column

        df[target_col] = le.fit_transform(df[target_col])

        # One Hot Encode categorical columns from passed list
        missing_cols = [c for c in cat_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Column(s) {missing_cols} not found in DataFrame.")

        # assert df[cat_cols].dtypes.isin(["object", "category"]).all(), "cat_cols must be categorical columns"
        assert (
            id_col not in cat_cols and target_col not in cat_cols
        ), "id_col and target_col should not be in cat_cols"

        for col in cat_cols:
            col_encoded = ohc.fit_transform(df[[col]])
            col_df = pd.DataFrame(
                col_encoded, columns=ohc.get_feature_names_out([col]), index=df.index
            )
            col_df.columns = [
                c.replace(" ", "_").replace("-", "_").upper() for c in col_df.columns
            ]
            df = pd.concat([df.drop(columns=[col]), col_df], axis=1)

        # rearrange columns to keep target at the end
        cols = [c for c in df.columns if c != target_col] + [target_col]
        df = df[cols]
        return df
