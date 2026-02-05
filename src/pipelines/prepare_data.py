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

        df = read_data(
            sql_query="""
                select
                    dc.idpuserid as HARBOURID,
                    dc.gender as GENDER,
                    year(sysdate()) - dc.yeardob as AGE,
                    dc.channel as CHANNEL,
                    de.name as EMPLOYERNAME,
                    case
                        when fcj.oastatus in ('Active', 'Completed') then 'Yes'
                        else 'No'
                    end as HASOA
                from prod_analytics.fact_clientjourney fcj
                inner join prod_analytics.dim_client dc on fcj.clientkey = dc.clientkey
                left join prod_analytics.dim_employer de on fcj.employerkey = de.employerkey
                where 1 = 1
                and pmgstatus in ('Active', 'Completed')
            """,
            schema_obj="input_data",  # Ignore schema validation for now
        )
        logger.info("Input data read. Shape: %s", df.shape)

        df = (
            df.pipe(self._dedup, idx_col="HARBOURID")
            .pipe(self._impute)
            # .pipe(self._feature_engineer, cols=["AGE"], target_col="HASOA") # omitted for now
            # Don't encode high cardinality columns here
            .pipe(
                self._encoder,
                cat_cols=["GENDER", "CHANNEL"],
                id_col="HARBOURID",
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
    def _dedup(df: pd.DataFrame, idx_col: str = "HARBOURID") -> pd.DataFrame:
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

    @staticmethod
    def _feature_engineer(
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
    def _encoder(
        df: pd.DataFrame,
        cat_cols: list[str],
        id_col: str = "HARBOURID",
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
