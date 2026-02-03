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
                    da.idpuserid as HARBOURID,
                    dc.GENDER,
                    cast(da.age as int) as AGE,
                    da.CHANNEL,
                    da.EMPLOYERNAME,
                    case
                        when da.oastatus in ('Active', 'Completed') then 'Yes'
                        else 'No'
                    end as HASOA
                from prod_quicksight.vw_custom_hq_daclientjourney_rls da -- you can link to different databases/schemas/tables here
                inner join prod_analytics.dim_client dc on da.idpuserid = dc.idpuserid
                where da.latestdastep != 'Not in DA Journey';
            """,
            schema_obj="input_data",
        )
        logger.info("Input data read. Shape: %s", df.shape)

        df = (
            df.pipe(self._dedup, idx_col="HARBOURID")
            .pipe(self._impute)
            .pipe(self._feature_engineer, cols=["AGE"], bin_width=10)
            .pipe(self._encoder, id_col="HARBOURID", target_col="HASOA")
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
    def _impute(df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values in DataFrame with median for numerical and mode for categorical columns.

        Assumes last column is target and does not impute it.
        """
        target_col = df.columns[-1]
        # exclude target column if exists from transformation
        numeric_cols = (
            df.drop(columns=[target_col]).select_dtypes(include="number").columns
        )
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        categorical_cols = (
            df.drop(columns=[target_col]).select_dtypes(include="object").columns
        )
        df[categorical_cols] = df[categorical_cols].fillna(
            df[categorical_cols].mode().iloc[0]
        )
        return df

    @staticmethod
    def _feature_engineer(
        df: pd.DataFrame, cols: list[str], bin_width: int = 10
    ) -> pd.DataFrame:
        """Feature engineer binned numerical columns.

        Assumes last column is target and does not transform it.

        """
        target_col = df.columns[-1]
        for col in cols:
            if col in df.columns and col != target_col:
                bin_labels = [
                    f"{i}-{i + (bin_width-1)}"
                    for i in range(0, int(df[col].max()) + 1, bin_width)
                ]

                df[f"{col}_BINNED"] = pd.cut(
                    df[col],
                    bins=range(
                        0,
                        int(df[col].max()) + (bin_width + 1),
                        bin_width,
                    ),
                    labels=bin_labels,
                    right=False,
                )
        return df

    @staticmethod
    def _encoder(
        df: pd.DataFrame, id_col: str = "HARBOURID", target_col: str = "HASOA"
    ) -> pd.DataFrame:
        """Encode target and categorical columns using Label Encoding and One-Hot Encoding.

        Args:
            df: Input DataFrame.
            id_col: Identifier column to exclude from encoding.
            target_col: Target column to label encode.

        """
        le = LabelEncoder()
        ohc = OneHotEncoder(sparse_output=False, drop="first")

        # Label encode target column

        df[target_col] = le.fit_transform(df[target_col])

        # One-Hot Encode categorical columns excluding id and target
        # Also dropping employername as it has high cardinality
        categorical_cols = (
            df.drop(columns=[id_col, target_col, "EMPLOYERNAME"])
            .select_dtypes(include="object")
            .columns
        )
        for col in categorical_cols:
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
