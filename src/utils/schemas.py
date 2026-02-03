import warnings

from pandera import Check, Column, DataFrameSchema

# Ignore some Pydantic user warnings
warnings.filterwarnings(
    "ignore",
    message='.*Field "model_server_url" has conflict with protected namespace.*',
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore", message=".*Valid config keys have changed in V2*", category=UserWarning
)


# Define individual schemas for OA datasets
# Snowflake field names are uppercase by default
input_data_schema = DataFrameSchema(
    {
        "HARBOURID": Column(
            str,
            checks=[
                Check.str_matches(
                    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
                )
            ],
        ),
        "GENDER": Column(str, checks=None, nullable=True),
        "AGE": Column(
            float, checks=Check.ge(18), nullable=True
        ),  # Our customers should be above 18
        "CHANNEL": Column(str, checks=None, nullable=False),
        "EMPLOYERNAME": Column(str, checks=None, nullable=True),
        "HASOA": Column(
            str, checks=Check.isin(["Yes", "No"])
        ),  # Assuming target is categorical with values 0, 1
    }
)

# Prepared data should be normalized and not contain nulls
prepared_data_schema = DataFrameSchema(
    {
        "HARBOURID": Column(
            str,
            checks=[
                Check.str_matches(
                    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
                )
            ],
        ),
        "GENDER": Column(str, checks=None),
        "AGE": Column(int, checks=Check.ge(18)),
        "CHANNEL": Column(str, checks=None),
        "EMPLOYERNAME": Column(str, checks=None),
        "HASOA": Column(
            int, checks=Check.isin([0, 1])
        ),  # Assuming target is categorical with values 0, 1
    }
)

# NOTE: This will first take in the feature-engineered prepared data to start with
# So might need to only show The original dataset columns plus prediction
output_data_schema = DataFrameSchema(
    {
        "HARBOURID": Column(
            str,
            checks=[
                Check.str_matches(
                    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
                )
            ],
        ),
        "GENDER": Column(str, checks=None),
        "AGE": Column(int, checks=Check.ge(18)),
        "CHANNEL": Column(str, checks=None),
        "EMPLOYERNAME": Column(str, checks=None),
        "HASOA": Column(
            int, checks=Check.isin([0, 1])
        ),  # Assuming target is categorical with values 0, 1
        "PREDICTION": Column(
            float, checks=Check.between(0, 1)
        ),  # Assuming prediction is a probability between 0 and 1
    }
)

# Dictionary of schemas
schemas = {
    "input_data": input_data_schema,
    "prepared_data": prepared_data_schema,
    "output_data": output_data_schema,
}
