import uuid
import pandas as pd
from time import time
import warnings
import re


def column_validation(entity: pd.DataFrame, attribute: pd.DataFrame, record: pd.DataFrame):

    # Convert all column names to lower case
    entity.columns = entity.columns.str.lower()
    attribute.columns = attribute.columns.str.lower()
    record.columns = record.columns.str.lower()

    # Reindex the dataframes
    entity = entity.reset_index(drop=True)
    attribute = attribute.reset_index(drop=True)
    record = record.reset_index(drop=True)

    # Ensure the correct columns are present
    try:
        entity = entity[["entity_uuid", "entity_name", "entity_description"]]
        attribute = attribute[["attribute_uuid",
                               "attribute_name", "attribute_description"]]
        record = record[['datetime', 'entity_uuid',
                         'attribute_uuid', 'record_value']]
    except KeyError:
        raise KeyError("The entity dataframe must contain entity_uuid, entity_name, and entity_description columns. The attribute dataframe must contain attribute_uuid, attribute_name, and attribute_description columns. The record dataframe must contain datetime, entity_uuid, attribute_uuid, and record_value columns.")

    return entity, attribute, record


def validate_entity(entity: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate the entity table from the data transformation process."""

    if entity["entity_uuid"].isnull().values.any():
        raise Exception(
            "The entity_uuid column must not contain null values.")
    if entity["entity_name"].isnull().values.any():
        raise Exception(
            "The entity_name column must not contain null values.")
    if entity["entity_uuid"].duplicated().any():
        print(
            f"WARNING: Duplicate entity_uuids detected. Dropping duplicates:{entity[entity['entity_uuid'].duplicated()]['entity_uuid'].values}")
        entity = entity.drop_duplicates(subset=["entity_uuid"])
    if entity["entity_name"].duplicated().any():
        print(
            f"WARNING: Duplicate entity_names detected. Dropping duplicates:{entity[entity['entity_name'].duplicated()]['entity_name'].values}")
        entity = entity.drop_duplicates(subset=["entity_name"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        entity["entity_uuid"] = entity["entity_uuid"].apply(
            lambda x: str(uuid.UUID(x)))
        entity["entity_description"] = entity["entity_description"].fillna(
            "").astype(str)

    return entity


def validate_attribute(attribute: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate the attribute table from the data transformation process."""

    if attribute["attribute_uuid"].isnull().values.any():
        raise Exception(
            "The attribute_uuid column must not contain null values.")
    if attribute["attribute_name"].isnull().values.any():
        raise Exception(
            "The attribute_name column must not contain null values.")
    if attribute["attribute_uuid"].duplicated().any():
        print(
            f"WARNING: Duplicate attribute_uuids detected. Dropping duplicates:{attribute[attribute['attribute_uuid'].duplicated()]['attribute_uuid'].values}")
        attribute = attribute.drop_duplicates(subset=["attribute_uuid"])
    if attribute["attribute_name"].duplicated().any():
        print(
            f"WARNING: Duplicate attribute_names detected. Dropping duplicates:{attribute[attribute['attribute_name'].duplicated()]['attribute_name'].values}")
        attribute = attribute.drop_duplicates(subset=["attribute_name"])

    attribute["attribute_uuid"] = attribute["attribute_uuid"].apply(
        lambda x: str(uuid.UUID(x)))
    attribute["attribute_name"] = attribute["attribute_name"].str.lower()
    attribute["attribute_description"] = attribute["attribute_description"].fillna(
        "").astype(str)

    return attribute


def validate_record(record: pd.DataFrame) -> pd.DataFrame:
    """Optimized clean and validate function for the record table."""

    # Check for null values in essential columns
    if record[["datetime", "entity_uuid", "attribute_uuid"]].isnull().any().any():
        raise Exception(
            "Columns 'datetime', 'entity_uuid', and 'attribute_uuid' must not contain null values.")

    # Drop rows where 'record_value' is null without affecting the original DataFrame
    record = record.dropna(subset=["record_value"])

    # Check for duplicates in key columns
    if record.duplicated(subset=["datetime", "entity_uuid", "attribute_uuid"]).any():
        raise Exception(
            "There can be no duplicate combinations of datetime, entity_uuid, and attribute_uuid.")

    # Convert datetime to string and validate format
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        record["datetime"] = record["datetime"].astype(str)

    valid_date_pattern = re.compile(
        r'^\d{4}(-\d{2}(-\d{2}( \d{2}:\d{2}(:\d{2})?)?)?)?$')
    if not record["datetime"].str.match(valid_date_pattern).all():
        raise Exception("Invalid datetime format detected.")

    # Validate and format UUID columns efficiently
    for col in ["entity_uuid", "attribute_uuid"]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            record[col] = record[col].map(lambda x: str(uuid.UUID(x)))

    # Ensure valid types in 'record_value' column
    valid_types = (float, int, str, list, bool, bytes)
    if not all(isinstance(x, valid_types) for x in record["record_value"]):
        raise Exception("Invalid types found in 'record_value' column.")

    return record


def uuid_match_validation(entity: pd.DataFrame, attribute: pd.DataFrame, record: pd.DataFrame):
    # Check that all entity_uuids and attribute_uuids in the record table exist in their respective tables
    if not record['entity_uuid'].isin(entity['entity_uuid']).all():
        raise ValueError(
            "All entity_uuids in the record table must exist in the entity table.")
    if not record['attribute_uuid'].isin(attribute['attribute_uuid']).all():
        raise ValueError(
            "All attribute_uuids in the record table must exist in the attribute table.")

    # Check that all entity_uuids and attribute_uuids in the entity and attribute tables are used in the record table
    if not entity['entity_uuid'].isin(record['entity_uuid']).all():
        raise ValueError(
            "All entity_uuids in the entity table must be used in the record table.")
    if not attribute['attribute_uuid'].isin(record['attribute_uuid']).all():
        print("WARNING: Not all attribute_uuids in the attribute table are used in the record table.")

    return entity, attribute, record


def validate_output(entity: pd.DataFrame, attribute: pd.DataFrame, record: pd.DataFrame) -> dict:
    """Clean and validate the three output tables from the data transformation process."""

    # Column validation
    t_columns = time()
    entity, attribute, record = column_validation(entity, attribute, record)
    print(f"Column validation: {time() - t_columns}")

    # Entity table validation
    t_entity = time()
    entity = validate_entity(entity)
    print(f"Entity table validation: {time() - t_entity}")

    # Attribute table validation
    t_attribute = time()
    attribute = validate_attribute(attribute)
    print(f"Attribute table validation: {time() - t_attribute}")

    # Record table validation
    t_record = time()
    record = validate_record(record)
    print(f"Record table validation: {time() - t_record}")

    # Check that all entity_uuids and attribute_uuids in the record table exist in their respective tables
    t_uuid_match = time()
    entity, attribute, record = uuid_match_validation(
        entity, attribute, record)
    print(f"UUID match validation: {time() - t_uuid_match}")

    return entity, attribute, record
