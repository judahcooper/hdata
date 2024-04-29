import pandas as pd
import os
import math
import zipfile
import io
import base64
from uuid import UUID
from typing import Any


class Source:
    """A class for loading data into the transformation pipeline"""

    def __init__(self, transformation_key, data):
        self.transformation_key: str = transformation_key
        self.data_input = data
        self.data_frame = self.load_data()
        self.data_chunks = self.split_data()
        self.zipped_chunks = self.zip_chunks()

    def load_data(self):
        if isinstance(self.data_input, pd.DataFrame):
            return self.data_input
        elif isinstance(self.data_input, str):
            if os.path.exists(self.data_input):
                file_extension = os.path.splitext(self.data_input)[1].lower()
                if file_extension in ['.csv']:
                    return pd.read_csv(self.data_input)
                elif file_extension in ['.parquet']:
                    return pd.read_parquet(self.data_input)
                elif file_extension in ['.xlsx']:
                    return pd.read_excel(self.data_input)
                elif file_extension in ['.json']:
                    return pd.read_json(self.data_input)
                else:
                    raise ValueError("Unsupported file format.")
            else:
                raise FileNotFoundError("The specified file does not exist.")
        else:
            raise TypeError("Unsupported input type. Please provide a pandas DataFrame or a file path.")

    def split_data(self, max_rows_per_chunk=30000):
        rows = self.data_frame.shape[0]
        if rows > max_rows_per_chunk:
            num_chunks = math.ceil(rows / max_rows_per_chunk)
            return [self.data_frame.iloc[i * max_rows_per_chunk : (i + 1) * max_rows_per_chunk] for i in range(num_chunks)]
        else:
            return [self.data_frame]

    def zip_chunks(self):
        encoded_zips = []

        for chunk in self.data_chunks:
            parquet_buffer = io.BytesIO()
            chunk.to_parquet(parquet_buffer, index=False)
            parquet_buffer.seek(0)
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.writestr(f"source.parquet", parquet_buffer.getvalue())
            zip_buffer.seek(0)
            encoded_zip = base64.b64encode(zip_buffer.read()).decode('utf-8')
            encoded_zips.append(encoded_zip)

        return encoded_zips


class Output:
    def __init__(self, entity, attribute, record):
        self.entity: pd.DataFrame = entity
        self.attribute: pd.DataFrame = attribute
        self.record: pd.DataFrame = record

    def get_entity_id(self, search_term: Any, entity_table: pd.DataFrame):
        # type cast search term to match the column type
        if type(search_term) != entity_table['entity_name'][0]:
            entity_type = type(entity_table['entity_name'][0])
            search_term = entity_type(search_term)

        matches = entity_table[entity_table['entity_name'] == search_term]
        # Return the first entity_id if matches are found, else return None
        if not matches.empty:
            return matches['entity_uuid'].iloc[0]

        raise ValueError(f"No attribute found for search_term: {search_term}")

    def get_attribute_id(self, search_term: Any, attribute_table: pd.DataFrame):
        if type(search_term) != attribute_table['attribute_name'][0]:
            entity_type = type(attribute_table['attribute_name'][0])
            search_term = entity_type(search_term)

        matches = attribute_table[attribute_table['attribute_name'] == search_term]
        # Return the first entity_id if matches are found, else return None
        if not matches.empty:
            return matches['attribute_uuid'].iloc[0]

        raise ValueError(f"No attribute found for search_term: {search_term}")

    def describe_entity(self, entity_id: str, attribute_table: pd.DataFrame, value_table: pd.DataFrame):
        try:
            entity_uuid = UUID(entity_id)
        except ValueError:
            raise ValueError("Entity ID must be a valid UUID")

        values = value_table[value_table['entity_uuid'] == str(entity_uuid)]
        # Join all attributes to the values_table
        attributes_and_values = pd.merge(left=attribute_table, right=values, on=['attribute_uuid'])
        return attributes_and_values[['attribute_name', 'record_value']]

    def describe_attribute(self, attribute_id: str, entity_table: pd.DataFrame, value_table: pd.DataFrame):
        try:
            attribute_uuid = UUID(attribute_id)
        except ValueError:
            raise ValueError("attribute_id must be a valid UUID")

        values = value_table[value_table['attribute_uuid'] == str(attribute_id)]

        attribute_entity_ids = values['entity_uuid'].unique()
        # Get all entities which have this attribute
        entity_ids = []
        for entity in entity_table.itertuples():
            if entity.entity_uuid in attribute_entity_ids:
                entity_ids.append(entity)

        entities = pd.DataFrame(entity_ids)

        values_and_entities = pd.merge(left=entities, right=values, on=['entity_uuid'])

        return values_and_entities[['entity_name', 'record_value']]

    def query(self, **kwargs):

        # Return all entities, attributes or records
        if 'all' in kwargs:
            if kwargs['all'] == 'entity':
                return self.entity[['entity_name', 'entity_description']]
            if kwargs['all'] == 'attribute':
                return self.attribute[['attribute_name', 'attribute_description']]
            if kwargs['all'] == 'record':
                return self.record[['record_value']]

        # Narrow down query
        filters = []

        if 'entity' in kwargs:
            if type(kwargs['entity']) == str:
                entity_id = self.get_entity_id(kwargs['entity'], self.entity)
                filters.append((self.record['entity_uuid'] == entity_id))
            if type(kwargs['entity']) == UUID:
                filters.append((self.record['entity_uuid'] == kwargs['entity']))
            if type(kwargs['entity']) == list:
                entity_ids = []
                for entity_name in kwargs['entity']:
                    entity_id = self.get_entity_id(entity_name, self.entity)
                    entity_ids.append(entity_id)
                filters.append((self.record['entity_uuid'].isin(entity_ids)))

        if 'attribute' in kwargs:
            if type(kwargs['attribute']) == str:
                attribute_id = self.get_attribute_id(kwargs['attribute'], self.attribute)
                filters.append((self.record['attribute_uuid'] == attribute_id))
            if type(kwargs['attribute']) == UUID:
                filters.append((self.record['attribute_uuid'] == kwargs['attribute']))
            if type(kwargs['attribute']) == list:
                attribute_ids = []
                for attribute_name in kwargs['attribute']:
                    attribute_id = self.get_attribute_id(attribute_name, self.attribute)
                    attribute_ids.append(attribute_id)
                filters.append((self.record['attribute_uuid'].isin(attribute_ids)))

        if 'after_date' in kwargs:
            filters.append((self.record['datetime'] > kwargs['after_date']))

        if 'before_date' in kwargs:
            filters.append((self.record['datetime'] < kwargs['before_date']))

        # Combine all filters using logical AND
        if filters:
            combined_filters = filters[0]
            for f in filters[1:]:
                combined_filters &= f
            df_filt = self.record[combined_filters]
        else:
            df_filt = self.record

        return df_filt.merge(self.entity, on=['entity_uuid']).merge(self.attribute, on=['attribute_uuid'])[['datetime', 'entity_name', 'attribute_name', 'record_value']]
