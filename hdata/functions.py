from .models import Source
from .force_validation import validate_output

import base64
import uuid
import io
import pandas as pd
import requests
import json
import time
from typing import List


def send(auth_token: str, transformation_key: str, raw_data: str, job_uuid: str):
    """Send data to the transformation API"""

    url = 'https://api.hyperdata.so/transform'
    data = {
        "raw_data": raw_data,
        "job_uuid": job_uuid
    }
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + auth_token,
        'x-api-key': transformation_key
    }

    response = requests.post(url, headers=headers,
                             data=json.dumps(data), stream=True)

    return response


def download(auth_token: str, transformation_key: str, process_uuid: str):
    """Download the transformed data"""

    params = {'process_uuid': process_uuid}
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + auth_token,
        'x-api-key': transformation_key
    }
    response = requests.get('https://api.hyperdata.so/download_output', params=params, headers=headers)

    output = requests.get(response.json()['data_url'], stream=True).json()

    entity = pd.read_parquet(io.BytesIO(base64.b64decode(output['entity'])))
    attribute = pd.read_parquet(io.BytesIO(base64.b64decode(output['attribute'])))
    record = pd.read_parquet(io.BytesIO(base64.b64decode(output['record'])))
    return entity, attribute, record


def apply_transformation(source: Source, auth_token: str, job_uuid: str):
    """Send data in chunks to the transformation API"""

    transformation_key = source.transformation_key
    entity = pd.DataFrame()
    attribute = pd.DataFrame()
    record = pd.DataFrame()

    for index, chunk in enumerate(source.zipped_chunks):

        response = send(auth_token, transformation_key, chunk, job_uuid)

        if response.status_code != 200:
            print(f"Process failed at chunk {index} with response: {response.content}. Trying again in 20s.")
            time.sleep(20)
            response = send(auth_token, transformation_key, chunk)
            if response.status_code != 200:
                print(f"Process failed at chunk {index} with response: {response.content}. Exiting.")
                return entity, attribute, record
        else:
            print(f"Process succeeded at chunk {index}.")

            new_entity, new_attribute, new_record = download(
                auth_token, transformation_key, response.json()['process_uuid'])

            if entity.empty:
                entity = pd.DataFrame(columns=new_entity.columns)
            if attribute.empty:
                attribute = pd.DataFrame(columns=new_attribute.columns)
            if record.empty:
                record = pd.DataFrame(columns=new_record.columns)

            entity = pd.concat([entity, new_entity], ignore_index=True, sort=False)
            attribute = pd.concat([attribute, new_attribute], ignore_index=True, sort=False)
            record = pd.concat([record, new_record], ignore_index=True, sort=False)

    return entity, attribute, record


def transform(sources: List[Source], auth_token: str):
    """Apply transformation to the source data"""

    entity = pd.DataFrame()
    attribute = pd.DataFrame()
    record = pd.DataFrame()

    # Initialize the job uuid
    job_uuid = str(uuid.uuid4())

    # Check if the sources are valid Source objects
    for source in sources:
        if not isinstance(source, Source):
            raise ValueError("Invalid source object. Use the hdata.Source class.")

    # Apply transformation to each source
    for source in sources:
        new_entity, new_attribute, new_record = apply_transformation(source, auth_token, job_uuid)

        if entity.empty:
            entity = pd.DataFrame(columns=new_entity.columns)
        if attribute.empty:
            attribute = pd.DataFrame(columns=new_attribute.columns)
        if record.empty:
            record = pd.DataFrame(columns=new_record.columns)

        entity = pd.concat([entity, new_entity], ignore_index=True, sort=False)
        attribute = pd.concat([attribute, new_attribute], ignore_index=True, sort=False)
        record = pd.concat([record, new_record], ignore_index=True, sort=False)

    # Validate the output
    entity, attribute, record = validate_output(entity, attribute, record)

    return entity, attribute, record
