from .models import Source

import base64
import io
import pandas as pd
import requests
import json
import time


def send(transformation_key: str, raw_data: str):
    """Send data to the transformation API"""

    url = 'https://api.hyperdata.network/transform'
    data = {"raw_data": raw_data}
    headers = {
        'Content-Type': 'application/json',
        'x-api-key': transformation_key
    }

    response = requests.post(url, headers=headers,
                             data=json.dumps(data), stream=True)

    return response


def download(transformation_key: str, process_uuid: str):
    """Download the transformed data"""

    params = {'process_uuid': process_uuid}
    headers = {'x-api-key': transformation_key}
    response = requests.get('https://api.hyperdata.network/download_output', params=params, headers=headers)

    output = requests.get(response.json()['data_url'], stream=True).json()

    entity = pd.read_parquet(io.BytesIO(base64.b64decode(output['entity'])))
    attribute = pd.read_parquet(io.BytesIO(base64.b64decode(output['attribute'])))
    record = pd.read_parquet(io.BytesIO(base64.b64decode(output['record'])))
    return entity, attribute, record


def transform(source: Source):
    """Send data in chunks to the transformation API"""

    transformation_key = source.transformation_key
    entity = pd.DataFrame()
    attribute = pd.DataFrame()
    record = pd.DataFrame()

    for index, chunk in enumerate(source.zipped_chunks):

        response = send(transformation_key, chunk)

        if response.status_code != 200:
            print(f"Process failed at chunk {index} with response: {response.content}. Trying again in 20s.")
            time.sleep(20)
            response = send(transformation_key, chunk)
            if response.status_code != 200:
                print(f"Process failed at chunk {index} with response: {response.content}. Exiting.")
                return entity, attribute, record
        else:
            print(f"Process succeeded at chunk {index}.")

            new_entity, new_attribute, new_record = download(transformation_key, response.json()['process_uuid'])

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
