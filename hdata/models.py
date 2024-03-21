import pandas as pd
import os
import math
import zipfile
import io
import base64


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
