import pandas as pd
import logging
from config.constant import Input_Data

logging.basicConfig(level=logging.INFO)

def data_ingestion():
    try:
        data = pd.read_csv(Input_Data)
        logging.info("Data Successfully Loaded")
        print(data.head(5))
        return data

    except Exception as e:
        logging.error(f"Error occurred while loading data: {e}")

if __name__ == "__main__":
    data_ingestion()