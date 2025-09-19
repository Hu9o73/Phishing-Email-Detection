from kagglehub import KaggleDatasetAdapter
import pandas as pd
import kagglehub
import chardet

class Dataloader:
    @staticmethod
    async def load_data() -> pd.DataFrame:
        file_path = kagglehub.dataset_load("advaithsrao/enron-fraud-email-dataset")
        return pd.read_csv(f"{file_path}/enron_data_fraud_labeled.csv")
