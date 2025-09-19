import pandas as pd
import kagglehub


class DataManager:
    def __init__(self):
        self.df: pd.DataFrame | None = None

    async def load_data(self) -> pd.DataFrame:
        try:
            file_path = kagglehub.dataset_download("advaithsrao/enron-fraud-email-dataset")
            dataframe = pd.read_csv(f"{file_path}/enron_data_fraud_labeled.csv")
        except Exception as e:
            raise e

        self.df = dataframe
        return dataframe

datamanager = DataManager()
