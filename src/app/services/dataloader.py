from kagglehub import KaggleDatasetAdapter
import pandas as pd
import kagglehub

class Dataloader:
    @staticmethod
    async def load_data() -> pd.DataFrame:
        return kagglehub.dataset_load(
            KaggleDatasetAdapter.PANDAS, "advaithsrao/enron-fraud-email-dataset", "enron_data_fraud_labeled.csv"
        )
