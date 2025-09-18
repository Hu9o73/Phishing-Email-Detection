import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)


import pandas as pd
import asyncio
import logging
from halo import Halo

from app.services.dataloader import Dataloader
from app.services.utils import clear_console

def logging_setup():
    logging.getLogger("kagglehub").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)


async def main():
    clear_console()
    logging_setup()

    spinner = Halo(text='Loading dataset...', spinner='dots')
    spinner.start()

    try:
        df = await Dataloader.load_data()
        spinner.succeed('Dataset loaded successfully!')
    except Exception as e:
        spinner.fail(f'Failed to load dataset: {e}')
        return

    print(df.head())


if __name__ == "__main__":
    asyncio.run(main())
