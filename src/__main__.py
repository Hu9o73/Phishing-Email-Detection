import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)

import asyncio  # noqa: E402
import logging  # noqa: E402

import pandas as pd  # noqa: E402

from app.services.utils import clear_console, menu  # noqa: E402

def logging_setup():
    logging.getLogger("kagglehub").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

async def main():
    clear_console()
    logging_setup()
    await menu()

if __name__ == "__main__":
    asyncio.run(main())
