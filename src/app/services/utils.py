import subprocess
import sys
from halo import Halo

from app.services.datamanager import datamanager

def clear_console():
    subprocess.run('cls' if sys.platform == 'win32' else 'clear', shell=True)

def ask_for_integer(min: int | None = None, max: int | None = None) -> int | None:
    try:
        user_input = int(input(""))
    except Exception:
        return None
    if min is not None and user_input < min:
        return None
    if max is not None and user_input > max:
        return None
    return user_input


async def load_the_dataset():
    spinner = Halo(text='Loading dataset...', spinner='dots')
    spinner.start()
    try:
        await datamanager.load_data()
        spinner.succeed('Dataset loaded successfully!')
    except Exception as e:
        spinner.fail(f'Failed to load dataset: {e}')
    return


async def get_info_about_dataset():
    print("--- Data Informations ---")
    if datamanager.df is None:
        print("Please load the dataset first.")
        return
    print(datamanager.df.info())
    return


async def menu():
    options = {
        1: ("Load the dataset", load_the_dataset),
        2: ("Get information about the dataset", get_info_about_dataset),
    }

    while True:
        clear_console()
        print("--- Phishing Email Detection ---")
        for key, (desc, _) in options.items():
            print(f"[{key}] {desc}")
        print("[0] Exit")

        choice = ask_for_integer(0, len(options))
        if choice is None:
            continue

        if choice == 0:
            print("Bye!")
            break

        _, action = options[choice]
        clear_console()
        await action()
        input("\nPress Enter to continue...")

