import subprocess
import sys
from halo import Halo

from app.services.datamanager import datamanager
from app.services.printers.data_information_printer import DataInformationPrinter

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

    spinner = Halo(text="Analyzing dataset...", spinner="dots")
    spinner.start()

    stats = await datamanager.get_comprehensive_stats()
    data_printer = DataInformationPrinter(stats)

    spinner.succeed(text="Analysis complete")

    clear_console()

    # Display each section
    data_printer.print_basic_info()
    data_printer.print_target_distribution()
    data_printer.print_text_statistics()
    data_printer.print_email_specific_stats()
    data_printer.print_data_quality_report()


async def preprocess_data():
    """Run lightweight preprocessing (missing-value handling) using the datamanager.

    This function will compute dataset statistics (async) and then call the
    datamanager's preprocessing routine which mutates `datamanager.df`.
    """
    if datamanager.df is None:
        print("Please load the dataset first.")
        return
    
    print("--- Data Preprocessing ---")

    # Ask for threshold for dropping columns
    threshold_input = input("Drop columns with completeness < (percent, default 50): ").strip()
    try:
        threshold = float(threshold_input) if threshold_input else 50.0
    except Exception:
        print("Invalid threshold value, using default 50")
        threshold = 50.0

    # Ask user whether to drop constant columns (default: yes)
    drop_input = input("Drop all constant columns? [Y/n] ").strip().lower()
    drop_constants = not (drop_input in ['N', 'No', 'n', 'no'])

    spinner = Halo(text='Handling quality issues...', spinner='dots')
    spinner.start()
    try:
        await datamanager.handle_quality_issues(drop_constants=drop_constants, threshold=threshold)
        spinner.succeed('Data quality issues treated!')
    except Exception as e:
        spinner.fail(f'Handling quality issues failed: {e}')

    # Ask whether to create ML feature columns (text-derived + domains + one-hot)
    fe_input = input("Create ML feature columns (text_length, flags, sender/recipient domains + one-hot)? [Y/n] ").strip().lower()
    create_features = not (fe_input in ['N', 'No', 'n', 'no'])
    if create_features:
        fe_spinner = Halo(text='Creating ML feature columns...', spinner='dots')
        fe_spinner.start()
        try:
            await datamanager.run_feature_engineering()
            fe_spinner.succeed('Feature creation complete!')
        except Exception as e:
            fe_spinner.fail(f'Feature creation failed: {e}')

    # Ask whether to vectorize text for ML
    vec_input = input("Vectorize text for ML now? [Y/n] ").strip().lower()
    do_vectorize = not (vec_input in ['N', 'No', 'n', 'no'])
    if do_vectorize == False:
        return
    print("\n--- Text Vectorization Parameters ---")
    ngram_input = input("n-gram range (min-max, default 1-2): ").strip()
    try:
        if ngram_input:
            parts = [int(p) for p in ngram_input.split("-")]
            ngram_range = (parts[0], parts[1]) if len(parts) >= 2 else (parts[0], parts[0])
        else:
            ngram_range = (1, 2)
    except Exception:
        print("Invalid ngram range, using default (1,2)")
        ngram_range = (1, 2)

    maxf_input = input("Max features for vectorizer (default 10000): ").strip()
    try:
        max_features = int(maxf_input) if maxf_input else 10000
    except Exception:
        print("Invalid max_features, using default 10000")
        max_features = 10000

    spinner = Halo(text='Vectorizing text...', spinner='dots')
    spinner.start()
    try:
        X = await datamanager.run_vectorization(vectorizer_type='tfidf',
                                                ngram_range=ngram_range,
                                                max_features=max_features)
        spinner.succeed('Vectorization complete!')
        # Report shape if available
        try:
            print(f"Feature matrix shape: {getattr(X, 'shape', None)}")
        except Exception:
            pass
    except Exception as e:
        spinner.fail(f'Vectorization failed: {e}')
    return


async def menu():
    options = {
        1: ("Load the dataset", load_the_dataset),
        2: ("Get information about the dataset", get_info_about_dataset),
        3: ("Preprocess data", preprocess_data)
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
