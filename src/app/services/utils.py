import subprocess
import sys

from halo import Halo

from app.services.datamanager import datamanager
from app.services.modelmanager import ModelManager
from app.services.inputs import ask_for_float, ask_for_integer, ask_for_range, ask_yes_no
from app.services.printers.data_information_printer import DataInformationPrinter


def clear_console():
    subprocess.run('cls' if sys.platform == 'win32' else 'clear', shell=True)

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
    """Preprocess the dataset by handling quality issues, feature engineering, and text vectorization."""
    if datamanager.df is None:
        print("Please load the dataset first.")
        return

    print("--- Data Preprocessing ---")

    threshold = ask_for_float("Drop columns with completeness < (percent)", default=50.0, min=0.0, max=100.0)

    drop_constants = ask_yes_no("Drop all constant columns?", default=True)

    spinner = Halo(text='Handling quality issues...', spinner='dots')
    spinner.start()
    try:
        await datamanager.handle_quality_issues(drop_constants=drop_constants, threshold=threshold)
        spinner.succeed('Data quality issues treated!')
    except Exception as e:
        spinner.fail(f'Handling quality issues failed: {e}')

    print("\n--- Feature Engineering ---")
    create_features = ask_yes_no(
        "Create ML feature columns (text_length, flags, sender/recipient domains + one-hot)?", default=True
    )
    if create_features:
        top_k = ask_for_integer("Max number of top domains to one-hot encode", default=50, min=1)
        fe_spinner = Halo(text='Creating ML feature columns...', spinner='dots')
        fe_spinner.start()
        try:
            await datamanager.run_feature_engineering(top_k_domains=top_k)
            fe_spinner.succeed('Feature creation complete!')
        except Exception as e:
            fe_spinner.fail(f'Feature creation failed: {e}')

    do_vectorize = ask_yes_no("Vectorize text for ML now?", default=True)
    if not do_vectorize:
        return

    print("\n--- Text Vectorization Parameters ---")
    ngram_range = ask_for_range("N-gram range (min-max)", default=(1, 2))
    max_features = ask_for_integer("Max features for vectorizer", default=10000, min=100)

    spinner = Halo(text='Vectorizing text...', spinner='dots')
    spinner.start()
    try:
        X = await datamanager.run_vectorization(ngram_range=ngram_range, max_features=max_features)
        spinner.succeed('Vectorization complete!')
        # Report shape if available
        try:
            print(f"Feature matrix shape: {getattr(X, 'shape', None)}")
        except Exception:
            pass
    except Exception as e:
        spinner.fail(f'Vectorization failed: {e}')
    return


modelmanager = ModelManager(datamanager)


async def train_model_menu():
    print("\nChoose a model to train:")
    print("1. Random Forest")
    print("2. Stochastic Gradient Descent (SGD)")

    choice = int(input("Enter choice: "))
    if choice == 1:
        modelmanager.train_model("random_forest")
    elif choice == 2:
        modelmanager.train_model("sgd")
    else:
        print("Invalid choice.")
        return

    save = input("Do you want to save this model? (y/n): ").lower()
    if save == "y":
        modelmanager.save_model()

async def load_saved_model():
    model_name = input("Enter the model name (e.g., RandomForest or SGDClassifier): ")
    modelmanager.load_model(model_name)

async def continue_training_model():
    modelmanager.continue_training()
    save = input("Save the updated model? (y/n): ").lower()
    if save == "y":
        modelmanager.save_model()


async def menu():
    options = {
        1: ("Load the dataset", load_the_dataset),
        2: ("Get information about the dataset", get_info_about_dataset),
        3: ("Preprocess data", preprocess_data),
        4: ("Train a classical ML model", train_model_menu),
        5: ("Load a saved model", load_saved_model),
        6: ("Continue training a loaded model", continue_training_model)
    }

    while True:
        clear_console()
        print("--- Phishing Email Detection ---")
        for key, (desc, _) in options.items():
            print(f"[{key}] {desc}")
        print("[0] Exit")

        choice = ask_for_integer("Choose", default=None, min=0, max=len(options))
        if choice is None:
            continue

        if choice == 0:
            print("Bye!")
            break

        _, action = options[choice]
        clear_console()
        await action()
        input("\nPress Enter to continue...")
